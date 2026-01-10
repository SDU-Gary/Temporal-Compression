"""Trainer for multi-temporal lighting compression model.

Implements:
    1. Dual optimizer setup (Adan for Gaussians, Adam for MLPs)
    2. Training loop with validation
    3. Checkpoint management
    4. TensorBoard logging
    5. Learning rate scheduling
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional, Dict
import json
from tqdm import tqdm

from models.full_model import MultiTimeCompressionModel
from training.losses import CombinedLoss
from training.metrics import MetricsTracker, compute_all_metrics
from utils.logger import TrainingLogger
from utils.config import Config


class Trainer:
    """Trainer for multi-temporal compression model.

    Manages training loop, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: MultiTimeCompressionModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Config,
        output_dir: Path,
        device: str = 'cuda'
    ):
        """Initialize trainer.

        Args:
            model: Compression model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration
            output_dir: Directory for checkpoints and logs
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device

        # Create output directories
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = self.output_dir / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = TrainingLogger(self.log_dir, name='training')

        # Setup TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))

        # Setup loss function
        loss_config = config.training.get('loss', {})
        self.criterion = CombinedLoss(
            w_reconstruction=loss_config.get('reconstruction', 1.0),
            w_temporal_smooth=loss_config.get('temporal_smooth', 0.01),
            w_energy=loss_config.get('energy_conservation', 0.0),
            temporal_norm=loss_config.get('temporal_norm', 'l2')
        )

        # Training config (must be set before _setup_scheduler)
        self.num_steps = config.training.get('num_steps', 10000)
        self.val_interval = config.training.get('val_interval', 500)
        self.save_interval = config.training.get('save_interval', 1000)
        self.log_interval = config.training.get('log_interval', 100)
        self.max_checkpoints = config.training.get('max_checkpoints', 3)

        # Setup optimizers
        self._setup_optimizers()

        # Setup learning rate scheduler
        self._setup_scheduler()

        # Training state
        self.current_step = 0
        self.current_epoch = 0
        self.best_val_psnr = 0.0
        self.best_val_loss = float('inf')

        # Early stopping
        early_stop_config = config.training.get('early_stopping', {})
        self.early_stopping_enabled = early_stop_config.get('enabled', False)
        self.early_stopping_patience = early_stop_config.get('patience', 10)
        self.early_stopping_min_delta = early_stop_config.get('min_delta', 1e-4)
        self.early_stopping_counter = 0

        # Metrics tracking
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()

    def _setup_optimizers(self):
        """Setup dual optimizers for Gaussians and MLPs."""
        optimizer_config = self.config.training.get('optimizer', {})

        # Optimizer for Gaussian parameters
        gaussian_config = optimizer_config.get('gaussians', {})
        gaussian_lr = gaussian_config.get('lr', 0.01)
        gaussian_type = gaussian_config.get('type', 'Adam')

        gaussian_params = list(self.model.gaussian_mixture.parameters())

        if gaussian_type.lower() == 'adan':
            try:
                from adan import Adan
                self.optimizer_gaussian = Adan(
                    gaussian_params,
                    lr=gaussian_lr,
                    betas=gaussian_config.get('betas', (0.98, 0.92, 0.99)),
                    eps=gaussian_config.get('eps', 1e-8),
                    weight_decay=gaussian_config.get('weight_decay', 0.02)
                )
            except ImportError:
                self.logger.warning("Adan optimizer not available, falling back to Adam")
                self.optimizer_gaussian = torch.optim.Adam(
                    gaussian_params,
                    lr=gaussian_lr,
                    betas=(0.9, 0.999),
                    weight_decay=0.0
                )
        else:
            self.optimizer_gaussian = torch.optim.Adam(
                gaussian_params,
                lr=gaussian_lr,
                betas=gaussian_config.get('betas', (0.9, 0.999)),
                weight_decay=gaussian_config.get('weight_decay', 0.0)
            )

        # Optimizer for MLP parameters
        mlp_config = optimizer_config.get('mlp', {})
        mlp_lr = mlp_config.get('lr', 0.001)

        mlp_params = (
            list(self.model.temporal_mlp.parameters()) +
            list(self.model.decoder_mlp.parameters())
        )

        self.optimizer_mlp = torch.optim.Adam(
            mlp_params,
            lr=mlp_lr,
            betas=mlp_config.get('betas', (0.9, 0.999)),
            weight_decay=mlp_config.get('weight_decay', 0.0)
        )

        # Store base learning rates for warmup
        self.base_lr_gaussian = gaussian_lr
        self.base_lr_mlp = mlp_lr

        self.logger.info(f"Setup optimizers:")
        self.logger.info(f"  Gaussian: {type(self.optimizer_gaussian).__name__} (lr={gaussian_lr})")
        self.logger.info(f"  MLP: {type(self.optimizer_mlp).__name__} (lr={mlp_lr})")

    def _setup_scheduler(self):
        """Setup learning rate schedulers with cosine annealing and warmup."""
        scheduler_config = self.config.training.get('scheduler', {})

        if not scheduler_config.get('enabled', True):
            self.scheduler_gaussian = None
            self.scheduler_mlp = None
            self.warmup_steps = 0
            return

        self.warmup_steps = scheduler_config.get('warmup_steps', 1000)

        # Cosine annealing for after warmup
        self.scheduler_gaussian = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_gaussian,
            T_max=self.num_steps - self.warmup_steps,
            eta_min=scheduler_config.get('min_lr', 1e-6)
        )

        # Cosine annealing for after warmup
        self.scheduler_mlp = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_mlp,
            T_max=self.num_steps - self.warmup_steps,
            eta_min=scheduler_config.get('min_lr', 1e-6)
        )

    def _update_learning_rate(self):
        """Update learning rate with warmup and cosine scheduling.

        Warmup: linearly increase from 0 to base_lr over warmup_steps
        After warmup: cosine annealing from base_lr to min_lr
        """
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr_scale = (self.current_step + 1) / self.warmup_steps
            for param_group in self.optimizer_gaussian.param_groups:
                param_group['lr'] = self.base_lr_gaussian * lr_scale
            for param_group in self.optimizer_mlp.param_groups:
                param_group['lr'] = self.base_lr_mlp * lr_scale
        elif self.scheduler_gaussian is not None:
            # Cosine annealing after warmup
            self.scheduler_gaussian.step()
            self.scheduler_mlp.step()

    def train_step(self, batch: Dict) -> Dict:
        """Execute single training step.

        Args:
            batch: Dictionary with 'position', 'sh_gt', 'sun_dirs'

        Returns:
            metrics: Dictionary with loss values
        """
        self.model.train()

        # Move data to device
        positions = batch['position'].to(self.device)  # [B, 3]
        sh_gt = batch['sh_gt'].to(self.device)  # [B, T, 27]
        sun_dirs_batch = batch['sun_dirs'].to(self.device)  # [B, T, 3]

        # All samples share the same sun directions (same time moments)
        # Extract just one copy for the model
        sun_dirs = sun_dirs_batch[0]  # [T, 3]

        # Zero gradients
        self.optimizer_gaussian.zero_grad()
        self.optimizer_mlp.zero_grad()

        # Forward pass
        sh_pred = self.model(positions, sun_dirs)  # [B, T, 27]

        # Compute loss
        loss, loss_dict = self.criterion(sh_pred, sh_gt)

        # Backward pass
        loss.backward()

        # Gradient clipping (optional)
        grad_clip = self.config.training.get('grad_clip', None)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

        # Optimizer step
        self.optimizer_gaussian.step()
        self.optimizer_mlp.step()

        # Learning rate scheduling
        self._update_learning_rate()

        # Compute additional metrics
        with torch.no_grad():
            metrics = compute_all_metrics(sh_pred, sh_gt, max_val=10.0)
            metrics.update(loss_dict)

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict:
        """Run validation on validation set.

        Returns:
            metrics: Dictionary with averaged metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        self.val_metrics.reset()

        for batch in self.val_loader:
            # Move data to device
            positions = batch['position'].to(self.device)
            sh_gt = batch['sh_gt'].to(self.device)
            sun_dirs_batch = batch['sun_dirs'].to(self.device)

            # All samples share the same sun directions
            sun_dirs = sun_dirs_batch[0]  # [T, 3]

            # Forward pass
            sh_pred = self.model(positions, sun_dirs)

            # Compute metrics
            batch_metrics = compute_all_metrics(sh_pred, sh_gt, max_val=10.0)

            # Update tracker
            self.val_metrics.update(batch_metrics, count=positions.shape[0])

        # Get averaged metrics
        avg_metrics = self.val_metrics.get_average()

        return avg_metrics

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'step': self.current_step,
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_gaussian_state_dict': self.optimizer_gaussian.state_dict(),
            'optimizer_mlp_state_dict': self.optimizer_mlp.state_dict(),
            'best_val_psnr': self.best_val_psnr,
            'config': self.config.config
        }

        if self.scheduler_gaussian is not None:
            checkpoint['scheduler_gaussian_state_dict'] = self.scheduler_gaussian.state_dict()
            checkpoint['scheduler_mlp_state_dict'] = self.scheduler_mlp.state_dict()

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model: {best_path}")

        # Keep only last N checkpoints (excluding best)
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the latest N."""
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_step_*.pt'),
            key=lambda x: int(x.stem.split('_')[-1])
        )

        # Keep only last N checkpoints
        if len(checkpoints) > self.max_checkpoints:
            for ckpt in checkpoints[:-self.max_checkpoints]:
                ckpt.unlink()
                self.logger.info(f"Removed old checkpoint: {ckpt.name}")

    def load_checkpoint(self, checkpoint_path: Path | str):
        """Load checkpoint and resume training.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_gaussian.load_state_dict(checkpoint['optimizer_gaussian_state_dict'])
        self.optimizer_mlp.load_state_dict(checkpoint['optimizer_mlp_state_dict'])

        if self.scheduler_gaussian is not None and 'scheduler_gaussian_state_dict' in checkpoint:
            self.scheduler_gaussian.load_state_dict(checkpoint['scheduler_gaussian_state_dict'])
            self.scheduler_mlp.load_state_dict(checkpoint['scheduler_mlp_state_dict'])

        # Load training state
        self.current_step = checkpoint['step']
        self.current_epoch = checkpoint['epoch']
        self.best_val_psnr = checkpoint.get('best_val_psnr', 0.0)

        self.logger.info(f"Loaded checkpoint from step {self.current_step}")

    def train(self):
        """Main training loop."""
        self.logger.info("=" * 70)
        self.logger.info("Starting training")
        self.logger.info("=" * 70)

        # Log model info
        num_probes = len(self.train_loader.dataset)
        num_moments = self.train_loader.dataset[0]['sh_gt'].shape[0]
        self.logger.log_model_info(self.model, num_probes, num_moments)

        # Log configuration
        self.logger.log_config(self.config.config)

        # Create infinite data loader iterator
        from itertools import cycle
        data_iterator = cycle(self.train_loader)

        # Single progress bar for all training steps
        pbar = tqdm(
            range(self.current_step, self.num_steps),
            desc="Training",
            initial=self.current_step,
            total=self.num_steps
        )

        # Reset training metrics
        self.train_metrics.reset()

        for step in pbar:
            # Get next batch
            batch = next(data_iterator)

            # Training step
            metrics = self.train_step(batch)

            # Update metrics tracker
            self.train_metrics.update(metrics, count=batch['position'].shape[0])

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['total']:.4f}",
                'psnr': f"{metrics['psnr']:.2f}",
                'lr': f"{self.optimizer_gaussian.param_groups[0]['lr']:.2e}"
            })

            # Logging
            if self.current_step % self.log_interval == 0:
                avg_metrics = self.train_metrics.get_average()

                # Log to console
                self.logger.log_step(self.current_step, avg_metrics, prefix="Train")

                # Log to TensorBoard
                for name, value in avg_metrics.items():
                    self.writer.add_scalar(f'train/{name}', value, self.current_step)

                # Log learning rates
                self.writer.add_scalar(
                    'lr/gaussian',
                    self.optimizer_gaussian.param_groups[0]['lr'],
                    self.current_step
                )
                self.writer.add_scalar(
                    'lr/mlp',
                    self.optimizer_mlp.param_groups[0]['lr'],
                    self.current_step
                )

                # Reset metrics
                self.train_metrics.reset()

            # Validation
            if self.current_step % self.val_interval == 0 and self.val_loader is not None:
                val_metrics = self.validate()

                # Log validation metrics
                self.logger.log_step(self.current_step, val_metrics, prefix="Val")

                for name, value in val_metrics.items():
                    self.writer.add_scalar(f'val/{name}', value, self.current_step)

                # Check if best model
                val_psnr = val_metrics.get('psnr', 0.0)
                val_loss = val_metrics.get('mse', float('inf'))

                is_best = val_psnr > self.best_val_psnr
                if is_best:
                    self.best_val_psnr = val_psnr
                    self.logger.info(f"New best PSNR: {val_psnr:.2f} dB")

                # Early stopping check (based on validation loss)
                if self.early_stopping_enabled:
                    if val_loss < (self.best_val_loss - self.early_stopping_min_delta):
                        # Improvement
                        self.best_val_loss = val_loss
                        self.early_stopping_counter = 0
                        self.logger.info(f"Validation loss improved to {val_loss:.6f}")
                    else:
                        # No improvement
                        self.early_stopping_counter += 1
                        self.logger.warning(
                            f"No validation improvement for {self.early_stopping_counter}/{self.early_stopping_patience} checks"
                        )

                        if self.early_stopping_counter >= self.early_stopping_patience:
                            self.logger.warning("=" * 70)
                            self.logger.warning("EARLY STOPPING TRIGGERED")
                            self.logger.warning(f"No improvement for {self.early_stopping_patience} validation checks")
                            self.logger.warning(f"Best validation loss: {self.best_val_loss:.6f}")
                            self.logger.warning(f"Best validation PSNR: {self.best_val_psnr:.2f} dB")
                            self.logger.warning("=" * 70)

                            # Save final checkpoint
                            self.save_checkpoint('checkpoint_early_stopped.pt')

                            # Exit training loop
                            break

                # Save checkpoint
                if self.current_step % self.save_interval == 0:
                    self.save_checkpoint(
                        f'checkpoint_step_{self.current_step}.pt',
                        is_best=is_best
                    )

            # Save regular checkpoint
            elif self.current_step % self.save_interval == 0:
                self.save_checkpoint(f'checkpoint_step_{self.current_step}.pt')

            self.current_step += 1

        # Final validation
        if self.val_loader is not None:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("Final Validation")
            self.logger.info("=" * 70)

            val_metrics = self.validate()
            self.logger.log_step(self.current_step, val_metrics, prefix="Final Val")

        # Save final checkpoint
        self.save_checkpoint('checkpoint_final.pt')

        # Close TensorBoard writer
        self.writer.close()

        self.logger.info("\n" + "=" * 70)
        self.logger.info("Training complete!")
        self.logger.info(f"Best validation PSNR: {self.best_val_psnr:.2f} dB")
        self.logger.info("=" * 70)
