"""Training script for dual-gaussian FBT model.

Trains on transmission tensor data with Charbonnier loss.
Validates on held-out probes to test spatial generalization.

Usage:
    python train_dual_gaussian_fbt.py --data ../../data_generation/output/transfer_tensor_validation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from models.dual_gaussian_fbt import DualGaussianFBT
from data.transfer_tensor_dataset import create_dataloaders


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (robust L2 loss).

    L = sqrt(x^2 + epsilon^2)

    More robust to outliers than MSE.
    """

    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.epsilon ** 2)
        return loss.mean()


class FBTTrainer:
    """Trainer for dual-gaussian FBT model."""

    def __init__(
        self,
        model: DualGaussianFBT,
        train_loader,
        val_loader,
        lr: float = 1e-3,
        device: str = 'cuda',
        log_dir: str = 'experiments/fbt_prototype'
    ):
        """Initialize trainer.

        Args:
            model: DualGaussianFBT model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            lr: Learning rate
            device: Device ('cuda' or 'cpu')
            log_dir: Directory for tensorboard logs and checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss and optimizer
        self.criterion = CharbonnierLoss(epsilon=1e-3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir / 'tensorboard')

        # Metrics
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            avg_loss: Average training loss
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:3d}")

        for batch in pbar:
            # Move to device
            probe_pos = batch['probe_position'].to(self.device)  # [B, 3]
            light_pos = batch['light_position'].to(self.device)  # [B, 3]
            sh_target = batch['sh_coeffs'].to(self.device)  # [B, 27]

            # Forward
            sh_pred = self.model(probe_pos, light_pos)  # [B, 27]

            # Loss
            loss = self.criterion(sh_pred, sh_target)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging
            epoch_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / num_batches
        return avg_loss

    def validate(self, epoch: int) -> dict:
        """Validate on held-out probes.

        Args:
            epoch: Current epoch number

        Returns:
            metrics: dict with validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        mae_total = 0.0
        rmse_total = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                probe_pos = batch['probe_position'].to(self.device)
                light_pos = batch['light_position'].to(self.device)
                sh_target = batch['sh_coeffs'].to(self.device)

                # Forward
                sh_pred = self.model(probe_pos, light_pos)

                # Metrics
                loss = self.criterion(sh_pred, sh_target)
                mae = torch.abs(sh_pred - sh_target).mean()
                rmse = torch.sqrt(((sh_pred - sh_target) ** 2).mean())

                val_loss += loss.item()
                mae_total += mae.item()
                rmse_total += rmse.item()
                num_batches += 1

        metrics = {
            'val_loss': val_loss / num_batches,
            'val_mae': mae_total / num_batches,
            'val_rmse': rmse_total / num_batches
        }

        return metrics

    def train(self, num_epochs: int, save_every: int = 50):
        """Full training loop.

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"  Device: {self.device}")
        print(f"  Train samples: {len(self.train_loader.dataset)}")
        print(f"  Val samples: {len(self.val_loader.dataset)}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters())}")

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Validate
            val_metrics = self.validate(epoch)
            self.val_losses.append(val_metrics['val_loss'])

            # Logging
            print(f"Epoch {epoch:3d}/{num_epochs}: "
                  f"train_loss={train_loss:.4f}, "
                  f"val_loss={val_metrics['val_loss']:.4f}, "
                  f"val_mae={val_metrics['val_mae']:.4f}, "
                  f"val_rmse={val_metrics['val_rmse']:.4f}")

            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_metrics['val_loss'], epoch)
            self.writer.add_scalar('Metrics/val_mae', val_metrics['val_mae'], epoch)
            self.writer.add_scalar('Metrics/val_rmse', val_metrics['val_rmse'], epoch)

            # Save best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(epoch, is_best=True)

            # Periodic checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, is_best=False)

        # Final checkpoint
        self.save_checkpoint(num_epochs, is_best=False)

        print(f"\nTraining complete!")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")
        print(f"  Checkpoints saved to: {self.log_dir / 'checkpoints'}")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint.

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = self.log_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }

        # Save latest
        torch.save(checkpoint, checkpoint_dir / 'latest.pth')

        # Save periodic
        if epoch % 100 == 0:
            torch.save(checkpoint, checkpoint_dir / f'epoch_{epoch:04d}.pth')

        # Save best
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best.pth')
            print(f"  Saved best model (val_loss={self.best_val_loss:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Train dual-gaussian FBT model")
    parser.add_argument('--data', type=str, required=True,
                        help='Data directory with transfer_tensor.npz')
    parser.add_argument('--output', type=str, default='experiments/fbt_prototype',
                        help='Output directory for logs and checkpoints')

    # Model hyperparameters
    parser.add_argument('--num_probe_gaussians', type=int, default=20,
                        help='Number of probe space Gaussians (default: 20)')
    parser.add_argument('--num_light_gaussians', type=int, default=10,
                        help='Number of light space Gaussians (default: 10)')
    parser.add_argument('--tucker_rank', type=int, default=3,
                        help='Tucker decomposition rank (default: 3)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers (default: 4)')

    # Dataset splits
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu, default: cuda)')

    args = parser.parse_args()

    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("=" * 80)
    print("Dual-Gaussian FBT Training")
    print("=" * 80)

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=args.data,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        normalize=True
    )

    # Get all positions for Gaussian initialization
    train_dataset = train_loader.dataset
    all_probe_positions = train_dataset.get_all_probe_positions()
    all_light_positions = train_dataset.get_all_light_positions()

    # Create model
    print("\nInitializing model...")
    model = DualGaussianFBT(
        num_probe_gaussians=args.num_probe_gaussians,
        num_light_gaussians=args.num_light_gaussians,
        tucker_rank=args.tucker_rank
    )

    # Initialize Gaussians from data
    model.initialize_from_data(all_probe_positions, all_light_positions)

    # Print model info
    num_params, size_mb = model.get_model_size()
    compression = model.get_compression_ratio(
        num_probes=len(all_probe_positions),
        num_lights=len(all_light_positions)
    )

    print(f"\nModel configuration:")
    print(f"  Probe Gaussians: {args.num_probe_gaussians}")
    print(f"  Light Gaussians: {args.num_light_gaussians}")
    print(f"  Tucker rank: {args.tucker_rank}")
    print(f"  Total parameters: {num_params}")
    print(f"  Model size: {size_mb:.2f} MB")
    print(f"  Compression ratio: {compression:.1f}×")

    # Create trainer
    trainer = FBTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        device=device,
        log_dir=args.output
    )

    # Save config
    config = vars(args)
    config['num_params'] = num_params
    config['model_size_mb'] = size_mb
    config['compression_ratio'] = compression
    config['timestamp'] = datetime.now().isoformat()

    with open(Path(args.output) / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Train
    trainer.train(num_epochs=args.epochs, save_every=100)

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
