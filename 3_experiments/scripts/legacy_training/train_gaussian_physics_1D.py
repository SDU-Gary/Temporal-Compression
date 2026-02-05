#!/usr/bin/env python3
"""Training script for 1D intensity modulation Gaussian-Physics Compression.

Trains the simplified 1D model (GaussianPhysicsCompression1D) on sinusoidal
intensity modulation dataset to validate method generalizability beyond TOD.

Expected results (K=30, rank=8):
- MAE < 0.03 (better than 5D due to simpler signal)
- Compression ratio: ~15.6× (7,140 params vs 111,132 naive)
- Parameter reduction: 14% fewer than 5D model (7,140 vs 8,340)
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import json
from datetime import datetime
import matplotlib.pyplot as plt

from models.gaussian_physics_1D import GaussianPhysicsCompression1D
from data.intensity_modulation_dataset import create_dataloaders
from training import BatchAdapter, GaussianPhysicsTrainer


def charbonnier_loss(pred, target, epsilon=0.001):
    """Charbonnier loss (smooth L1).

    More robust to outliers than MSE.

    Args:
        pred: [B, ...] predictions
        target: [B, ...] targets
        epsilon: Smoothing parameter

    Returns:
        loss: Scalar loss value
    """
    diff = pred - target
    loss = torch.sqrt(diff ** 2 + epsilon ** 2)
    return torch.mean(loss)


def mae_loss(pred, target):
    """Mean absolute error."""
    return torch.mean(torch.abs(pred - target))


def rmse_loss(pred, target):
    """Root mean squared error."""
    return torch.sqrt(torch.mean((pred - target) ** 2))


def temporal_smoothness_l1(model, probe_positions, intensities, top_k=3):
    """Temporal smoothness regularization (L1 second-order difference).

    Encourages smooth temporal transitions.

    Args:
        model: GaussianPhysicsCompression1D model
        probe_positions: [P, 3] probe positions
        intensities: [M] sorted intensity values
        top_k: Number of nearest Gaussians

    Returns:
        loss: Scalar temporal smoothness loss
    """
    M = len(intensities)
    if M < 3:
        return torch.tensor(0.0, device=probe_positions.device)

    # Compute SH for all moments
    sh_preds = []
    for intensity_val in intensities:
        intensity_tensor = torch.full((len(probe_positions), 1), intensity_val, device=probe_positions.device)
        sh_pred = model(probe_positions, intensity_tensor, top_k=top_k)  # [P, 27]
        sh_preds.append(sh_pred)

    sh_preds = torch.stack(sh_preds, dim=1)  # [P, M, 27]

    # Second-order difference: |sh(t-1) - 2*sh(t) + sh(t+1)|
    diff = sh_preds[:, :-2, :] - 2 * sh_preds[:, 1:-1, :] + sh_preds[:, 2:, :]  # [P, M-2, 27]
    temporal_loss = torch.mean(torch.abs(diff))

    return temporal_loss


def train_epoch(model, train_loader, optimizer, lambda_temporal, device, top_k=3):
    """Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        lambda_temporal: Temporal smoothness weight
        device: Device to use
        top_k: Number of nearest Gaussians

    Returns:
        metrics: Dict of training metrics
    """
    model.train()

    total_loss_accum = 0.0
    recon_loss_accum = 0.0
    temporal_loss_accum = 0.0
    num_batches = 0

    for batch in train_loader:
        positions = batch['probe_position'].to(device)  # [B, 3]
        intensity = batch['intensity'].to(device)  # [B, 1]
        sh_gt = batch['sh_coeffs'].to(device)  # [B, 27]

        # Forward pass
        sh_pred = model(positions, intensity, top_k=top_k)  # [B, 27]

        # Reconstruction loss (Charbonnier)
        recon_loss = charbonnier_loss(sh_pred, sh_gt)

        # Total loss (temporal regularization computed per-epoch, not per-batch)
        total_loss = recon_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate metrics
        total_loss_accum += total_loss.item()
        recon_loss_accum += recon_loss.item()
        num_batches += 1

    # Compute temporal smoothness (once per epoch, using training set probes)
    # This is expensive, so we do it only occasionally
    # For now, skip temporal loss to speed up training
    temporal_loss_val = 0.0

    metrics = {
        'total': total_loss_accum / num_batches,
        'recon': recon_loss_accum / num_batches,
        'temporal': temporal_loss_val
    }

    return metrics


def validate_epoch(model, val_loader, device, top_k=3):
    """Validate for one epoch.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        device: Device to use
        top_k: Number of nearest Gaussians

    Returns:
        metrics: Dict of validation metrics
    """
    model.eval()

    mae_accum = 0.0
    rmse_accum = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            positions = batch['probe_position'].to(device)
            intensity = batch['intensity'].to(device)
            sh_gt = batch['sh_coeffs'].to(device)

            # Forward pass
            sh_pred = model(positions, intensity, top_k=top_k)

            # Metrics
            mae = mae_loss(sh_pred, sh_gt)
            rmse = rmse_loss(sh_pred, sh_gt)

            mae_accum += mae.item()
            rmse_accum += rmse.item()
            num_batches += 1

    metrics = {
        'mae': mae_accum / num_batches,
        'rmse': rmse_accum / num_batches
    }

    return metrics


def plot_training_curve(train_losses, val_losses, output_dir):
    """Plot training curves."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(train_losses['total']) + 1)

    # Loss curves
    axes[0].plot(epochs, train_losses['total'], label='Train Total', alpha=0.7)
    axes[0].plot(epochs, train_losses['recon'], label='Train Recon', alpha=0.7)
    axes[0].plot(epochs, val_losses['mae'], label='Val MAE', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss / MAE')
    axes[0].set_title('Training Progress')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_yscale('log')

    # Validation metrics
    axes[1].plot(epochs, val_losses['mae'], label='MAE', alpha=0.7)
    axes[1].plot(epochs, val_losses['rmse'], label='RMSE', alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Error')
    axes[1].set_title('Validation Metrics')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curve.png', dpi=150)
    print(f"Training curve saved: {output_dir / 'training_curve.png'}")
    plt.close()


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use absolute paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent.parent.parent  # Go up to /home/kyrie/毕设
    data_root = project_root / 'data_generation/output/intensity_modulation_343'
    output_dir = project_root / 'multi_time_compression/PG-GCPL/experiments/experiment_groups/group2_intensity_modulation/01_baseline_training/K30_r8'

    # Hyperparameters
    num_gaussians = 30
    rank = 8
    top_k = 3
    lr = 1e-3
    lambda_temporal = 0.001
    num_epochs = 2000
    batch_size = 256

    print("=" * 80)
    print("1D Intensity Modulation Training: Gaussian-Physics Compression")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data: {data_root}")
    print(f"Output: {output_dir}")
    print(f"\nHyperparameters:")
    print(f"  Gaussians (K): {num_gaussians}")
    print(f"  Rank (r): {rank}")
    print(f"  Top-k: {top_k}")
    print(f"  Learning rate: {lr}")
    print(f"  Lambda temporal: {lambda_temporal}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\n[1/5] Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=str(data_root),
        batch_size=batch_size,
        num_workers=4,
        train_ratio=0.7,
        val_ratio=0.15
    )

    # Get dataset info for initialization
    train_dataset = train_loader.dataset
    probe_positions = train_dataset.get_full_probe_positions(normalized=True)

    # Initialize model
    print(f"\n[2/5] Initializing model...")
    model = GaussianPhysicsCompression1D(
        num_gaussians=num_gaussians,
        rank=rank,
        sh_dim=27
    ).to(device)

    # K-Means initialization
    print(f"\n[3/5] Initializing Gaussians with K-Means...")
    model.initialize_from_probes(probe_positions, method='kmeans', device=device)

    # Calculate compression ratio
    compression_ratio = model.get_compression_ratio(
        num_probes=343,
        num_moments=12
    )
    print(f"\nCompression analysis:")
    print(f"  Naive params: {343 * 12 * 27:,}")
    print(f"  Compressed params: {model.num_params():,}")
    print(f"  Compression ratio: {compression_ratio:.2f}×")

    # Training loop
    print(f"\n[4/5] Training for {num_epochs} epochs...")
    adapter = BatchAdapter(params_key='intensity')
    trainer = GaussianPhysicsTrainer(
        model=model,
        device=device,
        adapter=adapter,
        lr=lr,
        recon_loss="charbonnier",
        temporal_weight=lambda_temporal,
        temporal_loss_fn=lambda _model: torch.tensor(0.0, device=device),
        top_k=top_k,
        grad_clip=1.0,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        output_dir=output_dir,
        save_best=True,
        best_metric="mae",
        log_every=50,
    )

    train_losses = history["train"]
    val_losses = history["val"]
    best_val_mae = min(val_losses["mae"]) if val_losses["mae"] else float("inf")
    best_epoch = val_losses["mae"].index(best_val_mae) + 1 if val_losses["mae"] else 0

    print(f"\nTraining complete!")
    print(f"Best validation MAE: {best_val_mae:.6f} at epoch {best_epoch}")

    # Test evaluation
    print(f"\n[5/5] Evaluating on test set...")
    best_ckpt = output_dir / 'best_model.pt'
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt)['model_state_dict'])
    test_metrics = trainer.validate_epoch(test_loader)
    print(f"Test MAE: {test_metrics['mae']:.6f}")
    print(f"Test RMSE: {test_metrics['rmse']:.6f}")

    # Plot training curves
    plot_training_curve(train_losses, val_losses, output_dir)

    # Save final results
    results = {
        'hyperparameters': {
            'num_gaussians': num_gaussians,
            'rank': rank,
            'top_k': top_k,
            'learning_rate': lr,
            'lambda_temporal': lambda_temporal,
            'num_epochs': num_epochs,
            'batch_size': batch_size
        },
        'compression': {
            'naive_params': 343 * 12 * 27,
            'compressed_params': model.num_params(),
            'compression_ratio': compression_ratio
        },
        'best_epoch': best_epoch,
        'best_val_mae': best_val_mae,
        'test_metrics': test_metrics,
        'training_time': datetime.now().isoformat()
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'results.json'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
