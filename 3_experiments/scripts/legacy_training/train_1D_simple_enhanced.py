#!/usr/bin/env python3
"""Enhanced training script for 1D intensity modulation with COMPLETE logging.

ENHANCEMENTS:
1. Complete training log saved to file (every epoch)
2. Intermediate checkpoints (every 100 epochs)
3. Detailed metrics tracking (loss components, gradients, Gaussian coverage)
4. Real-time visualization updates
5. Comprehensive experiment metadata
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

from models.gaussian_physics_1D import GaussianPhysicsCompression1D
from data.intensity_modulation_dataset import create_dataloaders


def charbonnier_loss(pred, target, epsilon=0.001):
    """Charbonnier loss (smooth L1)."""
    diff = pred - target
    loss = torch.sqrt(diff ** 2 + epsilon ** 2)
    return torch.mean(loss)


def mae_loss(pred, target):
    """Mean absolute error."""
    return torch.mean(torch.abs(pred - target))


def rmse_loss(pred, target):
    """Root mean squared error."""
    return torch.sqrt(torch.mean((pred - target) ** 2))


def train_epoch(model, train_loader, optimizer, device, top_k=3, log_gradients=False):
    """Train for one epoch with detailed logging.

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to use
        top_k: Number of nearest Gaussians
        log_gradients: Whether to compute gradient statistics

    Returns:
        metrics: Dict with training metrics and diagnostics
    """
    model.train()

    total_loss_accum = 0.0
    recon_loss_accum = 0.0
    num_batches = 0

    # Gradient statistics
    grad_norms = []

    for batch in train_loader:
        positions = batch['probe_position'].to(device)  # [B, 3]
        intensity = batch['intensity'].to(device)  # [B, 1]
        sh_gt = batch['sh_coeffs'].to(device)  # [B, 27]

        # Forward pass
        sh_pred = model(positions, intensity, top_k=top_k)  # [B, 27]

        # Reconstruction loss (Charbonnier)
        recon_loss = charbonnier_loss(sh_pred, sh_gt)
        total_loss = recon_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Compute gradient norm before clipping
        if log_gradients:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate metrics
        total_loss_accum += total_loss.item()
        recon_loss_accum += recon_loss.item()
        num_batches += 1

    metrics = {
        'total': total_loss_accum / num_batches,
        'recon': recon_loss_accum / num_batches,
    }

    if log_gradients and grad_norms:
        metrics['grad_norm_mean'] = np.mean(grad_norms)
        metrics['grad_norm_max'] = np.max(grad_norms)

    return metrics


def validate_epoch(model, val_loader, device, top_k=3):
    """Validate for one epoch."""
    model.eval()

    mae_accum = 0.0
    rmse_accum = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []

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

            all_preds.append(sh_pred.cpu().numpy())
            all_targets.append(sh_gt.cpu().numpy())

    # Compute additional statistics
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Per-coefficient statistics
    per_coeff_mae = np.mean(np.abs(all_preds - all_targets), axis=0)

    metrics = {
        'mae': mae_accum / num_batches,
        'rmse': rmse_accum / num_batches,
        'per_coeff_mae_mean': per_coeff_mae.mean(),
        'per_coeff_mae_max': per_coeff_mae.max(),
        'pred_range': [all_preds.min(), all_preds.max()],
        'target_range': [all_targets.min(), all_targets.max()]
    }

    return metrics


def compute_gaussian_coverage(model, probe_positions, device, top_k=3):
    """Compute percentage of probes covered by Gaussians.

    Args:
        model: Trained model
        probe_positions: [P, 3] probe positions (normalized)
        device: Device
        top_k: Number of nearest Gaussians

    Returns:
        coverage: Float percentage of probes with non-zero Gaussian weights
    """
    model.eval()
    with torch.no_grad():
        positions_tensor = torch.from_numpy(probe_positions).float().to(device)

        # Use dummy intensity for coverage test
        dummy_intensity = torch.ones((len(positions_tensor), 1), device=device)

        # Compute Gaussian weights
        weights, _ = model.compute_gaussian_weights(positions_tensor, top_k=top_k)

        # Check coverage (probes with at least one non-zero weight)
        coverage = (weights.sum(dim=1) > 1e-6).float().mean().item()

    return coverage * 100.0  # Return as percentage


def plot_training_curve(train_losses, val_losses, output_dir, current_epoch=None):
    """Plot training curves with enhanced visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    epochs = range(1, len(train_losses['total']) + 1)

    # Loss curves
    axes[0, 0].plot(epochs, train_losses['total'], label='Train Total', alpha=0.7, linewidth=2)
    axes[0, 0].plot(epochs, train_losses['recon'], label='Train Recon', alpha=0.7, linewidth=2)
    if current_epoch:
        axes[0, 0].axvline(current_epoch, color='red', linestyle='--', alpha=0.5, label='Current')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_yscale('log')

    # Validation metrics
    axes[0, 1].plot(epochs, val_losses['mae'], label='MAE', alpha=0.7, linewidth=2, color='orange')
    axes[0, 1].plot(epochs, val_losses['rmse'], label='RMSE', alpha=0.7, linewidth=2, color='green')
    if current_epoch:
        axes[0, 1].axvline(current_epoch, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Error')
    axes[0, 1].set_title('Validation Metrics')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_yscale('log')

    # Gradient norms (if available) - NOTE: sparse data, only logged at specific intervals
    if 'grad_norm_mean' in train_losses and len(train_losses['grad_norm_mean']) > 0:
        # Gradient data is sparse, so we cannot plot against all epochs
        # Instead show text summary
        axes[1, 0].text(0.5, 0.5,
                       f"Gradient Statistics\n"
                       f"(logged at intervals)\n\n"
                       f"Mean norm: {train_losses['grad_norm_mean'][-1]:.4f}\n"
                       f"Max norm: {train_losses['grad_norm_max'][-1]:.4f}\n"
                       f"Samples: {len(train_losses['grad_norm_mean'])}",
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Gradient Statistics')

    # Per-coefficient MAE (if available)
    if 'per_coeff_mae_mean' in val_losses and len(val_losses['per_coeff_mae_mean']) > 0:
        axes[1, 1].plot(epochs, val_losses['per_coeff_mae_mean'], label='Mean', alpha=0.7, linewidth=2)
        axes[1, 1].plot(epochs, val_losses['per_coeff_mae_max'], label='Max', alpha=0.7, linewidth=2)
        if current_epoch:
            axes[1, 1].axvline(current_epoch, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].set_title('Per-Coefficient MAE')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curve.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use absolute paths - UPDATED to use simplified geometry dataset
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent.parent.parent  # Go up to /home/kyrie/毕设
    data_root = project_root / 'data_generation/output/intensity_modulation_343_simple'  # FIXED PATH

    # Create timestamped experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = project_root / 'multi_time_compression/PG-GCPL/experiments/experiment_groups/group2_intensity_modulation/01_baseline_training' / f'K30_r8_{timestamp}'

    # Hyperparameters
    num_gaussians = 30
    rank = 8
    top_k = 3
    lr = 1e-3
    num_epochs = 2000
    batch_size = 256

    # Logging control
    log_interval = 10  # Log every 10 epochs (instead of 50)
    checkpoint_interval = 100  # Save intermediate checkpoints
    gradient_log_interval = 50  # Compute gradient stats every 50 epochs

    print("=" * 80)
    print("1D Intensity Modulation Training: Gaussian-Physics Compression (ENHANCED)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data: {data_root}")
    print(f"Output: {output_dir}")
    print(f"\nHyperparameters:")
    print(f"  Gaussians (K): {num_gaussians}")
    print(f"  Rank (r): {rank}")
    print(f"  Top-k: {top_k}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"\nLogging:")
    print(f"  Log interval: {log_interval}")
    print(f"  Checkpoint interval: {checkpoint_interval}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create training log file
    log_file = output_dir / 'training_log.csv'
    log_fieldnames = ['epoch', 'train_total', 'train_recon', 'val_mae', 'val_rmse',
                      'grad_norm_mean', 'grad_norm_max', 'per_coeff_mae_mean',
                      'per_coeff_mae_max', 'gaussian_coverage']

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

    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Probes: {len(probe_positions)}")

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
    num_probes = len(probe_positions)
    num_moments = 12
    compression_ratio = model.get_compression_ratio(
        num_probes=num_probes,
        num_moments=num_moments
    )
    print(f"\nCompression analysis:")
    print(f"  Naive params: {num_probes * num_moments * 27:,}")
    print(f"  Compressed params: {model.num_params():,}")
    print(f"  Compression ratio: {compression_ratio:.2f}×")

    # Check initial Gaussian coverage
    initial_coverage = compute_gaussian_coverage(model, probe_positions, device, top_k=top_k)
    print(f"  Initial Gaussian coverage: {initial_coverage:.2f}%")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print(f"\n[4/5] Training for {num_epochs} epochs...")
    print(f"Complete training log will be saved to: {log_file}")

    train_losses = {'total': [], 'recon': [], 'grad_norm_mean': [], 'grad_norm_max': []}
    val_losses = {'mae': [], 'rmse': [], 'per_coeff_mae_mean': [], 'per_coeff_mae_max': []}

    best_val_mae = float('inf')
    best_epoch = 0

    # Initialize CSV log
    with open(log_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        # Train
        log_gradients = (epoch % gradient_log_interval == 0) or (epoch == 1)
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, top_k=top_k, log_gradients=log_gradients
        )

        # Validate
        val_metrics = validate_epoch(model, val_loader, device, top_k=top_k)

        # Record metrics
        train_losses['total'].append(train_metrics['total'])
        train_losses['recon'].append(train_metrics['recon'])
        val_losses['mae'].append(val_metrics['mae'])
        val_losses['rmse'].append(val_metrics['rmse'])
        val_losses['per_coeff_mae_mean'].append(val_metrics['per_coeff_mae_mean'])
        val_losses['per_coeff_mae_max'].append(val_metrics['per_coeff_mae_max'])

        if log_gradients:
            train_losses['grad_norm_mean'].append(train_metrics.get('grad_norm_mean', 0))
            train_losses['grad_norm_max'].append(train_metrics.get('grad_norm_max', 0))

        # Compute Gaussian coverage periodically
        gaussian_coverage = 0.0
        if epoch % checkpoint_interval == 0 or epoch == 1:
            gaussian_coverage = compute_gaussian_coverage(model, probe_positions, device, top_k=top_k)

        # Write to CSV log (EVERY EPOCH)
        with open(log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_fieldnames)
            writer.writerow({
                'epoch': epoch,
                'train_total': train_metrics['total'],
                'train_recon': train_metrics['recon'],
                'val_mae': val_metrics['mae'],
                'val_rmse': val_metrics['rmse'],
                'grad_norm_mean': train_metrics.get('grad_norm_mean', ''),
                'grad_norm_max': train_metrics.get('grad_norm_max', ''),
                'per_coeff_mae_mean': val_metrics['per_coeff_mae_mean'],
                'per_coeff_mae_max': val_metrics['per_coeff_mae_max'],
                'gaussian_coverage': gaussian_coverage if gaussian_coverage > 0 else ''
            })

        # Print progress
        if epoch % log_interval == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{num_epochs}: "
                  f"Train Loss={train_metrics['total']:.6f}, "
                  f"Val MAE={val_metrics['mae']:.6f}, "
                  f"Val RMSE={val_metrics['rmse']:.6f}, "
                  f"Pred range=[{val_metrics['pred_range'][0]:.2f}, {val_metrics['pred_range'][1]:.2f}]")

        # Save best model
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': best_val_mae,
                'train_loss': train_metrics['total'],
                'hyperparameters': {
                    'num_gaussians': num_gaussians,
                    'rank': rank,
                    'top_k': top_k,
                    'lr': lr
                }
            }, output_dir / 'best_model.pth')

        # Save intermediate checkpoints
        if epoch % checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_metrics['mae'],
                'train_loss': train_metrics['total']
            }, output_dir / f'checkpoint_epoch_{epoch:04d}.pth')

            # Update plot
            plot_training_curve(train_losses, val_losses, output_dir, current_epoch=epoch)

    print(f"\nTraining complete!")
    print(f"Best validation MAE: {best_val_mae:.6f} at epoch {best_epoch}")

    # Test evaluation
    print(f"\n[5/5] Evaluating on test set...")
    model.load_state_dict(torch.load(output_dir / 'best_model.pth')['model_state_dict'])
    test_metrics = validate_epoch(model, test_loader, device, top_k=top_k)
    print(f"Test MAE: {test_metrics['mae']:.6f}")
    print(f"Test RMSE: {test_metrics['rmse']:.6f}")
    print(f"Test pred range: [{test_metrics['pred_range'][0]:.2f}, {test_metrics['pred_range'][1]:.2f}]")
    print(f"Test target range: [{test_metrics['target_range'][0]:.2f}, {test_metrics['target_range'][1]:.2f}]")

    # Final Gaussian coverage
    final_coverage = compute_gaussian_coverage(model, probe_positions, device, top_k=top_k)
    print(f"Final Gaussian coverage: {final_coverage:.2f}%")

    # Final plot
    plot_training_curve(train_losses, val_losses, output_dir)

    # Save final results
    results = {
        'experiment_name': f'1D_intensity_modulation_K{num_gaussians}_r{rank}_{timestamp}',
        'dataset': str(data_root),
        'hyperparameters': {
            'num_gaussians': num_gaussians,
            'rank': rank,
            'top_k': top_k,
            'learning_rate': lr,
            'num_epochs': num_epochs,
            'batch_size': batch_size
        },
        'compression': {
            'num_probes': num_probes,
            'num_moments': num_moments,
            'naive_params': num_probes * num_moments * 27,
            'compressed_params': model.num_params(),
            'compression_ratio': float(compression_ratio)
        },
        'training': {
            'best_epoch': best_epoch,
            'best_val_mae': float(best_val_mae),
            'initial_gaussian_coverage': float(initial_coverage),
            'final_gaussian_coverage': float(final_coverage)
        },
        'test_metrics': {
            'mae': float(test_metrics['mae']),
            'rmse': float(test_metrics['rmse']),
            'per_coeff_mae_mean': float(test_metrics['per_coeff_mae_mean']),
            'per_coeff_mae_max': float(test_metrics['per_coeff_mae_max']),
            'pred_range': [float(x) for x in test_metrics['pred_range']],
            'target_range': [float(x) for x in test_metrics['target_range']]
        },
        'training_completed_at': datetime.now().isoformat()
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"  - results.json: Complete experiment metadata")
    print(f"  - training_log.csv: Per-epoch metrics")
    print(f"  - training_curve.png: Visualization")
    print(f"  - best_model.pth: Best checkpoint (epoch {best_epoch})")
    print(f"  - checkpoint_epoch_*.pth: Intermediate checkpoints")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
