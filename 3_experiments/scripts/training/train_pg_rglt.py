"""
Training script for PG-RGLT (Physics-Guided Light Trajectory) model.

Trains per-probe low-rank factorization on transmission tensor data.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

from models.pg_rglt import PGRGLT


class CharbonnierLoss(nn.Module):
    """Robust L2 loss with Charbonnier penalty."""

    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.epsilon ** 2)
        return loss.mean()


def load_transfer_tensor(data_path: str):
    """
    Load transmission tensor from npz file.

    Returns:
        tensor: [num_probes, num_lights, sh_dim] numpy array
        probe_positions: [num_probes, 3]
        light_positions: [num_lights, 3]
    """
    data = np.load(data_path)

    tensor = data['tensor']  # [343, 12, 27]
    probe_positions = data['probe_positions']  # [343, 3]
    light_positions = data['light_positions']  # [12, 3]

    print(f"Loaded transmission tensor from: {data_path}")
    print(f"  Tensor shape: {tensor.shape}")
    print(f"  Probe positions: {probe_positions.shape}")
    print(f"  Light positions: {light_positions.shape}")
    print(f"  Tensor stats: mean={tensor.mean():.4f}, std={tensor.std():.4f}")
    print(f"                min={tensor.min():.4f}, max={tensor.max():.4f}")
    print()

    return tensor, probe_positions, light_positions


def create_train_val_split(
    transfer_tensor: np.ndarray,
    train_ratio: float = 0.7,
    seed: int = 42
):
    """
    Split transmission tensor into train/val sets by probes.

    Args:
        transfer_tensor: [num_probes, num_lights, sh_dim]
        train_ratio: fraction of probes for training
        seed: random seed

    Returns:
        train_data: dict with train indices and data
        val_data: dict with val indices and data
    """
    num_probes, num_lights, sh_dim = transfer_tensor.shape

    # Split probes
    np.random.seed(seed)
    probe_indices = np.arange(num_probes)
    np.random.shuffle(probe_indices)

    num_train = int(num_probes * train_ratio)
    train_probe_indices = probe_indices[:num_train]
    val_probe_indices = probe_indices[num_train:]

    print(f"Data split:")
    print(f"  Train probes: {len(train_probe_indices)} ({train_ratio*100:.0f}%)")
    print(f"  Val probes: {len(val_probe_indices)} ({(1-train_ratio)*100:.0f}%)")
    print()

    # Create index pairs (probe_idx, light_idx) for all combinations
    train_pairs = []
    for probe_idx in train_probe_indices:
        for light_idx in range(num_lights):
            train_pairs.append((probe_idx, light_idx))

    val_pairs = []
    for probe_idx in val_probe_indices:
        for light_idx in range(num_lights):
            val_pairs.append((probe_idx, light_idx))

    train_data = {
        'probe_indices': train_probe_indices,
        'pairs': train_pairs,
        'size': len(train_pairs)
    }

    val_data = {
        'probe_indices': val_probe_indices,
        'pairs': val_pairs,
        'size': len(val_pairs)
    }

    return train_data, val_data


def train_epoch(
    model: PGRGLT,
    transfer_tensor: torch.Tensor,
    train_pairs: list,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    batch_size: int = 64,
    device: str = 'cuda'
) -> float:
    """
    Train for one epoch.

    Returns:
        avg_loss: average training loss
    """
    model.train()

    # Shuffle training pairs
    np.random.shuffle(train_pairs)

    total_loss = 0.0
    num_batches = 0

    for i in range(0, len(train_pairs), batch_size):
        batch_pairs = train_pairs[i:i+batch_size]

        # Extract probe and light indices
        probe_indices = torch.tensor([p for p, l in batch_pairs], dtype=torch.long, device=device)
        light_indices = torch.tensor([l for p, l in batch_pairs], dtype=torch.long, device=device)

        # Get ground truth SH coefficients
        target_sh = []
        for probe_idx, light_idx in batch_pairs:
            target_sh.append(transfer_tensor[probe_idx, light_idx])
        target_sh = torch.stack(target_sh, dim=0).to(device)  # [B, 27]

        # Forward pass
        pred_sh = model(probe_indices, light_indices)

        # Compute loss
        loss = criterion(pred_sh, target_sh)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def evaluate(
    model: PGRGLT,
    transfer_tensor: torch.Tensor,
    eval_pairs: list,
    criterion: nn.Module,
    batch_size: int = 64,
    device: str = 'cuda'
) -> dict:
    """
    Evaluate model on validation set.

    Returns:
        metrics: dict with loss, MAE, RMSE
    """
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    num_samples = 0

    for i in range(0, len(eval_pairs), batch_size):
        batch_pairs = eval_pairs[i:i+batch_size]

        # Extract probe and light indices
        probe_indices = torch.tensor([p for p, l in batch_pairs], dtype=torch.long, device=device)
        light_indices = torch.tensor([l for p, l in batch_pairs], dtype=torch.long, device=device)

        # Get ground truth
        target_sh = []
        for probe_idx, light_idx in batch_pairs:
            target_sh.append(transfer_tensor[probe_idx, light_idx])
        target_sh = torch.stack(target_sh, dim=0).to(device)

        # Forward pass
        pred_sh = model(probe_indices, light_indices)

        # Compute metrics
        loss = criterion(pred_sh, target_sh)
        mae = torch.abs(pred_sh - target_sh).mean()
        mse = ((pred_sh - target_sh) ** 2).mean()

        total_loss += loss.item() * len(batch_pairs)
        total_mae += mae.item() * len(batch_pairs)
        total_mse += mse.item() * len(batch_pairs)
        num_samples += len(batch_pairs)

    metrics = {
        'loss': total_loss / num_samples,
        'mae': total_mae / num_samples,
        'rmse': np.sqrt(total_mse / num_samples)
    }

    return metrics


def train_pg_rglt(
    data_path: str,
    rank: int = 5,
    num_epochs: int = 1000,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    train_ratio: float = 0.7,
    output_dir: str = None,
    device: str = 'cuda',
    svd_init: bool = True
):
    """
    Main training function for PG-RGLT.

    Args:
        data_path: path to transfer_tensor.npz
        rank: low-rank dimension
        num_epochs: number of training epochs
        learning_rate: optimizer learning rate
        batch_size: training batch size
        train_ratio: fraction of probes for training
        output_dir: directory to save results
        device: 'cuda' or 'cpu'
        svd_init: whether to initialize with SVD
    """
    # Setup output directory
    if output_dir is None:
        output_dir = Path(data_path).parent / 'pg_rglt_training'
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("PG-RGLT Training")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print()

    # Load data
    transfer_tensor_np, probe_positions, light_positions = load_transfer_tensor(data_path)
    num_probes, num_lights, sh_dim = transfer_tensor_np.shape

    # Convert to torch
    transfer_tensor = torch.tensor(transfer_tensor_np, dtype=torch.float32)

    # Create train/val split
    train_data, val_data = create_train_val_split(
        transfer_tensor_np,
        train_ratio=train_ratio
    )

    # Create model
    model = PGRGLT(
        num_probes=num_probes,
        num_light_positions=num_lights,
        sh_dim=sh_dim,
        rank=rank
    )

    # Print compression stats
    stats = model.get_compression_stats()
    print("Model Configuration:")
    print(f"  Rank: {rank}")
    print(f"  Parameters per probe: {stats['params_per_probe']}")
    print(f"  Total parameters: {stats['total_params']}")
    print(f"  Uncompressed size: {stats['uncompressed_size']}")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}×")
    print()

    # SVD initialization
    if svd_init:
        print("Initializing with SVD...")
        model.initialize_from_svd(transfer_tensor_np, rank=rank)

        # Analyze rank distribution
        rank_analysis = model.analyze_rank_distribution(transfer_tensor_np)
        print(f"Rank analysis (mean energy captured):")
        for r, energy in rank_analysis['mean_energy_at_rank'].items():
            print(f"  Rank {r}: {energy*100:.2f}%")
        print()

    # Move to device
    model = model.to(device)

    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = CharbonnierLoss()

    # Training loop
    print(f"Training for {num_epochs} epochs...")
    print()

    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': []
    }

    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Train
        train_loss = train_epoch(
            model=model,
            transfer_tensor=transfer_tensor,
            train_pairs=train_data['pairs'],
            optimizer=optimizer,
            criterion=criterion,
            batch_size=batch_size,
            device=device
        )

        # Validate
        val_metrics = evaluate(
            model=model,
            transfer_tensor=transfer_tensor,
            eval_pairs=val_data['pairs'],
            criterion=criterion,
            batch_size=batch_size,
            device=device
        )

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_rmse'].append(val_metrics['rmse'])

        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_metrics['loss']:.6f}")
            print(f"  Val MAE: {val_metrics['mae']:.6f}")
            print(f"  Val RMSE: {val_metrics['rmse']:.6f}")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), output_dir / 'best_model.pth')

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print()

    # Save final results
    final_metrics = evaluate(
        model=model,
        transfer_tensor=transfer_tensor,
        eval_pairs=val_data['pairs'],
        criterion=criterion,
        batch_size=batch_size,
        device=device
    )

    results = {
        'config': {
            'rank': rank,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'train_ratio': train_ratio,
            'svd_init': svd_init
        },
        'compression_stats': stats,
        'final_metrics': final_metrics,
        'best_val_loss': best_val_loss,
        'training_date': datetime.now().isoformat()
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train', alpha=0.7)
    axes[0].plot(history['val_loss'], label='Val', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Charbonnier Loss')
    axes[0].set_title('Training Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['val_mae'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Validation MAE')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history['val_rmse'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('RMSE')
    axes[2].set_title('Validation RMSE')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    print(f"Training curves saved: {output_dir / 'training_curves.png'}")

    # Save history
    np.savez(
        output_dir / 'training_history.npz',
        train_loss=history['train_loss'],
        val_loss=history['val_loss'],
        val_mae=history['val_mae'],
        val_rmse=history['val_rmse']
    )

    return model, results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train PG-RGLT model')
    parser.add_argument(
        '--data_path',
        type=str,
        default='/home/kyrie/毕设/data_generation/output/transfer_tensor_validation/transfer_tensor.npz',
        help='Path to transfer_tensor.npz'
    )
    parser.add_argument('--rank', type=int, default=5, help='Low-rank dimension')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Train split ratio')
    parser.add_argument('--no_svd_init', action='store_true', help='Disable SVD initialization')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    model, results = train_pg_rglt(
        data_path=args.data_path,
        rank=args.rank,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        output_dir=args.output_dir,
        device=args.device,
        svd_init=not args.no_svd_init
    )

    print("\nFinal Results:")
    print("=" * 70)
    for key, value in results['final_metrics'].items():
        print(f"{key}: {value:.6f}")
    print(f"Compression ratio: {results['compression_stats']['compression_ratio']:.2f}×")
    print("=" * 70)
