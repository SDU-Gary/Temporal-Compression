"""
Diagnostic script to analyze training results and identify performance bottlenecks
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from data.dataset import MultiTimeLightingDataset
from models.gaussian_mixture import GaussianMixture
from models.temporal_mlp import TemporalMLP
from models.decoder_mlp import DecoderMLP
from training.losses import CombinedLoss
from training.metrics import psnr, mse, mae
import yaml

def load_checkpoint_and_config(exp_dir):
    """Load checkpoint and config"""
    exp_path = Path(exp_dir)

    # Load config
    with open(exp_path / "config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Load best checkpoint
    ckpt_path = exp_path / "checkpoints" / "best_model.pt"
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    return config, checkpoint

def analyze_loss_components(config, checkpoint, dataset, device='cuda'):
    """Analyze loss components on validation set"""

    # Create models
    gaussian_model = GaussianMixture(
        num_gaussians=config['model']['num_gaussians'],
        latent_dim=config['model']['latent_dim_base']
    ).to(device)

    temporal_mlp = TemporalMLP(
        input_dim=config['model']['temporal_mlp']['input_dim'],
        hidden_dim=config['model']['temporal_mlp']['hidden_dim'],
        output_dim=config['model']['latent_dim_time'],
        num_layers=config['model']['temporal_mlp']['num_layers']
    ).to(device)

    decoder_mlp = DecoderMLP(
        input_dim=config['model']['decoder_mlp']['input_dim'],
        hidden_dim=config['model']['decoder_mlp']['hidden_dim'],
        output_dim=config['model']['decoder_mlp']['output_dim'],
        num_layers=config['model']['decoder_mlp']['num_layers']
    ).to(device)

    # Load weights
    gaussian_model.load_state_dict(checkpoint['gaussian_model'])
    temporal_mlp.load_state_dict(checkpoint['temporal_mlp'])
    decoder_mlp.load_state_dict(checkpoint['decoder_mlp'])

    # Set to eval mode
    gaussian_model.eval()
    temporal_mlp.eval()
    decoder_mlp.eval()

    # Create loss function
    loss_fn = CombinedLoss(
        w_reconstruction=config['training']['loss']['reconstruction'],
        w_temporal_smooth=config['training']['loss']['temporal_smooth'],
        w_energy=config['training']['loss']['energy_conservation'],
        temporal_norm=config['training']['loss']['temporal_norm']
    ).to(device)

    # Analyze all samples
    all_mse = []
    all_recon_loss = []
    all_temporal_loss = []
    all_total_loss = []

    from torch.utils.data import DataLoader
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    print("Analyzing validation set...")
    with torch.no_grad():
        for batch in val_loader:
            positions = batch['positions'].to(device)  # [B, 3]
            sun_dirs = batch['sun_dirs'][0].to(device)  # [T, 3]
            sh_coeffs_gt = batch['sh_coeffs'].to(device)  # [B, T, 27]

            # Forward pass
            latent_base = gaussian_model(positions)  # [B, F_base]
            sh_coeffs_pred = []

            for t in range(sun_dirs.shape[0]):
                sun_dir_t = sun_dirs[t:t+1].expand(positions.shape[0], -1)  # [B, 3]
                latent_time = temporal_mlp(latent_base, sun_dir_t)  # [B, F_time]
                sh_t = decoder_mlp(latent_time, positions, sun_dir_t)  # [B, 27]
                sh_coeffs_pred.append(sh_t)

            sh_coeffs_pred = torch.stack(sh_coeffs_pred, dim=1)  # [B, T, 27]

            # Compute losses
            total_loss, loss_dict = loss_fn(sh_coeffs_pred, sh_coeffs_gt)

            # Per-sample MSE
            sample_mse = torch.mean((sh_coeffs_pred - sh_coeffs_gt) ** 2, dim=(1, 2))

            all_mse.extend(sample_mse.cpu().numpy())
            all_recon_loss.extend([loss_dict['reconstruction']] * positions.shape[0])
            all_temporal_loss.extend([loss_dict['temporal_smooth']] * positions.shape[0])
            all_total_loss.extend([total_loss.item()] * positions.shape[0])

    all_mse = np.array(all_mse)
    all_recon_loss = np.array(all_recon_loss)
    all_temporal_loss = np.array(all_temporal_loss)
    all_total_loss = np.array(all_total_loss)

    return {
        'mse': all_mse,
        'recon_loss': all_recon_loss,
        'temporal_loss': all_temporal_loss,
        'total_loss': all_total_loss
    }

def main():
    exp_dir = "experiments/baseline_2k"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*70)
    print("TRAINING DIAGNOSTIC ANALYSIS")
    print("="*70)

    # Load checkpoint and config
    print("\n1. Loading checkpoint and config...")
    config, checkpoint = load_checkpoint_and_config(exp_dir)

    # Print training info
    print(f"\nTraining stopped at step: {checkpoint['step']}")
    print(f"Best validation PSNR: {checkpoint.get('best_val_psnr', 'N/A')}")

    # Load validation dataset
    print("\n2. Loading validation dataset...")
    dataset_path = Path(__file__).parent.parent / config['data']['dataset_path']
    val_dataset = MultiTimeLightingDataset(
        data_root=str(dataset_path),
        split='val',
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio']
    )
    print(f"Validation samples: {len(val_dataset)}")

    # Analyze loss components
    print("\n3. Analyzing loss components...")
    results = analyze_loss_components(config, checkpoint, val_dataset, device)

    # Print analysis
    print("\n" + "="*70)
    print("LOSS COMPONENT ANALYSIS")
    print("="*70)

    # Average losses
    avg_recon = np.mean(results['recon_loss'])
    avg_temporal = np.mean(results['temporal_loss'])
    avg_total = np.mean(results['total_loss'])

    # Calculate contributions with correct weighting
    w_temporal = config['training']['loss']['temporal_smooth']
    temporal_contribution = w_temporal * avg_temporal
    temporal_percentage = (temporal_contribution / avg_total) * 100

    print(f"\nAverage Reconstruction Loss: {avg_recon:.6f}")
    print(f"Average Temporal Smooth Loss (raw): {avg_temporal:.6f}")
    print(f"Average Total Loss: {avg_total:.6f}")
    print(f"\nTemporal smooth weight: {w_temporal}")
    print(f"Temporal smooth contribution: {temporal_contribution:.6f}")
    print(f"Temporal smooth percentage: {temporal_percentage:.2f}%")

    if temporal_percentage > 15:
        print(f"\n⚠️  WARNING: Temporal regularization is TOO STRONG!")
        print(f"   Recommended: <15%, Current: {temporal_percentage:.2f}%")
        print(f"   Suggestion: Reduce temporal_smooth to {w_temporal * 0.4:.3f} or lower")

    # MSE statistics
    print("\n" + "="*70)
    print("MSE DISTRIBUTION ANALYSIS")
    print("="*70)

    mse_mean = np.mean(results['mse'])
    mse_std = np.std(results['mse'])
    mse_min = np.min(results['mse'])
    mse_max = np.max(results['mse'])
    mse_median = np.median(results['mse'])

    print(f"\nMSE Statistics:")
    print(f"  Mean:   {mse_mean:.6f} (PSNR: {psnr(torch.tensor([mse_mean]), torch.tensor([0.0]), max_val=1.0):.2f} dB)")
    print(f"  Median: {mse_median:.6f} (PSNR: {psnr(torch.tensor([mse_median]), torch.tensor([0.0]), max_val=1.0):.2f} dB)")
    print(f"  Std:    {mse_std:.6f}")
    print(f"  Min:    {mse_min:.6f} (PSNR: {psnr(torch.tensor([mse_min]), torch.tensor([0.0]), max_val=1.0):.2f} dB)")
    print(f"  Max:    {mse_max:.6f} (PSNR: {psnr(torch.tensor([mse_max]), torch.tensor([0.0]), max_val=1.0):.2f} dB)")

    # Find worst samples
    worst_indices = np.argsort(results['mse'])[-10:]
    print(f"\nTop 10 worst samples (MSE):")
    for i, idx in enumerate(worst_indices[::-1], 1):
        sample_psnr = psnr(torch.tensor([results['mse'][idx]]), torch.tensor([0.0]), max_val=1.0)
        print(f"  {i}. Sample {idx}: MSE={results['mse'][idx]:.6f}, PSNR={sample_psnr:.2f} dB")

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if temporal_percentage > 20:
        print("\n🔴 CRITICAL: Regularization is severely limiting model performance")
        print(f"   Action: Create new config with temporal_smooth={w_temporal * 0.3:.3f} (70% reduction)")
    elif temporal_percentage > 15:
        print("\n🟡 WARNING: Regularization is too strong")
        print(f"   Action: Create new config with temporal_smooth={w_temporal * 0.5:.3f} (50% reduction)")
    else:
        print("\n🟢 Regularization is reasonable, investigate other factors")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
