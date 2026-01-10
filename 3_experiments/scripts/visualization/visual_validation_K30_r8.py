#!/usr/bin/env python3
"""
Week 6.2 可视化验证: SH重建质量评估

验证K30_r8模型的SH重建质量：
1. 从测试集随机采样10个样本
2. 使用模型重建SH系数
3. 计算MAE/RMSE指标
4. 生成误差可视化
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm

# Import model
from models.gaussian_physics_5D import GaussianPhysicsCompression5D
from data.transfer_tensor_dataset import TransferTensorDataset5D


def visualize_sh_errors(sh_errors_list, output_path):
    """
    可视化SH系数重建误差

    Args:
        sh_errors_list: List of [27] arrays (per-sample SH errors)
        output_path: Output path
    """
    sh_errors = np.array(sh_errors_list)  # [N, 27]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Per-sample MAE
    sample_mae = np.mean(np.abs(sh_errors), axis=1)
    axes[0, 0].bar(range(len(sample_mae)), sample_mae, color='steelblue')
    axes[0, 0].axhline(y=np.mean(sample_mae), color='r', linestyle='--',
                       label=f'Avg: {np.mean(sample_mae):.6f}')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].set_title('Per-Sample SH Reconstruction MAE')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Per-coefficient MAE
    coeff_mae = np.mean(np.abs(sh_errors), axis=0)
    axes[0, 1].bar(range(27), coeff_mae, color='coral')
    axes[0, 1].axhline(y=np.mean(coeff_mae), color='r', linestyle='--',
                       label=f'Avg: {np.mean(coeff_mae):.6f}')
    axes[0, 1].set_xlabel('SH Coefficient Index')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Per-Coefficient SH Reconstruction MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Error heatmap
    im = axes[1, 0].imshow(np.abs(sh_errors).T, aspect='auto', cmap='viridis')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('SH Coefficient Index')
    axes[1, 0].set_title('Absolute Error Heatmap')
    plt.colorbar(im, ax=axes[1, 0], label='Absolute Error')

    # 4. Error distribution
    all_errors = sh_errors.flatten()
    axes[1, 1].hist(all_errors, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=0, color='r', linestyle='--', label='Zero Error')
    axes[1, 1].set_xlabel('Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")


def visualize_reconstruction_samples(gt_list, pred_list, indices, output_path):
    """
    可视化GT vs Pred SH系数对比（前3个样本）

    Args:
        gt_list: List of [27] ground truth SH
        pred_list: List of [27] predicted SH
        indices: Sample indices
        output_path: Output path
    """
    n_show = min(3, len(gt_list))
    fig, axes = plt.subplots(n_show, 1, figsize=(12, 3 * n_show))

    if n_show == 1:
        axes = [axes]

    for i in range(n_show):
        gt = gt_list[i]
        pred = pred_list[i]

        x = np.arange(27)
        width = 0.35

        axes[i].bar(x - width/2, gt, width, label='Ground Truth', color='steelblue', alpha=0.7)
        axes[i].bar(x + width/2, pred, width, label='Prediction (K30_r8)', color='coral', alpha=0.7)

        mae = np.mean(np.abs(gt - pred))
        axes[i].set_xlabel('SH Coefficient Index')
        axes[i].set_ylabel('SH Value')
        axes[i].set_title(f'Sample {indices[i]} - MAE: {mae:.6f}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved reconstruction samples to {output_path}")


def main():
    # Configuration
    checkpoint_path = Path('/home/kyrie/毕设/multi_time_compression/experiments/week6_ablation_5D/K30_r8/best_model.pt')
    data_root = Path('/home/kyrie/毕设/data_generation/output/5D_parametric_validation')
    output_dir = Path('/home/kyrie/毕设/multi_time_compression/experiments/week6_visual_validation')
    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("\n[1/5] Loading K30_r8 model...")
    model = GaussianPhysicsCompression5D(num_gaussians=30, rank=8, sh_dim=27)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Load test dataset
    print("\n[2/5] Loading test dataset...")
    test_dataset = TransferTensorDataset5D(
        data_root=str(data_root),
        split='test',
        normalize_params=False
    )
    print(f"Test dataset: {len(test_dataset)} samples")

    # Randomly sample 10 test samples
    np.random.seed(2026)
    n_samples = min(10, len(test_dataset))
    test_indices = np.random.choice(len(test_dataset), size=n_samples, replace=False)
    print(f"Selected {n_samples} random samples for validation")

    # Evaluate
    sh_gt_list = []
    sh_pred_list = []
    sh_errors_list = []
    mae_list = []
    rmse_list = []

    print("\n[3/5] Evaluating SH reconstruction quality...")
    for idx in tqdm(test_indices, desc="Validating"):
        sample = test_dataset[idx]
        probe_pos = sample['probe_position']
        light_params = sample['light_params']
        sh_gt = sample['sh_coeffs']

        # Model prediction
        with torch.no_grad():
            position_t = probe_pos.unsqueeze(0).to(device)  # [1, 3]
            config_t = light_params.unsqueeze(0).to(device)  # [1, 5]

            sh_pred = model(position_t, config_t, top_k=3)  # [1, 27]

        # Convert to numpy
        sh_gt_np = sh_gt.cpu().numpy()
        sh_pred_np = sh_pred.cpu().numpy()[0]

        # Compute error
        error = sh_gt_np - sh_pred_np
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error ** 2))

        sh_gt_list.append(sh_gt_np)
        sh_pred_list.append(sh_pred_np)
        sh_errors_list.append(error)
        mae_list.append(mae)
        rmse_list.append(rmse)

    # Compute statistics
    avg_mae = np.mean(mae_list)
    std_mae = np.std(mae_list)
    avg_rmse = np.mean(rmse_list)
    std_rmse = np.std(rmse_list)

    print(f"\n{'='*70}")
    print(f"Visual Validation Results (K30_r8) - SH Reconstruction Quality")
    print(f"{'='*70}")
    print(f"Samples evaluated:    {n_samples}")
    print(f"Average SH MAE:       {avg_mae:.6f} ± {std_mae:.6f}")
    print(f"Average SH RMSE:      {avg_rmse:.6f} ± {std_rmse:.6f}")
    print(f"Min MAE:              {np.min(mae_list):.6f}")
    print(f"Max MAE:              {np.max(mae_list):.6f}")
    print(f"Target (MAE < 0.05):  {'✅ PASS' if avg_mae < 0.05 else '❌ FAIL'}")
    print(f"{'='*70}")

    # Save results
    print("\n[4/5] Saving results...")
    results = {
        'model': 'K30_r8',
        'checkpoint_epoch': int(checkpoint['epoch']),
        'n_samples': n_samples,
        'test_indices': test_indices.tolist(),
        'mae_list': [float(x) for x in mae_list],
        'rmse_list': [float(x) for x in rmse_list],
        'avg_mae': float(avg_mae),
        'std_mae': float(std_mae),
        'avg_rmse': float(avg_rmse),
        'std_rmse': float(std_rmse),
        'min_mae': float(np.min(mae_list)),
        'max_mae': float(np.max(mae_list)),
        'target_mae': 0.05,
        'pass': bool(avg_mae < 0.05)
    }

    results_path = output_dir / 'visual_validation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Create visualizations
    print("\n[5/5] Creating visualizations...")

    # Error analysis visualization
    viz_path = output_dir / 'sh_error_analysis.png'
    visualize_sh_errors(sh_errors_list, viz_path)

    # Reconstruction samples visualization
    samples_path = output_dir / 'reconstruction_samples.png'
    visualize_reconstruction_samples(sh_gt_list, sh_pred_list, test_indices, samples_path)

    print("\n✅ Visual validation complete!")
    print(f"   Average SH MAE: {avg_mae:.6f} {'< 0.05 ✓' if avg_mae < 0.05 else '>= 0.05 ✗'}")
    print(f"   Results: {results_path}")
    print(f"   Visualizations:")
    print(f"     - {viz_path}")
    print(f"     - {samples_path}")


if __name__ == '__main__':
    main()
