#!/usr/bin/env python3
"""
正确的SH可视化验证: 辐射场球面可视化

方法:
1. 在单位球面上采样N个方向
2. 使用GT SH和Pred SH重建每个方向的辐射值
3. 可视化为球面彩色点云
4. 对比GT vs Pred的视觉差异
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from mpl_toolkits.mplot3d import Axes3D

from models.gaussian_physics_5D import GaussianPhysicsCompression5D
from data.transfer_tensor_dataset import TransferTensorDataset5D

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data_generation')))
from utils.spherical_harmonics import reconstruct_from_sh, fibonacci_sphere


def visualize_radiance_sphere_3d(directions: np.ndarray,
                                 radiances_gt: np.ndarray,
                                 radiances_pred: np.ndarray,
                                 sample_idx: int,
                                 output_path: Path):
    """
    3D球面辐射场可视化 (GT vs Pred对比)

    Args:
        directions: [N, 3] 单位方向向量
        radiances_gt: [N, 3] GT辐射值 (RGB)
        radiances_pred: [N, 3] Pred辐射值 (RGB)
        sample_idx: 样本索引
        output_path: 输出路径
    """
    fig = plt.figure(figsize=(18, 6))

    # 1. Ground Truth球面
    ax1 = fig.add_subplot(131, projection='3d')
    colors_gt = np.clip(radiances_gt ** (1/2.2), 0, 1)  # Gamma校正用于显示
    ax1.scatter(directions[:, 0], directions[:, 1], directions[:, 2],
               c=colors_gt, s=20, alpha=0.6)
    ax1.set_title(f'Sample {sample_idx}\nGround Truth Radiance Field')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_box_aspect([1,1,1])

    # 2. Prediction球面
    ax2 = fig.add_subplot(132, projection='3d')
    colors_pred = np.clip(radiances_pred ** (1/2.2), 0, 1)
    ax2.scatter(directions[:, 0], directions[:, 1], directions[:, 2],
               c=colors_pred, s=20, alpha=0.6)
    ax2.set_title(f'Prediction (K30_r8)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_box_aspect([1,1,1])

    # 3. 误差球面 (放大5×)
    ax3 = fig.add_subplot(133, projection='3d')
    diff = np.abs(radiances_gt - radiances_pred) * 5.0
    colors_diff = np.clip(diff ** (1/2.2), 0, 1)
    ax3.scatter(directions[:, 0], directions[:, 1], directions[:, 2],
               c=colors_diff, s=20, alpha=0.6)
    ax3.set_title(f'Absolute Error ×5')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_box_aspect([1,1,1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved 3D radiance visualization to {output_path}")


def visualize_radiance_2d_unwrap(directions: np.ndarray,
                                 radiances_gt: np.ndarray,
                                 radiances_pred: np.ndarray,
                                 sample_idx: int,
                                 mae: float,
                                 output_path: Path):
    """
    2D展开辐射场可视化 (更清晰)

    将球面辐射场展开为2D图像 (theta, phi) → (row, col)
    """
    # 计算方向的球坐标
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]

    # theta: [0, pi], phi: [0, 2*pi]
    theta = np.arccos(np.clip(y, -1, 1))  # [0, pi]
    phi = np.arctan2(z, x) + np.pi  # [0, 2*pi]

    # 创建2D网格
    H, W = 64, 128
    theta_bins = np.linspace(0, np.pi, H + 1)
    phi_bins = np.linspace(0, 2 * np.pi, W + 1)

    # 分配到bin
    img_gt = np.zeros((H, W, 3))
    img_pred = np.zeros((H, W, 3))
    counts = np.zeros((H, W))

    for i in range(len(directions)):
        theta_idx = np.digitize(theta[i], theta_bins) - 1
        phi_idx = np.digitize(phi[i], phi_bins) - 1

        theta_idx = np.clip(theta_idx, 0, H - 1)
        phi_idx = np.clip(phi_idx, 0, W - 1)

        img_gt[theta_idx, phi_idx] += radiances_gt[i]
        img_pred[theta_idx, phi_idx] += radiances_pred[i]
        counts[theta_idx, phi_idx] += 1

    # 平均
    mask = counts > 0
    img_gt[mask] /= counts[mask, None]
    img_pred[mask] /= counts[mask, None]

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].imshow(np.clip(img_gt ** (1/2.2), 0, 1))
    axes[0].set_title(f'Sample {sample_idx} - Ground Truth\n(Unwrapped Sphere)')
    axes[0].axis('off')

    axes[1].imshow(np.clip(img_pred ** (1/2.2), 0, 1))
    axes[1].set_title(f'Prediction (K30_r8)\nMAE: {mae:.6f}')
    axes[1].axis('off')

    diff = np.abs(img_gt - img_pred) * 5.0
    axes[2].imshow(np.clip(diff ** (1/2.2), 0, 1))
    axes[2].set_title('Absolute Error ×5')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved 2D unwrapped radiance visualization to {output_path}")


def main():
    # Configuration
    checkpoint_path = Path('/home/kyrie/毕设/multi_time_compression/experiments/week6_ablation_5D/K30_r8/best_model.pt')
    data_root = Path('/home/kyrie/毕设/data_generation/output/5D_parametric_validation')
    output_dir = Path('/home/kyrie/毕设/multi_time_compression/experiments/week6_visual_validation/radiance_field')
    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("\n[1/4] Loading K30_r8 model...")
    model = GaussianPhysicsCompression5D(num_gaussians=30, rank=8, sh_dim=27)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load test dataset
    print("\n[2/4] Loading test dataset...")
    test_dataset = TransferTensorDataset5D(
        data_root=str(data_root),
        split='test',
        normalize_params=False
    )

    # Select samples (same as rendering comparison)
    test_indices = [121, 530, 495, 612, 597]
    print(f"Selected {len(test_indices)} samples for radiance field visualization")

    # Generate sphere sampling directions
    print("\n[3/4] Generating sphere samples...")
    N_samples = 512  # 球面采样点数
    directions = fibonacci_sphere(N_samples)  # [N, 3]
    print(f"Sampling {N_samples} directions on unit sphere")

    print("\n[4/4] Generating radiance field visualizations...")
    for idx in test_indices:
        sample = test_dataset[idx]
        probe_pos = sample['probe_position']
        light_params = sample['light_params']
        sh_gt = sample['sh_coeffs']

        # Model prediction
        with torch.no_grad():
            position_t = probe_pos.unsqueeze(0).to(device)
            config_t = light_params.unsqueeze(0).to(device)
            sh_pred = model(position_t, config_t, top_k=3)

        sh_gt_np = sh_gt.cpu().numpy()
        sh_pred_np = sh_pred.cpu().numpy()[0]

        # Reconstruct radiance from SH
        radiances_gt = reconstruct_from_sh(sh_gt_np, directions)  # [N, 3]
        radiances_pred = reconstruct_from_sh(sh_pred_np, directions)

        # Clamp negative values
        radiances_gt = np.maximum(radiances_gt, 0.0)
        radiances_pred = np.maximum(radiances_pred, 0.0)

        # Compute MAE
        mae = np.mean(np.abs(sh_gt_np - sh_pred_np))

        print(f"\nSample {idx}: SH MAE = {mae:.6f}")
        print(f"  Radiance range GT:   [{radiances_gt.min():.3f}, {radiances_gt.max():.3f}]")
        print(f"  Radiance range Pred: [{radiances_pred.min():.3f}, {radiances_pred.max():.3f}]")

        # 3D球面可视化
        viz_3d_path = output_dir / f'sample_{idx}_3d_sphere.png'
        visualize_radiance_sphere_3d(
            directions, radiances_gt, radiances_pred, idx, viz_3d_path
        )

        # 2D展开可视化
        viz_2d_path = output_dir / f'sample_{idx}_2d_unwrap.png'
        visualize_radiance_2d_unwrap(
            directions, radiances_gt, radiances_pred, idx, mae, viz_2d_path
        )

    print("\n✅ Radiance field visualization complete!")
    print(f"   Output directory: {output_dir}")
    print(f"   Generated 2 visualizations per sample:")
    print(f"     - 3D sphere view (sample_*_3d_sphere.png)")
    print(f"     - 2D unwrapped view (sample_*_2d_unwrap.png)")


if __name__ == '__main__':
    main()
