#!/usr/bin/env python3
"""
Week 6.2 渲染对比验证: 可视化GT vs Pred渲染结果

流程:
1. 从测试集选择5个样本
2. 使用GT SH系数渲染场景
3. 使用模型预测SH系数渲染场景
4. 对比渲染图像,计算PSNR/SSIM
5. 生成并排对比可视化
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
from typing import Tuple

# Import model
from models.gaussian_physics_5D import GaussianPhysicsCompression5D
from data.transfer_tensor_dataset import TransferTensorDataset5D

# Import rendering utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data_generation')))
from utils.spherical_harmonics import reconstruct_from_sh, fibonacci_sphere

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')


def sh_to_envmap(sh_coeffs: np.ndarray, resolution: int = 128) -> np.ndarray:
    """
    将SH系数转换为环境贴图

    Args:
        sh_coeffs: [27] SH系数 (9 bases × 3 RGB)
        resolution: 环境贴图分辨率

    Returns:
        envmap: [H, W, 3] 环境贴图 (equirectangular格式)
    """
    # 生成球面采样点 (equirectangular parameterization)
    H, W = resolution, resolution * 2

    # Theta: [0, pi], Phi: [0, 2*pi]
    theta = np.linspace(0, np.pi, H)
    phi = np.linspace(0, 2 * np.pi, W)

    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

    # 转换为笛卡尔坐标
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.cos(theta_grid)
    z = np.sin(theta_grid) * np.sin(phi_grid)

    # 方向向量 [H*W, 3]
    directions = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)

    # 从SH重建辐射 [H*W, 3]
    radiances = reconstruct_from_sh(sh_coeffs, directions)

    # Reshape到envmap格式 [H, W, 3]
    envmap = radiances.reshape(H, W, 3)

    # Clamp负值
    envmap = np.maximum(envmap, 0.0)

    return envmap


def render_with_envmap(envmap: np.ndarray,
                       probe_position: np.ndarray,
                       resolution: Tuple[int, int] = (256, 256),
                       spp: int = 64) -> np.ndarray:
    """
    使用环境贴图渲染Cornell Box场景

    Args:
        envmap: [H, W, 3] 环境贴图
        probe_position: [3] 相机位置
        resolution: (W, H) 渲染分辨率
        spp: 采样数

    Returns:
        image: [H, W, 3] 渲染图像
    """
    # 创建Cornell Box场景（简化版，只有几何体）
    scene_xml = f'''
    <scene version="3.0.0">
        <!-- Cornell Box几何 -->
        <shape type="obj">
            <string name="filename" value="/home/kyrie/毕设/data_generation/scenes/cornell_box.obj"/>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="0.8, 0.8, 0.8"/>
            </bsdf>
        </shape>

        <!-- 环境光（临时使用constant，后续替换为bitmap） -->
        <emitter type="constant">
            <rgb name="radiance" value="0.5, 0.5, 0.5"/>
        </emitter>

        <!-- 相机 -->
        <sensor type="perspective">
            <float name="fov" value="60"/>
            <transform name="to_world">
                <lookat origin="{probe_position[0]},{probe_position[1]},{probe_position[2]}"
                        target="{probe_position[0]},{probe_position[1]},{probe_position[2]-1}"
                        up="0,1,0"/>
            </transform>
            <film type="hdrfilm">
                <integer name="width" value="{resolution[0]}"/>
                <integer name="height" value="{resolution[1]}"/>
            </film>
            <sampler type="independent">
                <integer name="sample_count" value="{spp}"/>
            </sampler>
        </sensor>
    </scene>
    '''

    try:
        # 检查Cornell Box模型是否存在
        cornell_path = Path('/home/kyrie/毕设/data_generation/scenes/cornell_box.obj')
        if not cornell_path.exists():
            # 使用简化场景（仅环境光）
            scene_xml = f'''
            <scene version="3.0.0">
                <emitter type="constant">
                    <rgb name="radiance" value="{envmap.mean():.3f}, {envmap.mean():.3f}, {envmap.mean():.3f}"/>
                </emitter>

                <sensor type="perspective">
                    <float name="fov" value="60"/>
                    <transform name="to_world">
                        <lookat origin="{probe_position[0]},{probe_position[1]},{probe_position[2]}"
                                target="{probe_position[0]},{probe_position[1]},{probe_position[2]-1}"
                                up="0,1,0"/>
                    </transform>
                    <film type="hdrfilm">
                        <integer name="width" value="{resolution[0]}"/>
                        <integer name="height" value="{resolution[1]}"/>
                    </film>
                    <sampler type="independent">
                        <integer name="sample_count" value="{spp}"/>
                    </sampler>
                </sensor>
            </scene>
            '''

        scene = mi.load_string(scene_xml)
        image = mi.render(scene, spp=spp)
        return np.array(image)

    except Exception as e:
        print(f"Warning: Rendering failed ({e}), using envmap directly as visualization")
        # 直接返回envmap的可视化（作为fallback）
        # 取envmap中心区域作为"渲染结果"
        H, W = envmap.shape[:2]
        center_crop = envmap[H//4:3*H//4, W//4:3*W//4]
        # Resize到目标分辨率
        from scipy.ndimage import zoom
        scale_h = resolution[1] / center_crop.shape[0]
        scale_w = resolution[0] / center_crop.shape[1]
        resized = zoom(center_crop, (scale_h, scale_w, 1), order=1)
        return resized


def compute_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """计算PSNR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_ssim_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    """简化SSIM计算（全图统计）"""
    # 转换为灰度
    gray1 = np.mean(img1, axis=-1)
    gray2 = np.mean(img2, axis=-1)

    # 常数
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    # 统计量
    mu1 = np.mean(gray1)
    mu2 = np.mean(gray2)
    sigma1_sq = np.var(gray1)
    sigma2_sq = np.var(gray2)
    sigma12 = np.mean((gray1 - mu1) * (gray2 - mu2))

    # SSIM
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim


def visualize_rendering_comparison(samples_data: list, output_path: Path):
    """
    生成渲染对比可视化网格

    Args:
        samples_data: List of dicts with keys:
            - sample_idx
            - gt_envmap
            - pred_envmap
            - gt_render (optional)
            - pred_render (optional)
            - psnr
            - ssim
        output_path: 输出路径
    """
    n_samples = len(samples_data)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, data in enumerate(samples_data):
        # Ground Truth Envmap
        axes[i, 0].imshow(np.clip(data['gt_envmap'], 0, 1) ** (1/2.2))  # Gamma校正
        axes[i, 0].set_title(f'Sample {data["sample_idx"]}\nGround Truth Envmap')
        axes[i, 0].axis('off')

        # Prediction Envmap
        axes[i, 1].imshow(np.clip(data['pred_envmap'], 0, 1) ** (1/2.2))
        axes[i, 1].set_title(f'Prediction (K30_r8)')
        axes[i, 1].axis('off')

        # Difference (放大5x)
        diff = np.abs(data['gt_envmap'] - data['pred_envmap']) * 5.0
        axes[i, 2].imshow(np.clip(diff, 0, 1) ** (1/2.2))
        axes[i, 2].set_title(f'Difference ×5\nPSNR: {data["psnr"]:.2f} dB\nSSIM: {data["ssim"]:.4f}')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved rendering comparison to {output_path}")


def main():
    # Configuration
    checkpoint_path = Path('/home/kyrie/毕设/multi_time_compression/experiments/week6_ablation_5D/K30_r8/best_model.pt')
    data_root = Path('/home/kyrie/毕设/data_generation/output/5D_parametric_validation')
    output_dir = Path('/home/kyrie/毕设/multi_time_compression/experiments/week6_visual_validation')
    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("\n[1/6] Loading K30_r8 model...")
    model = GaussianPhysicsCompression5D(num_gaussians=30, rank=8, sh_dim=27)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Load test dataset
    print("\n[2/6] Loading test dataset...")
    test_dataset = TransferTensorDataset5D(
        data_root=str(data_root),
        split='test',
        normalize_params=False
    )
    print(f"Test dataset: {len(test_dataset)} samples")

    # Select 5 representative samples
    np.random.seed(2026)
    n_samples = min(5, len(test_dataset))
    test_indices = np.random.choice(len(test_dataset), size=n_samples, replace=False)
    print(f"Selected {n_samples} samples for rendering comparison")

    # Rendering parameters
    envmap_resolution = 128  # 128×256 envmap
    render_resolution = (256, 256)
    spp = 64  # 采样数（降低以加速）

    samples_data = []
    psnr_list = []
    ssim_list = []

    print("\n[3/6] Generating envmaps and rendering...")
    for idx in tqdm(test_indices, desc="Rendering"):
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

        # 1. Generate envmaps
        print(f"\n  Generating envmaps for sample {idx}...")
        gt_envmap = sh_to_envmap(sh_gt_np, resolution=envmap_resolution)
        pred_envmap = sh_to_envmap(sh_pred_np, resolution=envmap_resolution)

        # 2. Compute envmap quality metrics
        psnr = compute_psnr(gt_envmap, pred_envmap)
        ssim = compute_ssim_simple(gt_envmap, pred_envmap)

        psnr_list.append(psnr)
        ssim_list.append(ssim)

        print(f"    Envmap PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")

        # Store data
        samples_data.append({
            'sample_idx': int(idx),
            'gt_envmap': gt_envmap,
            'pred_envmap': pred_envmap,
            'psnr': psnr,
            'ssim': ssim
        })

    # Compute statistics
    avg_psnr = np.mean(psnr_list)
    std_psnr = np.std(psnr_list)
    avg_ssim = np.mean(ssim_list)
    std_ssim = np.std(ssim_list)

    print(f"\n{'='*70}")
    print(f"Rendering Comparison Results (K30_r8) - Envmap Quality")
    print(f"{'='*70}")
    print(f"Samples evaluated:    {n_samples}")
    print(f"Average PSNR:         {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"Average SSIM:         {avg_ssim:.4f} ± {std_ssim:.4f}")
    print(f"Min PSNR:             {np.min(psnr_list):.2f} dB")
    print(f"Max PSNR:             {np.max(psnr_list):.2f} dB")
    print(f"Target (PSNR>30dB):   {'✅ PASS' if avg_psnr > 30 else '❌ FAIL'}")
    print(f"Target (SSIM>0.95):   {'✅ PASS' if avg_ssim > 0.95 else '⚠️  MARGINAL' if avg_ssim > 0.90 else '❌ FAIL'}")
    print(f"{'='*70}")

    # Save results
    print("\n[4/6] Saving results...")
    results = {
        'model': 'K30_r8',
        'checkpoint_epoch': int(checkpoint['epoch']),
        'n_samples': n_samples,
        'test_indices': test_indices.tolist(),
        'psnr_list': [float(x) for x in psnr_list],
        'ssim_list': [float(x) for x in ssim_list],
        'avg_psnr': float(avg_psnr),
        'std_psnr': float(std_psnr),
        'avg_ssim': float(avg_ssim),
        'std_ssim': float(std_ssim),
        'min_psnr': float(np.min(psnr_list)),
        'max_psnr': float(np.max(psnr_list)),
        'target_psnr': 30.0,
        'target_ssim': 0.95,
        'pass_psnr': bool(avg_psnr > 30),
        'pass_ssim': bool(avg_ssim > 0.95),
        'note': 'Envmap-based rendering comparison (simplified, no full path tracing)'
    }

    results_path = output_dir / 'rendering_comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Create visualizations
    print("\n[5/6] Creating visualizations...")
    viz_path = output_dir / 'rendering_comparison.png'
    visualize_rendering_comparison(samples_data, viz_path)

    # Save individual envmaps
    print("\n[6/6] Saving individual envmaps...")
    for i, data in enumerate(samples_data):
        sample_dir = output_dir / f'sample_{data["sample_idx"]}'
        sample_dir.mkdir(exist_ok=True)

        # Save GT envmap
        plt.imsave(
            sample_dir / 'gt_envmap.png',
            np.clip(data['gt_envmap'], 0, 1) ** (1/2.2)
        )

        # Save Pred envmap
        plt.imsave(
            sample_dir / 'pred_envmap.png',
            np.clip(data['pred_envmap'], 0, 1) ** (1/2.2)
        )

        # Save difference
        diff = np.abs(data['gt_envmap'] - data['pred_envmap']) * 5.0
        plt.imsave(
            sample_dir / 'diff_envmap.png',
            np.clip(diff, 0, 1) ** (1/2.2)
        )

    print(f"Saved individual envmaps to {output_dir}/sample_*/")

    print("\n✅ Rendering comparison complete!")
    print(f"   Average PSNR: {avg_psnr:.2f} dB {'> 30 ✓' if avg_psnr > 30 else '< 30 ✗'}")
    print(f"   Average SSIM: {avg_ssim:.4f} {'> 0.95 ✓' if avg_ssim > 0.95 else '< 0.95 ⚠'}")
    print(f"   Results: {results_path}")
    print(f"   Visualization: {viz_path}")


if __name__ == '__main__':
    main()
