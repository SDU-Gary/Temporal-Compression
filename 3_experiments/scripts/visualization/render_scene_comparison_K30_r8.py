#!/usr/bin/env python3
"""
场景渲染对比: GT SH vs Pred SH光照下的Cornell Box渲染

Pipeline:
1. 将GT/Pred SH转换为高分辨率envmap
2. 使用envmap作为场景光照渲染Cornell Box
3. 对比渲染图像的PSNR/SSIM

与之前render_comparison的区别:
- 之前: 直接展示envmap图像 (模糊的球面贴图)
- 现在: 用envmap照亮场景，展示场景渲染结果 (真实相机视角)
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

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from models.gaussian_physics_5D import GaussianPhysicsCompression5D
from data.transfer_tensor_dataset import TransferTensorDataset5D

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data_generation')))
from utils.spherical_harmonics import reconstruct_from_sh


def sh_to_envmap(sh_coeffs: np.ndarray, resolution: int = 512) -> np.ndarray:
    """
    SH系数转高分辨率环境贴图

    Args:
        sh_coeffs: [27] SH coefficients
        resolution: envmap分辨率

    Returns:
        envmap: [H, W, 3] HDR环境贴图
    """
    H, W = resolution, resolution * 2  # Equirectangular: 1:2 aspect ratio

    # 生成theta/phi网格
    theta = np.linspace(0, np.pi, H)
    phi = np.linspace(0, 2 * np.pi, W)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

    # 转换为笛卡尔坐标方向
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.cos(theta_grid)  # Mitsuba uses Y-up
    z = np.sin(theta_grid) * np.sin(phi_grid)

    directions = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)  # [H*W, 3]

    # 用SH重建辐射
    radiances = reconstruct_from_sh(sh_coeffs, directions)  # [H*W, 3]

    # Clamp负值并reshape
    radiances = np.maximum(radiances, 0.0)
    envmap = radiances.reshape(H, W, 3)

    return envmap


def render_with_envmap(envmap_path: str,
                       scene_template_path: str,
                       output_path: str,
                       resolution: int = 256,
                       spp: int = 64):
    """
    用环境贴图渲染Cornell Box场景

    Args:
        envmap_path: 环境贴图路径 (.exr)
        scene_template_path: 场景模板路径 (.xml)
        output_path: 输出路径 (.exr)
        resolution: 渲染分辨率
        spp: 每像素采样数
    """
    # 读取场景模板
    with open(scene_template_path, 'r') as f:
        scene_xml = f.read()

    # 替换光源为envmap
    # 移除原有area light，添加envmap emitter
    # 找到Light shape并替换emitter
    scene_xml_modified = scene_xml.replace(
        '''<emitter type="area">
\t\t\t<rgb name="radiance" value="17, 12, 4" />
\t\t</emitter>''',
        ''  # 移除area light emitter
    )

    # 在场景根节点添加envmap emitter
    envmap_emitter = f'''
    <emitter type="envmap">
        <string name="filename" value="{envmap_path}"/>
        <float name="scale" value="1.0"/>
    </emitter>'''

    scene_xml_modified = scene_xml_modified.replace(
        '</scene>',
        envmap_emitter + '\n</scene>'
    )

    # 修改分辨率和SPP
    scene_xml_modified = scene_xml_modified.replace(
        '<default name="spp" value="64" />',
        f'<default name="spp" value="{spp}" />'
    )
    scene_xml_modified = scene_xml_modified.replace(
        '<default name="resx" value="1024" />',
        f'<default name="resx" value="{resolution}" />'
    )
    scene_xml_modified = scene_xml_modified.replace(
        '<default name="resy" value="1024" />',
        f'<default name="resy" value="{resolution}" />'
    )

    # 保存临时场景文件
    temp_scene_path = output_path.replace('.exr', '_scene.xml')
    with open(temp_scene_path, 'w') as f:
        f.write(scene_xml_modified)

    # 加载并渲染场景
    scene = mi.load_file(temp_scene_path)
    image = mi.render(scene)

    # 保存渲染结果
    mi.util.write_bitmap(output_path, image)

    # 清理临时文件
    os.remove(temp_scene_path)


def compute_image_metrics(img_gt: np.ndarray, img_pred: np.ndarray) -> dict:
    """
    计算图像质量指标

    Args:
        img_gt, img_pred: [H, W, 3] 图像 (linear RGB)

    Returns:
        metrics: {psnr, ssim, mae}
    """
    # MSE and PSNR
    mse = np.mean((img_gt - img_pred) ** 2)
    if mse < 1e-10:
        psnr = 100.0
    else:
        psnr = 10 * np.log10(img_gt.max() ** 2 / mse)

    # Simple SSIM (grayscale)
    gray_gt = np.mean(img_gt, axis=-1)
    gray_pred = np.mean(img_pred, axis=-1)

    mu_gt = np.mean(gray_gt)
    mu_pred = np.mean(gray_pred)
    sigma_gt = np.std(gray_gt)
    sigma_pred = np.std(gray_pred)
    sigma_gt_pred = np.mean((gray_gt - mu_gt) * (gray_pred - mu_pred))

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    ssim = ((2 * mu_gt * mu_pred + C1) * (2 * sigma_gt_pred + C2)) / \
           ((mu_gt**2 + mu_pred**2 + C1) * (sigma_gt**2 + sigma_pred**2 + C2))
    ssim = max(0.0, min(1.0, ssim))

    # MAE
    mae = np.mean(np.abs(img_gt - img_pred))

    return {
        'psnr': float(psnr),
        'ssim': float(ssim),
        'mae': float(mae)
    }


def tonemap_reinhard(img: np.ndarray, key: float = 0.18) -> np.ndarray:
    """
    Reinhard色调映射用于显示

    Args:
        img: [H, W, 3] HDR图像
        key: 目标亮度

    Returns:
        ldr: [H, W, 3] LDR图像 [0, 1]
    """
    luminance = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    avg_lum = np.exp(np.mean(np.log(luminance + 1e-8)))
    scaled_lum = (key / avg_lum) * luminance
    mapped_lum = scaled_lum / (1 + scaled_lum)

    # 保持色彩比例
    ldr = img * (mapped_lum / (luminance + 1e-8))[:, :, None]
    ldr = np.clip(ldr, 0, 1)

    return ldr


def main():
    # Configuration
    checkpoint_path = Path('/home/kyrie/毕设/multi_time_compression/experiments/week6_ablation_5D/K30_r8/best_model.pt')
    data_root = Path('/home/kyrie/毕设/data_generation/output/5D_parametric_validation')
    scene_template = Path('/home/kyrie/毕设/data_generation/scenes/cornell-box/scene.xml')
    output_dir = Path('/home/kyrie/毕设/multi_time_compression/experiments/week6_visual_validation/scene_rendering')
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

    # Load test dataset
    print("\n[2/5] Loading test dataset...")
    test_dataset = TransferTensorDataset5D(
        data_root=str(data_root),
        split='test',
        normalize_params=False
    )

    # Select test samples (same as radiance visualization)
    test_indices = [121, 530, 495, 612, 597]
    print(f"Selected {len(test_indices)} samples for scene rendering")

    # Render parameters
    render_resolution = 256  # 256×256像素 (平衡质量与速度)
    render_spp = 64  # 64 SPP
    envmap_resolution = 512  # 512×1024 envmap

    results = {
        'model': 'K30_r8',
        'checkpoint_epoch': checkpoint.get('epoch', -1),
        'n_samples': len(test_indices),
        'test_indices': test_indices,
        'render_settings': {
            'resolution': render_resolution,
            'spp': render_spp,
            'envmap_resolution': envmap_resolution
        },
        'per_sample_metrics': []
    }

    print("\n[3/5] Converting SH to envmaps...")
    for idx in tqdm(test_indices):
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

        # Convert to envmaps
        envmap_gt = sh_to_envmap(sh_gt_np, resolution=envmap_resolution)
        envmap_pred = sh_to_envmap(sh_pred_np, resolution=envmap_resolution)

        # Save envmaps (convert numpy to Mitsuba Bitmap)
        envmap_gt_path = output_dir / f'sample_{idx}_envmap_gt.exr'
        envmap_pred_path = output_dir / f'sample_{idx}_envmap_pred.exr'

        # Mitsuba expects [H, W, C] float32
        envmap_gt_mi = mi.Bitmap(envmap_gt.astype(np.float32))
        envmap_pred_mi = mi.Bitmap(envmap_pred.astype(np.float32))

        envmap_gt_mi.write(str(envmap_gt_path))
        envmap_pred_mi.write(str(envmap_pred_path))

    print("\n[4/5] Rendering scenes with Mitsuba...")
    for idx in tqdm(test_indices):
        envmap_gt_path = output_dir / f'sample_{idx}_envmap_gt.exr'
        envmap_pred_path = output_dir / f'sample_{idx}_envmap_pred.exr'

        render_gt_path = output_dir / f'sample_{idx}_render_gt.exr'
        render_pred_path = output_dir / f'sample_{idx}_render_pred.exr'

        # Render GT
        render_with_envmap(
            str(envmap_gt_path),
            str(scene_template),
            str(render_gt_path),
            resolution=render_resolution,
            spp=render_spp
        )

        # Render Pred
        render_with_envmap(
            str(envmap_pred_path),
            str(scene_template),
            str(render_pred_path),
            resolution=render_resolution,
            spp=render_spp
        )

    print("\n[5/5] Computing metrics and generating visualizations...")
    psnr_list = []
    ssim_list = []
    mae_list = []

    for idx in test_indices:
        render_gt_path = output_dir / f'sample_{idx}_render_gt.exr'
        render_pred_path = output_dir / f'sample_{idx}_render_pred.exr'

        # Load rendered images
        img_gt_bmp = mi.Bitmap(str(render_gt_path))
        img_pred_bmp = mi.Bitmap(str(render_pred_path))

        # Convert to numpy
        img_gt_np = np.array(img_gt_bmp)
        img_pred_np = np.array(img_pred_bmp)

        # Compute metrics
        metrics = compute_image_metrics(img_gt_np, img_pred_np)
        psnr_list.append(metrics['psnr'])
        ssim_list.append(metrics['ssim'])
        mae_list.append(metrics['mae'])

        results['per_sample_metrics'].append({
            'sample_idx': int(idx),
            **metrics
        })

        # Visualization (tone-mapped for display)
        img_gt_ldr = tonemap_reinhard(img_gt_np)
        img_pred_ldr = tonemap_reinhard(img_pred_np)
        diff = np.abs(img_gt_ldr - img_pred_ldr) * 5.0  # 放大5倍显示差异

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(np.clip(img_gt_ldr ** (1/2.2), 0, 1))
        axes[0].set_title(f'Sample {idx} - GT Render')
        axes[0].axis('off')

        axes[1].imshow(np.clip(img_pred_ldr ** (1/2.2), 0, 1))
        axes[1].set_title(f'Pred Render\nPSNR: {metrics["psnr"]:.2f} dB, SSIM: {metrics["ssim"]:.3f}')
        axes[1].axis('off')

        axes[2].imshow(np.clip(diff ** (1/2.2), 0, 1))
        axes[2].set_title('Absolute Diff ×5')
        axes[2].axis('off')

        plt.tight_layout()
        viz_path = output_dir / f'sample_{idx}_comparison.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nSample {idx}:")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.3f}")
        print(f"  MAE:  {metrics['mae']:.6f}")

    # Summary statistics
    results['avg_psnr'] = float(np.mean(psnr_list))
    results['std_psnr'] = float(np.std(psnr_list))
    results['avg_ssim'] = float(np.mean(ssim_list))
    results['std_ssim'] = float(np.std(ssim_list))
    results['avg_mae'] = float(np.mean(mae_list))

    # Save results
    results_path = output_dir / 'scene_rendering_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PSNR distribution
    axes[0].bar(range(len(test_indices)), psnr_list, alpha=0.7, color='steelblue')
    axes[0].axhline(y=30.0, color='red', linestyle='--', label='Target 30dB')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title(f'Scene Render PSNR (Avg: {results["avg_psnr"]:.2f} ± {results["std_psnr"]:.2f} dB)')
    axes[0].set_xticks(range(len(test_indices)))
    axes[0].set_xticklabels([str(idx) for idx in test_indices])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # SSIM distribution
    axes[1].bar(range(len(test_indices)), ssim_list, alpha=0.7, color='coral')
    axes[1].axhline(y=0.95, color='red', linestyle='--', label='Target 0.95')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title(f'Scene Render SSIM (Avg: {results["avg_ssim"]:.3f} ± {results["std_ssim"]:.3f})')
    axes[1].set_xticks(range(len(test_indices)))
    axes[1].set_xticklabels([str(idx) for idx in test_indices])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    summary_viz_path = output_dir / 'summary_metrics.png'
    plt.savefig(summary_viz_path, dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*80)
    print("Scene Rendering Results Summary")
    print("="*80)
    print(f"Average PSNR: {results['avg_psnr']:.2f} ± {results['std_psnr']:.2f} dB")
    print(f"Average SSIM: {results['avg_ssim']:.3f} ± {results['std_ssim']:.3f}")
    print(f"Average MAE:  {results['avg_mae']:.6f}")
    print("="*80)

    print(f"\n✅ Scene rendering comparison complete!")
    print(f"   Output directory: {output_dir}")
    print(f"   Results saved to: {results_path}")
    print(f"\nKey differences from previous envmap visualization:")
    print(f"  - Previous: Showed envmap image directly (blurry spherical map)")
    print(f"  - Current: Cornell Box scene rendered WITH envmap lighting")
    print(f"  - This shows: How GT vs Pred SH affect final scene appearance")


if __name__ == '__main__':
    main()
