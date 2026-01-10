#!/usr/bin/env python3
"""Render GT vs Predicted comparison images for 1D intensity modulation.

Renders multiple time moments (different light intensities) and creates
side-by-side comparison showing:
1. GT rendering from Mitsuba with actual light intensity
2. Visualization based on predicted SH coefficients
3. Absolute error map
"""

import sys
from pathlib import Path

# Add parent directories to path - MUST be before other imports
script_dir = Path(__file__).parent
pgcpl_src_dir = script_dir.parent.parent  # PG-GCPL/src
project_root = pgcpl_src_dir.parent.parent.parent  # /home/kyrie/毕设

sys.path.insert(0, str(pgcpl_src_dir))  # PG-GCPL/src
sys.path.insert(0, str(project_root))  # /home/kyrie/毕设

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from models.gaussian_physics_1D import GaussianPhysicsCompression1D
from data.intensity_modulation_dataset import IntensityModulationDataset
from data_generation.core.cornell_box_builder import build_cornell_box_with_area_light


def intensity_function(t, period=3.0):
    """Sinusoidal intensity modulation."""
    return 0.5 + 0.5 * np.sin(2 * np.pi * t / period)


def build_cornell_box_scene(intensity=1.0, spp=256):
    """Build Cornell Box scene with modulated area light."""
    scene_dict = build_cornell_box_with_area_light(
        intensity=intensity,
        color=[1.0, 1.0, 1.0],
        box_size=1.0,
        light_scale=0.3
    )

    # Add integrator settings
    scene_dict['integrator'] = {
        'type': 'path',
        'max_depth': 8
    }

    # Add sensor (camera)
    scene_dict['sensor'] = {
        'type': 'perspective',
        'fov': 45.0,
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[0, 0.5, 2.5],   # Camera position
            target=[0, 0.5, 0],      # Look at center of box
            up=[0, 1, 0]
        ),
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 512,
            'rfilter': {'type': 'gaussian'},
            'pixel_format': 'rgb'
        },
        'sampler': {
            'type': 'independent',
            'sample_count': spp
        }
    }

    return scene_dict


def render_gt_image(intensity, spp=256):
    """Render ground truth image with Mitsuba."""
    scene_dict = build_cornell_box_scene(intensity, spp)
    scene = mi.load_dict(scene_dict)

    # Render
    image = mi.render(scene)

    # Convert to numpy [H, W, 3]
    image_np = np.array(image)

    return image_np


def eval_sh_radiance(sh_coeffs, direction):
    """Evaluate radiance in a given direction using SH coefficients.

    Args:
        sh_coeffs: [27] or [N, 27] SH coefficients (9 bases × 3 RGB)
        direction: [3] or [N, 3] unit direction vector

    Returns:
        radiance: [3] or [N, 3] RGB radiance values
    """
    # SH basis evaluation (2nd order)
    # Y_0^0 = 0.282095
    # Y_1^{-1} = 0.488603 * y
    # Y_1^0 = 0.488603 * z
    # Y_1^1 = 0.488603 * x
    # Y_2^{-2} = 1.092548 * x * y
    # Y_2^{-1} = 1.092548 * y * z
    # Y_2^0 = 0.315392 * (3*z^2 - 1)
    # Y_2^1 = 1.092548 * x * z
    # Y_2^2 = 0.546274 * (x^2 - y^2)

    single_direction = direction.ndim == 1
    if single_direction:
        direction = direction[np.newaxis, :]  # [1, 3]

    single_sh = sh_coeffs.ndim == 1
    if single_sh:
        sh_coeffs = sh_coeffs[np.newaxis, :]  # [1, 27]

    x, y, z = direction[:, 0], direction[:, 1], direction[:, 2]

    # Compute SH basis functions
    Y = np.zeros((direction.shape[0], 9))
    Y[:, 0] = 0.282095
    Y[:, 1] = 0.488603 * y
    Y[:, 2] = 0.488603 * z
    Y[:, 3] = 0.488603 * x
    Y[:, 4] = 1.092548 * x * y
    Y[:, 5] = 1.092548 * y * z
    Y[:, 6] = 0.315392 * (3 * z**2 - 1)
    Y[:, 7] = 1.092548 * x * z
    Y[:, 8] = 0.546274 * (x**2 - y**2)

    # Evaluate radiance for each RGB channel
    radiance = np.zeros((sh_coeffs.shape[0], 3))
    for c in range(3):
        sh_c = sh_coeffs[:, c*9:(c+1)*9]  # [N, 9]
        radiance[:, c] = np.sum(Y * sh_c, axis=1)

    # Clamp to [0, inf)
    radiance = np.maximum(radiance, 0)

    if single_direction and single_sh:
        return radiance[0]
    elif single_sh:
        return radiance[0]
    else:
        return radiance


def render_probe_visualization(probe_positions, sh_coeffs_all, camera_pos, camera_target,
                                image_size=512, probe_size=0.03):
    """Render probe visualization (each probe as a colored sphere).

    Args:
        probe_positions: [N, 3] probe positions
        sh_coeffs_all: [N, 27] SH coefficients
        camera_pos: [3] camera position
        camera_target: [3] camera look-at target
        image_size: int, output image size
        probe_size: float, visual size of each probe

    Returns:
        image: [H, W, 3] rendered image
    """
    # Create image buffer
    image = np.zeros((image_size, image_size, 3))
    depth_buffer = np.full((image_size, image_size), np.inf)

    # Camera setup
    camera_dir = camera_target - camera_pos
    camera_dir = camera_dir / np.linalg.norm(camera_dir)

    # Simple orthographic projection for visualization
    up = np.array([0, 1, 0])
    right = np.cross(camera_dir, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, camera_dir)

    # For each probe, compute view direction and color
    for i, (pos, sh_coeffs) in enumerate(zip(probe_positions, sh_coeffs_all)):
        # Vector from probe to camera
        view_dir = camera_pos - pos
        dist = np.linalg.norm(view_dir)
        view_dir = view_dir / dist

        # Evaluate radiance in view direction
        color = eval_sh_radiance(sh_coeffs, view_dir)

        # Tone mapping (simple gamma correction)
        color = np.power(np.clip(color, 0, 1), 1/2.2)

        # Project to screen space
        rel_pos = pos - camera_target
        x_screen = np.dot(rel_pos, right)
        y_screen = np.dot(rel_pos, up)
        z_screen = np.dot(rel_pos, camera_dir)

        # Convert to pixel coordinates
        scale = 200  # Adjust for viewport
        px = int(image_size / 2 + x_screen * scale)
        py = int(image_size / 2 - y_screen * scale)

        # Draw probe as a small circle
        if 0 <= px < image_size and 0 <= py < image_size:
            radius_px = int(probe_size * scale / (1 + z_screen * 0.5))  # Perspective
            for dy in range(-radius_px, radius_px + 1):
                for dx in range(-radius_px, radius_px + 1):
                    if dx**2 + dy**2 <= radius_px**2:
                        px_draw = px + dx
                        py_draw = py + dy
                        if 0 <= px_draw < image_size and 0 <= py_draw < image_size:
                            if dist < depth_buffer[py_draw, px_draw]:
                                depth_buffer[py_draw, px_draw] = dist
                                image[py_draw, px_draw] = color

    return image


def create_comparison_figure(moments_data, output_path):
    """Create multi-moment GT vs Pred comparison figure.

    Args:
        moments_data: List of dicts with keys:
            - 'time': float
            - 'intensity': float
            - 'gt_image': [H, W, 3]
            - 'pred_probe_vis': [H, W, 3]
            - 'gt_sh': [N, 27]
            - 'pred_sh': [N, 27]
        output_path: Path to save figure
    """
    n_moments = len(moments_data)

    fig = plt.figure(figsize=(18, 4 * n_moments))
    gs = GridSpec(n_moments, 5, figure=fig, hspace=0.3, wspace=0.1)

    for i, data in enumerate(moments_data):
        time = data['time']
        intensity = data['intensity']
        gt_image = data['gt_image']
        pred_probe_vis = data['pred_probe_vis']

        # Compute SH-based error
        sh_error = np.abs(data['gt_sh'] - data['pred_sh']).mean()

        # 1. GT Mitsuba rendering
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(np.clip(gt_image, 0, 1))
        ax1.set_title(f't={time:.2f}s\nIntensity={intensity:.3f}\nGT Rendering', fontsize=10)
        ax1.axis('off')

        # 2. Predicted probe visualization
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.imshow(np.clip(pred_probe_vis, 0, 1))
        ax2.set_title(f'Predicted\nProbe Visualization', fontsize=10)
        ax2.axis('off')

        # 3. GT probe visualization (for fair comparison)
        gt_probe_vis = data.get('gt_probe_vis', None)
        if gt_probe_vis is not None:
            ax3 = fig.add_subplot(gs[i, 2])
            ax3.imshow(np.clip(gt_probe_vis, 0, 1))
            ax3.set_title(f'GT\nProbe Visualization', fontsize=10)
            ax3.axis('off')

        # 4. Error visualization (probe-level)
        ax4 = fig.add_subplot(gs[i, 3])
        # Visualize per-probe error as colored scatter
        errors_per_probe = np.abs(data['gt_sh'] - data['pred_sh']).mean(axis=1)
        probe_positions = data['probe_positions']
        scatter = ax4.scatter(probe_positions[:, 0], probe_positions[:, 2],
                             c=errors_per_probe, cmap='hot', s=20, vmin=0, vmax=0.01)
        ax4.set_xlabel('X')
        ax4.set_ylabel('Z')
        ax4.set_title(f'Per-Probe Error\n(Mean SH MAE: {sh_error:.2e})', fontsize=10)
        ax4.set_aspect('equal')
        plt.colorbar(scatter, ax=ax4, label='MAE')

        # 5. Statistics
        ax5 = fig.add_subplot(gs[i, 4])
        ax5.axis('off')

        stats_text = f"""
Time: {time:.2f}s
Intensity: {intensity:.3f}

SH Coefficient Error:
  Mean MAE: {sh_error:.2e}
  Max error: {np.abs(data['gt_sh'] - data['pred_sh']).max():.2e}

Per-Probe Error:
  Mean: {errors_per_probe.mean():.2e}
  Median: {np.median(errors_per_probe):.2e}
  95th: {np.percentile(errors_per_probe, 95):.2e}

Quality:
  {"✓ Excellent" if sh_error < 1e-4 else "✓ Good" if sh_error < 1e-3 else "⚠ Fair"}
        """
        ax5.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
                verticalalignment='center', transform=ax5.transAxes)

    plt.suptitle('Multi-Moment GT vs Predicted Rendering Comparison\n1D Intensity Modulation',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths (use global project_root)
    data_root = project_root / 'data_generation/output/intensity_modulation_343'
    experiment_dir = project_root / 'multi_time_compression/PG-GCPL/experiments/experiment_groups/group2_intensity_modulation/01_baseline_training/K30_r8'

    output_dir = experiment_dir / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("1D Intensity Modulation: GT vs Predicted Rendering Comparison")
    print("="*80)
    print(f"Device: {device}")
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")

    # Load model
    print("\n[1/5] Loading best model...")
    model = GaussianPhysicsCompression1D(
        num_gaussians=30,
        rank=8,
        sh_dim=27
    ).to(device)

    checkpoint = torch.load(experiment_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Load probe positions
    print("\n[2/5] Loading probe positions...")
    probes_data = np.load(data_root / 'probes.npz')
    probe_positions = probes_data['positions']  # [343, 3]
    probe_positions_normalized = (probe_positions - probe_positions.mean(axis=0)) / probe_positions.std(axis=0)
    print(f"   Loaded {len(probe_positions)} probes")

    # Select test moments (diverse intensities)
    print("\n[3/5] Selecting test moments...")
    test_times = [0.0, 0.75, 1.5, 2.25]  # 4 moments with different intensities
    test_intensities = [intensity_function(t, period=3.0) for t in test_times]
    print(f"   Selected {len(test_times)} moments:")
    for t, intensity in zip(test_times, test_intensities):
        print(f"     t={t:.2f}s → intensity={intensity:.3f}")

    # Camera setup
    camera_pos = np.array([0, 0.5, 2.5])
    camera_target = np.array([0, 0.5, 0])

    # Process each moment
    print("\n[4/5] Rendering comparisons...")
    moments_data = []

    for time, intensity in zip(test_times, test_intensities):
        print(f"\n   Processing t={time:.2f}s (intensity={intensity:.3f})...")

        # 1. Render GT with Mitsuba
        print(f"     [1/4] Rendering GT image...")
        gt_image = render_gt_image(intensity, spp=256)

        # 2. Load GT SH coefficients
        print(f"     [2/4] Loading GT SH coefficients...")
        # Find closest moment in dataset
        moment_idx = int(time / 0.25)  # Dataset has 0.25s intervals
        moment_dir = data_root / f'moment_{moment_idx:02d}'
        sh_data = np.load(moment_dir / 'sh_coeffs.npz')
        gt_sh = sh_data['sh_coeffs']  # [343, 27]

        # 3. Predict SH coefficients with model
        print(f"     [3/4] Predicting SH coefficients...")
        with torch.no_grad():
            positions_torch = torch.from_numpy(probe_positions_normalized).float().to(device)
            intensity_torch = torch.full((len(probe_positions), 1), intensity).float().to(device)
            pred_sh_torch = model(positions_torch, intensity_torch, top_k=3)
            pred_sh = pred_sh_torch.cpu().numpy()  # [343, 27]

        # 4. Render probe visualizations
        print(f"     [4/4] Rendering probe visualizations...")
        pred_probe_vis = render_probe_visualization(
            probe_positions, pred_sh, camera_pos, camera_target,
            image_size=512, probe_size=0.03
        )

        gt_probe_vis = render_probe_visualization(
            probe_positions, gt_sh, camera_pos, camera_target,
            image_size=512, probe_size=0.03
        )

        # Store data
        moments_data.append({
            'time': time,
            'intensity': intensity,
            'gt_image': gt_image,
            'pred_probe_vis': pred_probe_vis,
            'gt_probe_vis': gt_probe_vis,
            'gt_sh': gt_sh,
            'pred_sh': pred_sh,
            'probe_positions': probe_positions
        })

        print(f"     ✓ Moment complete (SH MAE: {np.abs(gt_sh - pred_sh).mean():.2e})")

    # Create comparison figure
    print("\n[5/5] Creating comparison figure...")
    create_comparison_figure(moments_data, output_dir / 'rendering_comparison_multi_moment.png')

    print("\n" + "="*80)
    print("Rendering comparison complete!")
    print(f"Output: {output_dir / 'rendering_comparison_multi_moment.png'}")
    print("="*80)


if __name__ == '__main__':
    main()
