"""
Generate 5D parametric light dataset for physics-guided validation.

Creates T[probe, light_config, sh_coeffs] tensor by rendering Cornell Box
with parametric light sources varying across 5D parameter space:
- sun_zenith, sun_azimuth (geometry)
- intensity, color_temp (photometry)
- cloud_cover (atmospheric)

Usage:
    python generate_5D_parametric.py --output output/5D_parametric_validation
    python generate_5D_parametric.py --resume --checkpoint output/5D_parametric_validation/checkpoint.npz
"""

import sys
from pathlib import Path

# Add utils and core to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))
sys.path.insert(0, str(Path(__file__).parent / "core"))

import argparse
import numpy as np
import mitsuba as mi
from tqdm import tqdm
import json
from datetime import datetime

from spherical_harmonics import fibonacci_sphere, fit_sh_coefficients
from parametric_light_builder import ParametricLightBuilder


# Set Mitsuba variant
try:
    mi.set_variant('cuda_ad_rgb')
    print("Using CUDA variant")
except:
    mi.set_variant('llvm_ad_rgb')
    print("CUDA not available, using LLVM variant")


def generate_probe_grid(
    box_size: float = 1.0,
    grid_resolution: int = 5,
    margin: float = 0.1
) -> np.ndarray:
    """
    Generate uniform 3D grid of probe positions (optimized for GPU).

    Args:
        box_size: Side length of Cornell Box
        grid_resolution: Probes per dimension (5^3 = 125 for GPU optimization)
        margin: Distance from walls

    Returns:
        probes: [N, 3] probe positions
    """
    half_size = box_size / 2.0

    x = np.linspace(-half_size + margin, half_size - margin, grid_resolution)
    y = np.linspace(-half_size + margin, half_size - margin, grid_resolution)
    z = np.linspace(margin, box_size - margin, grid_resolution)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    probes = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

    return probes


def build_cornell_box_base() -> dict:
    """
    Build Cornell Box scene without light source.

    Returns:
        scene_dict: Mitsuba scene dictionary (light will be added dynamically)
    """
    # Cornell Box geometry (reuse from cornell_box_builder.py)
    scene = {
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': 4  # Reduced for speed (still captures indirect lighting)
        },

        # Floor (white)
        'floor': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.scale([0.5, 0.5, 1]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}
            }
        },

        # Ceiling (white)
        'ceiling': {
            'type': 'rectangle',
            'to_world': (
                mi.ScalarTransform4f.translate([0, 0, 1]) @
                mi.ScalarTransform4f.rotate([1, 0, 0], 180) @
                mi.ScalarTransform4f.scale([0.5, 0.5, 1])
            ),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}
            }
        },

        # Back wall (white)
        'back_wall': {
            'type': 'rectangle',
            'to_world': (
                mi.ScalarTransform4f.translate([0, -0.5, 0.5]) @
                mi.ScalarTransform4f.rotate([1, 0, 0], -90) @
                mi.ScalarTransform4f.scale([0.5, 0.5, 1])
            ),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}
            }
        },

        # Left wall (red)
        'left_wall': {
            'type': 'rectangle',
            'to_world': (
                mi.ScalarTransform4f.translate([-0.5, 0, 0.5]) @
                mi.ScalarTransform4f.rotate([0, 1, 0], -90) @
                mi.ScalarTransform4f.scale([0.5, 0.5, 1])
            ),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.8, 0.2, 0.2]}
            }
        },

        # Right wall (green)
        'right_wall': {
            'type': 'rectangle',
            'to_world': (
                mi.ScalarTransform4f.translate([0.5, 0, 0.5]) @
                mi.ScalarTransform4f.rotate([0, 1, 0], 90) @
                mi.ScalarTransform4f.scale([0.5, 0.5, 1])
            ),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.2, 0.8, 0.2]}
            }
        },

        # Tall box
        'tall_box': {
            'type': 'cube',
            'to_world': (
                mi.ScalarTransform4f.translate([0.15, 0.2, 0.15]) @
                mi.ScalarTransform4f.rotate([0, 0, 1], -15) @
                mi.ScalarTransform4f.scale([0.15, 0.15, 0.3])
            ),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}
            }
        },

        # Short box
        'short_box': {
            'type': 'cube',
            'to_world': (
                mi.ScalarTransform4f.translate([-0.2, -0.15, 0.075]) @
                mi.ScalarTransform4f.rotate([0, 0, 1], 20) @
                mi.ScalarTransform4f.scale([0.15, 0.15, 0.15])
            ),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}
            }
        }
    }

    return scene


def bake_sh_at_probe(
    scene: mi.Scene,
    probe_position: np.ndarray,
    num_samples: int = 64,
    spp: int = 128  # Reduced from 256 for GPU optimization
) -> np.ndarray:
    """
    Bake SH coefficients at probe (optimized).

    Args:
        scene: Mitsuba scene
        probe_position: [3] probe position
        num_samples: Directional samples (64)
        spp: Samples per pixel (128 for speed)

    Returns:
        sh_coeffs: [27] SH coefficients
    """
    directions = fibonacci_sphere(num_samples)
    radiances = []

    for direction in directions:
        sensor = mi.load_dict({
            'type': 'perspective',
            'fov': 1.0,
            'to_world': mi.ScalarTransform4f.look_at(
                origin=probe_position,
                target=probe_position + direction,
                up=[0, 0, 1]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': 1,
                'height': 1,
                'rfilter': {'type': 'box'}
            }
        })

        image = mi.render(scene, sensor=sensor, spp=spp)
        radiance = np.array(image).flatten()
        radiances.append(radiance)

    radiances = np.array(radiances)
    sh_coeffs = fit_sh_coefficients(directions, radiances, max_order=2)

    return sh_coeffs


def save_checkpoint(
    output_dir: Path,
    config_idx: int,
    tensor: np.ndarray,
    configs: np.ndarray,
    probe_positions: np.ndarray
):
    """Save rendering checkpoint for resume capability."""
    checkpoint = {
        'config_idx': config_idx,
        'tensor': tensor,
        'configs': configs,
        'probe_positions': probe_positions,
        'timestamp': datetime.now().isoformat()
    }

    checkpoint_path = output_dir / 'checkpoint.npz'
    np.savez_compressed(checkpoint_path, **checkpoint)
    print(f"      Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load checkpoint to resume rendering."""
    data = np.load(checkpoint_path, allow_pickle=True)
    checkpoint = {
        'config_idx': int(data['config_idx']),
        'tensor': data['tensor'],
        'configs': data['configs'],
        'probe_positions': data['probe_positions']
    }
    return checkpoint


def generate_5D_parametric_dataset(
    output_dir: Path,
    num_probes: int = 125,
    num_geometric: int = 12,
    num_intensity: int = 20,
    num_atmospheric: int = 9,
    num_samples: int = 64,
    spp: int = 128,
    box_size: float = 1.0,
    checkpoint_interval: int = 10,
    resume_from: Path = None
):
    """
    Generate 5D parametric lighting dataset.

    Args:
        output_dir: Output directory
        num_probes: Number of probes (125 = 5^3, GPU optimized)
        num_geometric: Geometric configs (default 12)
        num_intensity: Intensity configs (default 20)
        num_atmospheric: Atmospheric configs (default 9)
        num_samples: Directional samples (64)
        spp: Samples per pixel (128 for speed)
        box_size: Cornell Box size (1.0m)
        checkpoint_interval: Save checkpoint every N configs
        resume_from: Checkpoint file to resume from

    Outputs:
        parametric_tensor.npz:
            - tensor: [num_probes, num_configs, 27]
            - probe_positions: [num_probes, 3]
            - light_configs: [num_configs, 5] (zenith, azimuth, intensity, temp, cloud)
            - metadata: dict
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize or resume
    if resume_from and resume_from.exists():
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = load_checkpoint(resume_from)
        probe_positions = checkpoint['probe_positions']
        configs = checkpoint['configs']
        tensor = checkpoint['tensor']
        start_config = checkpoint['config_idx'] + 1

        print(f"  Resuming from config {start_config}/{len(configs)}")
    else:
        # Generate probe grid (5×5×5 = 125)
        grid_resolution = int(np.round(num_probes ** (1/3)))
        probe_positions = generate_probe_grid(
            box_size=box_size,
            grid_resolution=grid_resolution,
            margin=0.1
        )
        actual_num_probes = len(probe_positions)

        print(f"\n[1/4] Generated {actual_num_probes} probes ({grid_resolution}^3 grid)")
        print(f"      Bounds: {probe_positions.min(axis=0)} to {probe_positions.max(axis=0)}")

        # Generate 5D light configurations (stratified)
        light_builder = ParametricLightBuilder(box_center=(0, 0, 0), box_size=box_size)
        configs = light_builder.sample_5D_stratified(
            num_geometric=num_geometric,
            num_intensity=num_intensity,
            num_atmospheric=num_atmospheric
        )
        num_configs = len(configs)

        print(f"\n[2/4] Generated {num_configs} light configurations (5D stratified)")
        print(f"      Geometric: {num_geometric}, Intensity: {num_intensity}, Atmospheric: {num_atmospheric}")

        # Initialize tensor
        tensor = np.zeros((actual_num_probes, num_configs, 27), dtype=np.float32)
        start_config = 0

    # Build base Cornell Box scene (without light)
    base_scene_dict = build_cornell_box_base()

    # Render loop
    print(f"\n[3/4] Rendering SH coefficients...")
    print(f"      Total: {len(probe_positions)} probes × {len(configs)} configs")
    print(f"      Estimated time: {len(configs) * 1.5:.0f} minutes ({len(configs) * 1.5 / 60:.1f} hours)")

    light_builder = ParametricLightBuilder(box_center=(0, 0, 0), box_size=box_size)

    for config_idx in tqdm(range(start_config, len(configs)), desc="Configs"):
        # Build light for this configuration
        zenith, azimuth, intensity, temp, cloud = configs[config_idx]
        light_dict = light_builder.build_5D_light(zenith, azimuth, intensity, temp, cloud)

        # Add light to scene
        scene_dict = base_scene_dict.copy()
        scene_dict['light'] = light_dict
        scene = mi.load_dict(scene_dict)

        # Render all probes for this light configuration
        for probe_idx, probe_pos in enumerate(probe_positions):
            sh_coeffs = bake_sh_at_probe(scene, probe_pos, num_samples=num_samples, spp=spp)
            tensor[probe_idx, config_idx, :] = sh_coeffs

        # Checkpoint every N configs
        if (config_idx + 1) % checkpoint_interval == 0:
            save_checkpoint(output_dir, config_idx, tensor, configs, probe_positions)

    # Save final results
    print(f"\n[4/4] Saving final dataset...")

    metadata = {
        'num_probes': len(probe_positions),
        'num_configs': len(configs),
        'num_geometric': num_geometric,
        'num_intensity': num_intensity,
        'num_atmospheric': num_atmospheric,
        'num_samples': num_samples,
        'spp': spp,
        'box_size': box_size,
        'generation_date': datetime.now().isoformat(),
        'parameter_ranges': {
            'sun_zenith': [0, 90],
            'sun_azimuth': [0, 360],
            'intensity': [0.5, 1.5],
            'color_temp': [3000, 8500],
            'cloud_cover': [0.0, 0.9]
        }
    }

    output_path = output_dir / 'parametric_tensor.npz'
    np.savez_compressed(
        output_path,
        tensor=tensor,
        probe_positions=probe_positions,
        light_configs=configs,
        metadata=metadata
    )

    # Save metadata as JSON
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset saved:")
    print(f"  Tensor: {output_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Shape: {tensor.shape}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Remove checkpoint
    checkpoint_path = output_dir / 'checkpoint.npz'
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"  Checkpoint removed")

    return tensor, probe_positions, configs, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate 5D parametric light dataset')
    parser.add_argument(
        '--output',
        type=str,
        default='output/5D_parametric_validation',
        help='Output directory'
    )
    parser.add_argument('--num_probes', type=int, default=125, help='Number of probes (5^3)')
    parser.add_argument('--num_geometric', type=int, default=12, help='Geometric configs')
    parser.add_argument('--num_intensity', type=int, default=20, help='Intensity configs')
    parser.add_argument('--num_atmospheric', type=int, default=9, help='Atmospheric configs')
    parser.add_argument('--num_samples', type=int, default=64, help='Directional samples')
    parser.add_argument('--spp', type=int, default=128, help='Samples per pixel')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Checkpoint interval')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')

    args = parser.parse_args()

    output_dir = Path(args.output)
    resume_from = output_dir / 'checkpoint.npz' if args.resume else None

    print("=" * 70)
    print("5D Parametric Light Dataset Generation")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Probes: {args.num_probes} ({int(args.num_probes**(1/3))}^3 grid)")
    print(f"Configs: {args.num_geometric + args.num_intensity + args.num_atmospheric}")
    print(f"  - Geometric: {args.num_geometric}")
    print(f"  - Intensity: {args.num_intensity}")
    print(f"  - Atmospheric: {args.num_atmospheric}")
    print(f"Sampling: {args.num_samples} directions × {args.spp} spp")
    print(f"Checkpoint: every {args.checkpoint_interval} configs")

    if args.resume and resume_from and resume_from.exists():
        print(f"Resume: YES (from {resume_from})")
    else:
        print(f"Resume: NO (fresh start)")

    print("=" * 70)

    tensor, probes, configs, metadata = generate_5D_parametric_dataset(
        output_dir=output_dir,
        num_probes=args.num_probes,
        num_geometric=args.num_geometric,
        num_intensity=args.num_intensity,
        num_atmospheric=args.num_atmospheric,
        num_samples=args.num_samples,
        spp=args.spp,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=resume_from
    )

    print("\n✓ Generation complete!")
