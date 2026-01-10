"""Generate transmission tensor for dual-gaussian FBT validation.

Creates T[probe, light, sh_coeffs] tensor by rendering Cornell Box
with dynamic point light at multiple positions.

Usage:
    python generate_transfer_tensor.py --output output/transfer_tensor_validation
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
from cornell_box_builder import (
    build_cornell_box_with_point_light,
    generate_circular_light_trajectory,
    validate_light_trajectory
)


# Set Mitsuba variant
try:
    mi.set_variant('cuda_ad_rgb')
    print("Using CUDA variant")
except:
    mi.set_variant('llvm_ad_rgb')
    print("CUDA not available, using LLVM variant")


def generate_probe_grid(
    box_size: float = 1.0,
    grid_resolution: int = 7,
    margin: float = 0.1
) -> np.ndarray:
    """Generate uniform 3D grid of probe positions within Cornell Box.

    Args:
        box_size: Side length of Cornell Box
        grid_resolution: Number of probes per dimension (7^3 = 343)
        margin: Distance from walls (avoid boundary artifacts)

    Returns:
        probes: [N, 3] probe positions in world coordinates

    Notes:
        - Grid spans [-0.5+margin, 0.5-margin] in x, y
        - Grid spans [0+margin, 1.0-margin] in z (floor to ceiling)
        - Total probes: grid_resolution^3 = 343 for resolution=7
    """
    half_size = box_size / 2.0

    # Create 1D grids for each dimension
    x = np.linspace(-half_size + margin, half_size - margin, grid_resolution)
    y = np.linspace(-half_size + margin, half_size - margin, grid_resolution)
    z = np.linspace(margin, box_size - margin, grid_resolution)

    # Create 3D meshgrid
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    # Flatten to [N, 3]
    probes = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

    return probes


def bake_sh_at_probe(
    scene: mi.Scene,
    probe_position: np.ndarray,
    num_samples: int = 64,
    spp: int = 256
) -> np.ndarray:
    """Bake spherical harmonics coefficients at probe position.

    Args:
        scene: Loaded Mitsuba scene
        probe_position: [3] probe world position
        num_samples: Number of sampling directions (Fibonacci sphere)
        spp: Samples per pixel for rendering

    Returns:
        sh_coeffs: [27] SH coefficients (9 bases × 3 RGB channels)

    Notes:
        - Uses fibonacci_sphere for uniform directional sampling
        - Renders tiny 1x1 pixel images for each direction
        - Fits 2nd order SH (9 bases) via least-squares
    """
    # Generate sampling directions
    directions = fibonacci_sphere(num_samples)

    # Sample radiance for each direction
    radiances = []

    for direction in directions:
        # Create sensor looking in this direction
        sensor = mi.load_dict({
            'type': 'perspective',
            'fov': 1.0,  # Small FOV (approximate single direction)
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

        # Render
        image = mi.render(scene, sensor=sensor, spp=spp)
        radiance = np.array(image).flatten()  # [3] RGB

        radiances.append(radiance)

    radiances = np.array(radiances)  # [num_samples, 3]

    # Fit SH coefficients
    sh_coeffs = fit_sh_coefficients(directions, radiances, max_order=2)

    return sh_coeffs


def generate_transfer_tensor(
    output_dir: Path,
    num_probes: int = 343,
    num_light_positions: int = 12,
    num_samples: int = 64,
    spp: int = 256,
    light_intensity: float = 50.0,
    box_size: float = 1.0
):
    """Generate complete transmission tensor T[probe, light, sh_coeffs].

    Args:
        output_dir: Directory to save output
        num_probes: Number of probe positions (default 343 = 7^3)
        num_light_positions: Number of light positions on trajectory (default 12)
        num_samples: Directional samples for SH baking (default 64)
        spp: Samples per pixel for rendering (default 256)
        light_intensity: Point light radiant intensity (default 50.0)
        box_size: Cornell Box size (default 1.0m)

    Outputs:
        transfer_tensor.npz:
            - tensor: [num_probes, num_light_positions, 27] transmission tensor
            - probe_positions: [num_probes, 3] probe world positions
            - light_positions: [num_light_positions, 3] light world positions
            - metadata: dict with generation parameters

    Rendering estimate:
        343 probes × 12 lights × 64 samples × 256 spp ≈ 10-12 hours on RTX 4090
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate probe positions (7×7×7 grid)
    grid_resolution = int(np.round(num_probes ** (1/3)))
    probe_positions = generate_probe_grid(
        box_size=box_size,
        grid_resolution=grid_resolution,
        margin=0.1
    )
    actual_num_probes = len(probe_positions)

    print(f"\n[1/4] Generated {actual_num_probes} probe positions ({grid_resolution}^3 grid)")
    print(f"      Probe bounds: {probe_positions.min(axis=0)} to {probe_positions.max(axis=0)}")

    # Generate light trajectory (circular path above box)
    light_positions = generate_circular_light_trajectory(
        center=(0.0, 0.0, 1.5),  # Above box center
        radius=1.2,  # Outside box (box is 1m × 1m)
        num_positions=num_light_positions,
        start_angle=0.0
    )

    # Validate trajectory
    is_valid, msg = validate_light_trajectory(light_positions, box_size=box_size)
    if not is_valid:
        raise ValueError(f"Invalid light trajectory: {msg}")

    print(f"\n[2/4] Generated {num_light_positions} light positions (circular trajectory)")
    print(f"      Light center: (0.0, 0.0, 1.5), radius: 1.2m")
    print(f"      Validation: {msg}")

    # Initialize transmission tensor
    tensor = np.zeros((actual_num_probes, num_light_positions, 27), dtype=np.float32)

    # Build base scene dictionary (will update light position)
    print(f"\n[3/4] Rendering transmission tensor...")
    print(f"      Total renders: {actual_num_probes} probes × {num_light_positions} lights = {actual_num_probes * num_light_positions}")
    print(f"      Estimated time: 10-12 hours on RTX 4090")

    start_time = datetime.now()

    # Iterate over light positions (outer loop for efficiency)
    for light_idx, light_pos in enumerate(tqdm(light_positions, desc="Light positions")):

        # Build scene with current light position
        scene = build_cornell_box_with_point_light(
            light_position=light_pos,
            light_intensity=light_intensity,
            box_size=box_size
        )

        # Bake SH at all probes for this light position
        for probe_idx, probe_pos in enumerate(tqdm(probe_positions, desc=f"  Probes (light {light_idx+1}/{num_light_positions})", leave=False)):

            sh_coeffs = bake_sh_at_probe(
                scene=scene,
                probe_position=probe_pos,
                num_samples=num_samples,
                spp=spp
            )

            tensor[probe_idx, light_idx, :] = sh_coeffs

        # Save intermediate checkpoint every 3 light positions
        if (light_idx + 1) % 3 == 0:
            checkpoint_path = output_dir / f"checkpoint_light_{light_idx+1:02d}.npz"
            np.savez_compressed(
                checkpoint_path,
                tensor=tensor[:, :light_idx+1, :],  # Partial tensor
                probe_positions=probe_positions,
                light_positions=light_positions[:light_idx+1],
            )
            print(f"\n      Checkpoint saved: {checkpoint_path}")

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print(f"\n[4/4] Rendering complete in {elapsed/3600:.2f} hours")

    # Compute statistics
    tensor_mean = tensor.mean()
    tensor_std = tensor.std()
    tensor_min = tensor.min()
    tensor_max = tensor.max()

    print(f"\nTensor statistics:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Mean: {tensor_mean:.4f}")
    print(f"  Std:  {tensor_std:.4f}")
    print(f"  Range: [{tensor_min:.4f}, {tensor_max:.4f}]")

    # Save final output
    output_path = output_dir / "transfer_tensor.npz"

    metadata = {
        'num_probes': actual_num_probes,
        'num_light_positions': num_light_positions,
        'num_samples': num_samples,
        'spp': spp,
        'light_intensity': light_intensity,
        'box_size': box_size,
        'grid_resolution': grid_resolution,
        'rendering_time_hours': elapsed / 3600,
        'tensor_stats': {
            'mean': float(tensor_mean),
            'std': float(tensor_std),
            'min': float(tensor_min),
            'max': float(tensor_max)
        },
        'created_at': end_time.isoformat()
    }

    np.savez_compressed(
        output_path,
        tensor=tensor,
        probe_positions=probe_positions,
        light_positions=light_positions,
        metadata=metadata
    )

    # Save metadata as JSON for easy reading
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nOutput saved to: {output_path}")
    print(f"Metadata saved to: {output_dir / 'metadata.json'}")

    return tensor, probe_positions, light_positions, metadata


def main():
    parser = argparse.ArgumentParser(description="Generate transmission tensor for FBT validation")
    parser.add_argument('--output', type=str, default='output/transfer_tensor_validation',
                        help='Output directory')
    parser.add_argument('--num_probes', type=int, default=343,
                        help='Number of probes (default: 343 = 7^3 grid)')
    parser.add_argument('--num_lights', type=int, default=12,
                        help='Number of light positions (default: 12)')
    parser.add_argument('--num_samples', type=int, default=64,
                        help='Directional samples for SH baking (default: 64)')
    parser.add_argument('--spp', type=int, default=256,
                        help='Samples per pixel (default: 256)')
    parser.add_argument('--light_intensity', type=float, default=50.0,
                        help='Point light intensity (default: 50.0)')
    parser.add_argument('--box_size', type=float, default=1.0,
                        help='Cornell Box size in meters (default: 1.0)')

    args = parser.parse_args()

    output_dir = Path(args.output)

    print("=" * 80)
    print("Transfer Tensor Generation for Dual-Gaussian FBT Validation")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Output directory: {output_dir}")
    print(f"  Num probes: {args.num_probes}")
    print(f"  Num light positions: {args.num_lights}")
    print(f"  Directional samples: {args.num_samples}")
    print(f"  Samples per pixel: {args.spp}")
    print(f"  Light intensity: {args.light_intensity}")
    print(f"  Box size: {args.box_size}m")

    generate_transfer_tensor(
        output_dir=output_dir,
        num_probes=args.num_probes,
        num_light_positions=args.num_lights,
        num_samples=args.num_samples,
        spp=args.spp,
        light_intensity=args.light_intensity,
        box_size=args.box_size
    )

    print("\n" + "=" * 80)
    print("Generation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
