"""
TPE Level 1 Toy Scene Dataset Generation

Generate validation datasets for TPE (Temporal Perturbation Embedding) using
controlled simple scenes to verify the core hypothesis: lighting changes can
be approximated by coordinate perturbations ε(t).

Usage:
    python generate_level1_toy_scenes.py \
        --scene shadow_plane \
        --output output/level1_tpe/shadow_plane \
        --spp 256 \
        --num-sh-samples 64
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import argparse
import json
from tqdm import tqdm
import mitsuba as mi

from core.lighting_modifier import LightingModifier
from core.sh_baker import bake_sh_at_probe
from utils.sun_position import calculate_sun_direction_simple


# 12 time steps: 6am to 6pm (hourly)
HOURS_12 = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]


# Scene configurations
SCENE_CONFIGS = {
    'shadow_plane': {
        'xml_path': 'scenes/toy_level1/shadow_plane.xml',
        'probe_type': 'planar',
        'description': 'Flat plane with cube casting moving shadow'
    },
    'rotating_sphere': {
        'xml_path': 'scenes/toy_level1/rotating_sphere.xml',
        'probe_type': 'spherical',
        'description': 'Sphere with rotating directional lighting'
    },
    'cornell_moving_light': {
        'xml_path': 'scenes/cornell-box/scene.xml',
        'probe_type': 'cubic',
        'description': 'Cornell box with directional sun (modified)'
    }
}


def get_probe_grid_planar(size=5, spacing=1.6, z=0.1):
    """
    Generate 5×5 planar grid of probe positions.

    Used for shadow_plane scene to sample lighting on ground plane.

    Args:
        size: Grid size per dimension (default 5 for 5×5=25 probes)
        spacing: Distance between adjacent probes (meters)
        z: Height above ground plane (meters)

    Returns:
        positions: [25, 3] numpy array of probe coordinates
    """
    x = np.linspace(-spacing*2, spacing*2, size)
    y = np.linspace(-spacing*2, spacing*2, size)
    xx, yy = np.meshgrid(x, y)

    positions = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        np.full(size*size, z)
    ])

    return positions


def get_probe_spherical_shell(n=25, radius=2.5):
    """
    Generate Fibonacci sphere sampling on shell.

    Used for rotating_sphere scene to sample lighting around central sphere.
    Uses golden ratio spiral for uniform distribution on sphere surface.

    Args:
        n: Number of samples
        radius: Sphere radius (meters)

    Returns:
        positions: [n, 3] numpy array of probe coordinates
    """
    indices = np.arange(n) + 0.5

    # Fibonacci lattice on sphere
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices

    # Convert to Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    positions = np.column_stack([x, y, z])

    return positions


def get_probe_cubic_grid(nx=4, ny=4, nz=4, bounds=[-0.8, 0.8], z_range=[0.3, 1.7]):
    """
    Generate 4×4×4 cubic grid of probe positions.

    Used for cornell_moving_light scene to sample inside enclosed box.

    Args:
        nx, ny, nz: Grid resolution per dimension
        bounds: [min, max] for x and y dimensions
        z_range: [min, max] for z dimension (height)

    Returns:
        positions: [64, 3] numpy array of probe coordinates
    """
    x = np.linspace(bounds[0], bounds[1], nx)
    y = np.linspace(bounds[0], bounds[1], ny)
    z = np.linspace(z_range[0], z_range[1], nz)

    xx, yy, zz = np.meshgrid(x, y, z)

    positions = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        zz.ravel()
    ])

    return positions


def get_probe_positions(probe_type: str) -> np.ndarray:
    """
    Get probe positions based on scene type.

    Args:
        probe_type: 'planar', 'spherical', or 'cubic'

    Returns:
        positions: [N, 3] probe positions
    """
    if probe_type == 'planar':
        return get_probe_grid_planar(size=5, spacing=1.6, z=0.1)
    elif probe_type == 'spherical':
        return get_probe_spherical_shell(n=25, radius=2.5)
    elif probe_type == 'cubic':
        return get_probe_cubic_grid(nx=4, ny=4, nz=4)
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")


def generate_level1_dataset(
    scene_name: str,
    scene_xml_path: Path,
    probe_positions: np.ndarray,
    output_dir: Path,
    spp: int = 256,
    num_sh_samples: int = 64
):
    """
    Generate Level 1 TPE validation dataset.

    For each of 12 time steps (hours 6-18):
        1. Calculate sun direction
        2. Modify scene lighting
        3. Bake SH coefficients at all probe positions
        4. Save results

    Args:
        scene_name: Scene identifier
        scene_xml_path: Path to scene XML file
        probe_positions: [N, 3] probe coordinates
        output_dir: Output directory path
        spp: Samples per pixel for rendering
        num_sh_samples: Number of directional samples for SH fitting
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save probe positions
    np.savez_compressed(output_dir / "probes.npz", positions=probe_positions)

    print(f"\n{'='*70}")
    print(f"Generating Level 1 Dataset: {scene_name}")
    print(f"{'='*70}")
    print(f"  Scene: {scene_xml_path}")
    print(f"  Probes: {len(probe_positions)}")
    print(f"  Time steps: {len(HOURS_12)}")
    print(f"  Quality: spp={spp}, sh_samples={num_sh_samples}")
    print(f"  Output: {output_dir}")
    print()

    # Generate data for each time step
    for hour in HOURS_12:
        print(f"\n--- Hour {hour:02d}:00 ---")

        # Calculate sun direction
        sun_dir = calculate_sun_direction_simple(hour)
        altitude = np.degrees(np.arcsin(sun_dir[1]))

        print(f"  Sun direction: [{sun_dir[0]:.3f}, {sun_dir[1]:.3f}, {sun_dir[2]:.3f}]")
        print(f"  Sun altitude: {altitude:.1f}°")

        # Modify scene lighting
        print(f"  Loading scene with modified lighting...")
        scene = LightingModifier.modify_scene_lighting(
            Path(scene_xml_path), sun_dir
        )

        # Bake SH coefficients at all probes
        print(f"  Baking SH coefficients for {len(probe_positions)} probes...")
        sh_coeffs_list = []

        for probe_pos in tqdm(probe_positions, desc="    Progress"):
            sh = bake_sh_at_probe(scene, probe_pos, num_sh_samples, spp)
            sh_coeffs_list.append(sh)

        # Save this moment's data
        moment_dir = output_dir / f"moment_{hour:02d}"
        moment_dir.mkdir(exist_ok=True)

        sh_coeffs = np.array(sh_coeffs_list)  # [N, 27]

        np.savez_compressed(
            moment_dir / "sh_coeffs.npz",
            coeffs=sh_coeffs,
            sun_dir=sun_dir
        )

        with open(moment_dir / "sun_direction.txt", 'w') as f:
            f.write(f"{sun_dir[0]:.6f} {sun_dir[1]:.6f} {sun_dir[2]:.6f}\n")

        print(f"  ✓ Saved to {moment_dir}/")

    # Save dataset configuration
    config = {
        'scene_name': scene_name,
        'scene_xml': str(scene_xml_path),
        'level': 1,
        'num_probes': len(probe_positions),
        'time_steps': HOURS_12,
        'num_moments': len(HOURS_12),
        'spp': spp,
        'num_sh_samples': num_sh_samples,
        'purpose': 'TPE Level 1 toy scene validation',
        'expected_tpe_behavior': SCENE_CONFIGS.get(scene_name, {}).get('description', '')
    }

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✓ Level 1 dataset generation completed!")
    print(f"  Output: {output_dir}")
    print(f"  Total moments: {len(HOURS_12)}")
    print(f"  Total probes: {len(probe_positions)}")
    print(f"  Storage: ~{estimate_storage_mb(len(probe_positions), len(HOURS_12)):.1f} MB")
    print(f"{'='*70}\n")


def estimate_storage_mb(num_probes: int, num_moments: int) -> float:
    """Estimate storage size in MB."""
    # Each SH coefficient array is [num_probes, 27] float32
    bytes_per_moment = num_probes * 27 * 4
    total_bytes = bytes_per_moment * num_moments
    return total_bytes / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(
        description='Generate TPE Level 1 toy scene validation datasets'
    )

    parser.add_argument(
        '--scene',
        type=str,
        required=True,
        choices=list(SCENE_CONFIGS.keys()),
        help='Scene to generate (shadow_plane, rotating_sphere, cornell_moving_light)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory path'
    )

    parser.add_argument(
        '--spp',
        type=int,
        default=256,
        help='Samples per pixel for rendering (default: 256)'
    )

    parser.add_argument(
        '--num-sh-samples',
        type=int,
        default=64,
        help='Number of directional samples for SH fitting (default: 64)'
    )

    args = parser.parse_args()

    # Set Mitsuba variant
    print("Initializing Mitsuba...")
    try:
        mi.set_variant('cuda_ad_rgb')
        print("  ✓ Using CUDA variant (GPU acceleration)")
    except Exception:
        mi.set_variant('llvm_ad_rgb')
        print("  ✓ Using LLVM variant (CPU mode)")

    # Get scene configuration
    scene_config = SCENE_CONFIGS[args.scene]
    scene_xml_path = Path(scene_config['xml_path'])

    if not scene_xml_path.exists():
        print(f"ERROR: Scene file not found: {scene_xml_path}")
        sys.exit(1)

    # Get probe positions
    probe_type = scene_config['probe_type']
    probe_positions = get_probe_positions(probe_type)

    # Generate dataset
    generate_level1_dataset(
        scene_name=args.scene,
        scene_xml_path=scene_xml_path,
        probe_positions=probe_positions,
        output_dir=Path(args.output),
        spp=args.spp,
        num_sh_samples=args.num_sh_samples
    )


if __name__ == "__main__":
    main()
