"""
TPE Level 2 Single Probe Validation Dataset Generation

Generate high-quality datasets for TPE critical validation using complex scenes.
This is the make-or-break test: if a scene passes all 3 criteria, TPE is viable.

Validation Criteria:
1. ||ε(t)||_max < 1.0m  (Taylor expansion validity)
2. avg(||ε(t+1) - ε(t)||) < 0.5m  (temporal smoothness)
3. correlation(||ε||, ||Δsun_dir||) > 0.5  (physical correlation)

Usage:
    python generate_level2_single_probe.py \
        --scene cornell-box \
        --output output/level2_tpe/cornell-box \
        --spp 512 \
        --num-sh-samples 128
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import argparse
import json
import mitsuba as mi

from core.lighting_modifier import LightingModifier
from core.sh_baker import bake_sh_at_probe
from utils.sun_position import calculate_sun_direction_simple


# 12 time steps: 6am to 6pm (hourly)
HOURS_12 = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]


# Strategic probe positions for Level 2 validation
SCENE_CONFIGS = {
    'cornell-box': {
        'xml_path': 'scenes/cornell-box/scene.xml',
        'probe_position': np.array([0.0, 1.0, 0.0]),
        'expected_epsilon_range': [0.3, 0.6],
        'rationale': 'Center of box, simple geometry, multi-bounce lighting'
    },
    'classroom': {
        'xml_path': 'scenes/classroom/scene.xml',
        'probe_position': np.array([-1.5, 0.5, 1.2]),
        'expected_epsilon_range': [0.5, 1.0],
        'rationale': 'Near window, environment map variation significant'
    },
    'house': {
        'xml_path': 'scenes/house/scene.xml',
        'probe_position': np.array([2.0, -1.0, 1.0]),
        'expected_epsilon_range': [0.6, 1.2],
        'rationale': 'Living room area, complex multi-room geometry'
    }
}


def generate_level2_dataset(
    scene_name: str,
    scene_xml_path: Path,
    probe_position: np.ndarray,
    output_dir: Path,
    spp: int = 512,
    num_sh_samples: int = 128
):
    """
    Generate Level 2 single-probe high-quality validation dataset.

    This dataset is used for the critical TPE hypothesis test. If the
    optimized perturbation vectors satisfy all 3 criteria, TPE is viable
    for this scene type.

    Args:
        scene_name: Scene identifier
        scene_xml_path: Path to scene XML file
        probe_position: [3] single probe coordinate
        output_dir: Output directory path
        spp: Samples per pixel for rendering (higher quality)
        num_sh_samples: Number of directional samples for SH fitting (higher quality)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save probe position
    np.savez_compressed(
        output_dir / "probes.npz",
        positions=probe_position.reshape(1, 3)
    )

    print(f"\n{'='*70}")
    print(f"Generating Level 2 Dataset: {scene_name}")
    print(f"{'='*70}")
    print(f"  Scene: {scene_xml_path}")
    print(f"  Probe position: [{probe_position[0]:.2f}, {probe_position[1]:.2f}, {probe_position[2]:.2f}]")
    print(f"  Time steps: {len(HOURS_12)}")
    print(f"  Quality: spp={spp}, sh_samples={num_sh_samples} (HIGH)")
    print(f"  Output: {output_dir}")
    print(f"\n  Purpose: TPE Level 2 CRITICAL VALIDATION")
    print(f"  Expected ||ε|| range: {SCENE_CONFIGS[scene_name]['expected_epsilon_range']}")
    print()

    sun_directions = []

    # Generate data for each time step
    for hour in HOURS_12:
        print(f"\n--- Hour {hour:02d}:00 ---")

        # Calculate sun direction
        sun_dir = calculate_sun_direction_simple(hour)
        sun_directions.append(sun_dir)

        altitude = np.degrees(np.arcsin(sun_dir[1]))

        print(f"  Sun direction: [{sun_dir[0]:.3f}, {sun_dir[1]:.3f}, {sun_dir[2]:.3f}]")
        print(f"  Sun altitude: {altitude:.1f}°")

        # Modify scene lighting
        print(f"  Loading scene with modified lighting...")
        scene = LightingModifier.modify_scene_lighting(
            Path(scene_xml_path), sun_dir
        )

        # High-quality SH baking for single probe
        print(f"  Baking high-quality SH coefficients...")
        print(f"    (This may take a while with {spp} SPP × {num_sh_samples} samples)")

        sh_coeffs = bake_sh_at_probe(
            scene, probe_position,
            num_samples=num_sh_samples,
            spp=spp
        )

        # Save this moment's data
        moment_dir = output_dir / f"moment_{hour:02d}"
        moment_dir.mkdir(exist_ok=True)

        np.savez_compressed(
            moment_dir / "sh_coeffs.npz",
            coeffs=sh_coeffs.reshape(1, 27),  # [1, 27]
            sun_dir=sun_dir
        )

        with open(moment_dir / "sun_direction.txt", 'w') as f:
            f.write(f"{sun_dir[0]:.6f} {sun_dir[1]:.6f} {sun_dir[2]:.6f}\n")

        print(f"  ✓ Saved to {moment_dir}/")

    # Save complete sun trajectory for correlation analysis
    print(f"\n  Saving sun trajectory...")
    np.savez_compressed(
        output_dir / "sun_trajectory.npz",
        directions=np.array(sun_directions),  # [12, 3]
        hours=np.array(HOURS_12)
    )

    # Save metadata with validation criteria
    scene_config = SCENE_CONFIGS[scene_name]

    metadata = {
        'scene_name': scene_name,
        'scene_xml': str(scene_xml_path),
        'level': 2,
        'probe_position': probe_position.tolist(),
        'time_steps': HOURS_12,
        'num_moments': len(HOURS_12),
        'spp': spp,
        'num_sh_samples': num_sh_samples,
        'expected_epsilon_range_meters': scene_config['expected_epsilon_range'],
        'probe_selection_rationale': scene_config['rationale'],
        'validation_criteria': {
            'max_perturbation_norm': 1.0,
            'avg_temporal_smoothness': 0.5,
            'sun_correlation_threshold': 0.5
        },
        'purpose': 'TPE Level 2 critical validation - MAKE OR BREAK TEST',
        'interpretation': {
            'if_pass': 'TPE hypothesis is VALID, proceed to Level 3 full system',
            'if_fail': 'TPE hypothesis is INVALID, abandon TPE approach immediately'
        }
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✓ Level 2 dataset generation completed!")
    print(f"  Output: {output_dir}")
    print(f"  Total moments: {len(HOURS_12)}")
    print(f"  Storage: ~{estimate_storage_mb(1, len(HOURS_12)):.1f} MB")
    print(f"\n  NEXT STEP: Run validation script to test TPE hypothesis")
    print(f"  → python scripts/validate_tpe_level2.py --dataset {output_dir}")
    print(f"{'='*70}\n")


def estimate_storage_mb(num_probes: int, num_moments: int) -> float:
    """Estimate storage size in MB."""
    # Each SH coefficient array is [num_probes, 27] float32
    bytes_per_moment = num_probes * 27 * 4
    total_bytes = bytes_per_moment * num_moments
    # Add overhead for sun_trajectory and metadata
    total_bytes += 1024 * 100  # ~100 KB overhead
    return total_bytes / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(
        description='Generate TPE Level 2 single-probe validation datasets'
    )

    parser.add_argument(
        '--scene',
        type=str,
        required=True,
        choices=list(SCENE_CONFIGS.keys()),
        help='Scene to generate (cornell-box, classroom, house)'
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
        default=512,
        help='Samples per pixel for rendering (default: 512, HIGH QUALITY)'
    )

    parser.add_argument(
        '--num-sh-samples',
        type=int,
        default=128,
        help='Number of directional samples for SH fitting (default: 128, HIGH QUALITY)'
    )

    args = parser.parse_args()

    # Set Mitsuba variant
    print("Initializing Mitsuba...")
    try:
        mi.set_variant('cuda_ad_rgb')
        print("  ✓ Using CUDA variant (GPU acceleration - RECOMMENDED for Level 2)")
    except Exception:
        mi.set_variant('llvm_ad_rgb')
        print("  ✓ Using LLVM variant (CPU mode - WARNING: will be slow)")

    # Get scene configuration
    scene_config = SCENE_CONFIGS[args.scene]
    scene_xml_path = Path(scene_config['xml_path'])

    if not scene_xml_path.exists():
        print(f"ERROR: Scene file not found: {scene_xml_path}")
        sys.exit(1)

    probe_position = scene_config['probe_position']

    print(f"\n  Scene rationale: {scene_config['rationale']}")

    # Generate dataset
    generate_level2_dataset(
        scene_name=args.scene,
        scene_xml_path=scene_xml_path,
        probe_position=probe_position,
        output_dir=Path(args.output),
        spp=args.spp,
        num_sh_samples=args.num_sh_samples
    )


if __name__ == "__main__":
    main()
