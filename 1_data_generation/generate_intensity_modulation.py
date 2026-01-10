#!/usr/bin/env python3
"""Generate light intensity modulation dataset for PG-GCPL validation.

This script generates a dataset with sinusoidal intensity variation of an area light,
designed to validate that PG-GCPL can compress arbitrary light source variations
(not just TOD sun direction changes).

Dataset Specification:
- 343 probes (7³ grid in Cornell Box)
- 12 time moments (covering 1 complete 3-second cycle)
- Sinusoidal intensity: I(t) = 0.5 + 0.5 * sin(2π * t / 3)
- SPP: 128 samples per pixel
- SH samples: 64 (for coefficient baking)
"""

import numpy as np
import mitsuba as mi
from pathlib import Path
import json
import sys
from typing import Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.probe_sampler import sample_probe_positions
from core.sh_baker import bake_sh_at_probe
from core.cornell_box_builder import build_cornell_box_with_area_light


def intensity_function(t: float, period: float = 3.0) -> float:
    """Sinusoidal intensity modulation.

    Args:
        t: Time in seconds
        period: Period of oscillation in seconds

    Returns:
        Intensity value in [0, 1]
    """
    return 0.5 + 0.5 * np.sin(2 * np.pi * t / period)


def generate_dataset(
    output_dir: str = 'output/intensity_modulation_343',
    num_probes: int = 343,
    num_moments: int = 12,
    period: float = 3.0,
    spp: int = 128,
    sh_samples: int = 64,
    verbose: bool = True
) -> None:
    """Generate light intensity modulation dataset.

    Args:
        output_dir: Output directory path
        num_probes: Number of probes (343 = 7³)
        num_moments: Number of time moments (12 for 1 complete cycle)
        period: Period of intensity oscillation in seconds
        spp: Samples per pixel for rendering
        sh_samples: Number of samples for SH coefficient baking
        verbose: Print progress information
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📊 INTENSITY MODULATION DATASET GENERATION")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"")
        print(f"Configuration:")
        print(f"  Probes: {num_probes} (7³ grid)")
        print(f"  Moments: {num_moments} (period: {period}s)")
        print(f"  SPP: {spp}")
        print(f"  SH samples: {sh_samples}")
        print(f"  Output: {output_dir}")
        print(f"")

    # Step 1: Sample probe positions (7³ grid in Cornell Box)
    if verbose:
        print(f"[1/4] Sampling probe positions...")

    # Cornell Box normalized bounds (x_min, x_max, y_min, y_max, z_min, z_max)
    # Box walls at X,Y ∈ [-0.5, 0.5], floor Z=0, ceiling Z=1.0
    # Use 0.1m margin to avoid placing probes too close to walls
    scene_bounds = (-0.4, 0.4, -0.4, 0.4, 0.15, 0.85)
    probe_positions = sample_probe_positions(
        num_probes=num_probes,
        scene_bounds=scene_bounds
    )

    # Save probe positions
    np.savez(
        output_dir / 'probes.npz',
        positions=probe_positions
    )

    if verbose:
        print(f"  ✓ Generated {len(probe_positions)} probe positions")
        print(f"  ✓ Bounds: x[{scene_bounds[0]:.1f}, {scene_bounds[1]:.1f}], "
              f"y[{scene_bounds[2]:.1f}, {scene_bounds[3]:.1f}], "
              f"z[{scene_bounds[4]:.1f}, {scene_bounds[5]:.1f}]")
        print(f"")

    # Step 2: Generate time moments
    if verbose:
        print(f"[2/4] Generating time moments...")

    time_moments = np.linspace(0, period, num_moments, endpoint=False)

    if verbose:
        print(f"  ✓ Time range: [0, {period}]s")
        print(f"  ✓ Time step: {period/num_moments:.3f}s")
        print(f"")

    # Step 3: For each time moment, render and bake SH
    if verbose:
        print(f"[3/4] Rendering and baking SH coefficients...")

    for idx, t in enumerate(time_moments):
        moment_dir = output_dir / f'moment_{idx:02d}'
        moment_dir.mkdir(exist_ok=True)

        # Calculate intensity at time t
        intensity = intensity_function(t, period)

        if verbose:
            print(f"  Moment {idx:02d}/{num_moments}: t={t:.3f}s, intensity={intensity:.4f}")

        # Build scene with modulated intensity
        scene_dict = build_cornell_box_with_area_light(
            intensity=intensity,
            color=[1.0, 1.0, 1.0]  # White light
        )
        scene = mi.load_dict(scene_dict)

        # Bake SH coefficients for all probes
        sh_coeffs_list = []
        for probe_idx, probe_pos in enumerate(probe_positions):
            sh_coeffs = bake_sh_at_probe(
                scene=scene,
                probe_position=probe_pos,
                num_samples=sh_samples,
                spp=spp
            )
            sh_coeffs_list.append(sh_coeffs)

            # Progress indicator every 50 probes
            if verbose and (probe_idx + 1) % 50 == 0:
                print(f"    Baked {probe_idx + 1}/{num_probes} probes...")

        sh_coeffs_array = np.array(sh_coeffs_list)  # [N, 27]

        # Save SH coefficients + intensity metadata
        np.savez(
            moment_dir / 'sh_coeffs.npz',
            sh_coeffs=sh_coeffs_array,
            intensity=intensity,
            time=t
        )

        # Save intensity value (for easy access)
        with open(moment_dir / 'intensity.txt', 'w') as f:
            f.write(f"{intensity:.6f}\n")

        if verbose:
            print(f"    ✓ Saved {len(sh_coeffs_array)} SH coefficient sets")

    if verbose:
        print(f"")

    # Step 4: Save metadata
    if verbose:
        print(f"[4/4] Saving metadata...")

    metadata = {
        'num_probes': int(num_probes),
        'num_moments': int(num_moments),
        'period': float(period),
        'intensity_function': 'sinusoidal',
        'intensity_formula': 'I(t) = 0.5 + 0.5 * sin(2π * t / T)',
        'intensity_range': [0.0, 1.0],
        'spp': int(spp),
        'sh_samples': int(sh_samples),
        'sh_order': 2,
        'sh_coeffs_dim': 27,
        'scene': 'cornell-box',
        'light_type': 'area_light_ceiling',
        'scene_bounds': list(scene_bounds),
        'time_range': [0.0, float(period)],
        'time_step': float(period / num_moments)
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"  ✓ Metadata saved")
        print(f"")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"✅ GENERATION COMPLETE")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"")
        print(f"Dataset structure:")
        print(f"  {output_dir}/")
        print(f"  ├── probes.npz           ({num_probes} probes × 3D positions)")
        print(f"  ├── metadata.json        (dataset configuration)")
        print(f"  └── moment_XX/           ({num_moments} time moments)")
        print(f"      ├── sh_coeffs.npz    ({num_probes} × 27 SH coefficients)")
        print(f"      └── intensity.txt    (scalar intensity value)")
        print(f"")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate light intensity modulation dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/intensity_modulation_343',
        help='Output directory path'
    )
    parser.add_argument(
        '--num-probes',
        type=int,
        default=343,
        help='Number of probes (default: 343 = 7³)'
    )
    parser.add_argument(
        '--num-moments',
        type=int,
        default=12,
        help='Number of time moments (default: 12)'
    )
    parser.add_argument(
        '--period',
        type=float,
        default=3.0,
        help='Period of intensity oscillation in seconds (default: 3.0)'
    )
    parser.add_argument(
        '--spp',
        type=int,
        default=128,
        help='Samples per pixel for rendering (default: 128)'
    )
    parser.add_argument(
        '--sh-samples',
        type=int,
        default=64,
        help='Number of samples for SH baking (default: 64)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Set Mitsuba variant
    mi.set_variant('cuda_ad_rgb')

    generate_dataset(
        output_dir=args.output_dir,
        num_probes=args.num_probes,
        num_moments=args.num_moments,
        period=args.period,
        spp=args.spp,
        sh_samples=args.sh_samples,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
