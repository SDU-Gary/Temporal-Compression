#!/usr/bin/env python3
"""
Emergency fix: Generate intensity modulation dataset with SIMPLE geometry.

Uses sphere instead of Cornell Box to avoid transform issues.
"""

import numpy as np
import mitsuba as mi
from pathlib import Path
import json
from tqdm import tqdm

# Add paths
import sys
sys.path.insert(0, str(Path(__file__).parent))
from core.probe_sampler import sample_probe_positions
from core.sh_baker import bake_sh_at_probe

mi.set_variant('cuda_ad_rgb')


def intensity_function(t, period=3.0):
    """Sinusoidal intensity: I(t) = 0.5 + 0.5 * sin(2πt/T)."""
    return 0.5 + 0.5 * np.sin(2 * np.pi * t / period)


def build_simple_scene_with_light(intensity=1.0):
    """Simple scene: sphere + floor + point light (TESTED TO WORK)."""
    return {
        'type': 'scene',
        'integrator': {'type': 'path', 'max_depth': 6},

        # Point light (variable intensity)
        'light': {
            'type': 'point',
            'position': [0, 0, 0.8],
            'intensity': {'type': 'rgb', 'value': [50.0 * intensity] * 3}
        },

        # Floor (large diffuse plane)
        'floor': {
            'type': 'sphere',  # Use large sphere as ground
            'center': [0, 0, -100.5],
            'radius': 100,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}
            }
        },

        # Central sphere (geometry reference)
        'center_sphere': {
            'type': 'sphere',
            'center': [0, 0, 0.3],
            'radius': 0.15,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}
            }
        }
    }


def generate_dataset_simple_geom(
    output_dir='output/intensity_modulation_343_simple',
    num_probes=343,
    num_moments=12,
    period=3.0,
    spp=128,
    sh_samples=64
):
    """Generate dataset with simple geometry (emergency fix)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("INTENSITY MODULATION DATASET - SIMPLE GEOMETRY")
    print("=" * 70)
    print(f"Probes: {num_probes}")
    print(f"Moments: {num_moments}")
    print(f"Using SIMPLE geometry (sphere + floor) to avoid transform issues")
    print()

    # 1. Sample probe positions
    print("[1/4] Sampling probe positions...")
    scene_bounds = (-0.4, 0.4, -0.4, 0.4, 0.1, 0.7)
    probe_positions = sample_probe_positions(num_probes, scene_bounds)
    np.savez(output_dir / 'probes.npz', positions=probe_positions)
    print(f"  ✓ Generated {len(probe_positions)} probes")

    # 2. Time moments
    print("[2/4] Generating time moments...")
    time_moments = np.linspace(0, period, num_moments, endpoint=False)
    print(f"  ✓ Time range: [0, {period}]s, step: {period/num_moments:.3f}s")

    # 3. Render and bake SH
    print("[3/4] Rendering and baking SH coefficients...")
    for idx, t in enumerate(time_moments):
        moment_dir = output_dir / f'moment_{idx:02d}'
        moment_dir.mkdir(exist_ok=True)

        intensity = intensity_function(t, period)
        print(f"  Moment {idx:02d}/{num_moments}: t={t:.3f}s, I={intensity:.4f}")

        # Build scene
        scene_dict = build_simple_scene_with_light(intensity=intensity)
        scene = mi.load_dict(scene_dict)

        # Bake SH for all probes
        sh_coeffs_list = []
        for probe_idx, probe_pos in enumerate(tqdm(probe_positions, desc="    Baking", leave=False)):
            sh_coeffs = bake_sh_at_probe(scene, probe_pos, num_samples=sh_samples, spp=spp)
            sh_coeffs_list.append(sh_coeffs)

        sh_coeffs_array = np.array(sh_coeffs_list)

        # Save
        np.savez(
            moment_dir / 'sh_coeffs.npz',
            sh_coeffs=sh_coeffs_array,
            intensity=intensity,
            time=t
        )

        with open(moment_dir / 'intensity.txt', 'w') as f:
            f.write(f"{intensity:.6f}\n")

        print(f"    ✓ SH range: [{sh_coeffs_array.min():.6f}, {sh_coeffs_array.max():.6f}]")

    # 4. Metadata
    print("[4/4] Saving metadata...")
    metadata = {
        'num_probes': num_probes,
        'num_moments': num_moments,
        'period': period,
        'intensity_function': 'sinusoidal',
        'intensity_range': [0.0, 1.0],
        'spp': spp,
        'sh_samples': sh_samples,
        'scene': 'simple_geometry',
        'scene_type': 'sphere_floor_point_light',
        'scene_bounds': list(scene_bounds),
        'note': 'Emergency fix: simple geometry to avoid Cornell Box transform issues'
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("=" * 70)
    print("✅ GENERATION COMPLETE")
    print(f"Output: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    generate_dataset_simple_geom(
        num_probes=343,
        num_moments=12,
        spp=128,
        sh_samples=64
    )
