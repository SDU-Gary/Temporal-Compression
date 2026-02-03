#!/usr/bin/env python3
"""Generate 1D intensity modulation dataset using Falcor.

This mirrors 1_data_generation/generate_intensity_modulation.py but renders with Falcor.
Outputs:
  - probes.npz
  - moment_XX/sh_coeffs.npz (sh_coeffs, intensity, time)
  - metadata.json
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
_UTILS = _ROOT / "1_data_generation" / "utils"
_FALCOR_GEN_UTILS = _ROOT / "1_data_generation" / "falcor" / "utils"
import sys
for p in (str(_UTILS), str(_FALCOR_GEN_UTILS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from spherical_harmonics import fibonacci_sphere, fit_sh_coefficients  # noqa: E402
from falcor_render import (  # noqa: E402
    build_testbed,
    configure_scene_for_sun,
    get_light,
    get_scene_bounds,
    load_scene,
    render_single_pixel,
    set_camera_look,
)


def intensity_function(t: float, period: float = 3.0) -> float:
    """Sinusoidal intensity modulation in [0, 1]."""
    return 0.5 + 0.5 * np.sin(2 * np.pi * t / period)


def generate_probe_grid(bounds, grid_resolution: int, margin: float) -> np.ndarray:
    """Uniform 3D grid of probes inside scene bounds."""
    min_pt, max_pt = bounds
    size = max_pt - min_pt
    if margin < 1.0:
        margin_vec = size * margin
    else:
        margin_vec = np.array([margin, margin, margin], dtype=np.float32)
    low = min_pt + margin_vec
    high = max_pt - margin_vec

    xs = np.linspace(low[0], high[0], grid_resolution)
    ys = np.linspace(low[1], high[1], grid_resolution)
    zs = np.linspace(low[2], high[2], grid_resolution)

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    probes = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
    return probes.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate 1D intensity modulation dataset using Falcor")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--scene", type=str, required=False,
                        default=str(_ROOT / "1_data_generation" / "falcor" / "scenes" / "cornell_box_sun.pyscene"))
    parser.add_argument("--grid-resolution", type=int, default=7)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--num-moments", type=int, default=12)
    parser.add_argument("--period", type=float, default=3.0)
    parser.add_argument("--num-sh-samples", type=int, default=64)
    parser.add_argument("--spp", type=int, default=64)
    parser.add_argument("--falcor-python-path", type=str, default=None)
    parser.add_argument("--max-probes", type=int, default=None)
    parser.add_argument("--max-moments", type=int, default=None)
    parser.add_argument("--flip-direction", action="store_true",
                        help="Flip sun direction if lighting appears inverted")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Falcor setup
    testbed, graph, falcor = build_testbed(
        width=1,
        height=1,
        spp=args.spp,
        falcor_python_path=args.falcor_python_path,
    )
    scene = load_scene(testbed, args.scene)
    configure_scene_for_sun(scene)
    sun = get_light(scene, name="Sun")

    # Fixed sun direction (straight down)
    sun_dir = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    if args.flip_direction:
        sun_dir = -sun_dir
    sun.direction = falcor.float3(float(sun_dir[0]), float(sun_dir[1]), float(sun_dir[2]))

    # Probes
    bounds = get_scene_bounds(scene)
    probes = generate_probe_grid(bounds, args.grid_resolution, args.margin)
    if args.max_probes is not None:
        probes = probes[: args.max_probes]

    np.savez(output_dir / "probes.npz", positions=probes)

    # Moments
    num_moments = args.num_moments if args.max_moments is None else min(args.num_moments, args.max_moments)
    times = np.linspace(0.0, args.period, num_moments, endpoint=False)
    directions = fibonacci_sphere(args.num_sh_samples)

    for moment_idx, t in enumerate(times):
        intensity = intensity_function(float(t), args.period)
        moment_dir = output_dir / f"moment_{moment_idx:02d}"
        moment_dir.mkdir(parents=True, exist_ok=True)

        # Update sun intensity (white light, scaled by intensity)
        sun.intensity = falcor.float3(float(intensity), float(intensity), float(intensity))

        sh_coeffs_list = []
        for probe_idx in tqdm(range(len(probes)), desc=f"Moment {moment_idx+1}/{num_moments}", leave=False):
            probe = probes[probe_idx]
            radiances = []
            for d in directions:
                set_camera_look(scene, probe, d)
                radiances.append(render_single_pixel(testbed, graph))
            radiances = np.array(radiances, dtype=np.float32)
            sh = fit_sh_coefficients(directions, radiances, max_order=2)
            sh_coeffs_list.append(sh)

        sh_coeffs = np.stack(sh_coeffs_list, axis=0)
        np.savez(
            moment_dir / "sh_coeffs.npz",
            sh_coeffs=sh_coeffs,
            intensity=float(intensity),
            time=float(t),
        )
        with open(moment_dir / "intensity.txt", "w") as f:
            f.write(f"{float(intensity):.6f}\n")

    metadata = {
        "engine": "Falcor",
        "scene": str(Path(args.scene).resolve()),
        "num_probes": int(len(probes)),
        "num_moments": int(num_moments),
        "period": float(args.period),
        "intensity_function": "sinusoidal",
        "intensity_formula": "I(t) = 0.5 + 0.5 * sin(2π * t / T)",
        "intensity_range": [0.0, 1.0],
        "spp": int(args.spp),
        "sh_samples": int(args.num_sh_samples),
        "sh_order": 2,
        "sh_coeffs_dim": 27,
        "grid_resolution": int(args.grid_resolution),
        "margin": float(args.margin),
        "bounds": {
            "min": bounds[0].tolist(),
            "max": bounds[1].tolist(),
        },
        "direction_convention": "Falcor DistantLight.direction uses ray direction toward scene",
        "generation_date": datetime.now().isoformat(),
    }

    with open(output_dir / "metadata.json", "w") as f:
        import json
        json.dump(metadata, f, indent=2)

    print(f"\nSaved dataset to: {output_dir}")


if __name__ == "__main__":
    main()
