#!/usr/bin/env python3
"""Generate 5D parametric dataset using Falcor (full Falcor pipeline).

This mirrors 1_data_generation/generate_5D_parametric.py but renders with Falcor.
Outputs:
  - parametric_tensor.npz (tensor, probe_positions, light_configs, metadata)
  - metadata.json
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple

import json
import numpy as np
from tqdm import tqdm

# Project utils
_ROOT = Path(__file__).resolve().parents[2]
_UTILS = _ROOT / "1_data_generation" / "utils"
_FALCOR_GEN = _ROOT / "1_data_generation" / "falcor"
_FALCOR_GEN_UTILS = _FALCOR_GEN / "utils"
_sys = __import__("sys")
if str(_UTILS) not in _sys.path:
    _sys.path.insert(0, str(_UTILS))
if str(_FALCOR_GEN_UTILS) not in _sys.path:
    _sys.path.insert(0, str(_FALCOR_GEN_UTILS))

from spherical_harmonics import fibonacci_sphere, fit_sh_coefficients  # noqa: E402

# Falcor helpers
from falcor_render import (  # noqa: E402
    build_testbed,
    configure_scene_for_sun,
    get_light,
    get_scene_bounds,
    load_scene,
    render_single_pixel,
    set_camera_look,
)
from light_params import (  # noqa: E402
    compute_sun_rgb,
    zenith_azimuth_to_direction,
)


def sample_5d_stratified() -> np.ndarray:
    """Stratified sampling (41 configs) matching the Mitsuba variant."""
    configs = []

    # Group 1: geometric
    zenith_vals = [15, 30, 45, 60]
    azimuth_vals = [0, 120, 240]
    for zenith in zenith_vals:
        for azimuth in azimuth_vals:
            configs.append([zenith, azimuth, 1.0, 5500, 0.0])

    # Group 2: intensity/temp
    intensity_vals = np.linspace(0.5, 1.5, 5)
    temp_vals = [3000, 4500, 6500, 8500]
    for intensity in intensity_vals:
        for temp in temp_vals:
            configs.append([30, 90, intensity, temp, 0.2])

    # Group 3: atmospheric
    cloud_vals = [0.0, 0.3, 0.6]
    zenith_atm = [15, 45, 75]
    for cloud in cloud_vals:
        for zenith in zenith_atm:
            configs.append([zenith, 180, 1.0, 5500, cloud])

    return np.array(configs, dtype=np.float32)


def generate_probe_grid(bounds: Tuple[np.ndarray, np.ndarray], grid_resolution: int, margin: float) -> np.ndarray:
    """Uniform 3D grid of probes inside scene bounds."""
    min_pt, max_pt = bounds
    size = max_pt - min_pt
    # If margin is given as fraction (<1), scale by scene size.
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


def save_checkpoint(path: Path, config_idx: int, tensor: np.ndarray, configs: np.ndarray, probes: np.ndarray):
    checkpoint = {
        "config_idx": int(config_idx),
        "tensor": tensor,
        "light_configs": configs,
        "probe_positions": probes,
        "timestamp": datetime.now().isoformat(),
    }
    np.savez_compressed(path, **checkpoint)


def load_checkpoint(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "config_idx": int(data["config_idx"]),
        "tensor": data["tensor"],
        "light_configs": data["light_configs"],
        "probe_positions": data["probe_positions"],
    }


def main():
    parser = argparse.ArgumentParser(description="Generate 5D parametric dataset using Falcor")
    parser.add_argument("--scene", type=str, required=False,
                        default=str(_ROOT / "1_data_generation" / "falcor" / "scenes" / "cornell_box_sun.pyscene"))
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--grid-resolution", type=int, default=5)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--num-sh-samples", type=int, default=64)
    parser.add_argument("--spp", type=int, default=64)
    parser.add_argument("--falcor-python-path", type=str, default=None,
                        help="Path to Falcor python bindings directory")
    parser.add_argument("--max-configs", type=int, default=None,
                        help="Limit number of light configs for quick tests")
    parser.add_argument("--max-probes", type=int, default=None,
                        help="Limit number of probes for quick tests")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
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

    # Probes
    bounds = get_scene_bounds(scene)
    probes = generate_probe_grid(bounds, args.grid_resolution, args.margin)
    if args.max_probes is not None:
        probes = probes[: args.max_probes]

    # Light configs
    configs = sample_5d_stratified()
    if args.max_configs is not None:
        configs = configs[: args.max_configs]

    P = probes.shape[0]
    M = configs.shape[0]
    tensor = np.zeros((P, M, 27), dtype=np.float32)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else (output_dir / "checkpoint.npz")
    start_config = 0
    if args.resume and ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path)
        tensor = ckpt["tensor"]
        start_config = ckpt["config_idx"]
        configs = ckpt["light_configs"]
        probes = ckpt["probe_positions"]
        P, M, _ = tensor.shape
        print(f"Resuming from checkpoint: config_idx={start_config}")

    directions = fibonacci_sphere(args.num_sh_samples)

    # Main loop
    for config_idx in range(start_config, M):
        zenith, azimuth, intensity, color_temp, cloud = configs[config_idx]

        # Update sun light
        direction = zenith_azimuth_to_direction(float(zenith), float(azimuth))
        # Falcor DistantLight expects direction of light rays (towards scene).
        # If results look inverted, flip direction here.
        sun.direction = falcor.float3(float(direction[0]), float(direction[1]), float(direction[2]))
        rgb = compute_sun_rgb(float(zenith), float(intensity), float(color_temp), float(cloud))
        sun.intensity = falcor.float3(float(rgb[0]), float(rgb[1]), float(rgb[2]))

        for probe_idx in tqdm(range(P), desc=f"Config {config_idx+1}/{M}", leave=False):
            probe = probes[probe_idx]
            radiances = []
            for d in directions:
                set_camera_look(scene, probe, d)
                rgb_sample = render_single_pixel(testbed, graph)
                radiances.append(rgb_sample)
            radiances = np.array(radiances, dtype=np.float32)
            sh = fit_sh_coefficients(directions, radiances, max_order=2)
            tensor[probe_idx, config_idx, :] = sh

        # Checkpoint after each config
        save_checkpoint(ckpt_path, config_idx + 1, tensor, configs, probes)

    # Save outputs
    metadata = {
        "engine": "Falcor",
        "scene": str(Path(args.scene).resolve()),
        "num_probes": int(P),
        "num_configs": int(M),
        "num_samples": int(args.num_sh_samples),
        "spp": int(args.spp),
        "grid_resolution": int(args.grid_resolution),
        "margin": float(args.margin),
        "bounds": {
            "min": bounds[0].tolist(),
            "max": bounds[1].tolist(),
        },
        "parameter_ranges": {
            "zenith": [0.0, 90.0],
            "azimuth": [0.0, 360.0],
            "intensity": [0.1, 2.0],
            "color_temp": [2500.0, 10000.0],
            "cloud_cover": [0.0, 0.9],
        },
        "direction_convention": "Falcor DistantLight.direction uses ray direction toward scene (verify sign if needed)",
        "generation_date": datetime.now().isoformat(),
    }

    np.savez_compressed(
        output_dir / "parametric_tensor.npz",
        tensor=tensor,
        probe_positions=probes,
        light_configs=configs,
        metadata=metadata,
    )
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved dataset to: {output_dir}")


if __name__ == "__main__":
    main()
