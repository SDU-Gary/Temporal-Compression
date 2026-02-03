#!/usr/bin/env python3
"""Validate SH reconstruction quality using Falcor rendering."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_UTILS = _ROOT / "1_data_generation" / "utils"
_FALCOR_GEN_UTILS = _ROOT / "1_data_generation" / "falcor" / "utils"
import sys
for p in (str(_UTILS), str(_FALCOR_GEN_UTILS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from spherical_harmonics import (  # noqa: E402
    fibonacci_sphere,
    fit_sh_coefficients,
    compute_sh_reconstruction_error,
)
from falcor_render import (  # noqa: E402
    build_testbed,
    configure_scene_for_sun,
    get_light,
    get_scene_bounds,
    load_scene,
    render_single_pixel,
    set_camera_look,
)
from light_params import compute_sun_rgb, zenith_azimuth_to_direction  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Falcor SH quality validation")
    parser.add_argument("--scene", type=str, required=False,
                        default=str(_ROOT / "1_data_generation" / "falcor" / "scenes" / "cornell_box_sun.pyscene"))
    parser.add_argument("--falcor-python-path", type=str, default=None)
    parser.add_argument("--num-sh-samples", type=int, default=64)
    parser.add_argument("--spp", type=int, default=64)
    args = parser.parse_args()

    testbed, graph, falcor = build_testbed(
        width=1, height=1, spp=args.spp, falcor_python_path=args.falcor_python_path
    )
    scene = load_scene(testbed, args.scene)
    configure_scene_for_sun(scene)
    sun = get_light(scene, name="Sun")

    # Use a mid-level configuration
    zenith, azimuth, intensity, color_temp, cloud = 30.0, 90.0, 1.0, 5500.0, 0.0
    direction = zenith_azimuth_to_direction(zenith, azimuth)
    sun.direction = falcor.float3(float(direction[0]), float(direction[1]), float(direction[2]))
    rgb = compute_sun_rgb(zenith, intensity, color_temp, cloud)
    sun.intensity = falcor.float3(float(rgb[0]), float(rgb[1]), float(rgb[2]))

    # Probe at scene center
    min_pt, max_pt = get_scene_bounds(scene)
    probe = (min_pt + max_pt) * 0.5

    dirs = fibonacci_sphere(args.num_sh_samples)
    radiances = []
    for d in dirs:
        set_camera_look(scene, probe, d)
        radiances.append(render_single_pixel(testbed, graph))
    radiances = np.array(radiances, dtype=np.float32)

    sh = fit_sh_coefficients(dirs, radiances, max_order=2)
    mse, rel_err = compute_sh_reconstruction_error(dirs, radiances, sh, max_order=2)

    print("Falcor SH reconstruction check:")
    print(f"  MSE: {mse:.6f}")
    print(f"  Relative error: {rel_err:.2f}%")


if __name__ == "__main__":
    main()
