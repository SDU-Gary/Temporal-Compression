#!/usr/bin/env python3
"""Linearity sanity check: SH(A+B) ≈ SH(A) + SH(B)."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_UTILS = _ROOT / "1_data_generation" / "utils"
_FALCOR_UTILS = _ROOT / "1_data_generation" / "falcor" / "utils"
import sys
for p in (str(_UTILS), str(_FALCOR_UTILS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from spherical_harmonics import fibonacci_sphere, fit_sh_coefficients  # noqa: E402
from falcor_render import (  # noqa: E402
    build_testbed,
    configure_scene_for_sun,
    get_light,
    get_scene_bounds,
    load_scene,
    render_sh_cubemap,
    render_single_pixel,
    set_camera_look,
    set_mogwai_renderer,
)
from light_params import compute_sun_rgb, zenith_azimuth_to_direction  # noqa: E402


def render_sh(
    testbed,
    graph,
    scene,
    probe,
    directions,
    sh_mode: str = "dir",
    cube_res: int = 32,
    num_frames: int = 4,
    seed_base: int | None = None,
    progress_every: int | None = None,
):
    if sh_mode == "cubemap":
        return render_sh_cubemap(
            testbed,
            graph,
            scene,
            probe,
            cube_res=cube_res,
            num_frames=num_frames,
            seed_base=seed_base,
        )
    radiances = []
    total = len(directions)
    for idx, d in enumerate(directions):
        set_camera_look(scene, probe, d)
        radiances.append(
            render_single_pixel(
                testbed,
                graph,
                num_frames=num_frames,
                seed_base=seed_base,
            )
        )
        if progress_every and (idx + 1) % progress_every == 0:
            print(f"  SH sample {idx + 1}/{total}", flush=True)
    radiances = np.array(radiances, dtype=np.float32)
    return fit_sh_coefficients(directions, radiances, max_order=2)


def main():
    def log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    # Mogwai mode: read from environment to avoid argparse conflicts.
    import __main__
    import builtins
    mogwai_renderer = globals().get("m")
    if mogwai_renderer is None:
        mogwai_renderer = getattr(__main__, "m", None)
    if mogwai_renderer is None:
        mogwai_renderer = getattr(builtins, "m", None)
    is_mogwai = mogwai_renderer is not None
    if is_mogwai:
        set_mogwai_renderer(mogwai_renderer)

    if is_mogwai:
        class Args:  # simple namespace
            pass
        args = Args()
        args.scene = os.environ.get(
            "FALCOR_SCENE",
            str(_ROOT / "1_data_generation/falcor/scenes/cornell_box_sun_point.pyscene"),
        )
        args.spp = int(os.environ.get("FALCOR_SPP", "512"))
        args.sh_mode = os.environ.get("FALCOR_SH_MODE", "dir").lower()
        args.cube_res = int(os.environ.get("FALCOR_CUBE_RES", "32"))
        args.num_sh_samples = int(os.environ.get("FALCOR_NUM_SH_SAMPLES", "512"))
        args.num_frames = int(os.environ.get("FALCOR_NUM_FRAMES", "32"))
        fixed_seed = os.environ.get("FALCOR_FIXED_SEED", "1")
        args.fixed_seed = int(fixed_seed) if fixed_seed != "" else None
        args.use_russian_roulette = False
        args.falcor_python_path = os.environ.get("FALCOR_PYTHON_PATH")
    else:
        parser = argparse.ArgumentParser(description="Falcor linearity check")
        parser.add_argument("--scene", type=str, default=str(_ROOT / "1_data_generation/falcor/scenes/cornell_box_sun_point.pyscene"))
        parser.add_argument("--spp", type=int, default=512)
        parser.add_argument("--sh-mode", type=str, default="dir", choices=["dir", "cubemap"])
        parser.add_argument("--cube-res", type=int, default=32)
        parser.add_argument("--num-sh-samples", type=int, default=512)
        parser.add_argument("--num-frames", type=int, default=32)
        parser.add_argument("--fixed-seed", type=int, default=1)
        parser.add_argument("--falcor-python-path", type=str, default=None)
        args = parser.parse_args()
        args.use_russian_roulette = False
        args.sh_mode = args.sh_mode.lower()

    args.sh_mode = args.sh_mode.lower()

    log("Linearity check start.")
    testbed, graph, falcor = build_testbed(
        width=1,
        height=1,
        spp=args.spp,
        falcor_python_path=args.falcor_python_path,
        fixed_seed=getattr(args, "fixed_seed", None),
        use_russian_roulette=False,
    )
    log("Testbed + RenderGraph ready.")
    scene = load_scene(testbed, args.scene)
    log("Scene loaded.")
    configure_scene_for_sun(scene)
    log("Scene configured.")

    sun = get_light(scene, name="Sun")
    lamp = get_light(scene, name="Lamp")

    bounds = get_scene_bounds(scene)
    probe = (bounds[0] + bounds[1]) * 0.5
    directions = None
    if args.sh_mode == "dir":
        directions = fibonacci_sphere(args.num_sh_samples)
        log(f"SH directions prepared: {len(directions)} samples.")
    else:
        log(f"SH mode: cubemap (cube_res={args.cube_res})")

    # Light A: sun only
    eps_light = 1e-6
    direction = zenith_azimuth_to_direction(45.0, 0.0)
    sun_rgb = compute_sun_rgb(45.0, 1.0, 5500.0, 0.0)
    sun.direction = falcor.float3(float(direction[0]), float(direction[1]), float(direction[2]))
    sun.intensity = falcor.float3(float(sun_rgb[0]), float(sun_rgb[1]), float(sun_rgb[2]))
    lamp.intensity = falcor.float3(eps_light, eps_light, eps_light)
    progress_every = max(1, args.num_sh_samples // 4) if args.sh_mode == "dir" else None
    log("Rendering SH for Light A (sun)...")
    sh_a = render_sh(
        testbed,
        graph,
        scene,
        probe,
        directions,
        sh_mode=args.sh_mode,
        cube_res=args.cube_res,
        num_frames=args.num_frames,
        seed_base=getattr(args, "fixed_seed", None),
        progress_every=progress_every,
    )

    # Light B: lamp only
    sun.intensity = falcor.float3(eps_light, eps_light, eps_light)
    lamp.position = falcor.float3(0.0, 0.3, 0.0)
    lamp.intensity = falcor.float3(1.0, 1.0, 1.0)
    log("Rendering SH for Light B (lamp)...")
    sh_b = render_sh(
        testbed,
        graph,
        scene,
        probe,
        directions,
        sh_mode=args.sh_mode,
        cube_res=args.cube_res,
        num_frames=args.num_frames,
        seed_base=getattr(args, "fixed_seed", None),
        progress_every=progress_every,
    )

    # Light A+B
    sun.intensity = falcor.float3(float(sun_rgb[0]), float(sun_rgb[1]), float(sun_rgb[2]))
    lamp.intensity = falcor.float3(1.0, 1.0, 1.0)
    log("Rendering SH for Light A+B...")
    sh_ab = render_sh(
        testbed,
        graph,
        scene,
        probe,
        directions,
        sh_mode=args.sh_mode,
        cube_res=args.cube_res,
        num_frames=args.num_frames,
        seed_base=getattr(args, "fixed_seed", None),
        progress_every=progress_every,
    )

    eps = np.mean(np.linalg.norm(sh_ab - (sh_a + sh_b)))
    rel = eps / (np.mean(np.linalg.norm(sh_ab)) + 1e-8)

    log("Linearity check done.")
    print("Linearity check:", flush=True)
    print(f"  eps = {eps:.6f}", flush=True)
    print(f"  rel = {rel:.6f}", flush=True)

    if is_mogwai:
        try:
            exit()
        except Exception:
            pass


if __name__ == "__main__" or "m" in globals():
    main()
