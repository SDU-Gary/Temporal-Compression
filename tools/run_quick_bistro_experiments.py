#!/usr/bin/env python3
"""Run a fast Bistro bake experiment (clean + emissive spheres) with low cost settings."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def _build_env() -> dict:
    env = os.environ.copy()
    env.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    env.setdefault("FALCOR_DEVICE_TYPE", "Vulkan")
    falcor_lib = "/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug"
    existing = env.get("LD_LIBRARY_PATH", "")
    if existing:
        env["LD_LIBRARY_PATH"] = f"{falcor_lib}:{existing}"
    else:
        env["LD_LIBRARY_PATH"] = falcor_lib
    env.setdefault("BISTRO_FBX", "/home/kyrie/毕设/1_data_generation/scenes/Bistro_v5_2/BistroExterior.fbx")
    return env


def _resolve_probe_file(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    return str(p) if p.exists() else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick Bistro bake experiment runner")
    parser.add_argument("--num-frames", type=int, default=600)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--frame-indices", type=str, default="")
    parser.add_argument("--frame-step", type=int, default=30, help="Sample every N frames (default: 30 => 20 frames).")
    parser.add_argument("--cube-res", type=int, default=8)
    parser.add_argument("--spp", type=int, default=256)
    parser.add_argument("--max-probes", type=int, default=200)
    parser.add_argument("--probe-uniform-ratio", type=float, default=0.7)
    parser.add_argument(
        "--probe-file",
        type=str,
        default="",
        help="Optional probe file; when omitted uses adaptive sampling with --max-probes.",
    )
    parser.add_argument("--fixed-seed", type=int, default=1)
    parser.add_argument("--radiance-clamp", type=float, default=20.0)
    parser.add_argument("--output-root", type=str, default="/home/kyrie/毕设/1_data_generation/output")
    parser.add_argument("--output-tag", type=str, default="quick_bistro")
    parser.add_argument("--scene-clean", type=str, default="/home/kyrie/毕设/1_data_generation/scenes/Bistro_v5_2/BistroExterior.pyscene")
    parser.add_argument("--scene-balls", type=str, default="/home/kyrie/毕设/1_data_generation/falcor/scenes/bistro_exterior_emissive_spheres.pyscene")
    parser.add_argument("--falcor-python-path", type=str, default="/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/python")
    parser.add_argument("--python-bin", type=str, default="/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/pythondist/bin/python3.10")
    args = parser.parse_args()

    env = _build_env()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    probe_file = _resolve_probe_file(args.probe_file)

    base_cmd = [
        args.python_bin,
        "/home/kyrie/毕设/1_data_generation/falcor/generate_bistro_temporal_spheres.py",
        "--num-frames", str(args.num_frames),
        "--fps", str(args.fps),
        "--sh-mode", "cubemap",
        "--cube-res", str(args.cube_res),
        "--spp", str(args.spp),
        "--probe-mode", "adaptive",
        "--max-probes", str(args.max_probes),
        "--probe-uniform-ratio", str(args.probe_uniform_ratio),
        "--fixed-seed", str(args.fixed_seed),
        "--falcor-python-path", args.falcor_python_path,
    ]

    if args.frame_indices:
        base_cmd += ["--frame-indices", args.frame_indices]
    elif args.frame_step and args.frame_step > 0:
        base_cmd += ["--frame-step", str(args.frame_step)]

    if probe_file:
        base_cmd += ["--probe-file", probe_file]

    out_clean = output_root / f"{args.output_tag}_clean"
    out_balls = output_root / f"{args.output_tag}_balls_clamp"

    cmd_clean = base_cmd + [
        "--scene", args.scene_clean,
        "--output", str(out_clean),
        "--radiance-clamp", "0",
    ]

    cmd_balls = base_cmd + [
        "--scene", args.scene_balls,
        "--output", str(out_balls),
        "--radiance-clamp", str(args.radiance_clamp),
    ]

    print("Launching quick clean Bistro bake...")
    p_clean = subprocess.Popen(cmd_clean, env=env)
    print("Launching quick emissive-sphere Bistro bake...")
    p_balls = subprocess.Popen(cmd_balls, env=env)

    ret_clean = p_clean.wait()
    ret_balls = p_balls.wait()
    if ret_clean != 0 or ret_balls != 0:
        raise SystemExit(f"One or both bakes failed: clean={ret_clean}, balls={ret_balls}")
    print("Quick bakes completed.")


if __name__ == "__main__":
    main()
