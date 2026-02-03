#!/usr/bin/env python3
"""Run emissive-sphere and clean Bistro bakes in parallel."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel Bistro data bakes")
    parser.add_argument("--num-frames", type=int, default=600)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--cube-res", type=int, default=16)
    parser.add_argument("--spp", type=int, default=128)
    parser.add_argument("--probe-file", type=str, default="")
    parser.add_argument("--max-probes", type=int, default=800)
    parser.add_argument("--probe-uniform-ratio", type=float, default=0.7)
    parser.add_argument("--output-clean", type=str, default="/home/kyrie/毕设/1_data_generation/output/bistro_clean_v2")
    parser.add_argument("--output-balls", type=str, default="/home/kyrie/毕设/1_data_generation/output/bistro_balls_clamp_v2")
    parser.add_argument("--scene-clean", type=str, default="/home/kyrie/毕设/1_data_generation/scenes/Bistro_v5_2/BistroExterior.pyscene")
    parser.add_argument("--scene-balls", type=str, default="/home/kyrie/毕设/1_data_generation/falcor/scenes/bistro_exterior_emissive_spheres.pyscene")
    parser.add_argument("--radiance-clamp", type=float, default=20.0)
    parser.add_argument("--falcor-python-path", type=str, default="/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/python")
    parser.add_argument("--python-bin", type=str, default="/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/pythondist/bin/python3.10")
    args = parser.parse_args()

    env = _build_env()

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
        "--use-env-light",
        "--falcor-python-path", args.falcor_python_path,
    ]

    if args.probe_file:
        base_cmd += ["--probe-file", args.probe_file]

    cmd_clean = base_cmd + [
        "--scene", args.scene_clean,
        "--output", args.output_clean,
        "--radiance-clamp", "0",
    ]

    cmd_balls = base_cmd + [
        "--scene", args.scene_balls,
        "--output", args.output_balls,
        "--radiance-clamp", str(args.radiance_clamp),
    ]

    print("Launching clean Bistro bake...")
    p_clean = subprocess.Popen(cmd_clean, env=env)
    print("Launching emissive-sphere Bistro bake...")
    p_balls = subprocess.Popen(cmd_balls, env=env)

    ret_clean = p_clean.wait()
    ret_balls = p_balls.wait()

    if ret_clean != 0 or ret_balls != 0:
        raise SystemExit(f"One or both bakes failed: clean={ret_clean}, balls={ret_balls}")
    print("Both bakes completed.")


if __name__ == "__main__":
    main()
