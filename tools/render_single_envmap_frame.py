#!/usr/bin/env python3
"""Render a single SH envmap through Falcor and save raw/tonemapped/sRGB outputs.

This script does not run training. It loads one SH sample from parametric_tensor.npz,
builds an envmap, applies auto-exposure (optional), renders once with high SPP,
then writes:
  - raw_linear.npy (HDR)
  - raw_linear.png (clipped to [0,1] for quick view)
  - tonemapped.png (Reinhard)
  - srgb.png (Reinhard + sRGB)
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2_src"))

from utils.rendering_utils import (
    sh_to_envmap,
    save_hdr,
    save_exr,
    openexr_available,
    compute_auto_exposure,
    tone_map_reinhard,
    srgb_encode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/home/kyrie/毕设/1_data_generation/output/probe_mode_compare_200configs/adaptive/multilight",
        help="Path containing parametric_tensor.npz",
    )
    parser.add_argument("--probe-idx", type=int, default=0)
    parser.add_argument("--config-idx", type=int, default=0)
    parser.add_argument("--scene", default=None, help="Optional .pyscene path")
    parser.add_argument("--spp", type=int, default=512)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--envmap-h", type=int, default=128)
    parser.add_argument("--envmap-w", type=int, default=256)
    parser.add_argument("--output-dir", default="/home/kyrie/毕设/metadata/render_debug")
    parser.add_argument("--no-auto-exposure", action="store_true")
    parser.add_argument("--exposure-percentile", type=float, default=95.0)
    parser.add_argument("--exposure-target", type=float, default=0.6)
    parser.add_argument("--exposure-min", type=float, default=0.05)
    parser.add_argument("--exposure-max", type=float, default=50.0)
    parser.add_argument("--falcor-python-path", default="/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/python")
    parser.add_argument("--falcor-python-bin", default="/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/pythondist/bin/python3.10")
    parser.add_argument("--vk-icd", default="/usr/share/vulkan/icd.d/nvidia_icd.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = Path(args.data_root) / "parametric_tensor.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_path}")

    npz = np.load(data_path, allow_pickle=True)
    tensor = npz["tensor"]
    if args.probe_idx >= tensor.shape[0] or args.config_idx >= tensor.shape[1]:
        raise IndexError(f"Invalid indices: probe {args.probe_idx}, config {args.config_idx}")

    sh = tensor[args.probe_idx, args.config_idx]
    envmap = sh_to_envmap(sh, H=args.envmap_h, W=args.envmap_w)

    if not args.no_auto_exposure:
        exposure = compute_auto_exposure(
            envmap,
            percentile=args.exposure_percentile,
            target=args.exposure_target,
            min_exposure=args.exposure_min,
            max_exposure=args.exposure_max,
        )
        envmap = envmap * exposure

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_exr = openexr_available()
    envmap_path = out_dir / ("envmap_exposed.exr" if use_exr else "envmap_exposed.hdr")
    if use_exr:
        save_exr(str(envmap_path), envmap)
    else:
        save_hdr(str(envmap_path), envmap)

    # Render with Falcor CLI to get HDR output from AccumulatePass
    cli = Path(__file__).resolve().parents[1] / "2_src" / "utils" / "falcor_render_cli.py"
    out_npy = out_dir / "raw_linear.npy"

    cmd = [
        args.falcor_python_bin,
        str(cli),
        "--envmap", str(envmap_path),
        "--out", str(out_npy),
        "--spp", str(args.spp),
        "--width", str(args.width),
        "--height", str(args.height),
        "--falcor-python-path", args.falcor_python_path,
        "--output-pass", "AccumulatePass.output",
        "--no-tonemap",
    ]
    if args.scene:
        cmd += ["--scene", args.scene]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug"
    env["FALCOR_DEVICE_TYPE"] = "Vulkan"
    env["FALCOR_GPU"] = "0"
    if args.vk_icd and Path(args.vk_icd).exists():
        env["VK_ICD_FILENAMES"] = args.vk_icd
    env["RERUN_KEEP_HDR"] = "1"

    subprocess.run(cmd, check=True, env=env)

    # Load and save three versions
    raw = np.load(out_npy)

    raw_png = out_dir / "raw_linear.png"
    raw_vis = np.clip(raw, 0.0, 1.0)
    Image.fromarray((raw_vis * 255).astype(np.uint8)).save(raw_png)

    tonemapped = tone_map_reinhard(raw)
    tone_png = out_dir / "tonemapped.png"
    Image.fromarray((np.clip(tonemapped, 0, 1) * 255).astype(np.uint8)).save(tone_png)

    srgb = srgb_encode(tonemapped)
    srgb_png = out_dir / "srgb.png"
    Image.fromarray((np.clip(srgb, 0, 1) * 255).astype(np.uint8)).save(srgb_png)

    print("Saved:")
    print(raw_png)
    print(tone_png)
    print(srgb_png)
    print(out_npy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
