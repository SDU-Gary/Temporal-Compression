"""CLI helper to render an envmap using Falcor in a Python 3.10 environment."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envmap", required=True, help="Path to envmap (.hdr/.exr)")
    parser.add_argument("--scene", default=None, help="Optional .pyscene path")
    parser.add_argument("--out", required=True, help="Output .npy path")
    parser.add_argument("--spp", type=int, default=32)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--falcor-python-path", default=None)
    parser.add_argument("--output-pass", default="ToneMapper.dst")
    parser.add_argument("--no-tonemap", action="store_true")
    args = parser.parse_args()

    # Ensure 2_src is on sys.path (for 'utils' package)
    src_root = Path(__file__).parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    # Optional Falcor python bindings path
    if args.falcor_python_path:
        if args.falcor_python_path not in sys.path:
            sys.path.insert(0, args.falcor_python_path)
        falcor_bin = Path(args.falcor_python_path).parent
        os.environ["LD_LIBRARY_PATH"] = f"{falcor_bin}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    from utils.rendering_utils import render_with_envmap

    image = render_with_envmap(
        args.envmap,
        scene_path=args.scene,
        spp=args.spp,
        resolution=(args.width, args.height),
        falcor_python_path=args.falcor_python_path,
        clamp_output=False,
        output_pass=args.output_pass,
        enable_tonemapper=not args.no_tonemap,
    )

    np.save(args.out, image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
