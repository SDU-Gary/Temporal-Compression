#!/usr/bin/env python3
"""Render a single preview frame for Bistro scene using Falcor PathTracer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

import sys
UTILS = ROOT / "1_data_generation" / "falcor" / "utils"
CORE_UTILS = ROOT / "2_src" / "utils"
for p in (str(UTILS), str(CORE_UTILS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from falcor_render import build_testbed, load_scene  # noqa: E402
from rendering_utils import (  # noqa: E402
    compute_auto_exposure,
    tone_map_reinhard,
    srgb_encode,
    save_exr,
)


def _parse_vec3(text: Optional[str]) -> Optional[np.ndarray]:
    if not text:
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got: {text}")
    return np.array([float(p) for p in parts], dtype=np.float32)


def _read_output(graph, width: int, height: int) -> np.ndarray:
    output = None
    try:
        output = graph.get_output("PathTracer.color")
    except Exception:
        output = graph.getOutput("PathTracer.color")
    if output is None:
        raise RuntimeError("Failed to get PathTracer.color output texture")
    if hasattr(output, "to_numpy"):
        img = output.to_numpy()
    else:
        img = output.toNumpy()
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 1:
        pixel_count = width * height
        if img.size % pixel_count != 0:
            raise RuntimeError(f"Unexpected output size {img.size} for {width}x{height}")
        channels = max(1, img.size // pixel_count)
        img = img.reshape(height, width, channels)
    if img.shape[-1] > 3:
        img = img[..., :3]
    return img


def _write_png_fallback(path: Path, img: np.ndarray) -> None:
    """Write an 8-bit RGB PNG using stdlib (no PIL dependency)."""
    import zlib
    import struct
    import binascii

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected HxWx3 image for PNG output.")
    height, width, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(height))
    compressor = zlib.compress(raw, level=6)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + tag
            + data
            + struct.pack("!I", binascii.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = b"".join([
        b"\x89PNG\r\n\x1a\n",
        _chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        _chunk(b"IDAT", compressor),
        _chunk(b"IEND", b""),
    ])
    with open(path, "wb") as f:
        f.write(png)


def _save_png(path: Path, image: np.ndarray) -> None:
    img = np.clip(image, 0.0, 1.0)
    img = (img * 255.0 + 0.5).astype(np.uint8)
    try:
        from PIL import Image  # type: ignore
        Image.fromarray(img).save(str(path))
    except Exception:
        _write_png_fallback(path, img)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Bistro preview frame")
    parser.add_argument("--scene", type=str, required=True, help="Path to Bistro .pyscene")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--spp", type=int, default=256)
    parser.add_argument("--accum-frames", type=int, default=1)
    parser.add_argument("--falcor-python-path", type=str, default=None)
    parser.add_argument("--cam-pos", type=str, default=None)
    parser.add_argument("--cam-target", type=str, default=None)
    parser.add_argument("--cam-up", type=str, default=None)
    parser.add_argument("--exposure", type=float, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    testbed, graph, falcor = build_testbed(
        width=args.width,
        height=args.height,
        spp=args.spp,
        falcor_python_path=args.falcor_python_path,
        fixed_seed=1,
        use_russian_roulette=False,
    )
    scene = load_scene(testbed, args.scene)

    cam_pos = _parse_vec3(args.cam_pos)
    cam_target = _parse_vec3(args.cam_target)
    cam_up = _parse_vec3(args.cam_up)
    if cam_pos is not None and cam_target is not None:
        cam = scene.camera
        cam.position = falcor.float3(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
        cam.target = falcor.float3(float(cam_target[0]), float(cam_target[1]), float(cam_target[2]))
        if cam_up is not None:
            cam.up = falcor.float3(float(cam_up[0]), float(cam_up[1]), float(cam_up[2]))

    try:
        testbed.resize_frame_buffer(args.width, args.height)
    except Exception:
        try:
            testbed.resizeFrameBuffer(args.width, args.height)
        except Exception:
            pass

    testbed.clock.pause()
    accum = None
    frames = max(1, int(args.accum_frames))
    for _ in range(frames):
        testbed.frame()
        img = _read_output(graph, args.width, args.height)
        if accum is None:
            accum = img
        else:
            accum = accum + img
    image = accum / float(frames)

    raw_path = output_dir / "bistro_raw_linear.exr"
    save_exr(str(raw_path), image)

    exposure = args.exposure
    if exposure is None:
        exposure = compute_auto_exposure(image)
    tone = tone_map_reinhard(image * float(exposure))
    srgb = srgb_encode(tone)

    _save_png(output_dir / "bistro_tonemapped.png", tone)
    _save_png(output_dir / "bistro_srgb.png", srgb)

    print(f"Saved preview to: {output_dir}")


if __name__ == "__main__":
    main()
