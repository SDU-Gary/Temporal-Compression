#!/usr/bin/env python3
"""Render one generic Falcor scene preview frame with optional camera override."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "1_data_generation" / "falcor" / "utils"
CORE_UTILS = ROOT / "2_src" / "utils"
for p in (str(UTILS), str(CORE_UTILS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from falcor_render import build_testbed, load_scene  # noqa: E402
from rendering_utils import compute_auto_exposure, save_exr, srgb_encode, tone_map_reinhard  # noqa: E402


def _parse_vec3(text: Optional[str]) -> Optional[np.ndarray]:
    if not text:
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got: {text}")
    return np.array([float(p) for p in parts], dtype=np.float32)


def _read_output(graph, width: int, height: int) -> np.ndarray:
    try:
        output = graph.get_output("PathTracer.color")
    except Exception:
        output = graph.getOutput("PathTracer.color")
    if output is None:
        raise RuntimeError("Failed to get PathTracer.color output texture")
    img = output.to_numpy() if hasattr(output, "to_numpy") else output.toNumpy()
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 1:
        pixel_count = width * height
        channels = max(1, img.size // pixel_count)
        img = img.reshape(height, width, channels)
    if img.shape[-1] > 3:
        img = img[..., :3]
    return img


def _save_png(path: Path, image: np.ndarray) -> None:
    img = np.clip(image, 0.0, 1.0)
    img = (img * 255.0 + 0.5).astype(np.uint8)
    try:
        from PIL import Image  # type: ignore

        Image.fromarray(img).save(str(path))
    except Exception:
        import binascii
        import struct
        import zlib

        height, width, _ = img.shape
        raw = b"".join(b"\x00" + img[y].tobytes() for y in range(height))

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack("!I", len(data))
                + tag
                + data
                + struct.pack("!I", binascii.crc32(tag + data) & 0xFFFFFFFF)
            )

        png = b"".join(
            [
                b"\x89PNG\r\n\x1a\n",
                chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)),
                chunk(b"IDAT", zlib.compress(raw, level=6)),
                chunk(b"IEND", b""),
            ]
        )
        path.write_bytes(png)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--spp", type=int, default=64)
    parser.add_argument("--accum-frames", type=int, default=4)
    parser.add_argument("--falcor-python-path", type=str, default=None)
    parser.add_argument("--cam-pos", type=str, default=None)
    parser.add_argument("--cam-target", type=str, default=None)
    parser.add_argument("--cam-up", type=str, default=None)
    parser.add_argument("--near-plane", type=float, default=None)
    parser.add_argument("--far-plane", type=float, default=None)
    parser.add_argument("--focal-length", type=float, default=None)
    parser.add_argument("--exposure", type=float, default=None)
    parser.add_argument("--stem", default="preview")
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
    cam = scene.camera
    if args.near_plane is not None:
        cam.nearPlane = float(args.near_plane)
    if args.far_plane is not None:
        cam.farPlane = float(args.far_plane)
    if args.focal_length is not None:
        cam.focalLength = float(args.focal_length)

    try:
        testbed.resize_frame_buffer(args.width, args.height)
    except Exception:
        try:
            testbed.resizeFrameBuffer(args.width, args.height)
        except Exception:
            pass

    testbed.clock.pause()
    accum = None
    for _ in range(max(1, int(args.accum_frames))):
        testbed.frame()
        image = _read_output(graph, args.width, args.height)
        accum = image if accum is None else accum + image
    image = accum / float(max(1, int(args.accum_frames)))

    stem = str(args.stem)
    save_exr(str(output_dir / f"{stem}_raw_linear.exr"), image)
    exposure = float(args.exposure) if args.exposure is not None else compute_auto_exposure(image)
    tone = tone_map_reinhard(image * exposure)
    srgb = srgb_encode(tone)
    _save_png(output_dir / f"{stem}_tonemapped.png", tone)
    _save_png(output_dir / f"{stem}_srgb.png", srgb)
    print(f"Saved preview to: {output_dir}")
    print(f"exposure={exposure:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
