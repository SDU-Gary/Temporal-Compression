#!/usr/bin/env python3
"""GPU probe-volume renderer using a compute shader (near real-time)."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_GEN = ROOT / "1_data_generation"
if str(DATA_GEN) not in os.sys.path:
    os.sys.path.insert(0, str(DATA_GEN))

from utils.spherical_harmonics import evaluate_sh_basis  # noqa: F401  (used to ensure module path)


def _compute_bounds(metadata, probes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(metadata, dict) and "bounds" in metadata:
        b = metadata["bounds"]
        return np.array(b["min"], dtype=np.float32), np.array(b["max"], dtype=np.float32)
    min_pt = probes.min(axis=0)
    max_pt = probes.max(axis=0)
    pad = 0.05 * (max_pt - min_pt + 1e-6)
    return min_pt - pad, max_pt + pad


def _build_probe_field_weights(
    probes: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    grid_res: int,
    k: int,
    chunk: int,
    weight_eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    grid_res = int(grid_res)
    k = int(k)
    lin = np.linspace(0.5 / grid_res, 1.0 - 0.5 / grid_res, grid_res, dtype=np.float32)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    centers = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    centers = bounds_min[None, :] + centers * (bounds_max - bounds_min)[None, :]

    out_idx = np.zeros((centers.shape[0], k), dtype=np.int32)
    out_w = np.zeros((centers.shape[0], k), dtype=np.float32)
    eps2 = float(weight_eps) ** 2
    for i in range(0, centers.shape[0], chunk):
        c = centers[i : i + chunk]
        d2 = ((c[:, None, :] - probes[None, :, :]) ** 2).sum(axis=2)
        k_eff = min(k, d2.shape[1])
        idx = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]
        sel_d2 = np.take_along_axis(d2, idx, axis=1)
        w = 1.0 / (sel_d2 + eps2)
        w = w / (w.sum(axis=1, keepdims=True) + 1e-6)
        out_idx[i : i + chunk, :k_eff] = idx.astype(np.int32)
        out_w[i : i + chunk, :k_eff] = w.astype(np.float32)
    return (
        out_idx.reshape(grid_res, grid_res, grid_res, k),
        out_w.reshape(grid_res, grid_res, grid_res, k),
    )


def _pack_field(field: np.ndarray) -> np.ndarray:
    """Pack SH field into [cells*9,4] with coeff-major RGB layout.

    Input layout: [R0..R8, G0..G8, B0..B8] per cell.
    Output layout: 9 float3 per cell where each entry is (R_i, G_i, B_i).
    """
    cells = field.reshape(-1, 27).astype(np.float32)
    packed = np.zeros((cells.shape[0] * 9, 4), dtype=np.float32)
    for i in range(9):
        packed[i::9, 0] = cells[:, i]
        packed[i::9, 1] = cells[:, 9 + i]
        packed[i::9, 2] = cells[:, 18 + i]
    return packed


def _save_png(path: Path, image: np.ndarray) -> None:
    img = np.clip(image, 0.0, 1.0)
    img = (img * 255.0 + 0.5).astype(np.uint8)
    try:
        from PIL import Image  # type: ignore
        Image.fromarray(img).save(str(path))
        return
    except Exception:
        # Minimal PNG writer fallback (no PIL dependency).
        import zlib
        import struct
        import binascii

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


def _resize_image(image: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Resize using PIL if available, otherwise nearest-neighbor with numpy."""
    try:
        from PIL import Image  # type: ignore
        up = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
        up = up.resize((out_w, out_h), resample=Image.BILINEAR)
        return np.array(up).astype(np.float32) / 255.0
    except Exception:
        h, w, _ = image.shape
        ys = np.linspace(0, h - 1, out_h).astype(np.int32)
        xs = np.linspace(0, w - 1, out_w).astype(np.int32)
        return image[ys[:, None], xs[None, :], :]


def _tone_map_reinhard(image: np.ndarray) -> np.ndarray:
    img = np.maximum(image, 0.0)
    return img / (1.0 + img)


def _srgb_encode(image: np.ndarray) -> np.ndarray:
    img = np.maximum(image, 0.0)
    threshold = 0.0031308
    low = 12.92 * img
    high = 1.055 * np.power(img, 1.0 / 2.4) - 0.055
    return np.where(img <= threshold, low, high)


def _compute_auto_exposure(
    image: np.ndarray,
    percentile: float = 95.0,
    target: float = 0.6,
    eps: float = 1e-6,
    min_exposure: float = 0.05,
    max_exposure: float = 50.0,
) -> float:
    lum = (
        0.2126 * image[..., 0]
        + 0.7152 * image[..., 1]
        + 0.0722 * image[..., 2]
    )
    key = float(np.percentile(lum, percentile))
    exposure = float(target / (key + eps))
    return float(np.clip(exposure, min_exposure, max_exposure))


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU probe-volume renderer")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--render-scale", type=float, default=1.0)
    parser.add_argument("--field-res", type=int, default=32)
    parser.add_argument("--field-knn", type=int, default=8)
    parser.add_argument("--weight-eps", type=float, default=0.1)
    parser.add_argument("--grid-chunk", type=int, default=4096)
    parser.add_argument("--falcor-python-path", default=None)
    parser.add_argument("--exposure", type=float, default=None)
    parser.add_argument("--auto-exposure", action="store_true", default=True)
    parser.add_argument("--no-auto-exposure", action="store_false", dest="auto_exposure")
    parser.add_argument("--exposure-percentile", type=float, default=95.0)
    parser.add_argument("--exposure-target", type=float, default=0.6)
    parser.add_argument("--video", action="store_true", default=True)
    parser.add_argument("--no-video", action="store_false", dest="video")
    parser.add_argument("--write-frames", action="store_true", default=False)
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--ao-strength", type=float, default=0.0)
    args = parser.parse_args()

    if args.falcor_python_path is None:
        args.falcor_python_path = "/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/python"
    if args.falcor_python_path and args.falcor_python_path not in os.sys.path:
        os.sys.path.insert(0, args.falcor_python_path)

    if "BISTRO_FBX" not in os.environ or os.environ.get("BISTRO_FBX", "") == "":
        os.environ["BISTRO_FBX"] = str(ROOT / "1_data_generation" / "scenes" / "Bistro_v5_2" / "BistroExterior.fbx")

    import falcor as fc

    data = np.load(args.dataset, allow_pickle=True)
    tensor = data["tensor"]
    probes = data["probe_positions"]
    num_frames = tensor.shape[1]
    metadata = data["metadata"].item() if "metadata" in data else {}
    bounds_min, bounds_max = _compute_bounds(metadata, probes)

    # Precompute grid weights
    field_knn, field_w = _build_probe_field_weights(
        probes,
        bounds_min,
        bounds_max,
        grid_res=args.field_res,
        k=args.field_knn,
        chunk=args.grid_chunk,
        weight_eps=args.weight_eps,
    )

    render_scale = max(0.1, float(args.render_scale))
    out_w = int(round(args.width * render_scale))
    out_h = int(round(args.height * render_scale))

    testbed_cls = getattr(fc, "Testbed", None)
    if testbed_cls is None and hasattr(fc, "falcor_ext"):
        testbed_cls = getattr(fc.falcor_ext, "Testbed", None)
    if testbed_cls is None:
        raise RuntimeError("Falcor Testbed not found. Check --falcor-python-path.")
    testbed = testbed_cls(width=out_w, height=out_h, create_window=False)
    graph = testbed.create_render_graph("ProbeVolumeGPU")
    graph.create_pass("GBufferRT", "GBufferRT", {"samplePattern": "Center", "sampleCount": 1})
    graph.mark_output("GBufferRT.posW")
    graph.mark_output("GBufferRT.normW")
    graph.mark_output("GBufferRT.diffuseOpacity")
    graph.mark_output("GBufferRT.emissive")
    graph.mark_output("GBufferRT.depth")
    graph.mark_output("GBufferRT.linearZ")
    testbed.render_graph = graph

    testbed.load_scene(args.scene)
    scene = testbed.scene

    # Create GPU resources
    device = testbed.device
    field_buf = device.create_structured_buffer(
        struct_size=16,
        element_count=(args.field_res ** 3) * 9,
        bind_flags=fc.ResourceBindFlags.ShaderResource | fc.ResourceBindFlags.UnorderedAccess,
    )
    output_tex = device.create_texture(
        format=fc.ResourceFormat.RGBA32Float,
        width=out_w,
        height=out_h,
        mip_levels=1,
        bind_flags=fc.ResourceBindFlags.UnorderedAccess | fc.ResourceBindFlags.ShaderResource,
    )

    shader_path = Path(__file__).parent / "sh_probe_volume.cs.slang"
    compute = fc.ComputePass(device, file=shader_path, cs_entry="main")
    compute.globals.gFieldSH = field_buf
    compute.globals.gOutput = output_tex
    compute.globals.gBoundsMin = fc.float3(float(bounds_min[0]), float(bounds_min[1]), float(bounds_min[2]))
    compute.globals.gBoundsMax = fc.float3(float(bounds_max[0]), float(bounds_max[1]), float(bounds_max[2]))
    compute.globals.gFieldRes = fc.uint3(int(args.field_res), int(args.field_res), int(args.field_res))
    compute.globals.gFrameDim = fc.uint2(int(out_w), int(out_h))
    compute.globals.gExposure = 1.0

    out_dir = Path(args.output_dir)
    frames_dir = out_dir / "frames"
    if args.write_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    video_path = None
    ffmpeg_proc = None
    if args.video:
        video_path = args.video_path or str(out_dir / "probe_render.mp4")
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{args.width}x{args.height}",
            "-r",
            str(args.fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]
        ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    exposure_value = float(args.exposure) if args.exposure is not None else None

    start = max(0, int(args.frame_start))
    step = max(1, int(args.frame_step))
    end = num_frames if args.max_frames is None else min(num_frames, start + int(args.max_frames) * step)
    for frame in range(start, end, step):
        t_sec = float(frame) / float(args.fps)
        try:
            testbed.clock.time = t_sec
        except Exception:
            try:
                testbed.clock.setTime(t_sec)
            except Exception:
                pass

        testbed.resize_frame_buffer(out_w, out_h)
        testbed.frame()

        # Build field (CPU) and upload to GPU
        field = (field_w[..., None] * tensor[field_knn, frame]).sum(axis=-2)
        packed = _pack_field(field)
        field_buf.from_numpy(packed)

        # Bind GBuffer outputs
        pos_tex = graph.get_output("GBufferRT.posW")
        norm_tex = graph.get_output("GBufferRT.normW")
        alb_tex = graph.get_output("GBufferRT.diffuseOpacity")
        emi_tex = graph.get_output("GBufferRT.emissive")
        dep_tex = graph.get_output("GBufferRT.depth")
        lin_tex = graph.get_output("GBufferRT.linearZ")

        compute.globals.gPosW = pos_tex
        compute.globals.gNormW = norm_tex
        compute.globals.gAlbedo = alb_tex
        compute.globals.gEmissive = emi_tex
        compute.globals.gDepth = dep_tex
        compute.globals.gLinearZ = lin_tex
        compute.globals.gAOStrength = float(args.ao_strength)

        if exposure_value is not None:
            compute.globals.gExposure = float(exposure_value)
        else:
            compute.globals.gExposure = 1.0

        compute.execute(threads_x=out_w, threads_y=out_h)

        img = output_tex.to_numpy()
        img = np.asarray(img, dtype=np.float32)
        if img.shape[-1] > 3:
            img = img[..., :3]

        if exposure_value is None and args.auto_exposure:
            exposure_value = _compute_auto_exposure(
                img,
                percentile=float(args.exposure_percentile),
                target=float(args.exposure_target),
            )

        # Tonemap + sRGB
        img = _srgb_encode(_tone_map_reinhard(img))

        if render_scale != 1.0:
            img = _resize_image(img, args.width, args.height)

        if args.video and ffmpeg_proc is not None:
            frame_u8 = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            if ffmpeg_proc.stdin is not None:
                ffmpeg_proc.stdin.write(frame_u8.tobytes())
        if args.write_frames:
            _save_png(frames_dir / f"frame_{frame:04d}.png", img)

        if frame % max(1, (10 * step)) == 0:
            print(f"Rendered frame {frame+1}/{num_frames}")

    if ffmpeg_proc is not None and ffmpeg_proc.stdin is not None:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        print(f"Saved video to: {video_path}")
    if args.write_frames:
        print(f"Saved frames to: {frames_dir}")


if __name__ == "__main__":
    main()
