#!/usr/bin/env python3
"""Render a probe-driven Bistro sequence and assemble a video.

Two modes:
1) global: use one probe as an environment map (fast preview)
2) volume: per-pixel KNN interpolation over all probes (irradiance volume)
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "2_src" / "utils"
DATA_GEN = ROOT / "1_data_generation"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))
if str(DATA_GEN) not in sys.path:
    sys.path.insert(0, str(DATA_GEN))

from rendering_utils import (  # noqa: E402
    sh_to_envmap,
    compute_auto_exposure,
    tone_map_reinhard,
    srgb_encode,
    save_hdr,
    save_exr,
    openexr_available,
)

from utils.spherical_harmonics import evaluate_sh_basis  # noqa: E402


def _save_png(path: Path, image: np.ndarray) -> None:
    img = np.clip(image, 0.0, 1.0)
    img = (img * 255.0 + 0.5).astype(np.uint8)
    try:
        from PIL import Image  # type: ignore
        Image.fromarray(img).save(str(path))
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


def _load_scene_camera(scene) -> np.ndarray:
    cam = scene.camera
    return np.array([cam.position.x, cam.position.y, cam.position.z], dtype=np.float32)


def _pick_probe_index(probes: np.ndarray, cam_pos: np.ndarray) -> int:
    d2 = ((probes - cam_pos[None, :]) ** 2).sum(axis=1)
    return int(np.argmin(d2))


def _compute_bounds(metadata, probes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(metadata, dict) and "bounds" in metadata:
        b = metadata["bounds"]
        return np.array(b["min"], dtype=np.float32), np.array(b["max"], dtype=np.float32)
    min_pt = probes.min(axis=0)
    max_pt = probes.max(axis=0)
    pad = 0.05 * (max_pt - min_pt + 1e-6)
    return min_pt - pad, max_pt + pad


def _build_probe_grid(
    probes: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    grid_res: int,
    k: int,
    chunk: int,
) -> np.ndarray:
    """Precompute KNN probe indices for each grid cell center (for fast lookup)."""
    grid_res = int(grid_res)
    k = int(k)
    lin = np.linspace(0.5 / grid_res, 1.0 - 0.5 / grid_res, grid_res, dtype=np.float32)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    centers = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    centers = bounds_min[None, :] + centers * (bounds_max - bounds_min)[None, :]

    out = np.zeros((centers.shape[0], k), dtype=np.int32)
    for i in range(0, centers.shape[0], chunk):
        c = centers[i : i + chunk]
        d2 = ((c[:, None, :] - probes[None, :, :]) ** 2).sum(axis=2)
        idx = np.argpartition(d2, kth=min(k - 1, d2.shape[1] - 1), axis=1)[:, :k]
        out[i : i + chunk] = idx.astype(np.int32)
    return out.reshape(grid_res, grid_res, grid_res, k)


def _build_probe_field_weights(
    probes: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    grid_res: int,
    k: int,
    chunk: int,
    weight_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute KNN indices + weights for a dense SH field."""
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


def _grid_lookup(
    pos: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    grid_res: int,
) -> np.ndarray:
    # Map to cell centers (i+0.5)/grid_res
    t = (pos - bounds_min[None, :]) / (bounds_max - bounds_min + 1e-6)[None, :]
    c = t * grid_res - 0.5
    base = np.floor(c).astype(np.int32)
    base = np.clip(base, 0, grid_res - 2)
    return base


def _compute_irradiance(
    normals: np.ndarray,
    sh_coeffs: np.ndarray,
) -> np.ndarray:
    """Evaluate diffuse irradiance from SH coefficients for given normals.

    normals: [N,3], sh_coeffs: [N, 27] or [N, K, 27]
    Returns: [N,3] or [N,K,3]
    """
    Y = evaluate_sh_basis(normals, max_order=2).astype(np.float32)
    # Lambertian convolution coefficients for l=0,1,2
    A0 = np.pi
    A1 = 2.0 * np.pi / 3.0
    A2 = np.pi / 4.0
    Y[:, 0] *= A0
    Y[:, 1:4] *= A1
    Y[:, 4:9] *= A2

    if sh_coeffs.ndim == 2:
        coeffs = sh_coeffs.reshape(-1, 3, 9)
        return np.einsum("ni,nci->nc", Y, coeffs)

    coeffs = sh_coeffs.reshape(sh_coeffs.shape[0], sh_coeffs.shape[1], 3, 9)
    return np.einsum("ni,nkci->nkc", Y, coeffs)


def _get_output(graph, name: str) -> np.ndarray:
    try:
        tex = graph.get_output(name)
    except Exception:
        tex = graph.getOutput(name)
    if hasattr(tex, "to_numpy"):
        arr = tex.to_numpy()
    else:
        arr = tex.toNumpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        # infer from size assuming RGBA
        raise RuntimeError(f"Unexpected 1D output for {name}")
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Bistro probe sequence to video")
    parser.add_argument("--dataset", required=True, help="parametric_tensor.npz")
    parser.add_argument("--scene", required=True, help="Bistro .pyscene")
    parser.add_argument("--output-dir", required=True, help="Output directory for frames/video")
    parser.add_argument("--mode", type=str, default="volume", choices=["global", "volume"])
    parser.add_argument("--probe-idx", type=int, default=None, help="Probe index (global mode only)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--spp", type=int, default=1)
    parser.add_argument("--envmap-h", type=int, default=128)
    parser.add_argument("--envmap-w", type=int, default=256)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--auto-exposure", action="store_true", default=True)
    parser.add_argument("--no-auto-exposure", action="store_false", dest="auto_exposure")
    parser.add_argument("--exposure", type=float, default=None)
    parser.add_argument("--exposure-samples", type=int, default=10)
    parser.add_argument("--knn", type=int, default=4)
    parser.add_argument("--grid-res", type=int, default=64)
    parser.add_argument("--grid-chunk", type=int, default=4096)
    parser.add_argument("--shade-chunk", type=int, default=200000)
    parser.add_argument("--weight-eps", type=float, default=0.1)
    parser.add_argument("--use-neighbor-cells", action="store_true", default=True)
    parser.add_argument("--no-neighbor-cells", action="store_false", dest="use_neighbor_cells")
    parser.add_argument("--volume-mode", type=str, default="field", choices=["knn", "field"])
    parser.add_argument("--field-res", type=int, default=32)
    parser.add_argument("--field-knn", type=int, default=8)
    parser.add_argument("--temporal-alpha", type=float, default=0.0)
    parser.add_argument("--image-temporal-alpha", type=float, default=0.0)
    parser.add_argument("--render-scale", type=float, default=1.0)
    parser.add_argument("--checkerboard", action="store_true", default=False)
    parser.add_argument("--checkerboard-reuse", action="store_true", default=True)
    parser.add_argument("--no-checkerboard-reuse", action="store_false", dest="checkerboard_reuse")
    parser.add_argument("--falcor-python-path", default=None)
    parser.add_argument("--python-bin", default=None)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    if args.falcor_python_path is None:
        args.falcor_python_path = "/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/python"
    if args.python_bin is None:
        args.python_bin = "/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/pythondist/bin/python3.10"

    # Ensure Bistro FBX is set for pyscene.
    if "BISTRO_FBX" not in os.environ or os.environ.get("BISTRO_FBX", "") == "":
        os.environ["BISTRO_FBX"] = str(ROOT / "1_data_generation" / "scenes" / "Bistro_v5_2" / "BistroExterior.fbx")

    # Load dataset
    data = np.load(args.dataset, allow_pickle=True)
    tensor = data["tensor"]  # [P, M, 27]
    probes = data["probe_positions"]
    num_frames = tensor.shape[1]
    metadata = data["metadata"].item() if "metadata" in data else {}
    bounds_min, bounds_max = _compute_bounds(metadata, probes)

    # Init Falcor for camera query
    if args.falcor_python_path not in sys.path:
        sys.path.insert(0, args.falcor_python_path)
    import falcor as fc

    render_scale = max(0.1, float(args.render_scale))
    out_w = int(round(args.width * render_scale))
    out_h = int(round(args.height * render_scale))
    testbed = fc.Testbed(width=out_w, height=out_h, create_window=False)
    graph = testbed.create_render_graph("ProbeVideoGraph")
    if args.mode == "global":
        graph.create_pass("PathTracer", "PathTracer", {
            "samplesPerPixel": int(min(16, max(1, args.spp))),
            "maxSurfaceBounces": 4,
            "useNEE": True,
        })
        graph.create_pass("VBufferRT", "VBufferRT", {"samplePattern": "Stratified", "sampleCount": 16})
        graph.create_pass("AccumulatePass", "AccumulatePass", {
            "enabled": True,
            "precisionMode": "Single",
            "autoReset": False,
            "maxFrameCount": int(max(1, math.ceil(args.spp / max(1, min(16, args.spp))))),
            "overflowMode": "Stop",
        })
        graph.add_edge("VBufferRT.vbuffer", "PathTracer.vbuffer")
        graph.add_edge("VBufferRT.viewW", "PathTracer.viewW")
        graph.add_edge("VBufferRT.mvec", "PathTracer.mvec")
        graph.add_edge("PathTracer.color", "AccumulatePass.input")
        graph.mark_output("AccumulatePass.output")
    else:
        graph.create_pass("GBufferRT", "GBufferRT", {"samplePattern": "Center", "sampleCount": 1})
        graph.mark_output("GBufferRT.posW")
        graph.mark_output("GBufferRT.normW")
        graph.mark_output("GBufferRT.diffuseOpacity")
        graph.mark_output("GBufferRT.emissive")
        graph.mark_output("GBufferRT.depth")
        graph.mark_output("GBufferRT.mask")
    testbed.render_graph = graph

    testbed.load_scene(args.scene)
    scene = testbed.scene
    cam_pos = _load_scene_camera(scene)

    if args.mode == "global":
        if args.probe_idx is None:
            args.probe_idx = _pick_probe_index(probes, cam_pos)

    out_dir = Path(args.output_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Exposure (global) to avoid flicker.
    exposure = args.exposure
    if exposure is None and args.auto_exposure and args.mode == "global":
        sample_idx = np.linspace(0, num_frames - 1, max(1, args.exposure_samples), dtype=int)
        exposures = []
        for idx in sample_idx:
            sh = tensor[args.probe_idx, idx]
            env = sh_to_envmap(sh, H=args.envmap_h, W=args.envmap_w)
            exposures.append(compute_auto_exposure(env))
        exposure = float(np.median(exposures)) if exposures else 1.0
    if exposure is None:
        exposure = 1.0

    use_exr = openexr_available()
    envmap_path = out_dir / ("temp_env.exr" if use_exr else "temp_env.hdr")

    # Precompute KNN grid for volume mode
    grid_knn = None
    field_knn = None
    field_w = None
    if args.mode == "volume":
        if args.volume_mode == "knn":
            grid_knn = _build_probe_grid(
                probes,
                bounds_min,
                bounds_max,
                grid_res=args.grid_res,
                k=args.knn,
                chunk=args.grid_chunk,
            )
        else:
            field_knn, field_w = _build_probe_field_weights(
                probes,
                bounds_min,
                bounds_max,
                grid_res=args.field_res,
                k=args.field_knn,
                chunk=args.grid_chunk,
                weight_eps=args.weight_eps,
            )

    volume_exposure = None
    prev_field = None
    prev_img = None
    for frame in range(0, num_frames, max(1, args.frame_step)):
        t_sec = float(frame) / float(args.fps)
        try:
            testbed.clock.time = t_sec
        except Exception:
            try:
                testbed.clock.setTime(t_sec)
            except Exception:
                pass

        if args.mode == "global":
            sh = tensor[args.probe_idx, frame]
            env = sh_to_envmap(sh, H=args.envmap_h, W=args.envmap_w)
            env = env * float(exposure)
            if use_exr:
                save_exr(str(envmap_path), env)
            else:
                save_hdr(str(envmap_path), env)

            if not scene.setEnvMap(str(envmap_path)):
                raise RuntimeError(f"Failed to set envmap: {envmap_path}")
            scene.envMap.intensity = 1.0
            scene.renderSettings.useEnvLight = True
            scene.renderSettings.useAnalyticLights = False
            scene.renderSettings.useEmissiveLights = False

            # Reset accumulation
            acc_pass = None
            try:
                acc_pass = graph.get_pass("AccumulatePass")
            except Exception:
                try:
                    acc_pass = graph.getPass("AccumulatePass")
                except Exception:
                    acc_pass = None
            if acc_pass is not None:
                try:
                    acc_pass.reset()
                except Exception:
                    pass

            per_frame_spp = int(min(16, max(1, args.spp)))
            accum_frames = int(math.ceil(float(args.spp) / float(per_frame_spp)))
            testbed.resize_frame_buffer(out_w, out_h)
            testbed.clock.pause()
            for _ in range(max(1, accum_frames)):
                testbed.frame()

            try:
                out_tex = graph.get_output("AccumulatePass.output")
            except Exception:
                out_tex = graph.getOutput("AccumulatePass.output")
            img = out_tex.to_numpy()
            img = np.asarray(img, dtype=np.float32)
            if img.ndim == 1:
                pixel_count = args.width * args.height
                channels = max(1, img.size // pixel_count)
                img = img.reshape(args.height, args.width, channels)
            if img.shape[-1] > 3:
                img = img[..., :3]
        else:
            testbed.resize_frame_buffer(out_w, out_h)
            testbed.frame()

            pos = _get_output(graph, "GBufferRT.posW")[..., :3]
            norm = _get_output(graph, "GBufferRT.normW")[..., :3]
            albedo = _get_output(graph, "GBufferRT.diffuseOpacity")[..., :3]
            emissive = _get_output(graph, "GBufferRT.emissive")[..., :3]
            depth = _get_output(graph, "GBufferRT.depth")
            if depth.ndim == 3:
                depth = depth[..., 0]
            mask_tex = _get_output(graph, "GBufferRT.mask")
            if mask_tex.ndim == 3:
                mask_tex = mask_tex[..., 0]
            valid = (
                (depth > 0.0) & (depth < 1.0)
                & np.isfinite(pos).all(axis=2)
                & (np.linalg.norm(norm, axis=2) > 1e-3)
            )
            if mask_tex is not None and float(mask_tex.max()) > 0.5:
                valid = valid & (mask_tex > 0.5)

            # Flatten valid pixels
            H, W, _ = pos.shape
            if args.checkerboard:
                yy, xx = np.indices((H, W))
                cb_mask = ((xx + yy + frame) & 1) == 0
                valid = valid & cb_mask
            idx = np.flatnonzero(valid.reshape(-1))
            if idx.size == 0:
                img = np.zeros((H, W, 3), dtype=np.float32)
            else:
                p = pos.reshape(-1, 3)[idx]
                n = norm.reshape(-1, 3)[idx]
                n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-6)
                a = albedo.reshape(-1, 3)[idx]
                e = emissive.reshape(-1, 3)[idx]

                if args.checkerboard and args.checkerboard_reuse and prev_img is not None:
                    out = prev_img.reshape(-1, 3).copy()
                else:
                    out = np.zeros((H * W, 3), dtype=np.float32)
                if args.volume_mode == "knn":
                    # Grid lookup -> KNN probe indices
                    base = _grid_lookup(p, bounds_min, bounds_max, args.grid_res)
                    if args.use_neighbor_cells:
                        # Collect candidate probes from 8 neighboring cells to reduce popping.
                        knn_list = []
                        for dx in (0, 1):
                            for dy in (0, 1):
                                for dz in (0, 1):
                                    cell = base + np.array([dx, dy, dz], dtype=np.int32)
                                    cell = np.clip(cell, 0, args.grid_res - 1)
                                    knn_list.append(grid_knn[cell[:, 0], cell[:, 1], cell[:, 2]])
                        knn_all = np.concatenate(knn_list, axis=1)
                    else:
                        knn_all = grid_knn[base[:, 0], base[:, 1], base[:, 2]]

                    pr = probes[knn_all]
                    d2 = ((p[:, None, :] - pr) ** 2).sum(axis=2)
                    # Keep nearest K probes
                    k = int(args.knn)
                    k = min(k, d2.shape[1])
                    sel = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
                    knn = np.take_along_axis(knn_all, sel, axis=1)
                    d2 = np.take_along_axis(d2, sel, axis=1)
                    w = 1.0 / (d2 + float(args.weight_eps) ** 2)
                    w = w / (w.sum(axis=1, keepdims=True) + 1e-6)

                    # Shade in chunks to keep memory reasonable
                    for s in range(0, idx.size, args.shade_chunk):
                        sl = slice(s, min(s + args.shade_chunk, idx.size))
                        ks = knn[sl]
                        ws = w[sl]
                        ns = n[sl]
                        sh = tensor[ks, frame]  # [N,K,27]
                        ir = _compute_irradiance(ns, sh)  # [N,K,3]
                        ir = (ws[:, :, None] * ir).sum(axis=1)
                        out[idx[sl]] = a[sl] * ir + e[sl]
                else:
                    # Field mode: build dense SH field, then trilinear sample per pixel.
                    field = (field_w[..., None] * tensor[field_knn, frame]).sum(axis=-2)
                    if args.temporal_alpha > 0.0 and prev_field is not None:
                        field = (1.0 - args.temporal_alpha) * field + args.temporal_alpha * prev_field
                    prev_field = field

                    # Map positions to grid coords
                    t = (p - bounds_min[None, :]) / (bounds_max - bounds_min + 1e-6)[None, :]
                    t = np.clip(t, 0.0, 0.999999)
                    c = t * (args.field_res - 1)
                    base = np.floor(c).astype(np.int32)
                    base = np.clip(base, 0, args.field_res - 2)
                    frac = c - base
                    fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
                    field_flat = field.reshape(-1, field.shape[-1])
                    stride_y = args.field_res
                    stride_x = args.field_res * args.field_res

                    for s in range(0, idx.size, args.shade_chunk):
                        sl = slice(s, min(s + args.shade_chunk, idx.size))
                        bx = base[sl, 0]
                        by = base[sl, 1]
                        bz = base[sl, 2]
                        fxs = fx[sl][:, None]
                        fys = fy[sl][:, None]
                        fzs = fz[sl][:, None]

                        base_idx = (bx * stride_x + by * stride_y + bz).astype(np.int32)
                        g000 = field_flat[base_idx]
                        g100 = field_flat[base_idx + stride_x]
                        g010 = field_flat[base_idx + stride_y]
                        g110 = field_flat[base_idx + stride_x + stride_y]
                        g001 = field_flat[base_idx + 1]
                        g101 = field_flat[base_idx + stride_x + 1]
                        g011 = field_flat[base_idx + stride_y + 1]
                        g111 = field_flat[base_idx + stride_x + stride_y + 1]

                        c00 = g000 * (1.0 - fxs) + g100 * fxs
                        c10 = g010 * (1.0 - fxs) + g110 * fxs
                        c01 = g001 * (1.0 - fxs) + g101 * fxs
                        c11 = g011 * (1.0 - fxs) + g111 * fxs
                        c0 = c00 * (1.0 - fys) + c10 * fys
                        c1 = c01 * (1.0 - fys) + c11 * fys
                        sh_interp = c0 * (1.0 - fzs) + c1 * fzs

                        ns = n[sl]
                        ir = _compute_irradiance(ns, sh_interp)
                        out[idx[sl]] = a[sl] * ir + e[sl]

                img = out.reshape(H, W, 3)

            if frame == 0:
                valid_ratio = float(idx.size) / float(H * W)
                print(f"[volume] valid_ratio={valid_ratio:.3f} depth_minmax=({depth.min():.3f},{depth.max():.3f}) "
                      f"albedo_mean={albedo.reshape(-1,3)[idx].mean(axis=0)} emissive_mean={emissive.reshape(-1,3)[idx].mean(axis=0)}")

        if args.mode == "volume":
            if args.auto_exposure and volume_exposure is None:
                volume_exposure = compute_auto_exposure(img)
            if volume_exposure is not None:
                img = img * float(volume_exposure)

        if args.image_temporal_alpha > 0.0:
            if prev_img is None:
                prev_img = img
            else:
                a = float(args.image_temporal_alpha)
                img = (1.0 - a) * img + a * prev_img
                prev_img = img

        # Tonemap + sRGB
        tonemapped = tone_map_reinhard(img)
        srgb = srgb_encode(tonemapped)

        if render_scale != 1.0:
            try:
                from PIL import Image  # type: ignore
                up = Image.fromarray((np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
                up = up.resize((args.width, args.height), resample=Image.BILINEAR)
                _save_png(frames_dir / f"frame_{frame:04d}.png", np.array(up) / 255.0)
            except Exception:
                _save_png(frames_dir / f"frame_{frame:04d}.png", srgb)
        else:
            _save_png(frames_dir / f"frame_{frame:04d}.png", srgb)

        print(f"Rendered frame {frame+1}/{num_frames}")

    # Assemble video
    video_path = out_dir / "probe_render.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    subprocess.run(cmd, check=True)
    if not args.keep_frames:
        for p in frames_dir.glob("frame_*.png"):
            p.unlink()
        frames_dir.rmdir()
    print(f"Saved video to: {video_path}")


if __name__ == "__main__":
    main()
