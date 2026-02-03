#!/usr/bin/env python3
"""One-click diagnostics for probe SH flicker issues."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_GEN = ROOT / "1_data_generation"
if str(DATA_GEN) not in os.sys.path:
    os.sys.path.insert(0, str(DATA_GEN))
DATA_UTILS = DATA_GEN / "utils"
if str(DATA_UTILS) not in os.sys.path:
    os.sys.path.insert(0, str(DATA_UTILS))

from utils.spherical_harmonics import evaluate_sh_basis  # noqa: F401


def _compute_bounds(metadata, probes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(metadata, dict) and "bounds" in metadata:
        b = metadata["bounds"]
        return np.array(b["min"], dtype=np.float32), np.array(b["max"], dtype=np.float32)
    min_pt = probes.min(axis=0)
    max_pt = probes.max(axis=0)
    pad = 0.05 * (max_pt - min_pt + 1e-6)
    return min_pt - pad, max_pt + pad


def _pick_probe_nearest_to_camera(scene, probes: np.ndarray) -> int:
    cam = scene.camera
    cam_pos = np.array([cam.position.x, cam.position.y, cam.position.z], dtype=np.float32)
    d2 = ((probes - cam_pos[None, :]) ** 2).sum(axis=1)
    return int(np.argmin(d2))


def _save_png(path: Path, image: np.ndarray) -> None:
    img = np.clip(image, 0.0, 1.0)
    img = (img * 255.0 + 0.5).astype(np.uint8)
    try:
        from PIL import Image  # type: ignore
        Image.fromarray(img).save(str(path))
        return
    except Exception:
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


def _plot_curve_png(values: np.ndarray, out_png: Path, width: int = 1200, height: int = 300) -> None:
    """Simple line plot without matplotlib."""
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    if v.size == 0:
        return
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if vmax - vmin < 1e-8:
        norm = np.full_like(v, 0.5, dtype=np.float32)
    else:
        norm = (v - vmin) / (vmax - vmin)

    img = np.ones((height, width, 3), dtype=np.float32)
    xs = np.linspace(0, width - 1, v.size).astype(np.int32)
    ys = (1.0 - norm) * (height - 1)
    ys = ys.astype(np.int32)

    # Draw polyline
    for i in range(1, v.size):
        x0, y0 = xs[i - 1], ys[i - 1]
        x1, y1 = xs[i], ys[i]
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            if 0 <= x < width and 0 <= y < height:
                img[y, x] = 0.0
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    _save_png(out_png, img)


def _hash_color(ids: np.ndarray) -> np.ndarray:
    # ids: [N] int -> color [N,3]
    x = ids.astype(np.uint32)
    x = (x ^ 0x9E3779B9) * 0x85EBCA6B
    r = ((x >> 0) & 255).astype(np.float32)
    g = ((x >> 8) & 255).astype(np.float32)
    b = ((x >> 16) & 255).astype(np.float32)
    return np.stack([r, g, b], axis=1) / 255.0


def test1_constant_sh(dataset: Path, out_dir: Path, falcor_py: str, scene: Path, max_frames: int) -> Path:
    data = np.load(dataset, allow_pickle=True)
    tensor = data["tensor"]
    probes = data["probe_positions"]
    num_frames = tensor.shape[1]
    const = np.zeros_like(tensor, dtype=np.float32)
    const[:, :, 0] = 1.0  # R0
    const[:, :, 9] = 1.0  # G0
    const[:, :, 18] = 1.0  # B0

    payload = {}
    for k in data.files:
        if k == "tensor":
            payload[k] = const
        else:
            payload[k] = data[k]

    const_path = out_dir / "constant_sh_dataset.npz"
    np.savez_compressed(const_path, **payload)

    cmd = [
        "/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/pythondist/bin/python3.10",
        str(ROOT / "tools" / "render_probe_video_gpu.py"),
        "--dataset", str(const_path),
        "--scene", str(scene),
        "--output-dir", str(out_dir / "test1_constant_sh"),
        "--width", "1280",
        "--height", "720",
        "--fps", "30",
        "--render-scale", "0.8",
        "--field-res", "32",
        "--field-knn", "8",
        "--frame-start", "0",
        "--max-frames", str(max_frames),
        "--no-auto-exposure",
        "--exposure", "1.0",
        "--no-video",
        "--write-frames",
        "--falcor-python-path", falcor_py,
    ]
    subprocess.run(cmd, check=True)
    return out_dir / "test1_constant_sh"


def test2_sh_trajectory(dataset: Path, out_dir: Path, falcor_py: str, scene: Path) -> Path:
    data = np.load(dataset, allow_pickle=True)
    tensor = data["tensor"]
    probes = data["probe_positions"]

    if falcor_py and falcor_py not in os.sys.path:
        os.sys.path.insert(0, falcor_py)
    import falcor as fc
    testbed = fc.Testbed(width=64, height=64, create_window=False)
    testbed.load_scene(str(scene))
    idx = _pick_probe_nearest_to_camera(testbed.scene, probes)

    dc = np.take(tensor[idx], [0, 9, 18], axis=1)
    # Use luminance of DC as scalar
    lum = 0.2126 * dc[:, 0] + 0.7152 * dc[:, 1] + 0.0722 * dc[:, 2]
    out_csv = out_dir / "test2_dc_curve.csv"
    np.savetxt(out_csv, lum, delimiter=",")

    out_png = out_dir / "test2_dc_curve.png"
    try:
        import matplotlib.pyplot as plt  # type: ignore
        plt.figure(figsize=(8, 3))
        plt.plot(lum)
        plt.title("Probe DC Luminance over Time")
        plt.xlabel("Frame")
        plt.ylabel("Luminance")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
    except Exception:
        _plot_curve_png(lum, out_png)
    return out_csv


def test3_static_consistency(dataset: Path, out_dir: Path, falcor_py: str, scene: Path, repeats: int) -> Path:
    # Render the same frame multiple times and measure coefficient variance.
    if falcor_py and falcor_py not in os.sys.path:
        os.sys.path.insert(0, falcor_py)
    import falcor as fc

    utils_path = ROOT / "1_data_generation" / "falcor" / "utils"
    if str(utils_path) not in os.sys.path:
        os.sys.path.insert(0, str(utils_path))
    from falcor_render import build_testbed, render_sh_cubemap

    testbed, graph, _ = build_testbed(width=64, height=64, spp=1, falcor_python_path=falcor_py, fixed_seed=1)
    testbed.load_scene(str(scene))
    scn = testbed.scene

    # choose a probe near camera
    data = np.load(dataset, allow_pickle=True)
    probes = data["probe_positions"]
    idx = _pick_probe_nearest_to_camera(scn, probes)
    probe = probes[idx]

    # Freeze time at t=0 so emissive spheres and camera stay static.
    try:
        testbed.clock.time = 0.0
    except Exception:
        try:
            testbed.clock.setTime(0.0)
        except Exception:
            pass
    try:
        testbed.clock.pause()
    except Exception:
        try:
            testbed.clock.setPaused(True)
        except Exception:
            pass

    coeffs = []
    for i in range(repeats):
        # Re-assert time in case the renderer advances it internally.
        try:
            testbed.clock.time = 0.0
        except Exception:
            try:
                testbed.clock.setTime(0.0)
            except Exception:
                pass
        coeffs.append(render_sh_cubemap(testbed, graph, scn, probe, cube_res=16, num_frames=1, seed_base=1))
    coeffs = np.stack(coeffs, axis=0)
    mean = coeffs.mean(axis=0)
    std = coeffs.std(axis=0)

    out_json = out_dir / "test3_static_consistency.json"
    out = {
        "probe_index": int(idx),
        "repeats": int(repeats),
        "mean_abs_std": float(np.mean(np.abs(std))),
        "max_abs_std": float(np.max(np.abs(std))),
    }
    out_json.write_text(json.dumps(out, indent=2))
    return out_json


def test4_id_debug(dataset: Path, out_dir: Path, falcor_py: str, scene: Path) -> Path:
    data = np.load(dataset, allow_pickle=True)
    probes = data["probe_positions"]
    metadata = data["metadata"].item() if "metadata" in data else {}
    bounds_min, bounds_max = _compute_bounds(metadata, probes)

    if falcor_py and falcor_py not in os.sys.path:
        os.sys.path.insert(0, falcor_py)
    import falcor as fc

    testbed = fc.Testbed(width=640, height=360, create_window=False)
    graph = testbed.create_render_graph("DiagGBuffer")
    graph.create_pass("GBufferRT", "GBufferRT", {"samplePattern": "Center", "sampleCount": 1})
    graph.mark_output("GBufferRT.posW")
    graph.mark_output("GBufferRT.depth")
    testbed.render_graph = graph
    testbed.load_scene(str(scene))

    testbed.frame()
    pos = graph.get_output("GBufferRT.posW").to_numpy()[..., :3]
    depth = graph.get_output("GBufferRT.depth").to_numpy()
    if depth.ndim == 3:
        depth = depth[..., 0]

    H, W, _ = pos.shape
    valid = (depth > 0.0) & (depth < 1.0) & np.isfinite(pos).all(axis=2)
    pos_flat = pos.reshape(-1, 3)
    idx = np.flatnonzero(valid.reshape(-1))

    # Build simple grid mapping for IDs using nearest probe to cell center
    field_res = 32
    lin = np.linspace(0.5 / field_res, 1.0 - 0.5 / field_res, field_res, dtype=np.float32)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    centers = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    centers = bounds_min[None, :] + centers * (bounds_max - bounds_min)[None, :]
    d2 = ((centers[:, None, :] - probes[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argmin(d2, axis=1).astype(np.int32).reshape(field_res, field_res, field_res)

    # map each pixel to cell
    t = (pos_flat[idx] - bounds_min[None, :]) / (bounds_max - bounds_min + 1e-6)[None, :]
    t = np.clip(t, 0.0, 0.999999)
    cell = (t * field_res).astype(np.int32)
    cell = np.clip(cell, 0, field_res - 1)
    ids = nearest[cell[:, 0], cell[:, 1], cell[:, 2]]
    colors = _hash_color(ids)

    img = np.zeros((H * W, 3), dtype=np.float32)
    img[idx] = colors
    img = img.reshape(H, W, 3)
    out_png = out_dir / "test4_probe_id.png"
    _save_png(out_png, img)
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all probe diagnostics")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--falcor-python-path", default="/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/python")
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--static-repeats", type=int, default=20)
    parser.add_argument("--test1", action="store_true", help="Run test1: constant SH render")
    parser.add_argument("--test2", action="store_true", help="Run test2: SH DC trajectory")
    parser.add_argument("--test3", action="store_true", help="Run test3: static consistency")
    parser.add_argument("--test4", action="store_true", help="Run test4: interpolation ID debug")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = Path(args.dataset)
    scene = Path(args.scene)

    if "BISTRO_FBX" not in os.environ or os.environ.get("BISTRO_FBX", "") == "":
        os.environ["BISTRO_FBX"] = str(ROOT / "1_data_generation" / "scenes" / "Bistro_v5_2" / "BistroExterior.fbx")

    run_all = not (args.test1 or args.test2 or args.test3 or args.test4)
    results = {}
    if run_all or args.test1:
        results["test1_constant_sh"] = str(test1_constant_sh(dataset, out_dir, args.falcor_python_path, scene, args.max_frames))
    if run_all or args.test2:
        results["test2_sh_trajectory"] = str(test2_sh_trajectory(dataset, out_dir, args.falcor_python_path, scene))
    if run_all or args.test3:
        results["test3_static_consistency"] = str(test3_static_consistency(dataset, out_dir, args.falcor_python_path, scene, args.static_repeats))
    if run_all or args.test4:
        results["test4_id_debug"] = str(test4_id_debug(dataset, out_dir, args.falcor_python_path, scene))

    (out_dir / "diagnostics_summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
