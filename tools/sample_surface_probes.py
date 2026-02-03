#!/usr/bin/env python3
"""Sample surface probes from GBufferRT posW with light-proximity weighting."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "1_data_generation" / "falcor" / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.probe_sampling import (  # noqa: E402
    compute_light_weights,
    load_light_trajectories,
    sample_with_decluster,
)


def _parse_vec3(text: Optional[str]) -> Optional[np.ndarray]:
    if not text:
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 3:
        return None
    return np.array([float(p) for p in parts], dtype=np.float32)


def _parse_float_env(name: str, default: float) -> float:
    val = os.environ.get(name, "")
    try:
        return float(val) if val != "" else float(default)
    except Exception:
        return float(default)


def _parse_int_env(name: str, default: int) -> int:
    val = os.environ.get(name, "")
    try:
        return int(val) if val != "" else int(default)
    except Exception:
        return int(default)


def _compute_trajectories(
    bounds: Tuple[np.ndarray, np.ndarray],
    num_frames: int,
    fps: float,
    height_base_ratio: float,
    height_amp_ratio: float,
    axis_margin_ratio: float,
    orth_offset_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_pt, max_pt = bounds
    size = max_pt - min_pt
    center = (min_pt + max_pt) * 0.5
    axis = 0 if size[0] >= size[2] else 2
    orth = 2 if axis == 0 else 0
    height = float(size[1])

    y_base = float(min_pt[1]) + height_base_ratio * height
    y_amp = height_amp_ratio * height
    axis_margin = axis_margin_ratio * float(size[axis])
    orth_offset = orth_offset_ratio * float(size[orth])

    start_axis = float(min_pt[axis]) + axis_margin
    end_axis = float(max_pt[axis]) - axis_margin

    phase = np.linspace(0.0, 2.0 * math.pi, num_frames, dtype=np.float32)

    red = np.repeat(center[None, :], num_frames, axis=0)
    red[:, 1] = y_base + y_amp * np.sin(phase)
    red[:, orth] -= orth_offset

    green = np.repeat(center[None, :], num_frames, axis=0)
    u = 0.5 - 0.5 * np.cos(phase)
    green[:, axis] = start_axis + u * (end_axis - start_axis)
    green[:, 1] = y_base + 0.3 * y_amp

    blue = np.repeat(center[None, :], num_frames, axis=0)
    blue[:, 1] = y_base + 0.1 * y_amp
    blue[:, orth] += orth_offset

    return red.astype(np.float32), green.astype(np.float32), blue.astype(np.float32)


def _write_ply(path: Path, points: np.ndarray) -> None:
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def _get_output(graph, name: str):
    try:
        tex = graph.get_output(name)
    except Exception:
        tex = graph.getOutput(name)
    if tex is None:
        raise RuntimeError(f"Missing output texture: {name}")
    return tex


def _to_numpy(tex, width: int, height: int) -> np.ndarray:
    if hasattr(tex, "to_numpy"):
        arr = tex.to_numpy()
    else:
        arr = tex.toNumpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        pixel_count = width * height
        if arr.size % pixel_count != 0:
            raise RuntimeError(f"Unexpected texture size {arr.size} for {width}x{height}")
        channels = max(1, arr.size // pixel_count)
        arr = arr.reshape(height, width, channels)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Depth-guided, light-aware probe sampling (Falcor GBufferRT)")
    parser.add_argument("--scene", required=True, help="Path to Bistro .pyscene")
    parser.add_argument("--output", required=True, help="Output .npy file for probes")
    parser.add_argument("--num-probes", type=int, default=800)
    parser.add_argument("--adaptive-ratio", type=float, default=0.7)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--num-cameras", type=int, default=3)
    parser.add_argument("--camera-offset", type=float, default=2.0)
    parser.add_argument("--camera-up-offset", type=float, default=1.5)
    parser.add_argument("--light-sample-step", type=int, default=10)
    parser.add_argument(
        "--light-trajectory",
        action="append",
        default=[],
        help="Path to .npy light trajectory (repeatable). Shape [F,3] or [F,L,3].",
    )
    parser.add_argument("--temporal-weight", type=float, default=0.5)
    parser.add_argument("--decluster", action="store_true")
    parser.add_argument("--decluster-factor", type=float, default=1.5)
    parser.add_argument("--decluster-seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--falcor-python-path", default=None)
    parser.add_argument("--save-ply", action="store_true")
    args = parser.parse_args()

    if args.falcor_python_path and args.falcor_python_path not in sys.path:
        sys.path.insert(0, args.falcor_python_path)

    # Ensure Bistro FBX path for pyscene.
    if "BISTRO_FBX" not in os.environ or os.environ.get("BISTRO_FBX", "") == "":
        os.environ["BISTRO_FBX"] = str(ROOT / "1_data_generation" / "scenes" / "Bistro_v5_2" / "BistroExterior.fbx")

    import falcor as fc

    # Build testbed + GBufferRT graph.
    testbed = fc.Testbed(width=args.width, height=args.height, create_window=False)
    graph = testbed.create_render_graph("ProbeGBuffer")
    graph.create_pass("GBufferRT", "GBufferRT", {"samplePattern": "Stratified", "sampleCount": 1})
    graph.mark_output("GBufferRT.posW")
    graph.mark_output("GBufferRT.depth")
    testbed.render_graph = graph

    # Load scene
    testbed.load_scene(args.scene)
    scene = testbed.scene

    cam = scene.camera
    cam_pos = np.array([cam.position.x, cam.position.y, cam.position.z], dtype=np.float32)
    cam_target = np.array([cam.target.x, cam.target.y, cam.target.z], dtype=np.float32)
    cam_up = np.array([cam.up.x, cam.up.y, cam.up.z], dtype=np.float32)
    forward = cam_target - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, cam_up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = cam_up / (np.linalg.norm(cam_up) + 1e-8)

    cam_positions = [cam_pos]
    if args.num_cameras >= 2:
        cam_positions.append(cam_pos + right * args.camera_offset + up * args.camera_up_offset)
    if args.num_cameras >= 3:
        cam_positions.append(cam_pos - right * args.camera_offset + up * args.camera_up_offset)

    points_all = []
    for pos in cam_positions:
        cam.position = fc.float3(float(pos[0]), float(pos[1]), float(pos[2]))
        cam.target = fc.float3(float(cam_target[0]), float(cam_target[1]), float(cam_target[2]))
        cam.up = fc.float3(float(up[0]), float(up[1]), float(up[2]))

        try:
            testbed.resize_frame_buffer(args.width, args.height)
        except Exception:
            try:
                testbed.resizeFrameBuffer(args.width, args.height)
            except Exception:
                pass
        testbed.clock.pause()
        testbed.frame()

        pos_tex = _get_output(graph, "GBufferRT.posW")
        depth_tex = _get_output(graph, "GBufferRT.depth")
        posw = _to_numpy(pos_tex, args.width, args.height)
        if posw.shape[-1] > 3:
            posw = posw[..., :3]
        depth = _to_numpy(depth_tex, args.width, args.height)
        if depth.ndim == 3:
            depth = depth[..., 0]

        valid = np.isfinite(posw).all(axis=2) & (depth > 0.0) & (depth < 1.0)
        pts = posw[valid]
        if pts.size > 0:
            points_all.append(pts.reshape(-1, 3))

    if not points_all:
        raise RuntimeError("No surface points collected. Check camera setup or GBuffer outputs.")

    points = np.concatenate(points_all, axis=0)

    # Compute weights based on light proximity.
    min_pt = np.array([scene.bounds.minPoint.x, scene.bounds.minPoint.y, scene.bounds.minPoint.z], dtype=np.float32)
    max_pt = np.array([scene.bounds.maxPoint.x, scene.bounds.maxPoint.y, scene.bounds.maxPoint.z], dtype=np.float32)
    num_frames = _parse_int_env("BISTRO_NUM_FRAMES", 600)
    fps = _parse_float_env("BISTRO_FPS", 30.0)
    h_base = _parse_float_env("BISTRO_HEIGHT_BASE_RATIO", 0.08)
    h_amp = _parse_float_env("BISTRO_HEIGHT_AMP_RATIO", 0.06)
    axis_margin = _parse_float_env("BISTRO_AXIS_MARGIN_RATIO", 0.15)
    orth_offset = _parse_float_env("BISTRO_ORTH_OFFSET_RATIO", 0.12)

    if args.light_trajectory:
        trajectories = load_light_trajectories(args.light_trajectory)
    else:
        red, green, blue = _compute_trajectories(
            (min_pt, max_pt),
            num_frames=num_frames,
            fps=fps,
            height_base_ratio=h_base,
            height_amp_ratio=h_amp,
            axis_margin_ratio=axis_margin,
            orth_offset_ratio=orth_offset,
        )
        trajectories = [red, green, blue]

    weights = compute_light_weights(
        points,
        trajectories,
        step=max(1, int(args.light_sample_step)),
        eps=float(args.eps),
        temporal_weight=float(args.temporal_weight),
    )

    total = int(args.num_probes)
    rng = np.random.default_rng(42)
    probes = sample_with_decluster(
        points,
        weights,
        total=total,
        adaptive_ratio=float(args.adaptive_ratio),
        rng=rng,
        decluster=bool(args.decluster),
        decluster_factor=float(args.decluster_factor),
        decluster_seed=int(args.decluster_seed),
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, probes)

    if args.save_ply:
        _write_ply(out_path.with_suffix(".ply"), probes)

    print(f"Saved {probes.shape[0]} probes to {out_path}")


if __name__ == "__main__":
    main()
