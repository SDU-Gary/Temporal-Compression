#!/usr/bin/env python3
"""Generate ZeroDay temporal SH dataset with strict probe validity mask."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
_UTILS = _ROOT / "1_data_generation" / "utils"
_FALCOR_UTILS = _ROOT / "1_data_generation" / "falcor" / "utils"
_CORE_UTILS = _ROOT / "2_src" / "utils"
for p in (str(_UTILS), str(_FALCOR_UTILS), str(_CORE_UTILS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from spherical_harmonics import fibonacci_sphere, fit_sh_coefficients  # noqa: E402
from falcor_render import (  # noqa: E402
    build_testbed,
    get_scene_bounds,
    load_scene,
    render_sh_cubemap,
    render_single_pixel,
    set_camera_look,
)
from probe_validity import (  # noqa: E402
    auto_grid_resolution,
    build_camera_record,
    dedupe_points_voxel,
    evaluate_frame_validity,
    generate_orbit_camera_poses,
    generate_probe_grid,
    make_surface_offset_candidates,
    normalize_rows,
    sample_probes_adaptive,
)


def _parse_frame_indices(num_frames: int, frame_indices: str, frame_step: int | None) -> List[int]:
    if frame_indices:
        items = []
        for token in frame_indices.split(","):
            token = token.strip()
            if token == "":
                continue
            items.append(int(token))
        indices = sorted(set([i for i in items if 0 <= i < num_frames]))
        return indices
    if frame_step is not None and frame_step > 0:
        return list(range(0, num_frames, frame_step))
    return list(range(num_frames))


def _set_scene_time(testbed, frame_idx: int, fps: float) -> None:
    t_sec = float(frame_idx) / float(max(fps, 1e-6))
    try:
        testbed.clock.time = t_sec
    except Exception:
        try:
            testbed.clock.setTime(t_sec)
        except Exception:
            pass


def _set_camera_pose(scene, falcor_module, pose: Dict[str, np.ndarray]) -> Dict[str, np.ndarray | float]:
    cam = scene.camera
    pos = np.asarray(pose["position"], dtype=np.float32)
    target = np.asarray(pose["target"], dtype=np.float32)
    up = np.asarray(pose["up"], dtype=np.float32)
    cam.position = falcor_module.float3(float(pos[0]), float(pos[1]), float(pos[2]))
    cam.target = falcor_module.float3(float(target[0]), float(target[1]), float(target[2]))
    cam.up = falcor_module.float3(float(up[0]), float(up[1]), float(up[2]))
    return build_camera_record(
        position=pos,
        target=target,
        up=up,
        frame_width=float(cam.frameWidth),
        frame_height=float(cam.frameHeight),
        focal_length=float(cam.focalLength),
        near_plane=float(cam.nearPlane),
    )


def _build_gbuffer_testbed(
    width: int,
    height: int,
    falcor_python_path: str | None,
    graph_name: str = "ZeroDayGBuffer",
):
    if falcor_python_path and falcor_python_path not in sys.path:
        sys.path.insert(0, falcor_python_path)
    import falcor  # noqa: WPS433

    testbed = falcor.Testbed(width=width, height=height, create_window=False)
    if hasattr(testbed, "create_render_graph"):
        graph = testbed.create_render_graph(graph_name)
    else:
        graph = testbed.createRenderGraph(graph_name)

    options = {
        "samplePattern": "Center",
        "sampleCount": 1,
        "useAlphaTest": True,
    }
    if hasattr(graph, "create_pass"):
        graph.create_pass("GBufferRT", "GBufferRT", options)
    else:
        graph.createPass("GBufferRT", "GBufferRT", options)

    if hasattr(graph, "mark_output"):
        graph.mark_output("GBufferRT.posW")
        graph.mark_output("GBufferRT.normW")
        graph.mark_output("GBufferRT.depth")
    else:
        graph.markOutput("GBufferRT.posW")
        graph.markOutput("GBufferRT.normW")
        graph.markOutput("GBufferRT.depth")

    try:
        testbed.render_graph = graph
    except Exception:
        try:
            testbed.setRenderGraph(graph)
        except Exception:
            pass
    return testbed, graph, falcor


def _resize_framebuffer(testbed, width: int, height: int) -> None:
    try:
        testbed.resize_frame_buffer(int(width), int(height))
    except Exception:
        try:
            testbed.resizeFrameBuffer(int(width), int(height))
        except Exception:
            pass


def _set_camera_pose_from_ray(scene, falcor_module, origin: np.ndarray, direction: np.ndarray) -> None:
    origin = np.asarray(origin, dtype=np.float32).reshape(3)
    forward = normalize_rows(np.asarray(direction, dtype=np.float32).reshape(3))
    up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(forward, up_hint))) > 0.95:
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = normalize_rows(np.cross(forward, up_hint))
    up = normalize_rows(np.cross(right, forward))
    target = origin + forward

    cam = scene.camera
    cam.position = falcor_module.float3(float(origin[0]), float(origin[1]), float(origin[2]))
    cam.target = falcor_module.float3(float(target[0]), float(target[1]), float(target[2]))
    cam.up = falcor_module.float3(float(up[0]), float(up[1]), float(up[2]))
    try:
        cam.nearPlane = 1e-4
    except Exception:
        pass


def _query_hit_distance(
    rq_testbed,
    rq_graph,
    rq_scene,
    falcor_module,
    origin: np.ndarray,
    direction: np.ndarray,
) -> float:
    _set_camera_pose_from_ray(rq_scene, falcor_module, origin, direction)
    rq_testbed.frame()
    gbuf = _capture_gbuffer(rq_graph, 1, 1)
    depth = float(gbuf["depth"][0, 0])
    if not np.isfinite(depth) or depth <= 0.0 or depth >= 1.0:
        return float("inf")
    hit_pos = np.asarray(gbuf["posW"][0, 0, :3], dtype=np.float32)
    if not np.isfinite(hit_pos).all():
        return float("inf")
    return float(np.linalg.norm(hit_pos - np.asarray(origin, dtype=np.float32)))


def _build_collision_directions(num_dirs: int) -> np.ndarray:
    base = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    num_dirs = max(1, int(num_dirs))
    if num_dirs <= base.shape[0]:
        return base[:num_dirs].copy()
    extra = fibonacci_sphere(num_dirs).astype(np.float32)
    return extra


def _evaluate_frame_validity_rayquery(
    probes: np.ndarray,
    base_valid: np.ndarray,
    camera_poses: Sequence[Dict[str, np.ndarray]],
    rq_testbed,
    rq_graph,
    rq_scene,
    falcor_module,
    visibility_margin: float,
    collision_distance: float,
    collision_signed_epsilon: float,
    collision_rays: int,
    max_visibility_cameras: int,
    origin_epsilon: float,
    refine_only_passing: bool,
) -> tuple[np.ndarray, Dict[str, int]]:
    probes = np.asarray(probes, dtype=np.float32)
    p_count = probes.shape[0]
    base = np.asarray(base_valid, dtype=bool).reshape(-1)
    if base.shape[0] != p_count:
        raise ValueError("base_valid size mismatch")

    if refine_only_passing:
        candidate_idx = np.flatnonzero(base)
        valid = np.zeros((p_count,), dtype=bool)
    else:
        candidate_idx = np.arange(p_count, dtype=np.int64)
        valid = np.zeros((p_count,), dtype=bool)

    if max_visibility_cameras > 0:
        cam_positions = [np.asarray(p["position"], dtype=np.float32) for p in camera_poses[: max_visibility_cameras]]
    else:
        cam_positions = [np.asarray(p["position"], dtype=np.float32) for p in camera_poses]
    if not cam_positions:
        return np.zeros((p_count,), dtype=bool), {"tested": 0, "visible_any": 0, "collision_any": 0, "valid": 0}

    dirs = _build_collision_directions(collision_rays)
    eps = max(float(origin_epsilon), float(collision_signed_epsilon), 1e-4)
    collision_any = 0
    visible_any = 0

    for idx in candidate_idx.tolist():
        probe = probes[int(idx)]

        colliding = False
        for d in dirs:
            d = normalize_rows(d)
            origin = probe + d * eps
            hit_dist = _query_hit_distance(rq_testbed, rq_graph, rq_scene, falcor_module, origin, d)
            if np.isfinite(hit_dist) and hit_dist <= float(collision_distance):
                colliding = True
                break
        if colliding:
            collision_any += 1
            continue

        is_visible = False
        for cam_pos in cam_positions:
            vec = probe - cam_pos
            dist_probe = float(np.linalg.norm(vec))
            if dist_probe <= 1e-6:
                is_visible = True
                break
            d = vec / max(dist_probe, 1e-6)
            origin = cam_pos + d * eps
            hit_dist = _query_hit_distance(rq_testbed, rq_graph, rq_scene, falcor_module, origin, d)
            if (not np.isfinite(hit_dist)) or (hit_dist + float(visibility_margin) >= dist_probe):
                is_visible = True
                break

        if is_visible:
            valid[int(idx)] = True
            visible_any += 1

    return valid, {
        "tested": int(candidate_idx.size),
        "visible_any": int(visible_any),
        "collision_any": int(collision_any),
        "valid": int(np.sum(valid)),
    }


def _get_output(graph, names: Sequence[str]):
    for name in names:
        try:
            tex = graph.get_output(name)
            if tex is not None:
                return tex
        except Exception:
            try:
                tex = graph.getOutput(name)
                if tex is not None:
                    return tex
            except Exception:
                pass
    raise RuntimeError(f"Failed to read graph output from candidates={list(names)}")


def _to_numpy(tex, width: int, height: int) -> np.ndarray:
    if hasattr(tex, "to_numpy"):
        arr = tex.to_numpy()
    else:
        arr = tex.toNumpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        pixel_count = int(width) * int(height)
        if arr.size % pixel_count != 0:
            raise RuntimeError(f"Unexpected flat texture size {arr.size} for {width}x{height}")
        channels = max(1, arr.size // pixel_count)
        arr = arr.reshape(height, width, channels)
    return arr


def _capture_gbuffer(graph, width: int, height: int) -> Dict[str, np.ndarray]:
    pos_tex = _get_output(graph, ("GBufferRT.posW",))
    nrm_tex = _get_output(graph, ("GBufferRT.normW", "GBufferRT.normWRoughnessMaterialID"))
    depth_tex = _get_output(graph, ("GBufferRT.depth",))

    posw = _to_numpy(pos_tex, width, height)
    normw = _to_numpy(nrm_tex, width, height)
    depth = _to_numpy(depth_tex, width, height)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if posw.shape[-1] > 3:
        posw = posw[..., :3]
    if normw.shape[-1] > 3:
        normw = normw[..., :3]
    return {
        "posW": np.asarray(posw, dtype=np.float32),
        "normW": np.asarray(normw, dtype=np.float32),
        "depth": np.asarray(depth, dtype=np.float32),
    }


def _collect_surface_pool(
    gb_testbed,
    gb_graph,
    gb_scene,
    falcor_module,
    camera_poses: Sequence[Dict[str, np.ndarray]],
    frame_indices: Sequence[int],
    fps: float,
    width: int,
    height: int,
    max_points_per_view: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1234)
    points: List[np.ndarray] = []
    normals: List[np.ndarray] = []

    for frame_idx in tqdm(frame_indices, desc="Collecting surface pool"):
        _set_scene_time(gb_testbed, int(frame_idx), fps)
        for pose in camera_poses:
            _set_camera_pose(gb_scene, falcor_module, pose)
            _resize_framebuffer(gb_testbed, width, height)
            gb_testbed.frame()
            gbuf = _capture_gbuffer(gb_graph, width, height)
            depth = gbuf["depth"]
            posw = gbuf["posW"]
            normw = gbuf["normW"]
            mask = (
                np.isfinite(depth)
                & (depth > 0.0)
                & (depth < 1.0)
                & np.isfinite(posw).all(axis=2)
                & np.isfinite(normw).all(axis=2)
            )
            if not np.any(mask):
                continue
            p = posw[mask].reshape(-1, 3)
            n = normw[mask].reshape(-1, 3)
            if int(max_points_per_view) > 0 and p.shape[0] > int(max_points_per_view):
                idx = rng.choice(p.shape[0], size=int(max_points_per_view), replace=False)
                p = p[idx]
                n = n[idx]
            points.append(p.astype(np.float32))
            normals.append(n.astype(np.float32))

    if not points:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(points, axis=0), np.concatenate(normals, axis=0)


def _pause_clock(testbed) -> None:
    try:
        testbed.clock.pause()
    except Exception:
        try:
            testbed.clock.setPaused(True)
        except Exception:
            pass


def _build_dummy_descriptor(dtype=np.float32) -> np.ndarray:
    desc = np.zeros((12,), dtype=dtype)
    desc[0] = 1.0
    desc[1:4] = 1.0
    return desc


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ZeroDay temporal SH dataset with strict valid_mask.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--scene",
        type=str,
        default=str(_ROOT / "1_data_generation" / "scenes" / "ZeroDay" / "MEASURE_SEVEN" / "MEASURE_SEVEN_COLORED_LIGHTS.fbx"),
    )
    parser.add_argument("--num-frames", type=int, default=250)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--frame-indices", type=str, default="")
    parser.add_argument("--frame-step", type=int, default=None)
    parser.add_argument("--validation-frame-step", type=int, default=1)
    parser.add_argument("--candidate-frame-step", type=int, default=5)

    parser.add_argument("--sh-mode", type=str, default="cubemap", choices=["dir", "cubemap"])
    parser.add_argument("--cube-res", type=int, default=32)
    parser.add_argument("--num-sh-samples", type=int, default=512)
    parser.add_argument("--spp", type=int, default=256)
    parser.add_argument("--accum-frames", type=int, default=1)
    parser.add_argument("--fixed-seed", type=int, default=1)
    parser.add_argument("--radiance-clamp", type=float, default=0.0)
    parser.add_argument("--falcor-python-path", type=str, default=None)

    parser.add_argument("--probe-mode", type=str, default="adaptive", choices=["grid", "adaptive"])
    parser.add_argument("--grid-resolution", type=int, default=8)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--max-probes", type=int, default=800)
    parser.add_argument("--probe-uniform-ratio", type=float, default=0.3)
    parser.add_argument("--probe-file", type=str, default=None)

    parser.add_argument("--gbuffer-width", type=int, default=640)
    parser.add_argument("--gbuffer-height", type=int, default=360)
    parser.add_argument("--gbuffer-num-cameras", type=int, default=8)
    parser.add_argument("--max-points-per-view", type=int, default=12000)
    parser.add_argument("--max-surface-candidates", type=int, default=800000)
    parser.add_argument("--surface-offset-min", type=float, default=0.03)
    parser.add_argument("--surface-offset-max", type=float, default=0.15)
    parser.add_argument("--candidate-voxel-size", type=float, default=0.05)

    parser.add_argument("--visibility-margin", type=float, default=0.02)
    parser.add_argument("--collision-distance", type=float, default=0.03)
    parser.add_argument("--collision-signed-epsilon", type=float, default=0.01)
    parser.add_argument(
        "--validity-method",
        type=str,
        default="rayquery_hybrid",
        choices=["gbuffer", "rayquery_hybrid", "rayquery_full"],
    )
    parser.add_argument("--rayquery-collision-rays", type=int, default=12)
    parser.add_argument("--rayquery-max-visibility-cameras", type=int, default=8)
    parser.add_argument("--rayquery-origin-epsilon", type=float, default=0.002)
    parser.add_argument("--rayquery-refine-only-passing", action="store_true")
    args = parser.parse_args()

    if args.validity_method == "rayquery_hybrid" and not args.rayquery_refine_only_passing:
        args.rayquery_refine_only_passing = True

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    desired_spp = int(args.spp)
    per_frame_spp = min(16, max(1, desired_spp))
    auto_frames = int(math.ceil(float(desired_spp) / float(per_frame_spp)))
    if args.accum_frames < auto_frames:
        args.accum_frames = auto_frames

    pt_testbed, pt_graph, falcor = build_testbed(
        width=1,
        height=1,
        spp=per_frame_spp,
        falcor_python_path=args.falcor_python_path,
        fixed_seed=args.fixed_seed,
        use_russian_roulette=False,
    )
    pt_scene = load_scene(pt_testbed, args.scene)

    gb_testbed, gb_graph, _ = _build_gbuffer_testbed(
        width=int(args.gbuffer_width),
        height=int(args.gbuffer_height),
        falcor_python_path=args.falcor_python_path,
        graph_name="ZeroDayGBuffer",
    )
    gb_testbed.load_scene(str(args.scene))
    gb_scene = gb_testbed.scene
    gb_scene.animated = True
    gb_scene.loopAnimations = False
    _pause_clock(gb_testbed)

    rq_testbed = None
    rq_graph = None
    rq_scene = None
    if args.validity_method in ("rayquery_hybrid", "rayquery_full"):
        rq_testbed, rq_graph, _ = _build_gbuffer_testbed(
            width=1,
            height=1,
            falcor_python_path=args.falcor_python_path,
            graph_name="ZeroDayRayQuery",
        )
        rq_testbed.load_scene(str(args.scene))
        rq_scene = rq_testbed.scene
        rq_scene.animated = True
        rq_scene.loopAnimations = False
        _pause_clock(rq_testbed)
        _resize_framebuffer(rq_testbed, 1, 1)

    pt_scene.animated = True
    pt_scene.loopAnimations = False
    pt_scene.renderSettings.useEmissiveLights = True
    _pause_clock(pt_testbed)

    bounds = get_scene_bounds(pt_scene)
    frame_indices = _parse_frame_indices(int(args.num_frames), args.frame_indices, args.frame_step)
    if len(frame_indices) == 0:
        raise ValueError("No frame indices selected.")
    validation_frame_step = max(1, int(args.validation_frame_step))
    candidate_frame_step = max(1, int(args.candidate_frame_step))
    validation_frames = frame_indices[::validation_frame_step]
    candidate_frames = frame_indices[::candidate_frame_step]

    base_cam = gb_scene.camera
    base_pose = {
        "position": np.array([base_cam.position.x, base_cam.position.y, base_cam.position.z], dtype=np.float32),
        "target": np.array([base_cam.target.x, base_cam.target.y, base_cam.target.z], dtype=np.float32),
        "up": np.array([base_cam.up.x, base_cam.up.y, base_cam.up.z], dtype=np.float32),
    }
    camera_poses = generate_orbit_camera_poses(
        bounds=bounds,
        num_cameras=max(1, int(args.gbuffer_num_cameras)),
        base_pose=base_pose,
    )

    rng = np.random.default_rng(42)
    probes = None
    candidate_points_count = 0
    if args.probe_file:
        probes = np.load(args.probe_file).astype(np.float32)
    elif args.probe_mode == "grid":
        if args.max_probes is not None:
            auto_res = auto_grid_resolution(args.max_probes)
            if auto_res != args.grid_resolution:
                args.grid_resolution = auto_res
        probes = generate_probe_grid(bounds, int(args.grid_resolution), float(args.margin))
        if args.max_probes is not None and probes.shape[0] > int(args.max_probes):
            order = rng.permutation(probes.shape[0])[: int(args.max_probes)]
            probes = probes[order]
    else:
        surf_points, surf_normals = _collect_surface_pool(
            gb_testbed=gb_testbed,
            gb_graph=gb_graph,
            gb_scene=gb_scene,
            falcor_module=falcor,
            camera_poses=camera_poses,
            frame_indices=candidate_frames,
            fps=float(args.fps),
            width=int(args.gbuffer_width),
            height=int(args.gbuffer_height),
            max_points_per_view=int(args.max_points_per_view),
        )
        if surf_points.shape[0] == 0:
            raise RuntimeError("Failed to collect any surface points from GBuffer.")

        if int(args.max_surface_candidates) > 0 and surf_points.shape[0] > int(args.max_surface_candidates):
            keep = rng.choice(surf_points.shape[0], size=int(args.max_surface_candidates), replace=False)
            surf_points = surf_points[keep]
            surf_normals = surf_normals[keep]

        near = make_surface_offset_candidates(
            points=surf_points,
            normals=surf_normals,
            offset_min=float(args.surface_offset_min),
            offset_max=float(args.surface_offset_max),
            rng=rng,
        )
        near = dedupe_points_voxel(
            points=near,
            voxel_size=float(args.candidate_voxel_size),
            bounds=bounds,
        )
        candidate_points_count = int(near.shape[0])
        probes = sample_probes_adaptive(
            bounds=bounds,
            surface_candidates=near,
            max_probes=int(args.max_probes),
            uniform_ratio=float(args.probe_uniform_ratio),
            margin=float(args.margin),
            rng=rng,
        )

    probes = np.asarray(probes, dtype=np.float32)
    if probes.ndim != 2 or probes.shape[1] != 3:
        raise ValueError(f"Invalid probes shape: {probes.shape}")

    valid_mask = np.ones((probes.shape[0],), dtype=np.float32)
    frame_pass_counts: List[int] = []
    frame_visible_counts: List[int] = []
    frame_collision_counts: List[int] = []
    frame_rq_tested_counts: List[int] = []
    frame_rq_visible_counts: List[int] = []
    frame_rq_collision_counts: List[int] = []
    frame_rq_valid_counts: List[int] = []
    start_validate = time.time()
    for frame_idx in tqdm(validation_frames, desc="Strict probe validation"):
        _set_scene_time(gb_testbed, int(frame_idx), float(args.fps))
        if rq_testbed is not None:
            _set_scene_time(rq_testbed, int(frame_idx), float(args.fps))
        frame_views: List[Dict[str, object]] = []
        for pose in camera_poses:
            camera_record = _set_camera_pose(gb_scene, falcor, pose)
            _resize_framebuffer(gb_testbed, int(args.gbuffer_width), int(args.gbuffer_height))
            gb_testbed.frame()
            gbuf = _capture_gbuffer(gb_graph, int(args.gbuffer_width), int(args.gbuffer_height))
            frame_views.append(
                {
                    "camera": camera_record,
                    "posW": gbuf["posW"],
                    "normW": gbuf["normW"],
                    "depth": gbuf["depth"],
                }
            )

        frame_valid_gbuf, stats = evaluate_frame_validity(
            probes=probes,
            frame_views=frame_views,
            visibility_margin=float(args.visibility_margin),
            collision_distance=float(args.collision_distance),
            collision_signed_epsilon=float(args.collision_signed_epsilon),
        )
        frame_valid = frame_valid_gbuf
        rq_stats = None
        if args.validity_method in ("rayquery_hybrid", "rayquery_full"):
            frame_valid, rq_stats = _evaluate_frame_validity_rayquery(
                probes=probes,
                base_valid=frame_valid_gbuf,
                camera_poses=camera_poses,
                rq_testbed=rq_testbed,
                rq_graph=rq_graph,
                rq_scene=rq_scene,
                falcor_module=falcor,
                visibility_margin=float(args.visibility_margin),
                collision_distance=float(args.collision_distance),
                collision_signed_epsilon=float(args.collision_signed_epsilon),
                collision_rays=int(args.rayquery_collision_rays),
                max_visibility_cameras=int(args.rayquery_max_visibility_cameras),
                origin_epsilon=float(args.rayquery_origin_epsilon),
                refine_only_passing=bool(args.rayquery_refine_only_passing),
            )
            frame_rq_tested_counts.append(int(rq_stats["tested"]))
            frame_rq_visible_counts.append(int(rq_stats["visible_any"]))
            frame_rq_collision_counts.append(int(rq_stats["collision_any"]))
            frame_rq_valid_counts.append(int(rq_stats["valid"]))

        valid_mask = valid_mask * frame_valid.astype(np.float32)
        frame_pass_counts.append(int(np.sum(frame_valid)))
        frame_visible_counts.append(int(stats["visible_any"]))
        frame_collision_counts.append(int(stats["collision_any"]))

    validate_seconds = float(time.time() - start_validate)
    valid_indices = np.where(valid_mask > 0.5)[0]
    num_valid = int(valid_indices.size)
    print(f"Valid probes after strict all-frame intersection: {num_valid}/{probes.shape[0]}", flush=True)

    directions = None
    if args.sh_mode == "dir":
        directions = fibonacci_sphere(int(args.num_sh_samples))

    P = int(probes.shape[0])
    M = int(len(frame_indices))
    N = 1
    tensor = np.zeros((P, M, 27), dtype=np.float32)
    light_configs = np.zeros((M, N, 12), dtype=np.float32)
    light_mask = np.ones((M, N), dtype=np.float32)
    dummy_desc = _build_dummy_descriptor(np.float32)

    def render_probe_sh(probe: np.ndarray) -> np.ndarray:
        if args.sh_mode == "cubemap":
            return render_sh_cubemap(
                pt_testbed,
                pt_graph,
                pt_scene,
                probe,
                cube_res=int(args.cube_res),
                num_frames=int(args.accum_frames),
                seed_base=int(args.fixed_seed),
                radiance_clamp=float(args.radiance_clamp) if float(args.radiance_clamp) > 0.0 else None,
            )
        if directions is None:
            raise RuntimeError("Directional SH mode selected but directions are not initialized.")
        radiances = []
        for d in directions:
            set_camera_look(pt_scene, probe, d)
            radiances.append(
                render_single_pixel(
                    pt_testbed,
                    pt_graph,
                    num_frames=int(args.accum_frames),
                    seed_base=int(args.fixed_seed),
                    radiance_clamp=float(args.radiance_clamp) if float(args.radiance_clamp) > 0.0 else None,
                )
            )
        return fit_sh_coefficients(directions, np.asarray(radiances, dtype=np.float32), max_order=2)

    start_bake = time.time()
    for out_idx, frame_idx in enumerate(frame_indices):
        _set_scene_time(pt_testbed, int(frame_idx), float(args.fps))
        light_configs[out_idx, 0, :] = dummy_desc
        for probe_idx in tqdm(valid_indices, desc=f"Bake frame {out_idx + 1}/{M}", leave=False):
            probe = probes[int(probe_idx)]
            tensor[int(probe_idx), out_idx, :] = render_probe_sh(probe)

    bake_seconds = float(time.time() - start_bake)
    num_samples = int(args.num_sh_samples)
    if args.sh_mode == "cubemap":
        num_samples = int(args.cube_res) * int(args.cube_res) * 6

    metadata = {
        "engine": "Falcor",
        "scene": str(Path(args.scene).resolve()),
        "num_probes": int(P),
        "num_valid_probes": int(num_valid),
        "num_configs": int(M),
        "num_frames": int(args.num_frames),
        "fps": float(args.fps),
        "duration_sec": float((args.num_frames - 1) / max(args.fps, 1e-6)),
        "frame_indices": list(frame_indices),
        "num_lights": int(N),
        "light_descriptor_dim": 12,
        "sh_mode": args.sh_mode,
        "cube_res": int(args.cube_res),
        "num_sh_samples": int(args.num_sh_samples),
        "num_samples": int(num_samples),
        "spp": int(desired_spp),
        "spp_per_frame": int(per_frame_spp),
        "accum_frames": int(args.accum_frames),
        "radiance_clamp": float(args.radiance_clamp),
        "bounds": {"min": bounds[0].tolist(), "max": bounds[1].tolist()},
        "probe_mode": args.probe_mode,
        "probe_uniform_ratio": float(args.probe_uniform_ratio),
        "probe_file": str(args.probe_file) if args.probe_file else None,
        "probe_validation": {
            "policy": "all_frames_intersection",
            "validity_method": str(args.validity_method),
            "candidate_frame_step": int(candidate_frame_step),
            "validation_frame_step": int(validation_frame_step),
            "num_candidate_frames": int(len(candidate_frames)),
            "num_validation_frames": int(len(validation_frames)),
            "num_validation_cameras": int(len(camera_poses)),
            "visibility_margin": float(args.visibility_margin),
            "collision_distance": float(args.collision_distance),
            "collision_signed_epsilon": float(args.collision_signed_epsilon),
            "frame_pass_counts": frame_pass_counts,
            "frame_visible_counts": frame_visible_counts,
            "frame_collision_counts": frame_collision_counts,
            "validation_seconds": validate_seconds,
        },
        "candidate_pool": {
            "gbuffer_width": int(args.gbuffer_width),
            "gbuffer_height": int(args.gbuffer_height),
            "max_points_per_view": int(args.max_points_per_view),
            "max_surface_candidates": int(args.max_surface_candidates),
            "surface_offset_range": [float(args.surface_offset_min), float(args.surface_offset_max)],
            "candidate_voxel_size": float(args.candidate_voxel_size),
            "candidate_points_count": int(candidate_points_count),
        },
        "bake_seconds": float(bake_seconds),
        "generation_date": datetime.now().isoformat(),
    }
    if args.validity_method in ("rayquery_hybrid", "rayquery_full"):
        metadata["probe_validation"]["rayquery"] = {
            "collision_rays": int(args.rayquery_collision_rays),
            "max_visibility_cameras": int(args.rayquery_max_visibility_cameras),
            "origin_epsilon": float(args.rayquery_origin_epsilon),
            "refine_only_passing": bool(args.rayquery_refine_only_passing),
            "frame_tested_counts": frame_rq_tested_counts,
            "frame_visible_counts": frame_rq_visible_counts,
            "frame_collision_counts": frame_rq_collision_counts,
            "frame_valid_counts": frame_rq_valid_counts,
        }

    np.savez_compressed(
        output_dir / "parametric_tensor.npz",
        tensor=tensor,
        probe_positions=probes,
        light_configs=light_configs,
        light_mask=light_mask,
        valid_mask=valid_mask.astype(np.float32),
        metadata=metadata,
    )
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved dataset to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
