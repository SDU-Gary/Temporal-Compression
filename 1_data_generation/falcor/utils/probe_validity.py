"""Probe candidate generation and strict validity checks for dynamic scenes."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def normalize_rows(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    if arr.ndim == 1:
        n = float(np.linalg.norm(arr))
        return arr / max(n, eps)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, eps)


def auto_grid_resolution(max_probes: int) -> int:
    res = int(math.ceil(float(max_probes) ** (1.0 / 3.0)))
    return max(2, res)


def _compute_margin_vec(min_pt: np.ndarray, max_pt: np.ndarray, margin: float) -> np.ndarray:
    size = np.maximum(max_pt - min_pt, 1e-6)
    if margin < 1.0:
        margin_vec = size * float(margin)
    else:
        margin_vec = np.array([margin, margin, margin], dtype=np.float32)
    return np.minimum(margin_vec.astype(np.float32), size * 0.49)


def generate_probe_grid(bounds: Tuple[np.ndarray, np.ndarray], grid_resolution: int, margin: float) -> np.ndarray:
    min_pt, max_pt = bounds
    margin_vec = _compute_margin_vec(min_pt, max_pt, float(margin))
    low = min_pt + margin_vec
    high = max_pt - margin_vec

    xs = np.linspace(low[0], high[0], int(grid_resolution), dtype=np.float32)
    ys = np.linspace(low[1], high[1], int(grid_resolution), dtype=np.float32)
    zs = np.linspace(low[2], high[2], int(grid_resolution), dtype=np.float32)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    probes = np.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], axis=1)
    return probes.astype(np.float32)


def _camera_basis(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.asarray(position, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    upv = np.asarray(up, dtype=np.float32)

    forward = normalize_rows(tgt - pos)
    right = np.cross(forward, upv)
    if float(np.linalg.norm(right)) < 1e-7:
        fallback = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(forward, fallback))) > 0.95:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = np.cross(forward, fallback)
    right = normalize_rows(right)
    up_ortho = normalize_rows(np.cross(right, forward))
    return forward, right, up_ortho


def build_camera_record(
    position: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    frame_width: float,
    frame_height: float,
    focal_length: float,
    near_plane: float,
) -> Dict[str, np.ndarray | float]:
    forward, right, up_ortho = _camera_basis(position, target, up)
    tan_half_fov_x = float(frame_width) / max(2.0 * float(focal_length), 1e-8)
    tan_half_fov_y = float(frame_height) / max(2.0 * float(focal_length), 1e-8)
    return {
        "position": np.asarray(position, dtype=np.float32),
        "forward": forward.astype(np.float32),
        "right": right.astype(np.float32),
        "up": up_ortho.astype(np.float32),
        "tan_half_fov_x": float(tan_half_fov_x),
        "tan_half_fov_y": float(tan_half_fov_y),
        "near_plane": float(max(near_plane, 1e-5)),
    }


def project_points(
    points: np.ndarray,
    camera: Dict[str, np.ndarray | float],
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.asarray(camera["position"], dtype=np.float32)
    forward = np.asarray(camera["forward"], dtype=np.float32)
    right = np.asarray(camera["right"], dtype=np.float32)
    up = np.asarray(camera["up"], dtype=np.float32)
    tan_x = float(camera["tan_half_fov_x"])
    tan_y = float(camera["tan_half_fov_y"])
    near_plane = float(camera["near_plane"])

    rel = np.asarray(points, dtype=np.float32) - pos[None, :]
    x_cam = np.sum(rel * right[None, :], axis=1)
    y_cam = np.sum(rel * up[None, :], axis=1)
    z_cam = np.sum(rel * forward[None, :], axis=1)

    ndc_x = x_cam / np.maximum(z_cam * tan_x, 1e-8)
    ndc_y = y_cam / np.maximum(z_cam * tan_y, 1e-8)
    in_front = z_cam > near_plane
    in_view = in_front & (np.abs(ndc_x) <= 1.0) & (np.abs(ndc_y) <= 1.0)

    px = ((ndc_x * 0.5 + 0.5) * (float(width) - 1.0)).astype(np.int32)
    py = ((1.0 - (ndc_y * 0.5 + 0.5)) * (float(height) - 1.0)).astype(np.int32)
    px = np.clip(px, 0, int(width) - 1)
    py = np.clip(py, 0, int(height) - 1)
    return in_view, px, py


def generate_orbit_camera_poses(
    bounds: Tuple[np.ndarray, np.ndarray],
    num_cameras: int,
    base_pose: Dict[str, np.ndarray] | None = None,
) -> List[Dict[str, np.ndarray]]:
    min_pt, max_pt = bounds
    center = 0.5 * (min_pt + max_pt)
    size = np.maximum(max_pt - min_pt, 1e-5)
    radius = float(max(size[0], size[2]) * 0.7)
    height = float(min_pt[1] + 0.35 * size[1])

    poses: List[Dict[str, np.ndarray]] = []
    if base_pose is not None:
        poses.append(
            {
                "position": np.asarray(base_pose["position"], dtype=np.float32),
                "target": np.asarray(base_pose["target"], dtype=np.float32),
                "up": normalize_rows(np.asarray(base_pose["up"], dtype=np.float32)),
            }
        )

    remaining = max(0, int(num_cameras) - len(poses))
    include_top = remaining >= 2
    orbit_count = max(0, remaining - (1 if include_top else 0))
    if orbit_count == 0 and remaining > 0:
        orbit_count = 1

    for i in range(orbit_count):
        angle = 2.0 * math.pi * float(i) / float(max(orbit_count, 1))
        pos = np.array(
            [
                center[0] + radius * math.cos(angle),
                height,
                center[2] + radius * math.sin(angle),
            ],
            dtype=np.float32,
        )
        poses.append(
            {
                "position": pos,
                "target": center.astype(np.float32),
                "up": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            }
        )

    if include_top:
        top = np.array([center[0], min_pt[1] + 0.95 * size[1], center[2]], dtype=np.float32)
        poses.append(
            {
                "position": top,
                "target": center.astype(np.float32),
                "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            }
        )
    return poses[: max(1, int(num_cameras))]


def make_surface_offset_candidates(
    points: np.ndarray,
    normals: np.ndarray,
    offset_min: float,
    offset_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32)
    nrm = normalize_rows(np.asarray(normals, dtype=np.float32))
    offsets = rng.uniform(float(offset_min), float(offset_max), size=(pts.shape[0], 1)).astype(np.float32)
    pos = pts + nrm * offsets
    neg = pts - nrm * offsets
    return np.concatenate([pos, neg], axis=0).astype(np.float32)


def dedupe_points_voxel(points: np.ndarray, voxel_size: float, bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] == 0:
        return pts
    min_pt, max_pt = bounds
    diag = float(np.linalg.norm(max_pt - min_pt))
    cell = max(float(voxel_size), diag * 1e-5, 1e-5)
    keys = np.floor((pts - min_pt[None, :]) / cell).astype(np.int64)
    _, keep_idx = np.unique(keys, axis=0, return_index=True)
    keep_idx = np.sort(keep_idx)
    return pts[keep_idx]


def sample_probes_adaptive(
    bounds: Tuple[np.ndarray, np.ndarray],
    surface_candidates: np.ndarray,
    max_probes: int,
    uniform_ratio: float,
    margin: float,
    rng: np.random.Generator,
) -> np.ndarray:
    min_pt, max_pt = bounds
    max_probes = max(1, int(max_probes))
    uniform_count = int(round(max_probes * float(uniform_ratio)))
    uniform_count = min(max(uniform_count, 0), max_probes)
    surface_count = max_probes - uniform_count

    margin_vec = _compute_margin_vec(min_pt, max_pt, float(margin))
    low = min_pt + margin_vec
    high = max_pt - margin_vec
    uniform_pts = rng.uniform(low, high, size=(uniform_count, 3)).astype(np.float32)

    surface_pool = np.asarray(surface_candidates, dtype=np.float32)
    if surface_count <= 0:
        probes = uniform_pts
    elif surface_pool.shape[0] == 0:
        extra = rng.uniform(low, high, size=(surface_count, 3)).astype(np.float32)
        probes = np.concatenate([uniform_pts, extra], axis=0)
    else:
        replace = surface_pool.shape[0] < surface_count
        idx = rng.choice(surface_pool.shape[0], size=surface_count, replace=replace)
        surface_pts = surface_pool[idx].astype(np.float32)
        probes = np.concatenate([uniform_pts, surface_pts], axis=0)

    order = rng.permutation(probes.shape[0])
    return probes[order].astype(np.float32)


def evaluate_frame_validity(
    probes: np.ndarray,
    frame_views: Sequence[Dict[str, object]],
    visibility_margin: float,
    collision_distance: float,
    collision_signed_epsilon: float,
) -> Tuple[np.ndarray, Dict[str, int]]:
    probes = np.asarray(probes, dtype=np.float32)
    p_count = probes.shape[0]
    visible_any = np.zeros((p_count,), dtype=bool)
    collision_any = np.zeros((p_count,), dtype=bool)

    for view in frame_views:
        camera = view["camera"]
        posw = np.asarray(view["posW"], dtype=np.float32)
        normw = np.asarray(view["normW"], dtype=np.float32)
        depth = np.asarray(view["depth"], dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        h, w = depth.shape[:2]
        pixel_ok = (
            np.isfinite(depth)
            & (depth > 0.0)
            & (depth < 1.0)
            & np.isfinite(posw).all(axis=2)
            & np.isfinite(normw).all(axis=2)
        )
        if not np.any(pixel_ok):
            continue

        in_view, px, py = project_points(probes, camera, w, h)
        if not np.any(in_view):
            continue
        probe_idx = np.flatnonzero(in_view)
        pix_x = px[probe_idx]
        pix_y = py[probe_idx]
        geom_ok = pixel_ok[pix_y, pix_x]
        if not np.any(geom_ok):
            continue
        probe_idx = probe_idx[geom_ok]
        pix_x = pix_x[geom_ok]
        pix_y = pix_y[geom_ok]

        cam_pos = np.asarray(camera["position"], dtype=np.float32)
        p = probes[probe_idx]
        surf = posw[pix_y, pix_x, :3]
        nrm = normalize_rows(normw[pix_y, pix_x, :3])

        d_probe = np.linalg.norm(p - cam_pos[None, :], axis=1)
        d_surf = np.linalg.norm(surf - cam_pos[None, :], axis=1)
        visible = d_probe + float(visibility_margin) < d_surf
        if np.any(visible):
            visible_any[probe_idx[visible]] = True

        delta = p - surf
        signed = np.abs(np.sum(delta * nrm, axis=1))
        dist = np.linalg.norm(delta, axis=1)
        colliding = (dist <= float(collision_distance)) | (signed <= float(collision_signed_epsilon))
        if np.any(colliding):
            collision_any[probe_idx[colliding]] = True

    frame_valid = visible_any & (~collision_any)
    stats = {
        "visible_any": int(np.sum(visible_any)),
        "collision_any": int(np.sum(collision_any)),
        "valid": int(np.sum(frame_valid)),
    }
    return frame_valid, stats


def reduce_valid_mask_strict(
    probes: np.ndarray,
    frame_stats: Iterable[np.ndarray],
) -> np.ndarray:
    mask = np.ones((int(np.asarray(probes).shape[0]),), dtype=np.float32)
    for valid in frame_stats:
        v = np.asarray(valid, dtype=bool).reshape(-1)
        mask = mask * v.astype(np.float32)
    return mask.astype(np.float32)
