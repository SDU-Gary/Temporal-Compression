#!/usr/bin/env python3
"""Visualize lamp positions and obstacle AABBs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _load_metadata(data_root: Path) -> dict:
    meta_path = data_root / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found at {meta_path}")
    return json.loads(meta_path.read_text())


def _denormalize_pos(pos: np.ndarray, min_pt: np.ndarray, max_pt: np.ndarray) -> np.ndarray:
    return 0.5 * (pos + 1.0) * (max_pt - min_pt) + min_pt


def _load_lamp_positions(data_root: Path, meta: dict) -> np.ndarray:
    if meta.get("lamp_positions"):
        return np.array(meta["lamp_positions"], dtype=np.float32)

    # Fallback: derive from light_configs (normalized positions)
    npz_path = data_root / "parametric_tensor.npz"
    data = np.load(npz_path, allow_pickle=True)
    light_configs = data["light_configs"]  # [M, N, 12]
    bounds = meta.get("bounds")
    if bounds is None:
        raise ValueError("bounds missing from metadata, cannot denormalize positions.")
    min_pt = np.array(bounds["min"], dtype=np.float32)
    max_pt = np.array(bounds["max"], dtype=np.float32)

    # Assume slot 1 is point light, pos stored in indices 4:7
    pos_norm = light_configs[:, 1, 4:7]
    return _denormalize_pos(pos_norm, min_pt, max_pt)


def _load_lamp_types(meta: dict, count: int) -> np.ndarray:
    types = meta.get("lamp_position_types")
    if types is None:
        return np.zeros((count,), dtype=np.int32)
    arr = np.array(types, dtype=np.int32)
    if arr.shape[0] != count:
        arr = np.resize(arr, (count,))
    return arr


def _aabb_distance(point: np.ndarray, aabb: Tuple[np.ndarray, np.ndarray]) -> float:
    mn, mx = aabb
    # Distance to AABB surface if outside; 0 if inside
    dx = max(mn[0] - point[0], 0.0, point[0] - mx[0])
    dy = max(mn[1] - point[1], 0.0, point[1] - mx[1])
    dz = max(mn[2] - point[2], 0.0, point[2] - mx[2])
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def _nearest_aabb_distance(point: np.ndarray, obstacles: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    if not obstacles:
        return float("nan")
    return min(_aabb_distance(point, aabb) for aabb in obstacles)


def _print_distribution_stats(points: np.ndarray, obstacles: List[Tuple[np.ndarray, np.ndarray]], grid_size: int) -> None:
    print(f"Point count: {points.shape[0]}")
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    print(f"Bounds (points): min={mins.tolist()} max={maxs.tolist()}")

    # Nearest-neighbor distance (rough density proxy)
    if points.shape[0] > 1:
        diff = points[:, None, :] - points[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        nn = dist.min(axis=1)
        print(f"NN distance: mean={float(nn.mean()):.4f} std={float(nn.std()):.4f} min={float(nn.min()):.4f} max={float(nn.max()):.4f}")

    # Voxel occupancy
    mins = mins - 1e-6
    maxs = maxs + 1e-6
    grid = np.floor((points - mins) / (maxs - mins) * grid_size).astype(int)
    grid = np.clip(grid, 0, grid_size - 1)
    vox_ids = grid[:, 0] + grid[:, 1] * grid_size + grid[:, 2] * grid_size * grid_size
    unique_vox = np.unique(vox_ids)
    occupancy = len(unique_vox) / float(grid_size ** 3)
    print(f"Voxel occupancy (grid={grid_size}): {len(unique_vox)}/{grid_size**3} ({occupancy:.4f})")

    # AABB distance distribution
    if obstacles:
        dists = np.array([_nearest_aabb_distance(p, obstacles) for p in points], dtype=np.float32)
        q = np.quantile(dists, [0.1, 0.25, 0.5, 0.75, 0.9])
        print("Dist to nearest AABB (m):",
              f"p10={q[0]:.4f} p25={q[1]:.4f} p50={q[2]:.4f} p75={q[3]:.4f} p90={q[4]:.4f}")


def _aabb_lines(aabb: Tuple[np.ndarray, np.ndarray]):
    mn, mx = aabb
    # 8 corners
    corners = np.array(
        [
            [mn[0], mn[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mx[1], mx[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mn[2]],
            [mx[0], mx[1], mx[2]],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    ]
    return corners, edges


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize lamp positions and obstacle AABBs")
    parser.add_argument("--data-root", required=True, help="Dataset root containing metadata.json")
    parser.add_argument("--save", type=str, default=None, help="Optional output PNG path")
    parser.add_argument("--show", action="store_true", help="Show interactive window")
    parser.add_argument("--grid-size", type=int, default=10, help="Grid size for density stats")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    meta = _load_metadata(data_root)
    lamp_positions = _load_lamp_positions(data_root, meta)
    lamp_types = _load_lamp_types(meta, lamp_positions.shape[0])

    bounds = meta.get("bounds", None)
    obstacles = meta.get("obstacles", [])
    obstacles = [(np.array(o["min"], dtype=np.float32), np.array(o["max"], dtype=np.float32)) for o in obstacles]

    try:
        import matplotlib.pyplot as plt  # type: ignore
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as exc:
        raise RuntimeError("matplotlib is required for visualization") from exc

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Color-code uniform vs surface samples
    uniform_mask = lamp_types == 0
    surface_mask = lamp_types == 1
    ax.scatter(
        lamp_positions[uniform_mask, 0],
        lamp_positions[uniform_mask, 1],
        lamp_positions[uniform_mask, 2],
        s=8,
        alpha=0.7,
        label="Uniform samples",
        color="#1f77b4",
    )
    ax.scatter(
        lamp_positions[surface_mask, 0],
        lamp_positions[surface_mask, 1],
        lamp_positions[surface_mask, 2],
        s=10,
        alpha=0.8,
        label="Surface-near samples",
        color="#ff7f0e",
    )

    # Plot scene bounds as wireframe if available
    if bounds is not None:
        min_pt = np.array(bounds["min"], dtype=np.float32)
        max_pt = np.array(bounds["max"], dtype=np.float32)
        corners, edges = _aabb_lines((min_pt, max_pt))
        for a, b in edges:
            ax.plot(
                [corners[a, 0], corners[b, 0]],
                [corners[a, 1], corners[b, 1]],
                [corners[a, 2], corners[b, 2]],
                color="gray",
                linewidth=1,
                alpha=0.5,
            )

    # Plot obstacle AABBs
    for aabb in obstacles:
        corners, edges = _aabb_lines(aabb)
        for a, b in edges:
            ax.plot(
                [corners[a, 0], corners[b, 0]],
                [corners[a, 1], corners[b, 1]],
                [corners[a, 2], corners[b, 2]],
                color="red",
                linewidth=1,
                alpha=0.8,
            )

    _print_distribution_stats(lamp_positions, obstacles, args.grid_size)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Lamp positions and obstacle AABBs")
    ax.legend(loc="upper right")

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        print(f"Saved visualization to {out_path}")

    if args.show or not args.save:
        plt.show()


if __name__ == "__main__":
    main()
