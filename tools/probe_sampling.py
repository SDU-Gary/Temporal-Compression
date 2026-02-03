"""Utilities for light-aware probe sampling."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


def load_light_trajectories(paths: Sequence[str]) -> List[np.ndarray]:
    """Load light trajectories from one or more .npy files.

    Supported shapes:
    - [F, 3] for a single light
    - [F, L, 3] for multiple lights in a single file
    """
    trajectories: List[np.ndarray] = []
    for path in paths:
        data = np.load(path)
        if data.ndim == 2 and data.shape[1] == 3:
            trajectories.append(data.astype(np.float32))
        elif data.ndim == 3 and data.shape[2] == 3:
            for i in range(data.shape[1]):
                trajectories.append(data[:, i, :].astype(np.float32))
        else:
            raise ValueError(f"Unsupported trajectory shape {data.shape} for {path}")
    if not trajectories:
        raise ValueError("No valid trajectories loaded")
    return trajectories


def compute_light_weights(
    points: np.ndarray,
    trajectories: Sequence[np.ndarray],
    step: int = 1,
    eps: float = 0.1,
    temporal_weight: float = 0.5,
    chunk: int = 50000,
) -> np.ndarray:
    """Compute light-aware weights for points across trajectories.

    Weight is based on mean inverse distance plus a temporal variance term.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if not trajectories:
        return np.ones((points.shape[0],), dtype=np.float32)

    eps2 = float(eps) ** 2
    weights = np.zeros((points.shape[0],), dtype=np.float32)

    for traj in trajectories:
        if traj.ndim != 2 or traj.shape[1] != 3:
            raise ValueError("trajectory must have shape [F, 3]")
        traj = traj[:: max(1, int(step))]
        for i in range(0, points.shape[0], chunk):
            pts = points[i : i + chunk]
            d2 = ((pts[:, None, :] - traj[None, :, :]) ** 2).sum(axis=2)
            inv = 1.0 / (d2 + eps2)
            mean = inv.mean(axis=1)
            var = inv.var(axis=1)
            weights[i : i + chunk] += mean + float(temporal_weight) * var

    return weights


def farthest_point_sampling(points: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """Select k points using farthest point sampling."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    n = points.shape[0]
    if k <= 0:
        raise ValueError("k must be positive")
    if k >= n:
        return points.copy()

    rng = np.random.default_rng(seed)
    start_idx = int(rng.integers(0, n))
    selected = np.empty((k,), dtype=np.int64)
    selected[0] = start_idx
    dist = ((points - points[start_idx]) ** 2).sum(axis=1)

    for i in range(1, k):
        idx = int(np.argmax(dist))
        selected[i] = idx
        dist = np.minimum(dist, ((points - points[idx]) ** 2).sum(axis=1))

    return points[selected]


def sample_with_decluster(
    points: np.ndarray,
    weights: np.ndarray,
    total: int,
    adaptive_ratio: float,
    rng: np.random.Generator,
    decluster: bool = False,
    decluster_factor: float = 1.5,
    decluster_seed: int = 42,
) -> np.ndarray:
    """Sample probes with weighted + uniform mix and optional declustering."""
    if total <= 0:
        raise ValueError("total must be positive")
    candidate_total = total
    if decluster:
        candidate_total = max(total, int(math.ceil(total * float(decluster_factor))))
    candidate_total = min(candidate_total, points.shape[0])

    weighted_n = int(round(candidate_total * float(adaptive_ratio)))
    weighted_n = min(weighted_n, points.shape[0])
    probs = weights / max(weights.sum(), 1e-8)
    weighted_idx = rng.choice(points.shape[0], size=weighted_n, replace=False, p=probs)
    remaining = np.setdiff1d(np.arange(points.shape[0]), weighted_idx, assume_unique=False)
    uniform_n = max(0, candidate_total - weighted_n)
    if remaining.size >= uniform_n:
        uniform_idx = rng.choice(remaining, size=uniform_n, replace=False)
    else:
        uniform_idx = rng.choice(points.shape[0], size=uniform_n, replace=True)
    candidates = np.concatenate([points[weighted_idx], points[uniform_idx]], axis=0).astype(np.float32)

    if decluster and candidates.shape[0] > total:
        return farthest_point_sampling(candidates, total, seed=decluster_seed)

    return candidates[:total]

