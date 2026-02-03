"""Probe field slicer for SH visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from utils.spherical_harmonics import evaluate_sh_basis


LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v) + 1e-8
    return v / norm


def l0_luminance(sh_coeffs: np.ndarray) -> np.ndarray:
    """Compute luminance from L0 (RGB) coefficients."""
    l0_rgb = sh_coeffs[:, [0, 9, 18]]
    return l0_rgb @ LUMA_WEIGHTS


def directional_luminance(sh_coeffs: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Compute luminance along a given direction from SH coefficients."""
    direction = _normalize(direction.astype(np.float32))
    Y = evaluate_sh_basis(direction.reshape(1, 3), max_order=2).astype(np.float32)  # [1, 9]
    coeffs = sh_coeffs.reshape(-1, 3, 9)
    radiance = np.sum(coeffs * Y[None, :, :], axis=2)  # [P, 3]
    return radiance @ LUMA_WEIGHTS


@dataclass
class SliceResult:
    heatmap: np.ndarray
    counts: np.ndarray
    extent: Tuple[float, float, float, float]  # (u_min, u_max, v_min, v_max)
    axis: str
    value: float
    tolerance: float


def slice_probe_field(
    probe_positions: np.ndarray,
    sh_coeffs: np.ndarray,
    axis: str = "y",
    value: float = 0.0,
    tolerance: float = 0.05,
    grid_size: int = 128,
    mode: str = "l0",
    direction: Optional[np.ndarray] = None,
) -> SliceResult:
    """Slice probes on a plane and build a heatmap of SH-derived values."""
    axis = axis.lower()
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be one of: x, y, z")

    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    other_axes = [i for i in range(3) if i != axis_idx]

    mask = np.abs(probe_positions[:, axis_idx] - value) <= tolerance
    if not np.any(mask):
        raise ValueError("No probes found within the slice tolerance.")

    positions_2d = probe_positions[mask][:, other_axes]
    sh_slice = sh_coeffs[mask]

    if mode == "l0":
        values = l0_luminance(sh_slice)
    elif mode == "dir":
        if direction is None:
            raise ValueError("direction is required when mode='dir'")
        values = directional_luminance(sh_slice, direction)
    else:
        raise ValueError("mode must be 'l0' or 'dir'")

    u = positions_2d[:, 0]
    v = positions_2d[:, 1]
    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())

    eps = 1e-8
    u_norm = (u - u_min) / (u_max - u_min + eps)
    v_norm = (v - v_min) / (v_max - v_min + eps)

    H = grid_size
    W = grid_size
    heatmap = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.int32)

    uu = np.clip((u_norm * (W - 1)).astype(np.int32), 0, W - 1)
    vv = np.clip((v_norm * (H - 1)).astype(np.int32), 0, H - 1)

    for idx in range(len(values)):
        heatmap[vv[idx], uu[idx]] += values[idx]
        counts[vv[idx], uu[idx]] += 1

    valid = counts > 0
    heatmap[valid] /= counts[valid]

    return SliceResult(
        heatmap=heatmap,
        counts=counts,
        extent=(u_min, u_max, v_min, v_max),
        axis=axis,
        value=value,
        tolerance=tolerance,
    )
