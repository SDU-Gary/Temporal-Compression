"""Coordinate utilities for consistent SH direction conventions."""

from __future__ import annotations

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def falcor_to_sh(direction: np.ndarray) -> np.ndarray:
    """Convert Falcor direction (Y-up) to SH basis direction.

    Current SH utilities also assume Y-up, so this is identity by default.
    Override here if a different SH basis convention is introduced.
    """
    return normalize(direction)


def sh_to_falcor(direction: np.ndarray) -> np.ndarray:
    """Convert SH basis direction back to Falcor direction."""
    return normalize(direction)


def zenith_azimuth_to_direction(zenith_deg: float, azimuth_deg: float) -> np.ndarray:
    """Convert zenith/azimuth (deg) to direction (Y-up)."""
    zenith_rad = np.deg2rad(zenith_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)
    x = np.sin(zenith_rad) * np.cos(azimuth_rad)
    z = np.sin(zenith_rad) * np.sin(azimuth_rad)
    y = np.cos(zenith_rad)
    return normalize(np.array([x, y, z], dtype=np.float32))


def normalize_pos(pos: np.ndarray, min_pt: np.ndarray, max_pt: np.ndarray) -> np.ndarray:
    """Normalize positions to [-1, 1] given scene bounds."""
    pos = np.asarray(pos, dtype=np.float32)
    min_pt = np.asarray(min_pt, dtype=np.float32)
    max_pt = np.asarray(max_pt, dtype=np.float32)
    return 2.0 * (pos - min_pt) / (max_pt - min_pt + 1e-8) - 1.0
