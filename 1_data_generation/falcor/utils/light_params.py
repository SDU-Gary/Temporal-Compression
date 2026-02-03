"""Utilities for converting 5D light parameters to Falcor DistantLight settings."""

from __future__ import annotations

import numpy as np

from pathlib import Path
import sys

# Reuse color temperature conversion from existing utils
_UTILS_PATH = Path(__file__).parents[2] / "utils"
if str(_UTILS_PATH) not in sys.path:
    sys.path.insert(0, str(_UTILS_PATH))

from color_temperature import kelvin_to_rgb  # noqa: E402


def zenith_azimuth_to_direction(zenith_deg: float, azimuth_deg: float) -> np.ndarray:
    """Convert zenith/azimuth to a unit direction (Y-up).

    zenith: 0 = +Y (up), 90 = horizon.
    azimuth: 0 = +X, 90 = +Z.
    """
    zenith_rad = np.deg2rad(zenith_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)

    x = np.sin(zenith_rad) * np.cos(azimuth_rad)
    z = np.sin(zenith_rad) * np.sin(azimuth_rad)
    y = np.cos(zenith_rad)

    direction = np.array([x, y, z], dtype=np.float32)
    direction /= np.linalg.norm(direction) + 1e-8
    return direction


def compute_attenuation(zenith_deg: float, cloud_cover: float) -> float:
    """Atmospheric attenuation (Rayleigh + clouds) to match Mitsuba heuristic."""
    zenith_rad = np.deg2rad(zenith_deg)
    air_mass = 1.0 / max(np.cos(zenith_rad), 0.01)

    rayleigh = np.exp(-0.1 * (air_mass - 1.0))
    cloud = 1.0 - 0.7 * cloud_cover

    attenuation = rayleigh * cloud
    return float(np.clip(attenuation, 0.05, 1.0))


def compute_sun_rgb(
    zenith_deg: float,
    intensity: float,
    color_temp: float,
    cloud_cover: float,
    base_radiance: float = 50.0,
) -> np.ndarray:
    """Compute RGB intensity for a DistantLight."""
    rgb = kelvin_to_rgb(color_temp)
    attenuation = compute_attenuation(zenith_deg, cloud_cover)
    return rgb * base_radiance * intensity * attenuation
