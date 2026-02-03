"""Utilities to build unified light descriptors from 1D/5D parameters."""

from __future__ import annotations

import torch


def _denormalize(params: torch.Tensor, param_min: torch.Tensor, param_max: torch.Tensor) -> torch.Tensor:
    """Denormalize params from [-1, 1] to original ranges."""
    return 0.5 * (params + 1.0) * (param_max - param_min) + param_min


def _kelvin_to_rgb_torch(kelvin: torch.Tensor) -> torch.Tensor:
    """Convert Kelvin to RGB (torch implementation of Tanner Helland algorithm)."""
    temp = torch.clamp(kelvin, 1000.0, 40000.0) / 100.0

    red = torch.where(
        temp <= 66.0,
        torch.full_like(temp, 255.0),
        329.698727446 * torch.pow(temp - 60.0, -0.1332047592),
    )

    green = torch.where(
        temp <= 66.0,
        99.4708025861 * torch.log(temp) - 161.1195681661,
        288.1221695283 * torch.pow(temp - 60.0, -0.0755148492),
    )

    blue = torch.where(
        temp >= 66.0,
        torch.full_like(temp, 255.0),
        torch.where(
            temp <= 19.0,
            torch.zeros_like(temp),
            138.5177312231 * torch.log(temp - 10.0) - 305.0447927307,
        ),
    )

    rgb = torch.stack(
        [
            torch.clamp(red, 0.0, 255.0),
            torch.clamp(green, 0.0, 255.0),
            torch.clamp(blue, 0.0, 255.0),
        ],
        dim=-1,
    )
    return rgb / 255.0


def _zenith_azimuth_to_dir_torch(zenith_deg: torch.Tensor, azimuth_deg: torch.Tensor) -> torch.Tensor:
    """Convert zenith/azimuth to direction (Y-up)."""
    zenith = torch.deg2rad(zenith_deg)
    azimuth = torch.deg2rad(azimuth_deg)
    x = torch.sin(zenith) * torch.cos(azimuth)
    z = torch.sin(zenith) * torch.sin(azimuth)
    y = torch.cos(zenith)
    dir_vec = torch.stack([x, y, z], dim=-1)
    return dir_vec / (torch.norm(dir_vec, dim=-1, keepdim=True) + 1e-8)


def build_descriptor_5d(
    params_norm: torch.Tensor,
    param_min: torch.Tensor,
    param_max: torch.Tensor,
    light_type: float = 0.0,
) -> torch.Tensor:
    """Build 12D descriptor from normalized 5D params.

    Descriptor layout:
      [Type, RGB(3), Pos(3), Dir(3), Meta1, Meta2]
    """
    raw = _denormalize(params_norm, param_min, param_max)

    zenith = raw[..., 0]
    azimuth = raw[..., 1]
    intensity = raw[..., 2]
    color_temp = raw[..., 3]
    cloud = raw[..., 4]

    rgb = _kelvin_to_rgb_torch(color_temp) * intensity.unsqueeze(-1)
    pos = torch.zeros_like(rgb)
    dir_vec = _zenith_azimuth_to_dir_torch(zenith, azimuth)

    type_val = torch.full_like(zenith.unsqueeze(-1), light_type)
    meta1 = cloud.unsqueeze(-1)
    meta2 = torch.zeros_like(meta1)

    desc = torch.cat([type_val, rgb, pos, dir_vec, meta1, meta2], dim=-1)
    return desc


def build_descriptor_1d(intensity: torch.Tensor, light_type: float = 1.0) -> torch.Tensor:
    """Build 12D descriptor from 1D intensity."""
    if intensity.dim() == 1:
        intensity = intensity.unsqueeze(-1)

    rgb = intensity.repeat(1, 3)
    pos = torch.zeros_like(rgb)
    dir_vec = torch.zeros_like(rgb)
    type_val = torch.full((intensity.shape[0], 1), light_type, device=intensity.device, dtype=intensity.dtype)
    meta1 = torch.zeros_like(type_val)
    meta2 = torch.zeros_like(type_val)
    return torch.cat([type_val, rgb, pos, dir_vec, meta1, meta2], dim=-1)
