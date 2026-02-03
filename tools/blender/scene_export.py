"""Helpers for exporting Blender-driven lighting setups to Falcor scene JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

DEFAULT_INTENSITY = 20.0
DEFAULT_RADIUS = 0.3


def infer_color_from_name(name: str) -> List[float]:
    """Infer a unit RGB color from object name tokens."""
    lowered = name.lower()
    if "red" in lowered or lowered.endswith("_r") or lowered.startswith("r_"):
        return [1.0, 0.0, 0.0]
    if "green" in lowered or lowered.endswith("_g") or lowered.startswith("g_"):
        return [0.0, 1.0, 0.0]
    if "blue" in lowered or lowered.endswith("_b") or lowered.startswith("b_"):
        return [0.0, 0.0, 1.0]
    return [1.0, 1.0, 1.0]


def ensure_list3(value: Sequence[float], name: str) -> List[float]:
    if len(value) != 3:
        raise ValueError(f"{name} must have 3 elements, got {len(value)}")
    return [float(value[0]), float(value[1]), float(value[2])]


def build_emissive_object(
    name: str,
    trajectory_path: str,
    color: Optional[Sequence[float]] = None,
    radius: Optional[float] = None,
    intensity: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a single emissive object entry for scene.json."""
    if color is None:
        color = infer_color_from_name(name)
    color_list = ensure_list3(color, "color")
    intensity_val = DEFAULT_INTENSITY if intensity is None else float(intensity)
    color_list = [float(c) * intensity_val for c in color_list]
    return {
        "name": name,
        "color": color_list,
        "radius": float(radius) if radius is not None else float(DEFAULT_RADIUS),
        "trajectory": trajectory_path,
    }


def build_scene_payload(
    scene_path: str,
    frames: int,
    fps: float,
    emissive_objects: Iterable[Dict[str, Any]],
    probes_file: Optional[str] = None,
    render: Optional[Dict[str, Any]] = None,
    lighting_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct a scene.json payload for Falcor generators."""
    if frames <= 0:
        raise ValueError("frames must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    payload: Dict[str, Any] = {
        "scene": str(scene_path),
        "frames": int(frames),
        "fps": float(fps),
        "emissive_objects": list(emissive_objects),
    }
    if probes_file:
        payload["probes"] = {"file": str(probes_file)}
    if render:
        payload["render"] = dict(render)
    if lighting_mode:
        payload["lighting_mode"] = str(lighting_mode)
    return payload


def write_scene_json(payload: Dict[str, Any], output_path: str | Path) -> Path:
    """Write scene.json and return the output path."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def write_trajectory_npy(trajectory: np.ndarray, output_path: str | Path) -> Path:
    """Write a trajectory to .npy and return the output path."""
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError("trajectory must have shape [F, 3]")
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, trajectory.astype(np.float32))
    return out_path
