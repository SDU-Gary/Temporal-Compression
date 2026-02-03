"""Utilities for loading Blender-exported scene.json files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _resolve_path(path_str: str | None, base_dir: Path) -> str | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def load_scene_json(path: str | Path) -> Dict[str, Any]:
    """Load a scene.json file and resolve relative paths to absolute."""
    json_path = Path(path)
    with open(json_path, "r") as f:
        cfg: Dict[str, Any] = json.load(f)

    base_dir = json_path.parent
    if "scene" in cfg:
        cfg["scene"] = _resolve_path(cfg["scene"], base_dir)

    probes = cfg.get("probes")
    if isinstance(probes, dict) and "file" in probes:
        probes["file"] = _resolve_path(probes["file"], base_dir)

    emissive = cfg.get("emissive_objects")
    if isinstance(emissive, list):
        for entry in emissive:
            if isinstance(entry, dict) and "trajectory" in entry:
                entry["trajectory"] = _resolve_path(entry["trajectory"], base_dir)

    return cfg


def apply_scene_overrides(args: Any, scene_cfg: Dict[str, Any] | None, defaults: Dict[str, Any]) -> None:
    """Apply scene.json overrides to an argparse-style namespace."""
    if not scene_cfg:
        return
    if scene_cfg.get("scene") and getattr(args, "scene", None) == defaults.get("scene"):
        args.scene = scene_cfg["scene"]
    if "frames" in scene_cfg and getattr(args, "num_frames", None) == defaults.get("num_frames"):
        args.num_frames = int(scene_cfg["frames"])
    if "fps" in scene_cfg and getattr(args, "fps", None) == defaults.get("fps"):
        args.fps = float(scene_cfg["fps"])
    render_cfg = scene_cfg.get("render", {}) if isinstance(scene_cfg, dict) else {}
    if "sh_mode" in render_cfg:
        args.sh_mode = str(render_cfg["sh_mode"])
    if "cube_res" in render_cfg:
        args.cube_res = int(render_cfg["cube_res"])
    if "num_sh_samples" in render_cfg:
        args.num_sh_samples = int(render_cfg["num_sh_samples"])
    if "spp" in render_cfg:
        args.spp = int(render_cfg["spp"])
    if "accum_frames" in render_cfg:
        args.accum_frames = int(render_cfg["accum_frames"])
    probes_cfg = scene_cfg.get("probes") if isinstance(scene_cfg, dict) else None
    if isinstance(probes_cfg, dict) and probes_cfg.get("file") and getattr(args, "probe_file", None) is None:
        args.probe_file = probes_cfg["file"]
    lighting_mode = scene_cfg.get("lighting_mode") if isinstance(scene_cfg, dict) else None
    if lighting_mode == "analytic_lights":
        args.use_analytic_lights = True
        args.use_emissive_lights = False
    elif lighting_mode == "emissive_mesh":
        args.use_emissive_lights = True
        args.use_analytic_lights = False
    if scene_cfg.get("emissive_objects") and not args.use_analytic_lights and not args.use_emissive_lights:
        args.use_emissive_lights = True
        args.use_analytic_lights = False


def load_emissive_objects(
    scene_cfg: Dict[str, Any] | None,
    fallback_intensity: float,
    fallback_radius: float,
    num_frames: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], int]:
    """Load emissive object trajectories and attributes from scene.json.

    Returns:
        light_trajs: list of [F, 3] arrays
        light_colors: list of [3] arrays
        light_radii: list of radii
        num_frames: possibly adjusted to shortest trajectory length
    """
    if not scene_cfg or not isinstance(scene_cfg.get("emissive_objects"), list):
        raise ValueError("scene_cfg missing emissive_objects")
    emissive = scene_cfg["emissive_objects"]
    trajectories: List[np.ndarray] = []
    colors: List[np.ndarray] = []
    radii: List[float] = []
    for entry in emissive:
        traj_path = entry.get("trajectory")
        if not traj_path:
            raise ValueError("emissive_objects entry missing trajectory")
        traj = np.load(traj_path).astype(np.float32)
        if traj.ndim != 2 or traj.shape[1] != 3:
            raise ValueError(f"Invalid trajectory shape for {traj_path}")
        trajectories.append(traj)
        colors.append(np.array(entry.get("color", [fallback_intensity, 0.0, 0.0]), dtype=np.float32))
        radii.append(float(entry.get("radius", fallback_radius)))
    max_len = min(t.shape[0] for t in trajectories)
    if max_len < num_frames:
        num_frames = max_len
    light_trajs = [t[: num_frames] for t in trajectories]
    return light_trajs, colors, radii, num_frames
