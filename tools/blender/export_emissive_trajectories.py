"""Blender script to export emissive sphere trajectories to Falcor scene.json.

Usage (from CLI):
  blender -b your_scene.blend -P tools/blender/export_emissive_trajectories.py -- \
    --output-dir /path/to/out \
    --scene 1_data_generation/falcor/scenes/bistro_exterior_emissive_spheres.pyscene
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import bpy
except ImportError:
    raise SystemExit("This script must be run inside Blender (bpy not found).")

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.blender.scene_export import build_emissive_object, build_scene_payload, write_scene_json, write_trajectory_npy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export emissive trajectories from Blender")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene", required=True, help="Falcor .pyscene path")
    parser.add_argument("--object-prefix", default="Ball", help="Object name prefix to export")
    parser.add_argument("--objects", default="", help="Comma-separated object names to export")
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--lighting-mode", default="emissive_mesh")
    return parser.parse_args()


def _infer_color(obj_name: str) -> List[float]:
    name = obj_name.lower()
    if "red" in name:
        return [1.0, 0.0, 0.0]
    if "green" in name:
        return [0.0, 1.0, 0.0]
    if "blue" in name:
        return [0.0, 0.0, 1.0]
    return [1.0, 1.0, 1.0]


def _read_custom_prop(obj, key: str, default):
    if key in obj:
        return obj[key]
    return default


def _export_object_trajectory(obj, frame_start: int, frame_end: int) -> np.ndarray:
    positions = []
    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        loc = obj.matrix_world.translation
        positions.append([loc.x, loc.y, loc.z])
    return np.asarray(positions, dtype=np.float32)


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    frame_start = args.frame_start if args.frame_start is not None else scene.frame_start
    frame_end = args.frame_end if args.frame_end is not None else scene.frame_end
    fps = float(args.fps) if args.fps is not None else float(scene.render.fps)
    frames = frame_end - frame_start + 1

    if args.objects:
        names = [n.strip() for n in args.objects.split(",") if n.strip()]
        objects = [bpy.data.objects[n] for n in names if n in bpy.data.objects]
    else:
        objects = [o for o in bpy.data.objects if o.type == "MESH" and o.name.startswith(args.object_prefix)]

    if not objects:
        raise SystemExit("No emissive objects found to export.")

    emissive_entries: List[Dict] = []
    for obj in objects:
        trajectory = _export_object_trajectory(obj, frame_start, frame_end)
        traj_path = out_dir / f"{obj.name}_traj.npy"
        write_trajectory_npy(trajectory, traj_path)

        radius = _read_custom_prop(obj, "emissive_radius", 0.5 * max(obj.dimensions))
        color = _read_custom_prop(obj, "emissive_color", _infer_color(obj.name))
        intensity = _read_custom_prop(obj, "emissive_intensity", None)
        entry = build_emissive_object(
            name=obj.name,
            trajectory_path=traj_path.name,
            color=color,
            radius=radius,
            intensity=intensity,
        )
        emissive_entries.append(entry)

    payload = build_scene_payload(
        scene_path=args.scene,
        frames=frames,
        fps=fps,
        emissive_objects=emissive_entries,
        lighting_mode=args.lighting_mode,
    )

    write_scene_json(payload, out_dir / "scene.json")
    print(f"Saved {len(emissive_entries)} emissive objects to {out_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
