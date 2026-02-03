#!/usr/bin/env python3
"""Generate a temporal SH dataset with moving emissive sphere lights in Bistro Exterior."""

from __future__ import annotations

import argparse
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
_UTILS = _ROOT / "1_data_generation" / "utils"
_FALCOR_UTILS = _ROOT / "1_data_generation" / "falcor" / "utils"
_CORE_UTILS = _ROOT / "2_src" / "utils"
import sys
for p in (str(_UTILS), str(_FALCOR_UTILS), str(_CORE_UTILS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from spherical_harmonics import fibonacci_sphere, fit_sh_coefficients  # noqa: E402
from falcor_render import (  # noqa: E402
    build_testbed,
    get_light,
    get_scene_bounds,
    load_scene,
    render_sh_cubemap,
    render_single_pixel,
    set_camera_look,
)
from coords import normalize_pos  # noqa: E402
from scene_io import apply_scene_overrides, load_emissive_objects, load_scene_json  # noqa: E402


def build_descriptor_point(
    rgb: np.ndarray,
    position_norm: np.ndarray,
    dtype=np.float32,
) -> np.ndarray:
    """12D descriptor for point/sphere light."""
    desc = np.zeros(12, dtype=dtype)
    desc[0] = 1.0  # type: point-like
    desc[1:4] = rgb
    desc[4:7] = position_norm
    desc[7:10] = 0.0
    desc[10] = 0.0
    desc[11] = 0.0
    return desc


def generate_probe_grid(bounds: Tuple[np.ndarray, np.ndarray], grid_resolution: int, margin: float) -> np.ndarray:
    min_pt, max_pt = bounds
    size = max_pt - min_pt
    if margin < 1.0:
        margin_vec = size * margin
    else:
        margin_vec = np.array([margin, margin, margin], dtype=np.float32)
    low = min_pt + margin_vec
    high = max_pt - margin_vec

    xs = np.linspace(low[0], high[0], grid_resolution)
    ys = np.linspace(low[1], high[1], grid_resolution)
    zs = np.linspace(low[2], high[2], grid_resolution)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    probes = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
    return probes.astype(np.float32)


def _auto_grid_resolution(max_probes: int) -> int:
    res = int(math.ceil(max_probes ** (1.0 / 3.0)))
    return max(2, res)


def _sample_uniform_positions(
    bounds: Tuple[np.ndarray, np.ndarray],
    count: int,
    margin: float,
    rng: np.random.Generator,
) -> np.ndarray:
    min_pt, max_pt = bounds
    min_pt = min_pt + margin
    max_pt = max_pt - margin
    positions = rng.uniform(min_pt, max_pt, size=(count, 3)).astype(np.float32)
    return positions


def _sample_aabb_surface_positions(
    bounds: Tuple[np.ndarray, np.ndarray],
    count: int,
    margin: float,
    offset_range: Tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    min_pt, max_pt = bounds
    min_pt = min_pt + margin
    max_pt = max_pt - margin
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    faces = rng.integers(0, 6, size=count)
    positions = rng.uniform(min_pt, max_pt, size=(count, 3)).astype(np.float32)
    offsets = rng.uniform(offset_range[0], offset_range[1], size=count).astype(np.float32)
    for i, face in enumerate(faces):
        if face == 0:
            positions[i, 0] = min_pt[0] + offsets[i]
        elif face == 1:
            positions[i, 0] = max_pt[0] - offsets[i]
        elif face == 2:
            positions[i, 1] = min_pt[1] + offsets[i]
        elif face == 3:
            positions[i, 1] = max_pt[1] - offsets[i]
        elif face == 4:
            positions[i, 2] = min_pt[2] + offsets[i]
        else:
            positions[i, 2] = max_pt[2] - offsets[i]
    return positions


def _compute_default_radius(bounds: Tuple[np.ndarray, np.ndarray]) -> float:
    min_pt, max_pt = bounds
    size = max_pt - min_pt
    base = float(min(size[0], size[2])) * 0.02
    return float(max(0.1, min(base, 1.0)))


def _compute_trajectories(
    bounds: Tuple[np.ndarray, np.ndarray],
    num_frames: int,
    fps: float,
    height_base_ratio: float,
    height_amp_ratio: float,
    axis_margin_ratio: float,
    orth_offset_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_pt, max_pt = bounds
    size = max_pt - min_pt
    center = (min_pt + max_pt) * 0.5
    axis = 0 if size[0] >= size[2] else 2
    orth = 2 if axis == 0 else 0
    height = float(size[1])

    y_base = float(min_pt[1]) + height_base_ratio * height
    y_amp = height_amp_ratio * height
    axis_margin = axis_margin_ratio * float(size[axis])
    orth_offset = orth_offset_ratio * float(size[orth])

    start_axis = float(min_pt[axis]) + axis_margin
    end_axis = float(max_pt[axis]) - axis_margin

    t_vals = np.linspace(0.0, (num_frames - 1) / fps, num_frames, dtype=np.float32)
    phase = 2.0 * math.pi * (t_vals / max(t_vals[-1], 1e-6))

    # Red: vertical motion
    red = np.repeat(center[None, :], num_frames, axis=0)
    red[:, 1] = y_base + y_amp * np.sin(phase)
    red[:, orth] -= orth_offset

    # Green: move along street axis (forward-back)
    green = np.repeat(center[None, :], num_frames, axis=0)
    u = 0.5 - 0.5 * np.cos(phase)  # 0 -> 1 -> 0
    green[:, axis] = start_axis + u * (end_axis - start_axis)
    green[:, 1] = y_base + 0.3 * y_amp

    # Blue: static
    blue = np.repeat(center[None, :], num_frames, axis=0)
    blue[:, 1] = y_base + 0.1 * y_amp
    blue[:, orth] += orth_offset

    return red.astype(np.float32), green.astype(np.float32), blue.astype(np.float32)


def _parse_frame_indices(num_frames: int, frame_indices: str, frame_step: int | None) -> List[int]:
    if frame_indices:
        items = []
        for token in frame_indices.split(","):
            token = token.strip()
            if token == "":
                continue
            items.append(int(token))
        indices = sorted(set([i for i in items if 0 <= i < num_frames]))
        return indices
    if frame_step is not None and frame_step > 0:
        return list(range(0, num_frames, frame_step))
    return list(range(num_frames))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate temporal SH dataset with emissive spheres (Bistro Exterior)")
    parser.add_argument("--output", type=str, required=True)
    default_scene = str(_ROOT / "1_data_generation/falcor/scenes/bistro_exterior_emissive_spheres.pyscene")
    parser.add_argument("--scene", type=str, default=default_scene)
    parser.add_argument("--scene-json", type=str, default=None, help="Optional Blender-exported scene.json")
    default_num_frames = 600
    default_fps = 30.0
    parser.add_argument("--num-frames", type=int, default=default_num_frames)
    parser.add_argument("--fps", type=float, default=default_fps)
    parser.add_argument("--frame-indices", type=str, default="")
    parser.add_argument("--frame-step", type=int, default=None)
    parser.add_argument("--grid-resolution", type=int, default=5)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--probe-mode", type=str, default="grid", choices=["grid", "adaptive"])
    parser.add_argument("--probe-uniform-ratio", type=float, default=0.7)
    parser.add_argument("--probe-surface-offset-min", type=float, default=0.05)
    parser.add_argument("--probe-surface-offset-max", type=float, default=0.5)
    parser.add_argument("--max-probes", type=int, default=None)
    parser.add_argument("--probe-file", type=str, default=None, help="Optional .npy probe positions (overrides probe-mode)")
    parser.add_argument("--sh-mode", type=str, default="cubemap", choices=["dir", "cubemap"])
    parser.add_argument("--cube-res", type=int, default=16)
    parser.add_argument("--num-sh-samples", type=int, default=512)
    parser.add_argument("--accum-frames", type=int, default=1)
    parser.add_argument("--spp", type=int, default=128)
    parser.add_argument("--fixed-seed", type=int, default=1)
    parser.add_argument("--radiance-clamp", type=float, default=0.0, help="Clamp radiance before SH projection (0 = disabled)")
    parser.add_argument("--falcor-python-path", type=str, default=None)
    parser.add_argument("--ball-radius", type=float, default=None)
    parser.add_argument("--ball-intensity", type=float, default=20.0)
    parser.add_argument("--height-base-ratio", type=float, default=0.08)
    parser.add_argument("--height-amp-ratio", type=float, default=0.06)
    parser.add_argument("--axis-margin-ratio", type=float, default=0.15)
    parser.add_argument("--orth-offset-ratio", type=float, default=0.12)
    parser.add_argument("--use-env-light", action="store_true")
    parser.add_argument("--use-emissive-lights", action="store_true")
    parser.add_argument("--use-analytic-lights", action="store_true")
    parser.add_argument("--auto-bounds", action="store_true", default=True, help="Auto-compute Bistro bounds for animation")
    parser.add_argument("--no-auto-bounds", action="store_false", dest="auto_bounds")
    args = parser.parse_args()

    scene_cfg = load_scene_json(args.scene_json) if args.scene_json else None
    defaults = {"scene": default_scene, "num_frames": default_num_frames, "fps": default_fps}
    apply_scene_overrides(args, scene_cfg, defaults)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Falcor PathTracer clamps samplesPerPixel to [1, 16]. To reach higher SPP,
    # increase the accumulation frames while keeping per-frame spp <= 16.
    desired_spp = int(args.spp)
    per_frame_spp = min(16, max(1, desired_spp))
    auto_frames = int(math.ceil(float(desired_spp) / float(per_frame_spp)))
    if args.accum_frames < auto_frames:
        args.accum_frames = auto_frames

    testbed, graph, falcor = build_testbed(
        width=1,
        height=1,
        spp=per_frame_spp,
        falcor_python_path=args.falcor_python_path,
        fixed_seed=args.fixed_seed,
        use_russian_roulette=False,
    )
    # Compute bounds from base scene if requested and bounds not provided.
    if args.auto_bounds and not os.environ.get("BISTRO_BOUNDS_MIN"):
        base_scene = os.environ.get(
            "BISTRO_BASE_SCENE",
            str(_ROOT / "1_data_generation" / "scenes" / "Bistro_v5_2" / "BistroExterior.pyscene"),
        )
        try:
            base = load_scene(testbed, base_scene)
            base_bounds = get_scene_bounds(base)
            os.environ["BISTRO_BOUNDS_MIN"] = ",".join([f"{v:.6f}" for v in base_bounds[0].tolist()])
            os.environ["BISTRO_BOUNDS_MAX"] = ",".join([f"{v:.6f}" for v in base_bounds[1].tolist()])
        except Exception:
            pass

    # Sync scene parameters into env for pyscene animation.
    os.environ["BISTRO_FBX"] = os.environ.get("BISTRO_FBX", "")
    if os.environ["BISTRO_FBX"] == "":
        os.environ["BISTRO_FBX"] = str(_ROOT / "1_data_generation" / "scenes" / "Bistro_v5_2" / "BistroExterior.fbx")
    os.environ["BISTRO_NUM_FRAMES"] = str(args.num_frames)
    os.environ["BISTRO_FPS"] = str(args.fps)
    if args.ball_radius is not None:
        os.environ["BISTRO_BALL_RADIUS"] = str(args.ball_radius)
    else:
        # Avoid empty string that breaks float() in pyscene.
        if "BISTRO_BALL_RADIUS" in os.environ and os.environ["BISTRO_BALL_RADIUS"] == "":
            del os.environ["BISTRO_BALL_RADIUS"]
    os.environ["BISTRO_BALL_INTENSITY"] = str(args.ball_intensity)
    os.environ["BISTRO_HEIGHT_BASE_RATIO"] = str(args.height_base_ratio)
    os.environ["BISTRO_HEIGHT_AMP_RATIO"] = str(args.height_amp_ratio)
    os.environ["BISTRO_AXIS_MARGIN_RATIO"] = str(args.axis_margin_ratio)
    os.environ["BISTRO_ORTH_OFFSET_RATIO"] = str(args.orth_offset_ratio)

    scene = load_scene(testbed, args.scene)

    scene.renderSettings.useEnvLight = bool(args.use_env_light)
    scene.renderSettings.useEmissiveLights = True if args.use_emissive_lights or not args.use_analytic_lights else False
    scene.renderSettings.useAnalyticLights = bool(args.use_analytic_lights)
    scene.animated = True
    scene.loopAnimations = False

    bounds = get_scene_bounds(scene)
    radius = float(args.ball_radius) if args.ball_radius is not None else _compute_default_radius(bounds)

    if scene_cfg and isinstance(scene_cfg.get("emissive_objects"), list):
        light_trajs, light_colors, light_radii, args.num_frames = load_emissive_objects(
            scene_cfg,
            fallback_intensity=args.ball_intensity,
            fallback_radius=radius,
            num_frames=args.num_frames,
        )
    else:
        red_traj, green_traj, blue_traj = _compute_trajectories(
            bounds=bounds,
            num_frames=args.num_frames,
            fps=args.fps,
            height_base_ratio=args.height_base_ratio,
            height_amp_ratio=args.height_amp_ratio,
            axis_margin_ratio=args.axis_margin_ratio,
            orth_offset_ratio=args.orth_offset_ratio,
        )
        light_trajs = [red_traj, green_traj, blue_traj]
        light_colors = [
            np.array([args.ball_intensity, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, args.ball_intensity, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, args.ball_intensity], dtype=np.float32),
        ]
        light_radii = [radius, radius, radius]
    spheres = None
    if args.use_analytic_lights:
        spheres = []
        emissive = scene_cfg.get("emissive_objects") if scene_cfg else None
        for idx, color in enumerate(light_colors):
            name = f"Ball{idx}"
            if isinstance(emissive, list) and idx < len(emissive):
                name = emissive[idx].get("name", name)
            light = get_light(scene, name=name)
            spheres.append(light)
            if hasattr(light, "radius"):
                light.radius = float(light_radii[idx])
            light.intensity = falcor.float3(float(color[0]), float(color[1]), float(color[2]))

    frame_indices = _parse_frame_indices(args.num_frames, args.frame_indices, args.frame_step)
    if len(frame_indices) == 0:
        raise ValueError("No frame indices selected.")

    probes = None
    if args.probe_file:
        probes = np.load(args.probe_file).astype(np.float32)
    else:
        rng = np.random.default_rng(42)
        if args.probe_mode == "adaptive":
            if args.max_probes is None:
                raise ValueError("probe_mode=adaptive requires --max-probes to be set.")
            uniform_count = int(round(args.max_probes * float(args.probe_uniform_ratio)))
            surface_count = max(0, int(args.max_probes) - uniform_count)
            uniform_res = _auto_grid_resolution(max(1, uniform_count))
            probes_uniform = generate_probe_grid(bounds, uniform_res, args.margin)
            if probes_uniform.shape[0] > uniform_count:
                order = rng.permutation(probes_uniform.shape[0])[:uniform_count]
                probes_uniform = probes_uniform[order]
            elif probes_uniform.shape[0] < uniform_count:
                needed = uniform_count - probes_uniform.shape[0]
                extra = _sample_uniform_positions(bounds, needed, args.margin, rng)
                probes_uniform = np.concatenate([probes_uniform, extra], axis=0)

            offset_range = (float(args.probe_surface_offset_min), float(args.probe_surface_offset_max))
            probes_surface = _sample_aabb_surface_positions(bounds, surface_count, args.margin, offset_range, rng)
            probes = np.concatenate([probes_uniform, probes_surface], axis=0)
            order = rng.permutation(probes.shape[0])
            probes = probes[order]
        else:
            if args.max_probes is not None:
                auto_res = _auto_grid_resolution(args.max_probes)
                if auto_res != args.grid_resolution:
                    args.grid_resolution = auto_res
            probes = generate_probe_grid(bounds, args.grid_resolution, args.margin)
            if args.max_probes is not None and probes.shape[0] > args.max_probes:
                order = rng.permutation(probes.shape[0])[: args.max_probes]
                probes = probes[order]

    directions = None
    if args.sh_mode == "dir":
        directions = fibonacci_sphere(args.num_sh_samples)

    P = probes.shape[0]
    M = len(frame_indices)
    N = len(light_trajs)

    tensor = np.zeros((P, M, 27), dtype=np.float32)
    light_configs = np.zeros((M, N, 12), dtype=np.float32)
    light_mask = np.ones((M, N), dtype=np.float32)

    def render_probe_sh(probe: np.ndarray) -> np.ndarray:
        if args.sh_mode == "cubemap":
            return render_sh_cubemap(
                testbed,
                graph,
                scene,
                probe,
                cube_res=args.cube_res,
                num_frames=args.accum_frames,
                seed_base=args.fixed_seed,
                radiance_clamp=args.radiance_clamp if args.radiance_clamp > 0.0 else None,
            )
        if directions is None:
            raise RuntimeError("Directional SH sampling selected but directions are not initialized.")
        radiances = []
        for d in directions:
            set_camera_look(scene, probe, d)
            radiances.append(
                render_single_pixel(
                    testbed,
                    graph,
                    num_frames=args.accum_frames,
                    seed_base=args.fixed_seed,
                    radiance_clamp=args.radiance_clamp if args.radiance_clamp > 0.0 else None,
                )
            )
        radiances = np.array(radiances, dtype=np.float32)
        return fit_sh_coefficients(directions, radiances, max_order=2)

    start_time = time.time()
    display_total = len(frame_indices)
    for out_idx, frame_idx in enumerate(frame_indices):
        # Update animation time (seconds).
        t_sec = float(frame_idx) / float(args.fps)
        try:
            testbed.clock.time = t_sec
        except Exception:
            try:
                testbed.clock.setTime(t_sec)
            except Exception:
                pass

        positions = [traj[frame_idx] for traj in light_trajs]

        if spheres is not None:
            for light, pos in zip(spheres, positions):
                light.position = falcor.float3(float(pos[0]), float(pos[1]), float(pos[2]))

        # Descriptors
        for light_idx, (pos, color) in enumerate(zip(positions, light_colors)):
            light_configs[out_idx, light_idx] = build_descriptor_point(
                color,
                normalize_pos(pos, bounds[0], bounds[1]),
            )

        for probe_idx in tqdm(range(P), desc=f"Frame {out_idx + 1}/{display_total}", leave=False):
            probe = probes[probe_idx]
            sh = render_probe_sh(probe)
            tensor[probe_idx, out_idx, :] = sh

        elapsed = time.time() - start_time
        avg_per = elapsed / max(1, out_idx + 1)
        eta = avg_per * (M - out_idx - 1)
        print(
            f"Frame {out_idx + 1}/{display_total} (src {frame_idx + 1}/{args.num_frames}): "
            f"elapsed={avg_per * (out_idx + 1):.1f}s, eta={eta:.1f}s",
            flush=True,
        )

    num_samples = int(args.num_sh_samples)
    if args.sh_mode == "cubemap":
        num_samples = int(args.cube_res) * int(args.cube_res) * 6

    metadata = {
        "engine": "Falcor",
        "scene": str(Path(args.scene).resolve()),
        "scene_json": str(Path(args.scene_json).resolve()) if args.scene_json else None,
        "num_probes": int(P),
        "num_configs": int(M),
        "num_frames": int(args.num_frames),
        "fps": float(args.fps),
        "duration_sec": float((args.num_frames - 1) / args.fps),
        "frame_indices": frame_indices,
        "num_lights": int(N),
        "light_descriptor_dim": 12,
        "sh_mode": args.sh_mode,
        "cube_res": int(args.cube_res),
        "num_sh_samples": int(args.num_sh_samples),
        "num_samples": num_samples,
        "spp": int(desired_spp),
        "spp_per_frame": int(per_frame_spp),
        "accum_frames": int(args.accum_frames),
        "radiance_clamp": float(args.radiance_clamp),
        "bounds": {"min": bounds[0].tolist(), "max": bounds[1].tolist()},
        "probe_mode": args.probe_mode,
        "probe_uniform_ratio": float(args.probe_uniform_ratio),
        "probe_surface_offset_range": [
            float(args.probe_surface_offset_min),
            float(args.probe_surface_offset_max),
        ],
        "probe_file": str(args.probe_file) if args.probe_file else None,
        "ball_radius": float(radius),
        "ball_intensity": float(args.ball_intensity),
        "lighting_mode": "emissive_mesh" if not args.use_analytic_lights else "analytic_lights",
        "trajectory_params": {
            "height_base_ratio": float(args.height_base_ratio),
            "height_amp_ratio": float(args.height_amp_ratio),
            "axis_margin_ratio": float(args.axis_margin_ratio),
            "orth_offset_ratio": float(args.orth_offset_ratio),
        },
        "generation_date": datetime.now().isoformat(),
    }

    np.savez_compressed(
        output_dir / "parametric_tensor.npz",
        tensor=tensor,
        probe_positions=probes,
        light_configs=light_configs,
        light_mask=light_mask,
        metadata=metadata,
    )
    with open(output_dir / "metadata.json", "w") as f:
        import json
        json.dump(metadata, f, indent=2)

    print(f"Saved dataset to: {output_dir}")


if __name__ == "__main__":
    main()
