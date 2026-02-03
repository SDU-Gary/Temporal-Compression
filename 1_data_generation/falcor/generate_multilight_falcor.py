#!/usr/bin/env python3
"""Generate multi-light dataset using Falcor with unified 12D light descriptors."""

from __future__ import annotations

import argparse
import os
import time
import math
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
    configure_scene_for_sun,
    get_light,
    get_scene_bounds,
    load_scene,
    render_sh_cubemap,
    render_single_pixel,
    set_camera_look,
    set_mogwai_renderer,
)
from light_params import compute_sun_rgb, zenith_azimuth_to_direction  # noqa: E402
from coords import normalize_pos  # noqa: E402


def build_descriptor_sun(
    rgb: np.ndarray,
    direction: np.ndarray,
    cloud: float,
    dtype=np.float32,
) -> np.ndarray:
    """12D descriptor for distant light."""
    desc = np.zeros(12, dtype=dtype)
    desc[0] = 0.0  # type: distant
    desc[1:4] = rgb
    desc[4:7] = 0.0  # position unused
    desc[7:10] = direction
    desc[10] = cloud
    desc[11] = 0.0
    return desc


def build_descriptor_point(
    rgb: np.ndarray,
    position_norm: np.ndarray,
    dtype=np.float32,
) -> np.ndarray:
    """12D descriptor for point light."""
    desc = np.zeros(12, dtype=dtype)
    desc[0] = 1.0  # type: point
    desc[1:4] = rgb
    desc[4:7] = position_norm
    desc[7:10] = 0.0  # direction unused
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


def _cornell_box_obstacles() -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return AABBs for Cornell box interior obstacles (large + small box)."""
    def aabb_from_box(scale, translation, rot_y):
        # Base cube in [-0.5, 0.5]^3, scaled and rotated around Y.
        corners = np.array(
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5,  0.5],
                [-0.5,  0.5, -0.5],
                [-0.5,  0.5,  0.5],
                [ 0.5, -0.5, -0.5],
                [ 0.5, -0.5,  0.5],
                [ 0.5,  0.5, -0.5],
                [ 0.5,  0.5,  0.5],
            ],
            dtype=np.float32,
        )
        corners *= np.array(scale, dtype=np.float32)
        c, s = float(np.cos(rot_y)), float(np.sin(rot_y))
        rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
        corners = corners @ rot.T
        corners += np.array(translation, dtype=np.float32)
        return corners.min(axis=0), corners.max(axis=0)

    obstacles = []
    # Large box
    obstacles.append(
        aabb_from_box(
            scale=(0.165, 0.33, 0.165),
            translation=(-0.093, 0.165, -0.071),
            rot_y=-1.27,
        )
    )
    # Small box
    obstacles.append(
        aabb_from_box(
            scale=(0.165, 0.165, 0.165),
            translation=(0.09, 0.0825, 0.111),
            rot_y=-0.29,
        )
    )
    return obstacles


def _point_in_aabb(point: np.ndarray, aabb: Tuple[np.ndarray, np.ndarray], margin: float) -> bool:
    mn, mx = aabb
    return np.all(point >= (mn - margin)) and np.all(point <= (mx + margin))


def _push_out_of_aabb(point: np.ndarray, aabb: Tuple[np.ndarray, np.ndarray], margin: float) -> np.ndarray:
    """Move point to nearest outside position of AABB (axis-aligned), with margin."""
    mn, mx = aabb
    p = point.copy()
    # Distances to faces
    d_min = (p - (mn - margin))
    d_max = ((mx + margin) - p)
    # Choose smallest displacement axis
    disp = []
    for axis in range(3):
        if d_min[axis] < d_max[axis]:
            disp.append((d_min[axis], axis, -1))
        else:
            disp.append((d_max[axis], axis, 1))
    disp.sort(key=lambda x: x[0])
    _, axis, sign = disp[0]
    if sign < 0:
        p[axis] = mn[axis] - margin
    else:
        p[axis] = mx[axis] + margin
    return p


def _relocate_probe(
    point: np.ndarray,
    obstacles: List[Tuple[np.ndarray, np.ndarray]],
    bounds: Tuple[np.ndarray, np.ndarray],
    min_dist: float = 0.02,
    max_iters: int = 4,
) -> Tuple[np.ndarray, bool]:
    """Relocate probe if inside obstacle AABB."""
    if not obstacles:
        return point, True
    p = point.copy()
    for _ in range(max_iters):
        inside_any = False
        for aabb in obstacles:
            mn, mx = aabb
            if np.all(p >= mn) and np.all(p <= mx):
                inside_any = True
                dist_min = p - mn
                dist_max = mx - p
                axis = int(np.argmin(np.minimum(dist_min, dist_max)))
                if dist_min[axis] < dist_max[axis]:
                    p[axis] = mn[axis] - min_dist
                else:
                    p[axis] = mx[axis] + min_dist
        # Clamp to scene bounds
        p = np.minimum(np.maximum(p, bounds[0]), bounds[1])
        if not inside_any:
            break
    valid = True
    for aabb in obstacles:
        mn, mx = aabb
        if np.all(p >= mn) and np.all(p <= mx):
            valid = False
            break
    return p, valid


def _relocate_probes(
    probes: np.ndarray,
    obstacles: List[Tuple[np.ndarray, np.ndarray]],
    bounds: Tuple[np.ndarray, np.ndarray],
    min_dist: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    if not obstacles:
        valid = np.ones((probes.shape[0],), dtype=np.float32)
        return probes, valid
    relocated = np.zeros_like(probes, dtype=np.float32)
    valid = np.ones((probes.shape[0],), dtype=np.float32)
    for i, p in enumerate(probes):
        rp, ok = _relocate_probe(p, obstacles, bounds, min_dist=min_dist)
        relocated[i] = rp.astype(np.float32)
        valid[i] = 1.0 if ok else 0.0
    return relocated, valid


def _auto_grid_resolution(max_probes: int) -> int:
    res = int(math.ceil(max_probes ** (1.0 / 3.0)))
    return max(2, res)


def _sample_uniform_positions(
    bounds: Tuple[np.ndarray, np.ndarray],
    count: int,
    obstacles: List[Tuple[np.ndarray, np.ndarray]],
    margin: float,
    rng: np.random.Generator,
) -> np.ndarray:
    min_pt, max_pt = bounds
    min_pt = min_pt + margin
    max_pt = max_pt - margin
    positions = []
    max_attempts = max(200, count * 50)
    attempts = 0
    while len(positions) < count and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(min_pt, max_pt)
        if any(_point_in_aabb(p, aabb, margin) for aabb in obstacles):
            continue
        positions.append(p.astype(np.float32))
    if len(positions) < count:
        # Fallback: accept points without obstacle test.
        needed = count - len(positions)
        extra = rng.uniform(min_pt, max_pt, size=(needed, 3)).astype(np.float32)
        positions.extend(list(extra))
    return np.stack(positions, axis=0)


def _sample_aabb_surface_positions(
    bounds: Tuple[np.ndarray, np.ndarray],
    count: int,
    obstacles: List[Tuple[np.ndarray, np.ndarray]],
    margin: float,
    offset_range: Tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    if not obstacles:
        return _sample_uniform_positions(bounds, count, obstacles, margin, rng)
    min_pt, max_pt = bounds
    min_pt = min_pt + margin
    max_pt = max_pt - margin
    positions = []
    max_attempts = max(200, count * 50)
    attempts = 0
    while len(positions) < count and attempts < max_attempts:
        attempts += 1
        aabb = obstacles[int(rng.integers(0, len(obstacles)))]
        mn, mx = aabb
        axis = int(rng.integers(0, 3))
        side = int(rng.integers(0, 2))
        coord = mn[axis] if side == 0 else mx[axis]
        other_axes = [i for i in range(3) if i != axis]
        p = np.zeros(3, dtype=np.float32)
        p[axis] = coord
        for ax in other_axes:
            p[ax] = rng.uniform(mn[ax], mx[ax])
        normal = -1.0 if side == 0 else 1.0
        offset = rng.uniform(offset_range[0], offset_range[1])
        p[axis] += normal * offset
        # Clamp to scene bounds
        p = np.minimum(np.maximum(p, min_pt), max_pt)
        if any(_point_in_aabb(p, ob, margin) for ob in obstacles):
            continue
        positions.append(p.astype(np.float32))
    if len(positions) < count:
        needed = count - len(positions)
        extra = _sample_uniform_positions(bounds, needed, obstacles, margin, rng)
        positions.extend(list(extra))
    return np.stack(positions, axis=0)


def generate_lamp_positions_mixed(
    bounds: Tuple[np.ndarray, np.ndarray],
    count: int,
    obstacles: List[Tuple[np.ndarray, np.ndarray]],
    uniform_ratio: float = 0.7,
    margin: float = 0.02,
    offset_range: Tuple[float, float] = (0.01, 0.03),
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate lamp positions with 70/30 uniform/surface mix.

    Returns:
        positions: [count, 3]
        types: [count] (0=uniform, 1=surface)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    uniform_count = int(round(count * uniform_ratio))
    surface_count = max(0, count - uniform_count)
    uniform_pts = _sample_uniform_positions(bounds, uniform_count, obstacles, margin, rng)
    surface_pts = _sample_aabb_surface_positions(
        bounds, surface_count, obstacles, margin, offset_range, rng
    )
    positions = np.concatenate([uniform_pts, surface_pts], axis=0)
    types = np.concatenate(
        [
            np.zeros((uniform_pts.shape[0],), dtype=np.int32),
            np.ones((surface_pts.shape[0],), dtype=np.int32),
        ],
        axis=0,
    )
    order = np.arange(positions.shape[0])
    rng.shuffle(order)
    positions = positions[order]
    types = types[order]
    return positions.astype(np.float32), types


def main():
    import __main__
    import builtins
    mogwai_renderer = globals().get("m")
    if mogwai_renderer is None:
        mogwai_renderer = getattr(__main__, "m", None)
    if mogwai_renderer is None:
        mogwai_renderer = getattr(builtins, "m", None)
    is_mogwai = mogwai_renderer is not None
    if is_mogwai:
        set_mogwai_renderer(mogwai_renderer)

    if is_mogwai:
        class Args:
            pass
        args = Args()
        args.output = os.environ.get("FALCOR_OUTPUT", str(_ROOT / "1_data_generation/output/multilight_falcor"))
        args.scene = os.environ.get(
            "FALCOR_SCENE",
            str(_ROOT / "1_data_generation/falcor/scenes/cornell_box_sun_point.pyscene"),
        )
        args.grid_resolution = int(os.environ.get("FALCOR_GRID_RES", "5"))
        args.margin = float(os.environ.get("FALCOR_MARGIN", "0.08"))
        args.sh_mode = os.environ.get("FALCOR_SH_MODE", "dir").lower()
        args.cube_res = int(os.environ.get("FALCOR_CUBE_RES", "32"))
        args.num_sh_samples = int(os.environ.get("FALCOR_NUM_SH_SAMPLES", "512"))
        args.num_frames = int(os.environ.get("FALCOR_NUM_FRAMES", "32"))
        args.spp = int(os.environ.get("FALCOR_SPP", "512"))
        fixed_seed = os.environ.get("FALCOR_FIXED_SEED", "1")
        args.fixed_seed = int(fixed_seed) if fixed_seed != "" else None
        args.use_russian_roulette = False
        args.probe_mode = os.environ.get("FALCOR_PROBE_MODE", "grid")
        args.probe_uniform_ratio = float(os.environ.get("FALCOR_PROBE_UNIFORM_RATIO", "0.7"))
        args.probe_surface_offset_min = float(os.environ.get("FALCOR_PROBE_SURFACE_OFFSET_MIN", "0.05"))
        args.probe_surface_offset_max = float(os.environ.get("FALCOR_PROBE_SURFACE_OFFSET_MAX", "0.5"))
        args.config_mode = os.environ.get("FALCOR_CONFIG_MODE", "mix")
        args.uniform_ratio = float(os.environ.get("FALCOR_UNIFORM_RATIO", "0.7"))
        max_configs = os.environ.get("FALCOR_MAX_CONFIGS", "")
        max_probes = os.environ.get("FALCOR_MAX_PROBES", "")
        args.max_configs = int(max_configs) if max_configs else None
        args.max_probes = int(max_probes) if max_probes else None
        args.falcor_python_path = os.environ.get("FALCOR_PYTHON_PATH")
    else:
        parser = argparse.ArgumentParser(description="Generate multi-light dataset (Falcor)")
        parser.add_argument("--output", type=str, required=True)
        parser.add_argument("--scene", type=str, default=str(_ROOT / "1_data_generation/falcor/scenes/cornell_box_sun_point.pyscene"))
        parser.add_argument("--grid-resolution", type=int, default=5)
        parser.add_argument("--margin", type=float, default=0.08)
        parser.add_argument("--sh-mode", type=str, default="dir", choices=["dir", "cubemap"])
        parser.add_argument("--cube-res", type=int, default=32)
        parser.add_argument("--num-sh-samples", type=int, default=512)
        parser.add_argument("--num-frames", type=int, default=32)
        parser.add_argument("--spp", type=int, default=512)
        parser.add_argument("--max-configs", type=int, default=None)
        parser.add_argument("--max-probes", type=int, default=None)
        parser.add_argument("--fixed-seed", type=int, default=1)
        parser.add_argument("--probe-mode", type=str, default="grid", choices=["grid", "adaptive"])
        parser.add_argument("--probe-uniform-ratio", type=float, default=0.7)
        parser.add_argument("--probe-surface-offset-min", type=float, default=0.05)
        parser.add_argument("--probe-surface-offset-max", type=float, default=0.5)
        parser.add_argument("--config-mode", type=str, default="mix", choices=["grid", "mix"])
        parser.add_argument("--uniform-ratio", type=float, default=0.7)
        parser.add_argument("--falcor-python-path", type=str, default=None)
        args = parser.parse_args()
        args.use_russian_roulette = False
        args.sh_mode = args.sh_mode.lower()

    args.sh_mode = args.sh_mode.lower()
    args.probe_mode = args.probe_mode.lower()
    if args.sh_mode not in ("dir", "cubemap"):
        raise ValueError(f"Unsupported sh_mode: {args.sh_mode}")
    if args.probe_mode not in ("grid", "adaptive"):
        raise ValueError(f"Unsupported probe_mode: {args.probe_mode}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    testbed, graph, falcor = build_testbed(
        width=1,
        height=1,
        spp=args.spp,
        falcor_python_path=args.falcor_python_path,
        fixed_seed=getattr(args, "fixed_seed", None),
        use_russian_roulette=False,
    )
    scene = load_scene(testbed, args.scene)
    configure_scene_for_sun(scene)

    sun = get_light(scene, name="Sun")
    lamp = get_light(scene, name="Lamp")

    bounds = get_scene_bounds(scene)
    scene_name = Path(args.scene).name.lower()
    obstacles: List[Tuple[np.ndarray, np.ndarray]] = []
    if "cornell_box" in scene_name:
        obstacles = _cornell_box_obstacles()

    rng = np.random.default_rng(42)
    probes = None
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
            extra = _sample_uniform_positions(bounds, needed, obstacles, args.margin, rng)
            probes_uniform = np.concatenate([probes_uniform, extra], axis=0)

        offset_range = (float(args.probe_surface_offset_min), float(args.probe_surface_offset_max))
        probes_surface = _sample_aabb_surface_positions(
            bounds, surface_count, obstacles, args.margin, offset_range, rng
        )
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

    probes, valid_mask = _relocate_probes(probes, obstacles, bounds, min_dist=0.02)

    directions = None
    if args.sh_mode == "dir":
        directions = fibonacci_sphere(args.num_sh_samples)

    configs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    lamp_positions = None
    lamp_position_types = None
    if args.config_mode == "mix":
        if args.max_configs is None:
            raise ValueError("config_mode=mix requires --max-configs to be set.")
        rng = np.random.default_rng(42)
        lamp_positions, lamp_position_types = generate_lamp_positions_mixed(
            bounds,
            args.max_configs,
            obstacles,
            uniform_ratio=args.uniform_ratio,
            margin=0.02,
            rng=rng,
        )
        sun_zeniths = rng.uniform(15.0, 75.0, size=args.max_configs)
        sun_azimuths = rng.uniform(0.0, 360.0, size=args.max_configs)
        lamp_intensities = rng.uniform(0.5, 1.2, size=args.max_configs)
        clouds = rng.uniform(0.0, 0.3, size=args.max_configs)
        for i in range(args.max_configs):
            configs.append(
                (
                    np.array([sun_zeniths[i], sun_azimuths[i]], dtype=np.float32),
                    lamp_positions[i],
                    np.array([lamp_intensities[i]], dtype=np.float32),
                    np.array([clouds[i]], dtype=np.float32),
                )
            )
    else:
        # Config lists (simple stratified example)
        sun_zeniths = [15, 45, 75]
        sun_azimuths = [0, 120, 240]
        lamp_positions = [
            np.array([0.0, 0.3, 0.0], dtype=np.float32),
            np.array([0.2, 0.3, 0.1], dtype=np.float32),
            np.array([-0.2, 0.3, -0.1], dtype=np.float32),
        ]
        lamp_intensities = [0.5, 1.0]

        for z in sun_zeniths:
            for a in sun_azimuths:
                for lp in lamp_positions:
                    for li in lamp_intensities:
                        configs.append((np.array([z, a]), lp, np.array([li], dtype=np.float32), np.array([0.0], dtype=np.float32)))

        if args.max_configs is not None:
            configs = configs[: args.max_configs]

    P = probes.shape[0]
    M = len(configs)
    N = 2  # fixed slots: sun + lamp

    def _fmt_time(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        m, s = divmod(int(seconds + 0.5), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    if is_mogwai:
        valid_count = int(valid_mask.sum())
        print(
            "Generation config: "
            f"sh_mode={args.sh_mode}, cube_res={args.cube_res}, "
            f"num_frames={args.num_frames}, num_sh_samples={args.num_sh_samples}, "
            f"probe_mode={args.probe_mode}, grid_res={args.grid_resolution}, "
            f"probes={P} (valid={valid_count}), configs={M}, "
            f"config_mode={args.config_mode}, uniform_ratio={args.uniform_ratio}",
            flush=True,
        )

    tensor = np.zeros((P, M, 27), dtype=np.float32)
    light_configs = np.zeros((M, N, 12), dtype=np.float32)
    light_mask = np.ones((M, N), dtype=np.float32)

    def clamp_lamp_intensity(intensity: float, lamp_pos: np.ndarray) -> float:
        max_irradiance = float(os.environ.get("FALCOR_MAX_LAMP_IRR", "4.0"))
        min_dist = float(os.environ.get("FALCOR_MIN_LAMP_DIST", "0.05"))
        dists = np.linalg.norm(probes - lamp_pos[None, :], axis=1)
        d_min = max(float(dists.min()), min_dist)
        max_intensity = max_irradiance * (d_min * d_min)
        return float(min(intensity, max_intensity))

    if lamp_positions is None:
        lamp_positions = np.array([cfg[1] for cfg in configs], dtype=np.float32)
    if lamp_position_types is None:
        lamp_position_types = np.zeros((len(configs),), dtype=np.int32)

    lamp_intensities_out: List[float] = []

    def render_probe_sh(probe: np.ndarray) -> np.ndarray:
        if args.sh_mode == "cubemap":
            return render_sh_cubemap(
                testbed,
                graph,
                scene,
                probe,
                cube_res=args.cube_res,
                num_frames=args.num_frames,
                seed_base=getattr(args, "fixed_seed", None),
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
                    num_frames=args.num_frames,
                    seed_base=getattr(args, "fixed_seed", None),
                )
            )
        radiances = np.array(radiances, dtype=np.float32)
        return fit_sh_coefficients(directions, radiances, max_order=2)

    start_time = time.time()
    for config_idx, (sun_angles, lamp_pos, lamp_intensity, cloud) in enumerate(configs):
        zenith, azimuth = sun_angles.tolist()
        direction = zenith_azimuth_to_direction(float(zenith), float(azimuth))
        sun_rgb = compute_sun_rgb(float(zenith), 1.0, 5500.0, float(cloud))

        # Set scene lights
        sun.direction = falcor.float3(float(direction[0]), float(direction[1]), float(direction[2]))
        sun.intensity = falcor.float3(float(sun_rgb[0]), float(sun_rgb[1]), float(sun_rgb[2]))

        lamp.position = falcor.float3(float(lamp_pos[0]), float(lamp_pos[1]), float(lamp_pos[2]))
        lamp_i = clamp_lamp_intensity(float(lamp_intensity[0]), lamp_pos)
        lamp_rgb = np.array([lamp_i] * 3, dtype=np.float32)
        lamp.intensity = falcor.float3(float(lamp_rgb[0]), float(lamp_rgb[1]), float(lamp_rgb[2]))
        lamp_intensities_out.append(lamp_i)

        # Descriptor
        sun_desc = build_descriptor_sun(sun_rgb, direction, float(cloud))
        lamp_desc = build_descriptor_point(lamp_rgb, normalize_pos(lamp_pos, bounds[0], bounds[1]))
        light_configs[config_idx, 0] = sun_desc
        light_configs[config_idx, 1] = lamp_desc

        # Render probes
        if is_mogwai:
            print(f"Config {config_idx + 1}/{M}: rendering {P} probes...", flush=True)
            for probe_idx in range(P):
                if (probe_idx + 1) % max(1, P // 4) == 0:
                    print(f"  Probe {probe_idx + 1}/{P}", flush=True)
                probe = probes[probe_idx]
                sh = render_probe_sh(probe)
                tensor[probe_idx, config_idx, :] = sh
            print(f"Config {config_idx + 1}/{M}: done", flush=True)
            elapsed = time.time() - start_time
            avg_per = elapsed / max(1, config_idx + 1)
            eta = avg_per * (M - config_idx - 1)
            print(
                f"Progress: {config_idx + 1}/{M} configs, "
                f"elapsed={_fmt_time(elapsed)}, eta={_fmt_time(eta)}",
                flush=True,
            )
        else:
            for probe_idx in tqdm(range(P), desc=f"Config {config_idx+1}/{M}", leave=False):
                probe = probes[probe_idx]
                sh = render_probe_sh(probe)
                tensor[probe_idx, config_idx, :] = sh

    num_samples = int(args.num_sh_samples)
    if args.sh_mode == "cubemap":
        num_samples = int(args.cube_res) * int(args.cube_res) * 6

    metadata = {
        "engine": "Falcor",
        "scene": str(Path(args.scene).resolve()),
        "num_probes": int(P),
        "num_valid_probes": int(valid_mask.sum()),
        "num_configs": int(M),
        "num_lights": int(N),
        "light_descriptor_dim": 12,
        "sh_mode": args.sh_mode,
        "cube_res": int(args.cube_res),
        "num_frames": int(args.num_frames),
        "num_sh_samples": int(args.num_sh_samples),
        "num_samples": num_samples,
        "spp": int(args.spp),
        "bounds": {"min": bounds[0].tolist(), "max": bounds[1].tolist()},
        "probe_mode": args.probe_mode,
        "probe_uniform_ratio": float(args.probe_uniform_ratio),
        "probe_surface_offset_range": [
            float(args.probe_surface_offset_min),
            float(args.probe_surface_offset_max),
        ],
        "config_mode": args.config_mode,
        "uniform_ratio": float(args.uniform_ratio),
        "lamp_positions": lamp_positions.tolist() if lamp_positions is not None else None,
        "lamp_position_types": lamp_position_types.tolist() if lamp_position_types is not None else None,
        "lamp_intensities": lamp_intensities_out,
        "valid_mask": valid_mask.tolist(),
        "obstacles": [
            {"min": aabb[0].tolist(), "max": aabb[1].tolist()} for aabb in obstacles
        ],
        "max_lamp_irradiance": float(os.environ.get("FALCOR_MAX_LAMP_IRR", "4.0")),
        "min_lamp_distance": float(os.environ.get("FALCOR_MIN_LAMP_DIST", "0.05")),
        "generation_date": datetime.now().isoformat(),
    }

    np.savez_compressed(
        output_dir / "parametric_tensor.npz",
        tensor=tensor,
        probe_positions=probes,
        light_configs=light_configs,
        light_mask=light_mask,
        valid_mask=valid_mask,
        metadata=metadata,
    )
    with open(output_dir / "metadata.json", "w") as f:
        import json
        json.dump(metadata, f, indent=2)

    print(f"Saved dataset to: {output_dir}")

    if is_mogwai:
        try:
            exit()
        except Exception:
            pass


if __name__ == "__main__" or "m" in globals():
    main()
