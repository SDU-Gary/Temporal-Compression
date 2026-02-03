#!/usr/bin/env python3
"""Interactive SH lobe visualization using Polyscope."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
_UTILS = _ROOT / "1_data_generation" / "utils"
if str(_UTILS) not in sys.path:
    sys.path.insert(0, str(_UTILS))


def _load_metadata(data_root: Path) -> dict:
    meta_path = data_root / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found at {meta_path}")
    return json.loads(meta_path.read_text())


def _denormalize_pos(pos: np.ndarray, min_pt: np.ndarray, max_pt: np.ndarray) -> np.ndarray:
    return 0.5 * (pos + 1.0) * (max_pt - min_pt) + min_pt


def _load_lamp_positions(data_root: Path, meta: dict) -> np.ndarray:
    if meta.get("lamp_positions"):
        return np.array(meta["lamp_positions"], dtype=np.float32)

    # Fallback: derive from light_configs (normalized positions)
    npz_path = data_root / "parametric_tensor.npz"
    data = np.load(npz_path, allow_pickle=True)
    light_configs = data["light_configs"]  # [M, N, 12]
    bounds = meta.get("bounds")
    if bounds is None:
        raise ValueError("bounds missing from metadata, cannot denormalize positions.")
    min_pt = np.array(bounds["min"], dtype=np.float32)
    max_pt = np.array(bounds["max"], dtype=np.float32)

    # Assume slot 1 is point light, pos stored in indices 4:7
    pos_norm = light_configs[:, 1, 4:7]
    return _denormalize_pos(pos_norm, min_pt, max_pt)


def _aabb_lines(aabb: Tuple[np.ndarray, np.ndarray]):
    mn, mx = aabb
    corners = np.array(
        [
            [mn[0], mn[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mx[1], mx[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mn[2]],
            [mx[0], mx[1], mx[2]],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    ]
    return corners, edges


def _register_aabb_wire(name: str, aabb: Tuple[np.ndarray, np.ndarray], color=(0.8, 0.1, 0.1)) -> None:
    corners, edges = _aabb_lines(aabb)
    nodes = corners
    edges = np.array(edges, dtype=np.int32)
    import polyscope as ps
    ps.register_curve_network(name, nodes, edges, color=color, radius=0.0015)


def _icosphere(subdivisions: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    t = (1.0 + 5.0 ** 0.5) / 2.0
    verts = np.array(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ],
        dtype=np.float32,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int32,
    )

    def midpoint_cache():
        cache: Dict[Tuple[int, int], int] = {}

        def mid(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            if key in cache:
                return cache[key]
            v = (verts[a] + verts[b]) * 0.5
            v = v / (np.linalg.norm(v) + 1e-8)
            cache[key] = len(verts_list)
            verts_list.append(v.astype(np.float32))
            return cache[key]

        return mid

    for _ in range(subdivisions):
        verts_list = [v for v in verts]
        mid = midpoint_cache()
        new_faces = []
        for tri in faces:
            a = mid(tri[0], tri[1])
            b = mid(tri[1], tri[2])
            c = mid(tri[2], tri[0])
            new_faces.extend(
                [
                    [tri[0], a, c],
                    [tri[1], b, a],
                    [tri[2], c, b],
                    [a, b, c],
                ]
            )
        verts = np.array(verts_list, dtype=np.float32)
        faces = np.array(new_faces, dtype=np.int32)

    return verts, faces


def _colormap(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    c0 = np.array([0.231, 0.298, 0.753], dtype=np.float32)  # blue
    c1 = np.array([0.865, 0.865, 0.865], dtype=np.float32)  # light
    c2 = np.array([0.706, 0.016, 0.150], dtype=np.float32)  # red
    mid = t < 0.5
    out = np.zeros((t.shape[0], 3), dtype=np.float32)
    t0 = (t[mid] * 2.0)[:, None]
    t1 = ((t[~mid] - 0.5) * 2.0)[:, None]
    out[mid] = c0 * (1 - t0) + c1 * t0
    out[~mid] = c1 * (1 - t1) + c2 * t1
    return out


def _compute_lobe_vertices(
    base_dirs: np.ndarray,
    Y: np.ndarray,
    sh_coeffs: np.ndarray,
    base_radius: float,
    scale: float,
    vis_mode: str = "linear",
    vis_clip: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    # sh_coeffs: [P, 27]
    cR = sh_coeffs[:, 0:9]
    cG = sh_coeffs[:, 9:18]
    cB = sh_coeffs[:, 18:27]

    # Compute L0 for color (per probe)
    l0 = np.stack([cR[:, 0], cG[:, 0], cB[:, 0]], axis=1)
    l0_luma = l0 @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    # Vectorized evaluation: Y [V,9], cX [P,9] -> [V,P]
    Lr = Y @ cR.T
    Lg = Y @ cG.T
    Lb = Y @ cB.T
    luma = 0.2126 * Lr + 0.7152 * Lg + 0.0722 * Lb
    luma = np.maximum(0.0, luma)

    if vis_mode == "log":
        luma = np.log1p(luma)
        l0_luma = np.log1p(np.maximum(0.0, l0_luma))

    if vis_clip < 1.0:
        clip_val = float(np.quantile(luma, vis_clip))
        if clip_val > 0.0:
            luma = np.minimum(luma, clip_val)
        clip_l0 = float(np.quantile(l0_luma, vis_clip))
        if clip_l0 > 0.0:
            l0_luma = np.minimum(l0_luma, clip_l0)

    r = base_radius + scale * luma
    verts = base_dirs[:, None, :] * r[..., None]
    verts = np.transpose(verts, (1, 0, 2))
    return verts.astype(np.float32), l0_luma


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive SH lobe visualization (Polyscope)")
    parser.add_argument("--data-root", required=True, help="Dataset root containing parametric_tensor.npz")
    parser.add_argument("--config-idx", type=int, default=0, help="Initial config index")
    parser.add_argument("--stride", type=int, default=1, help="Show every Nth probe")
    parser.add_argument("--subdivisions", type=int, default=2, help="Icosphere subdivisions (0-3)")
    parser.add_argument("--base-radius", type=float, default=0.03, help="Base radius for lobes")
    parser.add_argument("--scale", type=float, default=0.08, help="Scale for lobe deformation")
    parser.add_argument("--lamp-radius", type=float, default=0.05, help="Radius for lamp marker sphere")
    parser.add_argument("--vis-mode", type=str, default="linear", choices=["linear", "log"], help="Visualization mode")
    parser.add_argument("--vis-clip", type=float, default=1.0, help="Quantile clip for lobe deformation (e.g., 0.95)")
    parser.add_argument("--title", type=str, default="SH Lobes Viewer", help="Window title")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    meta = _load_metadata(data_root)
    data = np.load(data_root / "parametric_tensor.npz", allow_pickle=True)
    tensor = data["tensor"]  # [P, M, 27]
    probe_positions = data["probe_positions"]  # [P, 3]
    valid_mask = None
    if "valid_mask" in data:
        valid_mask = np.array(data["valid_mask"], dtype=np.float32).reshape(-1)

    P, M, _ = tensor.shape
    config_idx = max(0, min(int(args.config_idx), M - 1))

    lamp_positions = _load_lamp_positions(data_root, meta)

    # Precompute base sphere
    base_dirs, faces = _icosphere(subdivisions=args.subdivisions)
    base_dirs = base_dirs.astype(np.float32)

    # SH basis for base sphere
    from spherical_harmonics import evaluate_sh_basis
    Y = evaluate_sh_basis(base_dirs, max_order=2).astype(np.float32)

    # Subsample probes (respect valid_mask if present)
    stride = max(1, int(args.stride))
    if valid_mask is not None:
        valid_indices = np.where(valid_mask > 0.5)[0]
    else:
        valid_indices = np.arange(P)
    probe_indices = valid_indices[::stride]
    probe_positions = probe_positions[probe_indices]

    # Compute initial lobes
    sh_coeffs = tensor[probe_indices, config_idx, :]
    local_vertices, l0_luma = _compute_lobe_vertices(
        base_dirs,
        Y,
        sh_coeffs,
        args.base_radius,
        args.scale,
        vis_mode=args.vis_mode,
        vis_clip=float(args.vis_clip),
    )

    # Normalize color range
    l0_min = float(l0_luma.min()) if l0_luma.size else 0.0
    l0_max = float(l0_luma.max()) if l0_luma.size else 1.0
    denom = max(1e-6, l0_max - l0_min)
    colors = _colormap((l0_luma - l0_min) / denom)

    try:
        import polyscope as ps
    except Exception as exc:
        raise RuntimeError("polyscope is required. Install via: pip install polyscope") from exc

    ps.init()
    ps.set_program_name(args.title)
    ps.set_up_dir("y_up")

    # Register probe meshes
    probe_meshes: List[str] = []
    for i, pos in enumerate(probe_positions):
        verts = local_vertices[i] + pos[None, :]
        name = f"probe_{i}"
        mesh = ps.register_surface_mesh(name, verts, faces, smooth_shade=True)
        mesh.set_color(colors[i].tolist())
        probe_meshes.append(name)

    # Scene AABB wireframe
    bounds = meta.get("bounds")
    if bounds is not None:
        min_pt = np.array(bounds["min"], dtype=np.float32)
        max_pt = np.array(bounds["max"], dtype=np.float32)
        _register_aabb_wire("Scene AABB", (min_pt, max_pt), color=(0.6, 0.6, 0.6))

    # Obstacle AABBs
    obstacles = meta.get("obstacles", [])
    for idx, obs in enumerate(obstacles):
        mn = np.array(obs["min"], dtype=np.float32)
        mx = np.array(obs["max"], dtype=np.float32)
        _register_aabb_wire(f"Obstacle_{idx}", (mn, mx), color=(1.0, 0.2, 0.2))

    # Lamp marker mesh
    lamp_marker = None
    lamp_pos = lamp_positions[config_idx]
    lamp_verts = base_dirs * float(args.lamp_radius) + lamp_pos[None, :]
    lamp_marker = ps.register_surface_mesh("Lamp", lamp_verts, faces, smooth_shade=True)
    lamp_marker.set_color((1.0, 0.1, 0.1))

    state = {"config_idx": config_idx}

    def update_config(new_idx: int) -> None:
        nonlocal l0_min, l0_max
        sh = tensor[probe_indices, new_idx, :]
        local_v, l0 = _compute_lobe_vertices(
            base_dirs,
            Y,
            sh,
            args.base_radius,
            args.scale,
            vis_mode=args.vis_mode,
            vis_clip=float(args.vis_clip),
        )
        l0_min = float(l0.min()) if l0.size else 0.0
        l0_max = float(l0.max()) if l0.size else 1.0
        denom = max(1e-6, l0_max - l0_min)
        cols = _colormap((l0 - l0_min) / denom)
        for i, name in enumerate(probe_meshes):
            mesh = ps.get_surface_mesh(name)
            mesh.update_vertex_positions(local_v[i] + probe_positions[i][None, :])
            mesh.set_color(cols[i].tolist())

        # Update lamp marker
        lp = lamp_positions[new_idx]
        lamp_mesh = ps.get_surface_mesh("Lamp")
        lamp_mesh.update_vertex_positions(base_dirs * float(args.lamp_radius) + lp[None, :])

    def ui_callback() -> None:
        import polyscope.imgui as imgui
        changed, value = imgui.SliderInt("config_idx", state["config_idx"], 0, M - 1)
        if changed and value != state["config_idx"]:
            state["config_idx"] = int(value)
            update_config(state["config_idx"])

        imgui.Text(f"probes: {probe_positions.shape[0]} / {P}")
        imgui.Text(f"stride: {stride}")
        imgui.Text(f"vis_mode: {args.vis_mode} clip={args.vis_clip:.2f}")

    ps.set_user_callback(ui_callback)
    ps.show()


if __name__ == "__main__":
    main()
