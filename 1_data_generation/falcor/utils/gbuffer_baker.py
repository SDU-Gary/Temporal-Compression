"""Reusable Falcor G-Buffer baking utilities.

This module provides a scene-agnostic baking pipeline for offline image-loss
supervision datasets consumed by ``GBufferSupervisionDataset``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

import numpy as np
from tqdm import tqdm

try:
    import torch
except Exception:  # pragma: no cover - keep npz-only bake usable without torch.
    torch = None

try:
    from .falcor_render import build_testbed, get_scene_bounds, load_scene
    from .probe_validity import generate_orbit_camera_poses
except Exception:  # pragma: no cover - direct script import fallback
    from falcor_render import build_testbed, get_scene_bounds, load_scene
    from probe_validity import generate_orbit_camera_poses


LightProvider = Callable[[int, int, int], Any]
FrameHook = Callable[[Any, Any, int, int], None]
SceneHook = Callable[[Any, Any], None]


_GBUFFER_OUTPUT_CANDIDATES: Dict[str, tuple[str, ...]] = {
    "posW": ("GBufferRT.posW",),
    "normW": ("GBufferRT.normW", "GBufferRT.normWRoughnessMaterialID"),
    "albedo": (
        "GBufferRT.diffuseOpacity",
        "GBufferRT.baseColor",
        "GBufferRT.diffuseReflectance",
        "GBufferRT.albedo",
    ),
    "depth": ("GBufferRT.depth",),
    "gt_linear": ("PathTracer.color",),
}


def _graph_create_pass(graph: Any, name: str, pass_type: str, options: Mapping[str, Any]) -> None:
    try:
        if hasattr(graph, "create_pass"):
            graph.create_pass(name, pass_type, dict(options))
            return
        graph.createPass(name, pass_type, dict(options))
    except Exception:
        # Ignore if pass already exists.
        pass


def _graph_add_edge(graph: Any, src: str, dst: str) -> None:
    try:
        if hasattr(graph, "add_edge"):
            graph.add_edge(src, dst)
            return
        graph.addEdge(src, dst)
    except Exception:
        pass


def _graph_mark_output(graph: Any, name: str) -> None:
    try:
        if hasattr(graph, "mark_output"):
            graph.mark_output(name)
            return
        graph.markOutput(name)
    except Exception:
        pass


def _resize_framebuffer(testbed: Any, width: int, height: int) -> None:
    try:
        testbed.resize_frame_buffer(int(width), int(height))
    except Exception:
        try:
            testbed.resizeFrameBuffer(int(width), int(height))
        except Exception:
            pass


def _set_scene_time(testbed: Any, frame_idx: int, fps: float) -> None:
    t_sec = float(frame_idx) / float(max(fps, 1e-6))
    try:
        testbed.clock.time = t_sec
    except Exception:
        try:
            testbed.clock.setTime(t_sec)
        except Exception:
            pass


def _set_camera_pose(scene: Any, falcor_module: Any, pose: Mapping[str, np.ndarray]) -> None:
    pos = np.asarray(pose["position"], dtype=np.float32).reshape(3)
    target = np.asarray(pose["target"], dtype=np.float32).reshape(3)
    up = np.asarray(pose["up"], dtype=np.float32).reshape(3)
    cam = scene.camera
    cam.position = falcor_module.float3(float(pos[0]), float(pos[1]), float(pos[2]))
    cam.target = falcor_module.float3(float(target[0]), float(target[1]), float(target[2]))
    cam.up = falcor_module.float3(float(up[0]), float(up[1]), float(up[2]))


def _get_output(graph: Any, names: Sequence[str]):
    for name in names:
        try:
            tex = graph.get_output(name)
            if tex is not None:
                return tex
        except Exception:
            try:
                tex = graph.getOutput(name)
                if tex is not None:
                    return tex
            except Exception:
                pass
    raise RuntimeError(f"Failed to read graph output from candidates={list(names)}")


def _to_numpy(tex: Any, width: int, height: int) -> np.ndarray:
    if hasattr(tex, "to_numpy"):
        arr = tex.to_numpy()
    else:
        arr = tex.toNumpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        pixel_count = int(width) * int(height)
        if arr.size % pixel_count != 0:
            raise RuntimeError(
                f"Unexpected flat texture size {arr.size} for frame shape {width}x{height}"
            )
        channels = max(1, arr.size // pixel_count)
        arr = arr.reshape(int(height), int(width), channels)
    return arr


def _ensure_hwc3(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC tensor, got shape={arr.shape}")
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr.astype(np.float32, copy=False)


def _build_default_camera_poses(scene: Any, bounds: tuple[np.ndarray, np.ndarray], num_cameras: int) -> list[dict[str, np.ndarray]]:
    cam = scene.camera
    base_pose = {
        "position": np.array([cam.position.x, cam.position.y, cam.position.z], dtype=np.float32),
        "target": np.array([cam.target.x, cam.target.y, cam.target.z], dtype=np.float32),
        "up": np.array([cam.up.x, cam.up.y, cam.up.z], dtype=np.float32),
    }
    return generate_orbit_camera_poses(
        bounds=bounds,
        num_cameras=max(1, int(num_cameras)),
        base_pose=base_pose,
    )


def _canonical_light_payload(
    payload: Any,
    *,
    default_light_params: np.ndarray,
    default_light_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if payload is None:
        return default_light_params, default_light_mask

    light_params: Any = None
    light_mask: Any = None

    if isinstance(payload, Mapping):
        light_params = payload.get("light_params", None)
        light_mask = payload.get("light_mask", None)
    elif isinstance(payload, (tuple, list)):
        if len(payload) >= 1:
            light_params = payload[0]
        if len(payload) >= 2:
            light_mask = payload[1]
    else:
        light_params = payload

    if light_params is None:
        light_params = default_light_params
    if light_mask is None:
        light_mask = default_light_mask

    lp = np.asarray(light_params, dtype=np.float32)
    if lp.ndim == 1:
        lp = lp.reshape(1, -1)
    if lp.ndim != 2:
        raise ValueError(f"light_params must be [S,F] or [F], got shape={lp.shape}")

    lm = np.asarray(light_mask, dtype=np.float32).reshape(-1)
    if lm.shape[0] != lp.shape[0]:
        raise ValueError(
            "light_mask size mismatch: "
            f"light_mask={lm.shape[0]} vs light_params slots={lp.shape[0]}"
        )
    return lp.astype(np.float32, copy=False), lm.astype(np.float32, copy=False)


def _capture_sample(
    graph: Any,
    *,
    width: int,
    height: int,
    include_depth: bool,
    strict_albedo: bool,
) -> Dict[str, np.ndarray]:
    pos_tex = _get_output(graph, _GBUFFER_OUTPUT_CANDIDATES["posW"])
    nrm_tex = _get_output(graph, _GBUFFER_OUTPUT_CANDIDATES["normW"])
    gt_tex = _get_output(graph, _GBUFFER_OUTPUT_CANDIDATES["gt_linear"])

    pos = _ensure_hwc3(_to_numpy(pos_tex, width, height))
    norm = _ensure_hwc3(_to_numpy(nrm_tex, width, height))
    gt = _ensure_hwc3(_to_numpy(gt_tex, width, height))

    albedo = None
    try:
        albedo_tex = _get_output(graph, _GBUFFER_OUTPUT_CANDIDATES["albedo"])
        albedo = _ensure_hwc3(_to_numpy(albedo_tex, width, height))
    except Exception:
        if strict_albedo:
            raise
        albedo = np.ones_like(gt, dtype=np.float32)

    depth = None
    if include_depth:
        try:
            depth_tex = _get_output(graph, _GBUFFER_OUTPUT_CANDIDATES["depth"])
            depth_arr = _to_numpy(depth_tex, width, height)
            if depth_arr.ndim == 3:
                depth_arr = depth_arr[..., 0]
            depth = np.asarray(depth_arr, dtype=np.float32)
        except Exception:
            depth = np.full((height, width), np.nan, dtype=np.float32)

    valid_mask = (
        np.isfinite(pos).all(axis=2)
        & np.isfinite(norm).all(axis=2)
        & np.isfinite(albedo).all(axis=2)
        & np.isfinite(gt).all(axis=2)
    )
    if depth is not None:
        valid_mask = valid_mask & np.isfinite(depth) & (depth > 0.0) & (depth < 1.0)

    payload = {
        "posW": np.asarray(pos, dtype=np.float32),
        "normW": np.asarray(norm, dtype=np.float32),
        "albedo": np.asarray(albedo, dtype=np.float32),
        "gt_linear": np.asarray(gt, dtype=np.float32),
        "valid_mask": valid_mask.astype(np.float32),
    }
    if include_depth and depth is not None:
        payload["depth"] = depth.astype(np.float32)
    return payload


def _save_payload(sample_path: Path, payload: Mapping[str, Any], output_format: str) -> None:
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = str(output_format).strip().lower()
    if fmt == "pt":
        if torch is None:
            raise RuntimeError("output_format='pt' requires torch to be installed in the current Python env.")
        torch.save(dict(payload), sample_path)
        return
    if fmt == "npz":
        np_payload: Dict[str, Any] = {}
        for k, v in payload.items():
            if torch is not None and torch.is_tensor(v):
                np_payload[k] = v.detach().cpu().numpy()
            elif hasattr(v, "detach") and hasattr(v, "cpu") and hasattr(v, "numpy"):
                # Supports tensor-like objects without requiring torch import.
                np_payload[k] = v.detach().cpu().numpy()
            else:
                np_payload[k] = np.asarray(v) if isinstance(v, (list, tuple)) else v
        np.savez_compressed(sample_path, **np_payload)
        return
    raise ValueError(f"Unsupported output_format: {output_format}")


def ensure_gbuffer_graph_outputs(graph: Any) -> None:
    """Attach GBuffer outputs to an existing PathTracer testbed graph."""
    _graph_create_pass(
        graph,
        "GBufferRT",
        "GBufferRT",
        {
            "samplePattern": "Center",
            "sampleCount": 1,
            "useAlphaTest": True,
        },
    )

    # Keep a deterministic VBuffer feeding PathTracer if available.
    _graph_add_edge(graph, "VBufferRT.vbuffer", "PathTracer.vbuffer")
    _graph_add_edge(graph, "VBufferRT.viewW", "PathTracer.viewW")
    _graph_add_edge(graph, "VBufferRT.mvec", "PathTracer.mvec")

    for key in ("posW", "normW", "albedo", "depth", "gt_linear"):
        for name in _GBUFFER_OUTPUT_CANDIDATES[key]:
            _graph_mark_output(graph, name)


def build_light_config_provider_from_npz(
    npz_path: str | Path,
    *,
    light_configs_key: str = "light_configs",
    light_mask_key: str = "light_mask",
    frame_indices_key: str = "frame_indices",
    strict_frame_map: bool = True,
) -> LightProvider:
    """Build a ``frame_idx -> (light_params, light_mask)`` provider from NPZ."""
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"light config npz not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        if light_configs_key not in data.files:
            raise KeyError(f"Missing '{light_configs_key}' in {path}")
        light_configs = np.asarray(data[light_configs_key], dtype=np.float32)
        if light_configs.ndim != 3:
            raise ValueError(
                f"Expected light_configs shape [M,S,F], got {light_configs.shape} from {path}"
            )

        if light_mask_key in data.files:
            light_mask = np.asarray(data[light_mask_key], dtype=np.float32)
        else:
            light_mask = np.ones(light_configs.shape[:2], dtype=np.float32)
        if light_mask.ndim != 2:
            raise ValueError(f"Expected light_mask shape [M,S], got {light_mask.shape} from {path}")

        frame_indices = None
        if frame_indices_key in data.files:
            frame_indices = np.asarray(data[frame_indices_key], dtype=np.int64).reshape(-1)
        elif "metadata" in data.files:
            try:
                md_raw = data["metadata"]
                md_obj = md_raw.item() if isinstance(md_raw, np.ndarray) and md_raw.dtype == object else md_raw
                if isinstance(md_obj, Mapping) and "frame_indices" in md_obj:
                    frame_indices = np.asarray(md_obj["frame_indices"], dtype=np.int64).reshape(-1)
            except Exception:
                frame_indices = None

    num_cfg = int(light_configs.shape[0])
    if frame_indices is not None and frame_indices.shape[0] == num_cfg:
        frame_to_cfg = {int(f): i for i, f in enumerate(frame_indices.tolist())}
    else:
        frame_to_cfg = {i: i for i in range(num_cfg)}

    def provider(frame_idx: int, out_idx: int, _view_idx: int) -> Dict[str, np.ndarray]:
        cfg_idx = frame_to_cfg.get(int(frame_idx), None)
        if cfg_idx is None:
            if strict_frame_map:
                raise KeyError(
                    f"frame_idx={frame_idx} not found in light config map from {path}. "
                    "Use --strict-light-frame-map=false to fallback by output order."
                )
            cfg_idx = int(np.clip(int(out_idx), 0, num_cfg - 1))
        return {
            "light_params": light_configs[int(cfg_idx)],
            "light_mask": light_mask[int(cfg_idx)],
        }

    return provider


def bake_gbuffer_dataset(
    *,
    output_dir: str | Path,
    scene_path: str | Path,
    frame_indices: Sequence[int],
    fps: float = 30.0,
    width: int = 640,
    height: int = 360,
    spp: int = 16,
    accum_frames: int = 1,
    fixed_seed: int | None = 1,
    falcor_python_path: str | None = None,
    camera_mode: str = "scene",
    num_cameras: int = 1,
    camera_poses: Sequence[Mapping[str, np.ndarray]] | None = None,
    light_provider: LightProvider | None = None,
    scene_setup_hook: SceneHook | None = None,
    frame_hook: FrameHook | None = None,
    output_format: str = "pt",
    include_depth: bool = False,
    strict_albedo: bool = True,
    default_light_params: np.ndarray | None = None,
    default_light_mask: np.ndarray | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> Dict[str, Any]:
    """Bake per-frame/view G-Buffer supervision samples.

    Output payload keys are aligned with ``GBufferSupervisionDataset``:
    ``posW``, ``normW``, ``albedo``, ``gt_linear``, ``light_params``,
    ``light_mask``, and ``valid_mask``.
    """
    frame_list = [int(v) for v in frame_indices]
    if not frame_list:
        raise ValueError("frame_indices is empty")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_frame_spp = min(16, max(1, int(spp)))
    auto_accum = int(np.ceil(float(max(1, int(spp))) / float(per_frame_spp)))
    accum = max(int(accum_frames), auto_accum)

    testbed, graph, falcor = build_testbed(
        width=int(width),
        height=int(height),
        spp=int(per_frame_spp),
        falcor_python_path=falcor_python_path,
        fixed_seed=fixed_seed,
        use_russian_roulette=False,
    )
    ensure_gbuffer_graph_outputs(graph)
    scene = load_scene(testbed, str(scene_path))

    if scene_setup_hook is not None:
        scene_setup_hook(scene, testbed)

    bounds = get_scene_bounds(scene)
    if camera_poses is not None and len(camera_poses) > 0:
        cam_poses = list(camera_poses)
    elif str(camera_mode).strip().lower() == "orbit":
        cam_poses = _build_default_camera_poses(scene, bounds, max(1, int(num_cameras)))
    else:
        cam = scene.camera
        cam_poses = [
            {
                "position": np.array([cam.position.x, cam.position.y, cam.position.z], dtype=np.float32),
                "target": np.array([cam.target.x, cam.target.y, cam.target.z], dtype=np.float32),
                "up": np.array([cam.up.x, cam.up.y, cam.up.z], dtype=np.float32),
            }
        ]
    if not cam_poses:
        raise RuntimeError("No camera poses available for baking")

    if default_light_params is None:
        default_light_params = np.zeros((1, 12), dtype=np.float32)
    if default_light_mask is None:
        default_light_mask = np.ones((int(default_light_params.shape[0]),), dtype=np.float32)
    default_light_params = np.asarray(default_light_params, dtype=np.float32)
    default_light_mask = np.asarray(default_light_mask, dtype=np.float32)

    sample_paths: list[str] = []
    frame_iter: Iterable[tuple[int, int]]
    frame_iter = enumerate(frame_list)
    if progress:
        frame_iter = tqdm(frame_iter, total=len(frame_list), desc="Bake GBuffer")

    for out_idx, frame_idx in frame_iter:
        _set_scene_time(testbed, int(frame_idx), float(fps))
        if frame_hook is not None:
            frame_hook(scene, testbed, int(frame_idx), int(out_idx))

        for view_idx, pose in enumerate(cam_poses):
            _set_camera_pose(scene, falcor, pose)
            _resize_framebuffer(testbed, int(width), int(height))
            for _ in range(int(accum)):
                testbed.frame()

            sample = _capture_sample(
                graph,
                width=int(width),
                height=int(height),
                include_depth=bool(include_depth),
                strict_albedo=bool(strict_albedo),
            )
            lp_raw = None
            if light_provider is not None:
                lp_raw = light_provider(int(frame_idx), int(out_idx), int(view_idx))
            light_params, light_mask = _canonical_light_payload(
                lp_raw,
                default_light_params=default_light_params,
                default_light_mask=default_light_mask,
            )

            sample["light_params"] = light_params
            sample["light_mask"] = light_mask
            sample["frame_idx"] = np.int32(frame_idx)
            sample["view_idx"] = np.int32(view_idx)

            ext = "pt" if str(output_format).lower() == "pt" else "npz"
            sample_name = f"sample_f{int(frame_idx):06d}_v{int(view_idx):03d}.{ext}"
            sample_path = out_dir / sample_name
            if sample_path.exists() and (not overwrite):
                raise FileExistsError(
                    f"Sample already exists: {sample_path}. "
                    "Use overwrite=True to replace existing files."
                )
            _save_payload(sample_path, sample, output_format=output_format)
            sample_paths.append(str(sample_path))

    manifest_path = out_dir / "samples_manifest.txt"
    manifest_path.write_text("\n".join(sample_paths) + "\n", encoding="utf-8")

    metadata = {
        "engine": "Falcor",
        "scene": str(Path(scene_path).resolve()),
        "output_dir": str(out_dir.resolve()),
        "output_format": str(output_format),
        "num_samples": int(len(sample_paths)),
        "num_frames": int(len(frame_list)),
        "frame_indices": [int(v) for v in frame_list],
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "spp": int(spp),
        "spp_per_frame": int(per_frame_spp),
        "accum_frames": int(accum),
        "camera_mode": str(camera_mode),
        "num_cameras_per_frame": int(len(cam_poses)),
        "include_depth": bool(include_depth),
        "strict_albedo": bool(strict_albedo),
        "bounds": {
            "min": [float(v) for v in np.asarray(bounds[0]).reshape(3).tolist()],
            "max": [float(v) for v in np.asarray(bounds[1]).reshape(3).tolist()],
        },
        "sample_manifest": str(manifest_path.resolve()),
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata
