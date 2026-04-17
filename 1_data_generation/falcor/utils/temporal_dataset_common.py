"""Shared helpers for Falcor temporal dataset generation scripts."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np
from tqdm import tqdm


def parse_frame_indices(num_frames: int, frame_indices: str, frame_step: int | None) -> List[int]:
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


def write_temporal_metadata(path: str | Path, metadata: dict[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


def resolve_num_samples(sh_mode: str, cube_res: int, num_sh_samples: int) -> int:
    if str(sh_mode) == "cubemap":
        return int(cube_res) * int(cube_res) * 6
    return int(num_sh_samples)


def bake_temporal_tensor(
    *,
    probes: np.ndarray,
    frame_indices: Sequence[int],
    num_lights: int,
    light_descriptor_dim: int,
    render_probe_sh: Callable[[np.ndarray], np.ndarray],
    per_frame_setup: Callable[[int, int], np.ndarray | None] | None = None,
    valid_indices: Iterable[int] | np.ndarray | None = None,
    probe_progress_desc_fn: Callable[[int, int], str] | None = None,
    on_frame_progress: Callable[[int, int, float, float], None] | None = None,
) -> Dict[str, Any]:
    probes_arr = np.asarray(probes, dtype=np.float32)
    if probes_arr.ndim != 2 or probes_arr.shape[1] != 3:
        raise ValueError(f"Invalid probes shape: {probes_arr.shape}")

    frame_list = [int(x) for x in frame_indices]
    p_count = int(probes_arr.shape[0])
    m_count = int(len(frame_list))

    tensor = np.zeros((p_count, m_count, 27), dtype=np.float32)
    light_configs = np.zeros((m_count, int(num_lights), int(light_descriptor_dim)), dtype=np.float32)
    light_mask = np.ones((m_count, int(num_lights)), dtype=np.float32)

    if valid_indices is None:
        probe_iter: List[int] = list(range(p_count))
    else:
        probe_iter = [int(i) for i in list(valid_indices)]
        probe_iter = [idx for idx in probe_iter if 0 <= idx < p_count]

    start_bake = time.time()
    for out_idx, frame_idx in enumerate(frame_list):
        if per_frame_setup is not None:
            descriptors = per_frame_setup(int(frame_idx), int(out_idx))
            if descriptors is not None:
                descriptors_arr = np.asarray(descriptors, dtype=np.float32)
                expected_shape = (int(num_lights), int(light_descriptor_dim))
                if descriptors_arr.shape != expected_shape:
                    raise ValueError(
                        f"per_frame_setup descriptors shape mismatch: got {descriptors_arr.shape}, expected {expected_shape}"
                    )
                light_configs[out_idx, :, :] = descriptors_arr

        if probe_progress_desc_fn is not None:
            desc = str(probe_progress_desc_fn(int(out_idx), int(m_count)))
        else:
            desc = f"Frame {int(out_idx) + 1}/{int(m_count)}"

        for probe_idx in tqdm(probe_iter, desc=desc, leave=False):
            probe = probes_arr[int(probe_idx)]
            sh = np.asarray(render_probe_sh(probe), dtype=np.float32).reshape(-1)
            if sh.shape[0] != 27:
                raise ValueError(f"render_probe_sh must return 27 coefficients, got shape={sh.shape}")
            tensor[int(probe_idx), int(out_idx), :] = sh

        if on_frame_progress is not None:
            elapsed = float(time.time() - start_bake)
            avg_per = elapsed / max(1, int(out_idx) + 1)
            eta = avg_per * (int(m_count) - int(out_idx) - 1)
            on_frame_progress(int(out_idx), int(frame_idx), float(elapsed), float(eta))

    bake_seconds = float(time.time() - start_bake)
    return {
        "tensor": tensor,
        "light_configs": light_configs,
        "light_mask": light_mask,
        "bake_seconds": bake_seconds,
    }


def build_base_temporal_metadata(
    *,
    scene: str,
    num_probes: int,
    num_configs: int,
    num_frames: int,
    fps: float,
    frame_indices: Sequence[int],
    num_lights: int,
    light_descriptor_dim: int,
    sh_mode: str,
    cube_res: int,
    num_sh_samples: int,
    num_samples: int,
    spp: int,
    spp_per_frame: int,
    accum_frames: int,
    radiance_clamp: float,
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
    duration_sec: float,
) -> Dict[str, Any]:
    return {
        "engine": "Falcor",
        "scene": str(Path(scene).resolve()),
        "num_probes": int(num_probes),
        "num_configs": int(num_configs),
        "num_frames": int(num_frames),
        "fps": float(fps),
        "duration_sec": float(duration_sec),
        "frame_indices": [int(i) for i in frame_indices],
        "num_lights": int(num_lights),
        "light_descriptor_dim": int(light_descriptor_dim),
        "sh_mode": str(sh_mode),
        "cube_res": int(cube_res),
        "num_sh_samples": int(num_sh_samples),
        "num_samples": int(num_samples),
        "spp": int(spp),
        "spp_per_frame": int(spp_per_frame),
        "accum_frames": int(accum_frames),
        "radiance_clamp": float(radiance_clamp),
        "bounds": {
            "min": [float(x) for x in list(bounds_min)],
            "max": [float(x) for x in list(bounds_max)],
        },
        "generation_date": datetime.now().isoformat(),
    }


def save_temporal_dataset(
    *,
    output_dir: str | Path,
    tensor: np.ndarray,
    probe_positions: np.ndarray,
    light_configs: np.ndarray,
    light_mask: np.ndarray,
    metadata: Dict[str, Any],
    extra_arrays: Dict[str, Any] | None = None,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "tensor": tensor,
        "probe_positions": probe_positions,
        "light_configs": light_configs,
        "light_mask": light_mask,
        "metadata": metadata,
    }
    if isinstance(extra_arrays, dict):
        payload.update(extra_arrays)
    npz_path = out_dir / "parametric_tensor.npz"
    np.savez_compressed(npz_path, **payload)
    write_temporal_metadata(out_dir / "metadata.json", metadata)
    return npz_path
