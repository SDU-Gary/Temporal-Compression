#!/usr/bin/env python3
"""Precompute model-predicted probe SH for the live Falcor comparison demo.

This script runs in the project Python environment because it depends on
PyTorch/model code. The Falcor live demo then consumes the produced NPZ with
Falcor's embedded Python, avoiding a Python ABI/PyTorch mismatch at runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark_realtime_pipeline import (  # noqa: E402
    ModelProvider,
    _normalize_positions,
)


def _resolve_dataset_npz(dataset: str) -> Path:
    p = Path(dataset)
    if p.is_dir():
        p = p / "parametric_tensor.npz"
    if not p.exists():
        raise FileNotFoundError(f"Dataset npz not found: {p}")
    return p


def _parse_frames(num_frames: int, args: argparse.Namespace) -> List[int]:
    start = max(0, int(args.frame_start))
    step = max(1, int(args.frame_step))
    frames = list(range(start, num_frames, step))
    if args.max_frames is not None:
        frames = frames[: max(0, int(args.max_frames))]
    if not frames:
        raise ValueError("No frames selected. Check frame-start/frame-step/max-frames.")
    return frames


def _metadata_to_jsonable(metadata: Any) -> Any:
    if isinstance(metadata, np.ndarray) and metadata.shape == ():
        metadata = metadata.item()
    if isinstance(metadata, dict):
        out: Dict[str, Any] = {}
        for k, v in metadata.items():
            if isinstance(v, np.ndarray):
                out[str(k)] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                out[str(k)] = v.item()
            else:
                out[str(k)] = v
        return out
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Precompute model SH for Falcor live comparison demo")
    p.add_argument("--dataset", required=True, help="Dataset root or parametric_tensor.npz")
    p.add_argument("--checkpoint", required=True, help="Model checkpoint to use for Pred")
    p.add_argument("--output", required=True, help="Output NPZ path")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--top-k", type=int, default=None)
    soft_group = p.add_mutually_exclusive_group()
    soft_group.add_argument("--training-soft-routing", dest="training_soft_routing", action="store_true")
    soft_group.add_argument("--no-training-soft-routing", dest="training_soft_routing", action="store_false")
    p.set_defaults(training_soft_routing=None)
    p.add_argument("--routing-soft-topk", type=int, default=None)
    p.add_argument("--routing-temperature", type=float, default=None)
    p.add_argument(
        "--routing-profile-json",
        default="",
        help="Routing profile JSON; accepts the same format as benchmark_realtime_pipeline.py",
    )
    p.add_argument("--sync-cuda", action="store_true", default=False)
    p.add_argument("--no-compress", action="store_true", default=False)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_npz = _resolve_dataset_npz(args.dataset)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    data = np.load(dataset_npz, allow_pickle=True)
    tensor = np.asarray(data["tensor"], dtype=np.float32)
    probes = np.asarray(data["probe_positions"], dtype=np.float32)
    light_configs = np.asarray(data["light_configs"], dtype=np.float32)
    light_mask = (
        np.asarray(data["light_mask"], dtype=np.float32)
        if "light_mask" in data
        else np.ones(light_configs.shape[:2], dtype=np.float32)
    )
    metadata = _metadata_to_jsonable(data["metadata"]) if "metadata" in data else {}

    if "valid_mask" in data:
        valid_mask = np.asarray(data["valid_mask"], dtype=np.float32).reshape(-1)
        valid_idx = np.where(valid_mask > 0.5)[0]
        if valid_idx.size > 0 and valid_idx.size < probes.shape[0]:
            probes = probes[valid_idx]
            tensor = tensor[valid_idx]

    num_frames = min(int(tensor.shape[1]), int(light_configs.shape[0]))
    light_configs = light_configs[:num_frames]
    light_mask = light_mask[:num_frames]
    frames = _parse_frames(num_frames, args)

    probes_norm, _, _ = _normalize_positions(probes)
    provider = ModelProvider.create(
        checkpoint=checkpoint,
        probe_positions_norm_np=probes_norm,
        light_configs_np=light_configs,
        light_mask_np=light_mask,
        device_str=str(args.device),
        top_k_override=args.top_k,
        training_soft_routing_override=args.training_soft_routing,
        routing_soft_topk_override=args.routing_soft_topk,
        routing_temperature_override=args.routing_temperature,
        routing_profile_json=args.routing_profile_json,
        sync_cuda=bool(args.sync_cuda),
    )

    sh_model = np.zeros((len(frames), probes.shape[0], 27), dtype=np.float32)
    infer_ms = np.zeros((len(frames),), dtype=np.float32)
    t_all = time.perf_counter()
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        sh_model[i] = provider.infer_frame(frame)
        infer_ms[i] = (time.perf_counter() - t0) * 1000.0
        if i == 0 or (i + 1) == len(frames) or (i + 1) % max(1, len(frames) // 20) == 0:
            print(f"[{i + 1:4d}/{len(frames)}] frame={frame:04d} infer={infer_ms[i]:.3f} ms")

    routing_profile = {
        "top_k": int(provider.top_k),
        "training_soft_routing": bool(provider.training_soft_routing),
        "routing_soft_topk": (
            int(provider.routing_soft_topk)
            if provider.routing_soft_topk is not None
            else None
        ),
        "routing_temperature": float(provider.routing_temperature),
    }
    run_meta = {
        "dataset": str(dataset_npz),
        "checkpoint": str(checkpoint),
        "num_frames": int(num_frames),
        "num_selected_frames": int(len(frames)),
        "num_probes": int(probes.shape[0]),
        "routing_profile": routing_profile,
        "dataset_metadata": metadata,
        "elapsed_s": float(time.perf_counter() - t_all),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_fn = np.savez if bool(args.no_compress) else np.savez_compressed
    save_fn(
        out_path,
        frames=np.asarray(frames, dtype=np.int64),
        sh_model=sh_model,
        infer_ms=infer_ms,
        metadata_json=np.asarray(json.dumps(run_meta, ensure_ascii=False)),
    )
    print(f"Saved Pred SH: {out_path}")
    print(
        f"Inference mean={float(np.mean(infer_ms)):.3f} ms, "
        f"p95={float(np.percentile(infer_ms, 95)):.3f} ms"
    )


if __name__ == "__main__":
    main()
