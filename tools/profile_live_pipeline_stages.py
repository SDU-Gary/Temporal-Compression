#!/usr/bin/env python3
"""Profile live PG-CPL pipeline stages.

This script intentionally runs in the project Python environment, not Falcor's
embedded Python. It profiles the neural/model-side stages directly and can
optionally launch the existing Falcor worker to profile GPU upload, field-build,
shading, and synchronization under the same route.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "2_src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tools.benchmark_realtime_pipeline import (  # noqa: E402
    ModelProvider,
    _default_falcor_python_bin,
    _default_falcor_python_path,
    _normalize_positions,
)
from tools.profiler_common import export_csv, export_json, summarize_latency  # noqa: E402


def _resolve_dataset_npz(dataset: str) -> Path:
    p = Path(dataset)
    if p.is_dir():
        p = p / "parametric_tensor.npz"
    if not p.exists():
        raise FileNotFoundError(f"Dataset npz not found: {p}")
    return p


def _build_frame_schedule(
    num_frames: int,
    *,
    frame_start: int,
    frame_step: int,
    max_frames: Optional[int],
    warmup_frames: int,
    benchmark_frames: int,
) -> Tuple[List[int], int, List[int]]:
    frames = list(range(max(0, int(frame_start)), int(num_frames), max(1, int(frame_step))))
    if max_frames is not None:
        frames = frames[: max(0, int(max_frames))]
    if not frames:
        raise ValueError("No frames selected. Check frame-start/frame-step/max-frames.")
    warmup_n = min(max(0, int(warmup_frames)), max(0, len(frames) - 1))
    bench = frames[warmup_n:]
    if benchmark_frames is not None:
        bench = bench[: max(1, int(benchmark_frames))]
    if not bench:
        raise ValueError("No benchmark frames remain after warmup.")
    all_frames = frames[:warmup_n] + bench
    return all_frames, warmup_n, bench


def _sync_if_cuda(torch_mod: Any, device: Any, enabled: bool) -> None:
    if bool(enabled) and getattr(device, "type", None) == "cuda":
        torch_mod.cuda.synchronize(device)


def _sum_stage_means(summary: Dict[str, Dict[str, float]], names: List[str]) -> float:
    total = 0.0
    for name in names:
        payload = summary.get(name, {})
        value = float(payload.get("mean", 0.0) or 0.0)
        if np.isfinite(value):
            total += value
    return float(total)


def _load_dataset_arrays(dataset_npz: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(dataset_npz, allow_pickle=True)
    tensor = np.asarray(data["tensor"], dtype=np.float32)
    probes = np.asarray(data["probe_positions"], dtype=np.float32)
    light_configs = np.asarray(data["light_configs"], dtype=np.float32)
    light_mask = (
        np.asarray(data["light_mask"], dtype=np.float32)
        if "light_mask" in data
        else np.ones(light_configs.shape[:2], dtype=np.float32)
    )

    if "valid_mask" in data:
        valid_mask = np.asarray(data["valid_mask"], dtype=np.float32).reshape(-1)
        valid_idx = np.where(valid_mask > 0.5)[0]
        if valid_idx.size > 0 and valid_idx.size < probes.shape[0]:
            probes = probes[valid_idx]
            tensor = tensor[valid_idx]

    num_frames = min(int(tensor.shape[1]), int(light_configs.shape[0]))
    tensor = tensor[:, :num_frames, :]
    light_configs = light_configs[:num_frames]
    light_mask = light_mask[:num_frames]
    return tensor, probes, light_configs, light_mask


def profile_model_side(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    import torch

    dataset_npz = _resolve_dataset_npz(args.dataset)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    tensor, probes, light_configs, light_mask = _load_dataset_arrays(dataset_npz)
    num_frames = min(int(tensor.shape[1]), int(light_configs.shape[0]))
    all_frames, warmup_n, bench_frames = _build_frame_schedule(
        num_frames,
        frame_start=args.frame_start,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        warmup_frames=args.warmup_frames,
        benchmark_frames=args.benchmark_frames,
    )

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
        sync_cuda=False,
    )

    rows: List[Dict[str, Any]] = []
    sh_model = np.zeros((len(all_frames), probes.shape[0], 27), dtype=np.float32)
    route_profile = {
        "top_k": int(provider.top_k),
        "training_soft_routing": bool(provider.training_soft_routing),
        "routing_soft_topk": (
            int(provider.routing_soft_topk) if provider.routing_soft_topk is not None else None
        ),
        "routing_temperature": float(provider.routing_temperature),
    }

    P = int(provider.probe_positions_norm.shape[0])
    with torch.no_grad():
        for i, frame in enumerate(all_frames):
            is_bench = i >= warmup_n

            _sync_if_cuda(torch, provider.device, args.sync_cuda)
            t0 = time.perf_counter()
            params = provider.light_configs[frame].unsqueeze(0).expand(P, -1, -1)
            mask = provider.light_mask[frame].unsqueeze(0).expand(P, -1)
            batch_index = torch.arange(P, device=provider.device, dtype=torch.long)
            _ = batch_index
            _sync_if_cuda(torch, provider.device, args.sync_cuda)
            t1 = time.perf_counter()

            routing = None
            t2 = time.perf_counter()
            routing = provider.model.compute_gaussian_routing(
                provider.probe_positions_norm,
                top_k=int(provider.top_k),
                training_soft_routing=bool(provider.training_soft_routing),
                routing_soft_topk=provider.routing_soft_topk,
                routing_temperature=float(provider.routing_temperature),
            )
            _sync_if_cuda(torch, provider.device, args.sync_cuda)
            t3 = time.perf_counter()

            pred = provider.model.forward_with_routing(routing, params, mask)
            if provider.scaler is not None:
                pred = provider.scaler.inverse(pred)
            _sync_if_cuda(torch, provider.device, args.sync_cuda)
            t4 = time.perf_counter()

            pred_np = pred.detach().cpu().numpy().astype(np.float32, copy=False)
            falcor_probe_buffer = np.ascontiguousarray(pred_np.reshape(-1))
            _ = falcor_probe_buffer
            t5 = time.perf_counter()

            sh_model[i] = pred_np

            row = {
                "schedule_index": int(i),
                "frame": int(frame),
                "is_benchmark": int(is_bench),
                "input_prepare_ms": float((t1 - t0) * 1000.0),
                "spatial_routing_ms": float((t3 - t2) * 1000.0),
                "conditional_decode_ms": float((t4 - t3) * 1000.0),
                "probe_field_organize_ms": float((t5 - t4) * 1000.0),
            }
            row["model_side_total_ms"] = float(
                row["input_prepare_ms"]
                + row["spatial_routing_ms"]
                + row["conditional_decode_ms"]
                + row["probe_field_organize_ms"]
            )
            if is_bench:
                rows.append(row)
            if i == 0 or (i + 1) == len(all_frames) or (i + 1) % max(1, len(all_frames) // 10) == 0:
                print(
                    f"[model {i + 1:4d}/{len(all_frames)}] frame={frame:04d} "
                    f"route={row['spatial_routing_ms']:.3f} decode={row['conditional_decode_ms']:.3f} "
                    f"pack={row['probe_field_organize_ms']:.3f} ms"
                )

    csv_path = export_csv(out_dir / "model_stage_frames.csv", rows)

    stage_names = [
        "input_prepare_ms",
        "spatial_routing_ms",
        "conditional_decode_ms",
        "probe_field_organize_ms",
        "model_side_total_ms",
    ]
    stage_summary = {name: summarize_latency([float(r[name]) for r in rows]) for name in stage_names}

    model_sh_npz = out_dir / "model_stage_pred_sh.npz"
    np.savez(
        model_sh_npz,
        frames=np.asarray(all_frames, dtype=np.int64),
        sh_model=sh_model,
        route_profile=np.asarray(json.dumps(route_profile, ensure_ascii=False)),
    )

    return {
        "dataset_npz": str(dataset_npz),
        "checkpoint": str(checkpoint),
        "num_probes": int(probes.shape[0]),
        "num_frames": int(num_frames),
        "all_frames": [int(f) for f in all_frames],
        "warmup_count": int(warmup_n),
        "benchmark_frames": [int(f) for f in bench_frames],
        "routing_profile": route_profile,
        "sync_cuda": bool(args.sync_cuda),
        "csv": str(csv_path),
        "model_sh_npz": str(model_sh_npz),
        "latency_ms": stage_summary,
    }


def run_falcor_worker(args: argparse.Namespace, out_dir: Path, model_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not bool(args.profile_falcor):
        return None
    scene = Path(args.scene)
    if not scene.exists():
        raise FileNotFoundError(f"Scene not found: {scene}")

    worker_script = Path(args.worker_script)
    if not worker_script.exists():
        raise FileNotFoundError(f"Falcor worker script not found: {worker_script}")

    frames_json = out_dir / "falcor_frames.json"
    frames_json.write_text(
        json.dumps(
            {
                "all_frames": model_result["all_frames"],
                "warmup_count": int(model_result["warmup_count"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    worker_out = out_dir / "falcor_worker"
    worker_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.falcor_python_bin),
        str(worker_script),
        "--dataset-npz",
        str(model_result["dataset_npz"]),
        "--scene",
        str(scene),
        "--output-dir",
        str(worker_out),
        "--routes",
        "model",
        "--model-sh-npz",
        str(model_result["model_sh_npz"]),
        "--frames-json",
        str(frames_json),
        "--width",
        str(int(args.width)),
        "--height",
        str(int(args.height)),
        "--fps",
        str(float(args.fps)),
        "--render-scale",
        str(float(args.render_scale)),
        "--field-res",
        str(int(args.field_res)),
        "--field-knn",
        str(int(args.field_knn)),
        "--weight-eps",
        str(float(args.weight_eps)),
        "--grid-chunk",
        str(int(args.grid_chunk)),
        "--field-builder",
        str(args.field_builder),
        "--gbuffer-mode",
        str(args.gbuffer_mode),
        "--gbuffer-pass",
        str(args.gbuffer_pass),
        "--normal-transform",
        str(args.normal_transform),
        "--cosine-mode",
        str(args.cosine_mode),
        "--falcor-python-path",
        str(args.falcor_python_path),
        "--output-scale",
        str(float(args.output_scale)),
        "--sync-every",
        str(max(0, int(args.falcor_sync_every))),
    ]
    cmd.append("--sync-gpu" if bool(args.falcor_sync_gpu) else "--no-sync-gpu")

    print("[falcor] launching worker:")
    print(" ".join(cmd))
    env = os.environ.copy()
    if args.falcor_python_path:
        py_path = Path(args.falcor_python_path)
        cand_dirs = [py_path.parent, py_path.parent.parent, py_path]
        old_ld = env.get("LD_LIBRARY_PATH", "")
        parts = [str(d) for d in cand_dirs if d.exists()] + [p for p in old_ld.split(":") if p]
        seen = set()
        merged: List[str] = []
        for part in parts:
            if part and part not in seen:
                merged.append(part)
                seen.add(part)
        env["LD_LIBRARY_PATH"] = ":".join(merged)
    subprocess.run(cmd, check=True, env=env)

    summary_path = worker_out / "falcor_summary.json"
    frames_path = worker_out / "falcor_frames.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Falcor worker summary missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["summary_path"] = str(summary_path)
    summary["frames_csv"] = str(frames_path) if frames_path.exists() else None
    return summary


def build_combined_summary(model_result: Dict[str, Any], falcor_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    model_lat = model_result["latency_ms"]
    mapped: Dict[str, Any] = {
        "input_preparation": model_lat["input_prepare_ms"],
        "spatial_routing": model_lat["spatial_routing_ms"],
        "conditional_decoding": model_lat["conditional_decode_ms"],
        "probe_field_organization_model_side": model_lat["probe_field_organize_ms"],
    }

    if falcor_result is not None:
        model_route = falcor_result.get("results", {}).get("model", {})
        mapped.update(
            {
                "gbuffer_pass": falcor_result.get("gbuffer", {}),
                "resource_upload": model_route.get("upload_ms", {}),
                "probe_field_build_cpu_or_submit": model_route.get("field_ms", {}),
                "probe_field_build_gpu_submit": model_route.get("field_build_submit_ms", {}),
                "probe_field_build_gpu_sync": model_route.get("field_build_sync_ms", {}),
                "falcor_shade_submit": model_route.get("shade_submit_ms", {}),
                "falcor_shade_sync_or_readback": model_route.get("shade_sync_ms", {}),
                "falcor_shade_total": model_route.get("shade_ms", {}),
            }
        )

    mean_names = [
        "input_prepare_ms",
        "spatial_routing_ms",
        "conditional_decode_ms",
        "probe_field_organize_ms",
    ]
    end_to_end_mean = _sum_stage_means(model_lat, mean_names)
    if falcor_result is not None:
        model_route = falcor_result.get("results", {}).get("model", {})
        for payload in (
            falcor_result.get("gbuffer", {}),
            model_route.get("upload_ms", {}),
            model_route.get("field_ms", {}),
            model_route.get("field_build_submit_ms", {}),
            model_route.get("field_build_sync_ms", {}),
            model_route.get("shade_submit_ms", {}),
            model_route.get("shade_sync_ms", {}),
        ):
            value = float(payload.get("mean", 0.0) or 0.0) if isinstance(payload, dict) else 0.0
            if np.isfinite(value):
                end_to_end_mean += value

    return {
        "model_side": model_result,
        "falcor": falcor_result,
        "requested_stage_mapping_ms": mapped,
        "estimated_end_to_end_mean_ms": float(end_to_end_mean),
        "estimated_end_to_end_fps": float(1000.0 / end_to_end_mean) if end_to_end_mean > 0 else float("nan"),
        "notes": [
            "Model-side timings synchronize CUDA by default for per-stage attribution.",
            "probe_field_organization_model_side includes GPU->CPU pred SH materialization and contiguous Falcor probe-buffer layout.",
            "Falcor resource_upload/field/shade timings come from benchmark_realtime_pipeline_worker.py.",
            "If falcor_sync_gpu is false, Falcor submit timings are enqueue costs; enable --falcor-sync-gpu for blocking readback/sync timing.",
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Profile PG-CPL live pipeline stages")
    p.add_argument("--dataset", required=True, help="Dataset root or parametric_tensor.npz")
    p.add_argument("--checkpoint", required=True, help="Model checkpoint")
    p.add_argument("--output-dir", default="3_experiments/results/demo/pipeline_stage_profile")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--warmup-frames", type=int, default=10)
    p.add_argument("--benchmark-frames", type=int, default=60)
    p.add_argument("--top-k", type=int, default=None)
    soft_group = p.add_mutually_exclusive_group()
    soft_group.add_argument("--training-soft-routing", dest="training_soft_routing", action="store_true")
    soft_group.add_argument("--no-training-soft-routing", dest="training_soft_routing", action="store_false")
    p.set_defaults(training_soft_routing=None)
    p.add_argument("--routing-soft-topk", type=int, default=None)
    p.add_argument("--routing-temperature", type=float, default=None)
    p.add_argument("--routing-profile-json", default="")
    sync_group = p.add_mutually_exclusive_group()
    sync_group.add_argument("--sync-cuda", dest="sync_cuda", action="store_true")
    sync_group.add_argument("--no-sync-cuda", dest="sync_cuda", action="store_false")
    p.set_defaults(sync_cuda=True)

    p.add_argument("--profile-falcor", action="store_true", default=False)
    p.add_argument("--scene", default="", help="Required when --profile-falcor is set")
    p.add_argument("--falcor-python-bin", default=_default_falcor_python_bin())
    p.add_argument("--falcor-python-path", default=_default_falcor_python_path())
    p.add_argument("--worker-script", default=str((ROOT / "tools" / "benchmark_realtime_pipeline_worker.py").resolve()))
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--render-scale", type=float, default=1.0)
    p.add_argument("--field-res", type=int, default=32)
    p.add_argument("--field-knn", type=int, default=8)
    p.add_argument("--weight-eps", type=float, default=0.1)
    p.add_argument("--grid-chunk", type=int, default=4096)
    p.add_argument("--field-builder", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--gbuffer-mode", choices=["realtime", "reuse_first"], default="realtime")
    p.add_argument("--gbuffer-pass", choices=["raster", "rt", "auto"], default="raster")
    p.add_argument("--normal-transform", choices=["identity", "swap_yz", "swap_xz", "swap_xy", "flip_x", "flip_y", "flip_z"], default="identity")
    p.add_argument("--cosine-mode", choices=["irradiance", "radiance"], default="irradiance")
    p.add_argument("--output-scale", type=float, default=1.0)
    falcor_sync = p.add_mutually_exclusive_group()
    falcor_sync.add_argument("--falcor-sync-gpu", dest="falcor_sync_gpu", action="store_true")
    falcor_sync.add_argument("--falcor-no-sync-gpu", dest="falcor_sync_gpu", action="store_false")
    p.set_defaults(falcor_sync_gpu=True)
    p.add_argument("--falcor-sync-every", type=int, default=0)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_result = profile_model_side(args, out_dir)
    falcor_result = run_falcor_worker(args, out_dir, model_result)
    combined = build_combined_summary(model_result, falcor_result)
    summary_path = export_json(out_dir / "pipeline_stage_summary.json", combined)

    print(f"Saved model frame timings: {model_result['csv']}")
    if falcor_result is not None:
        print(f"Saved Falcor timings: {falcor_result['summary_path']}")
    print(f"Saved combined summary: {summary_path}")
    print(
        "Estimated mean end-to-end: "
        f"{combined['estimated_end_to_end_mean_ms']:.3f} ms "
        f"({combined['estimated_end_to_end_fps']:.1f} FPS)"
    )


if __name__ == "__main__":
    main()
