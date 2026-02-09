#!/usr/bin/env python3
"""Benchmark real-time probe rendering pipeline with interchangeable SH source.

Goal:
- Keep rendering pipeline identical.
- Compare two SH sources under same scene/GBuffer/compute shader path:
  1) Ground-truth SH from dataset tensor
  2) Model-predicted SH from checkpoint inference

Outputs:
- benchmark_summary.json: aggregate latency/FPS stats
- benchmark_frames.csv: per-frame timing breakdown
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import glob
import importlib.machinery
import importlib.util
import json
import subprocess
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "2_src"
DATA_GEN = ROOT / "1_data_generation"
for p in (SRC, ROOT, DATA_GEN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

FALCOR_BIN_ROOT = ROOT / "Falcor" / "build" / "linux-gcc" / "bin"


def _first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _default_falcor_python_path() -> str:
    candidates = [
        FALCOR_BIN_ROOT / "Release" / "python",
        FALCOR_BIN_ROOT / "Debug" / "python",
    ]
    for c in candidates:
        if c.exists() and list(c.glob("falcor/falcor_ext*.so")):
            return str(c)
    picked = _first_existing(candidates)
    return str(picked if picked is not None else candidates[0])


def _default_falcor_python_bin() -> str:
    candidates = [
        FALCOR_BIN_ROOT / "Release" / "pythondist" / "bin" / "python3.10",
        FALCOR_BIN_ROOT / "Debug" / "pythondist" / "bin" / "python3.10",
    ]
    picked = _first_existing(candidates)
    return str(picked if picked is not None else candidates[0])


def _compute_bounds(metadata: Any, probes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(metadata, dict) and "bounds" in metadata:
        b = metadata["bounds"]
        return np.array(b["min"], dtype=np.float32), np.array(b["max"], dtype=np.float32)
    min_pt = probes.min(axis=0)
    max_pt = probes.max(axis=0)
    pad = 0.05 * (max_pt - min_pt + 1e-6)
    return min_pt - pad, max_pt + pad


def _normalize_positions(probes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pmin = probes.min(axis=0)
    pmax = probes.max(axis=0)
    norm = 2.0 * (probes - pmin[None, :]) / (pmax - pmin + 1e-8)[None, :] - 1.0
    return norm.astype(np.float32), pmin.astype(np.float32), pmax.astype(np.float32)


def _build_probe_field_weights(
    probes: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    grid_res: int,
    k: int,
    chunk: int,
    weight_eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    grid_res = int(grid_res)
    k = int(k)
    lin = np.linspace(0.5 / grid_res, 1.0 - 0.5 / grid_res, grid_res, dtype=np.float32)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    centers = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    centers = bounds_min[None, :] + centers * (bounds_max - bounds_min)[None, :]

    out_idx = np.zeros((centers.shape[0], k), dtype=np.int32)
    out_w = np.zeros((centers.shape[0], k), dtype=np.float32)
    eps2 = float(weight_eps) ** 2

    for i in range(0, centers.shape[0], chunk):
        c = centers[i : i + chunk]
        d2 = ((c[:, None, :] - probes[None, :, :]) ** 2).sum(axis=2)
        k_eff = min(k, d2.shape[1])
        idx = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]
        sel_d2 = np.take_along_axis(d2, idx, axis=1)
        w = 1.0 / (sel_d2 + eps2)
        w = w / (w.sum(axis=1, keepdims=True) + 1e-6)
        out_idx[i : i + chunk, :k_eff] = idx.astype(np.int32)
        out_w[i : i + chunk, :k_eff] = w.astype(np.float32)

    return (
        out_idx.reshape(grid_res, grid_res, grid_res, k),
        out_w.reshape(grid_res, grid_res, grid_res, k),
    )


def _pack_field(field: np.ndarray) -> np.ndarray:
    """Pack SH field into [cells*9,4] with coeff-major RGB layout.

    Input per cell layout: [R0..R8, G0..G8, B0..B8]
    Output per cell layout: 9 float3 entries (R_i,G_i,B_i), padded to float4.
    """
    cells = field.reshape(-1, 27).astype(np.float32)
    packed = np.zeros((cells.shape[0] * 9, 4), dtype=np.float32)
    for i in range(9):
        packed[i::9, 0] = cells[:, i]
        packed[i::9, 1] = cells[:, 9 + i]
        packed[i::9, 2] = cells[:, 18 + i]
    return packed


def _tone_map_reinhard(image: np.ndarray) -> np.ndarray:
    x = np.maximum(image, 0.0)
    return x / (1.0 + x)


def _srgb_encode(image: np.ndarray) -> np.ndarray:
    img = np.maximum(image, 0.0)
    threshold = 0.0031308
    low = 12.92 * img
    high = 1.055 * np.power(img, 1.0 / 2.4) - 0.055
    return np.where(img <= threshold, low, high)


def _summarize_ms(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "count": 0,
        }
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "count": int(arr.size),
    }


def _fps_from_ms(ms_mean: float) -> float:
    if not np.isfinite(ms_mean) or ms_mean <= 0.0:
        return float("nan")
    return float(1000.0 / ms_mean)


@dataclass
class ModelProvider:
    model: Any
    top_k: int
    probe_positions_norm: Any
    light_configs: Any
    light_mask: Any
    scaler: Optional[Any]
    device: Any
    sync_cuda: bool

    @classmethod
    def create(
        cls,
        checkpoint: Path,
        probe_positions_norm_np: np.ndarray,
        light_configs_np: np.ndarray,
        light_mask_np: np.ndarray,
        device_str: str,
        top_k_override: Optional[int],
        sync_cuda: bool,
    ) -> "ModelProvider":
        import torch
        from data.sh_scaler import AdaptiveSHScaler
        from models.gaussian_physics_unified import GaussianPhysicsCompressionUnified

        if device_str == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")

        device = torch.device(device_str)
        ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        meta = ckpt.get("meta") if isinstance(ckpt, dict) else None

        K = int(state["mu"].shape[0])
        rank = int(state["U"].shape[-1])
        embed_dim = int(state["coeffs"].shape[-1])
        sh_dim = int(state["U"].shape[1])

        model_meta = meta.get("model", {}) if isinstance(meta, dict) else {}
        top_k = int(model_meta.get("top_k", 3))
        if top_k_override is not None:
            top_k = int(top_k_override)

        light_dim = int(model_meta.get("light_dim", 12))
        intensity_dim = int(model_meta.get("intensity_dim", 3))
        intensity_offset = int(model_meta.get("intensity_offset", 1))
        enable_film = bool(model_meta.get("enable_film", True))

        model = GaussianPhysicsCompressionUnified(
            num_gaussians=K,
            rank=rank,
            sh_dim=sh_dim,
            light_dim=light_dim,
            embed_dim=embed_dim,
            intensity_dim=intensity_dim,
            intensity_offset=intensity_offset,
            enable_film=enable_film,
        ).to(device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint state mismatch. missing={missing}, unexpected={unexpected}"
            )
        model.eval()

        scaler = None
        scaler_meta = meta.get("sh_scaler") if isinstance(meta, dict) else None
        if isinstance(scaler_meta, dict):
            scaler = AdaptiveSHScaler(
                l0_mean=np.array(scaler_meta.get("l0_mean", [0.0, 0.0, 0.0]), dtype=np.float32),
                l0_std=np.array(scaler_meta.get("l0_std", [1.0, 1.0, 1.0]), dtype=np.float32),
                ho_rms=np.array(scaler_meta.get("ho_rms", [1.0, 1.0, 1.0]), dtype=np.float32),
                eps=float(scaler_meta.get("eps", 1e-6)),
            )

        probe_positions_norm = torch.from_numpy(probe_positions_norm_np).to(device=device, dtype=torch.float32)
        light_configs = torch.from_numpy(light_configs_np).to(device=device, dtype=torch.float32)
        light_mask = torch.from_numpy(light_mask_np).to(device=device, dtype=torch.float32)

        return cls(
            model=model,
            top_k=top_k,
            probe_positions_norm=probe_positions_norm,
            light_configs=light_configs,
            light_mask=light_mask,
            scaler=scaler,
            device=device,
            sync_cuda=sync_cuda,
        )

    def infer_frame(self, frame_idx: int) -> np.ndarray:
        import torch

        P = int(self.probe_positions_norm.shape[0])
        params = self.light_configs[frame_idx].unsqueeze(0).expand(P, -1, -1)
        mask = self.light_mask[frame_idx].unsqueeze(0).expand(P, -1)

        if self.sync_cuda and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        with torch.no_grad():
            pred = self.model(self.probe_positions_norm, params, top_k=self.top_k, light_mask=mask)
            if self.scaler is not None:
                pred = self.scaler.inverse(pred)

        if self.sync_cuda and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        return pred.detach().cpu().numpy().astype(np.float32)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark GT vs Model SH realtime rendering pipeline")
    parser.add_argument("--dataset", required=True, help="Dataset root or path to parametric_tensor.npz")
    parser.add_argument("--scene", required=True, help="Falcor .pyscene path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=None, help="Required for --route model/both")
    parser.add_argument("--route", choices=["gt", "model", "both"], default="both")

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--render-scale", type=float, default=1.0)

    parser.add_argument("--field-res", type=int, default=32)
    parser.add_argument("--field-knn", type=int, default=8)
    parser.add_argument("--weight-eps", type=float, default=0.1)
    parser.add_argument("--grid-chunk", type=int, default=4096)
    parser.add_argument("--field-builder", choices=["cpu", "gpu"], default="cpu",
                        help="Field construction backend (commit1 metadata only; logic unchanged)")
    parser.add_argument("--gbuffer-mode", choices=["realtime", "reuse_first"], default="realtime",
                        help="GBuffer mode (commit1 metadata only; logic unchanged)")

    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--benchmark-frames", type=int, default=300)

    parser.add_argument("--falcor-python-path", default=_default_falcor_python_path())
    parser.add_argument("--split-runtime", action="store_true", default=True,
                        help="Use split runtime: model inference in current Python, Falcor render in external Python")
    parser.add_argument("--no-split-runtime", action="store_false", dest="split_runtime")
    parser.add_argument("--falcor-python-bin",
                        default=_default_falcor_python_bin(),
                        help="Python binary used to run Falcor worker in split-runtime mode")
    parser.add_argument("--worker-script",
                        default=str((ROOT / "tools" / "benchmark_realtime_pipeline_worker.py").resolve()),
                        help="Path to Falcor worker script (Python 3.10)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--top-k", type=int, default=None, help="Override model top_k from checkpoint meta")

    parser.add_argument("--sync-gpu", action="store_true", default=False,
                        help="Force per-frame GPU synchronization/readback (slow, debugging only)")
    parser.add_argument("--no-sync-gpu", action="store_false", dest="sync_gpu")
    parser.add_argument("--sync-every", type=int, default=0,
                        help="If >0, synchronize/readback every N benchmark frames (sampling mode)")

    parser.add_argument("--compute-image-metrics", action="store_true", default=False,
                        help="Compute PSNR/SSIM between GT route and Model route rendered images")
    parser.add_argument("--save-frame-metrics-every", type=int, default=30,
                        help="Only compute image metrics every N benchmark frames")
    return parser


def _resolve_dataset_npz(dataset_arg: str) -> Path:
    p = Path(dataset_arg)
    if p.is_dir():
        p = p / "parametric_tensor.npz"
    if not p.exists():
        raise FileNotFoundError(f"Dataset npz not found: {p}")
    return p


def _build_frame_schedule(num_frames: int, args: argparse.Namespace) -> Tuple[List[int], int, List[int]]:
    frame_indices = list(range(max(0, int(args.frame_start)), num_frames, max(1, int(args.frame_step))))
    if args.max_frames is not None:
        frame_indices = frame_indices[: max(0, int(args.max_frames))]
    if not frame_indices:
        raise ValueError("No frames selected. Check frame_start/frame_step/max_frames")

    warmup_n = min(max(0, int(args.warmup_frames)), len(frame_indices) - 1)
    bench_frames = frame_indices[warmup_n:]
    if args.benchmark_frames is not None:
        bench_frames = bench_frames[: max(1, int(args.benchmark_frames))]
    if not bench_frames:
        raise ValueError("No benchmark frames after warmup")

    all_frames = frame_indices[:warmup_n] + bench_frames
    return all_frames, warmup_n, bench_frames


def _run_split_runtime(
    args: argparse.Namespace,
    out_dir: Path,
    npz_path: Path,
    scene_path: Path,
    routes: List[str],
) -> None:
    data = np.load(npz_path, allow_pickle=True)
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

    num_probes, num_frames_tensor, sh_dim = tensor.shape
    if sh_dim != 27:
        raise ValueError(f"Expected SH dim=27, got {sh_dim}")

    num_frames_light = int(light_configs.shape[0])
    num_frames = min(int(num_frames_tensor), num_frames_light)
    if num_frames < num_frames_tensor:
        tensor = tensor[:, :num_frames, :]
    if num_frames < num_frames_light:
        light_configs = light_configs[:num_frames]
        light_mask = light_mask[:num_frames]

    probe_positions_norm, _, _ = _normalize_positions(probes)
    all_frames, warmup_n, bench_frames = _build_frame_schedule(num_frames, args)

    split_dir = out_dir / "_split_runtime"
    split_dir.mkdir(parents=True, exist_ok=True)

    model_sh_npz: Optional[Path] = None
    infer_ms_full: Optional[np.ndarray] = None

    if "model" in routes:
        model_provider = ModelProvider.create(
            checkpoint=Path(args.checkpoint),
            probe_positions_norm_np=probe_positions_norm,
            light_configs_np=light_configs,
            light_mask_np=light_mask,
            device_str=args.device,
            top_k_override=args.top_k,
            sync_cuda=args.sync_gpu,
        )

        sh_model = np.zeros((len(all_frames), num_probes, 27), dtype=np.float32)
        infer_ms = np.zeros((len(all_frames),), dtype=np.float64)
        for i, frame in enumerate(all_frames):
            t0 = time.perf_counter()
            sh_model[i] = model_provider.infer_frame(frame)
            infer_ms[i] = (time.perf_counter() - t0) * 1000.0
            if i % max(1, len(all_frames) // 20) == 0:
                print(f"[model {i+1}/{len(all_frames)}] frame={frame}")

        model_sh_npz = split_dir / "model_sh.npz"
        np.savez(
            model_sh_npz,
            frames=np.asarray(all_frames, dtype=np.int64),
            sh_model=sh_model,
            infer_ms=infer_ms,
        )
        infer_ms_full = infer_ms

    frames_json = split_dir / "frames.json"
    with open(frames_json, "w", encoding="utf-8") as f:
        json.dump({"all_frames": all_frames, "warmup_count": warmup_n}, f)

    worker_out = split_dir / "falcor_worker"
    worker_out.mkdir(parents=True, exist_ok=True)

    worker_script = Path(args.worker_script)
    if not worker_script.exists():
        raise FileNotFoundError(f"Falcor worker script not found: {worker_script}")

    cmd = [
        str(args.falcor_python_bin),
        str(worker_script),
        "--dataset-npz", str(npz_path),
        "--scene", str(scene_path),
        "--output-dir", str(worker_out),
        "--routes", ",".join(routes),
        "--frames-json", str(frames_json),
        "--width", str(args.width),
        "--height", str(args.height),
        "--fps", str(args.fps),
        "--render-scale", str(args.render_scale),
        "--field-res", str(args.field_res),
        "--field-knn", str(args.field_knn),
        "--weight-eps", str(args.weight_eps),
        "--grid-chunk", str(args.grid_chunk),
        "--field-builder", str(args.field_builder),
        "--gbuffer-mode", str(args.gbuffer_mode),
        "--falcor-python-path", str(args.falcor_python_path),
    ]
    if model_sh_npz is not None:
        cmd.extend(["--model-sh-npz", str(model_sh_npz)])
    cmd.append("--sync-gpu" if args.sync_gpu else "--no-sync-gpu")
    cmd.extend(["--sync-every", str(max(0, int(args.sync_every)))])

    print("[split-runtime] launching Falcor worker:")
    print(" ".join(cmd))

    # Ensure Falcor shared libs are visible to the worker interpreter.
    worker_env = os.environ.copy()
    if args.falcor_python_path:
        py_path = Path(args.falcor_python_path)
        cand_dirs = [py_path.parent, py_path.parent.parent, py_path]
        ld_old = worker_env.get("LD_LIBRARY_PATH", "")
        ld_parts = [str(d) for d in cand_dirs if d.exists()] + [p for p in ld_old.split(":") if p]
        seen = set()
        ld_merged = []
        for p in ld_parts:
            if p not in seen:
                ld_merged.append(p)
                seen.add(p)
        worker_env["LD_LIBRARY_PATH"] = ":".join(ld_merged)

    subprocess.run(cmd, check=True, env=worker_env)

    worker_summary_path = worker_out / "falcor_summary.json"
    worker_csv_path = worker_out / "falcor_frames.csv"
    if not worker_summary_path.exists():
        raise FileNotFoundError(f"Worker summary missing: {worker_summary_path}")

    with open(worker_summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    rows: List[Dict[str, Any]] = []
    if worker_csv_path.exists():
        import csv

        with open(worker_csv_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    summary["execution_mode"] = "split_runtime"
    summary["field_builder"] = str(args.field_builder)
    summary["gbuffer_mode"] = str(args.gbuffer_mode)
    summary["split_runtime"] = {
        "falcor_python_bin": str(args.falcor_python_bin),
        "worker_script": str(worker_script),
        "worker_output_dir": str(worker_out),
        "falcor_python_path": str(args.falcor_python_path),
        "field_builder": str(args.field_builder),
        "gbuffer_mode": str(args.gbuffer_mode),
        "sync_gpu_forced": bool(args.sync_gpu),
        "sync_every": int(max(0, int(args.sync_every))),
    }

    if "model" in routes and infer_ms_full is not None:
        infer_map = {int(fr): float(ms) for fr, ms in zip(all_frames, infer_ms_full)}
        infer_bench = np.asarray([infer_map[int(fr)] for fr in bench_frames], dtype=np.float64)

        full_total: List[float] = []
        full_with_gbuffer: List[float] = []

        for row in rows:
            try:
                fr = int(float(row.get("frame", "-1")))
            except Exception:
                continue
            if fr not in infer_map:
                continue
            infer_ms = float(infer_map[fr])
            row["model_infer_ms"] = f"{infer_ms:.6f}"

            try:
                src = float(row.get("model_source_ms", 0.0))
                total = float(row.get("model_total_ms", 0.0))
                with_g = float(row.get("model_with_gbuffer_ms", 0.0))
            except Exception:
                continue

            total_full = total - src + infer_ms
            with_g_full = with_g - src + infer_ms
            row["model_total_full_ms"] = f"{total_full:.6f}"
            row["model_with_gbuffer_full_ms"] = f"{with_g_full:.6f}"
            full_total.append(total_full)
            full_with_gbuffer.append(with_g_full)

        model_result = summary.get("results", {}).get("model", {})
        model_result["inference_ms"] = _summarize_ms(infer_bench)
        model_result["total_full_ms"] = _summarize_ms(full_total)
        model_result["with_gbuffer_full_ms"] = _summarize_ms(full_with_gbuffer)
        model_result["fps_route_only_full"] = _fps_from_ms(model_result["total_full_ms"]["mean"])
        model_result["fps_with_gbuffer_full"] = _fps_from_ms(model_result["with_gbuffer_full_ms"]["mean"])
        summary.setdefault("results", {})["model"] = model_result

        if "gt" in summary.get("results", {}):
            gt_total = float(summary["results"]["gt"]["total_ms"]["mean"])
            gt_wgb = float(summary["results"]["gt"]["with_gbuffer_ms"]["mean"])
            m_total = float(model_result["total_full_ms"]["mean"])
            m_wgb = float(model_result["with_gbuffer_full_ms"]["mean"])
            summary["comparison_full"] = {
                "model_vs_gt_total_speed_ratio": float(gt_total / m_total) if m_total > 0 else float("nan"),
                "model_vs_gt_e2e_speed_ratio": float(gt_wgb / m_wgb) if m_wgb > 0 else float("nan"),
                "delta_total_ms": float(m_total - gt_total),
                "delta_e2e_ms": float(m_wgb - gt_wgb),
            }

    out_sum = out_dir / "benchmark_summary.json"
    out_csv = out_dir / "benchmark_frames.csv"

    with open(out_sum, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Saved merged summary: {out_sum}")
    print(f"Saved merged frame timings: {out_csv}")


def _import_falcor_with_testbed(falcor_python_path: Optional[str]):
    """Import Falcor with robust extension/lib compatibility handling.

    Notes:
    - Falcor python bindings are CPython-version specific (e.g. cpython-310).
    - This helper detects ABI mismatch early and prints an actionable error.
    """
    if falcor_python_path and falcor_python_path not in sys.path:
        sys.path.insert(0, falcor_python_path)

    # Try to make Falcor shared libs discoverable for extension loading.
    if falcor_python_path:
        py_path = Path(falcor_python_path)
        cand_dirs = [
            py_path.parent,          # .../Debug
            py_path.parent.parent,   # .../bin
            py_path,                 # .../Debug/python
        ]
        loaded_dirs: List[str] = []
        for d in cand_dirs:
            lib_path = d / "libFalcor.so"
            if lib_path.exists():
                try:
                    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                    loaded_dirs.append(str(d))
                except OSError:
                    pass

        if loaded_dirs:
            old_ld = os.environ.get("LD_LIBRARY_PATH", "")
            parts = [p for p in loaded_dirs + old_ld.split(":") if p]
            # preserve order while dedup
            seen = set()
            dedup = []
            for p in parts:
                if p not in seen:
                    dedup.append(p)
                    seen.add(p)
            os.environ["LD_LIBRARY_PATH"] = ":".join(dedup)

    try:
        import falcor as fc  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to import falcor. Check --falcor-python-path and Falcor build."
        ) from exc

    if hasattr(fc, "Testbed"):
        return fc

    if not falcor_python_path:
        raise RuntimeError(
            "Falcor imported but Testbed missing. Provide --falcor-python-path."
        )

    ext_candidates = sorted(glob.glob(os.path.join(str(falcor_python_path), "falcor", "falcor_ext*.so")))
    if not ext_candidates:
        raise RuntimeError(
            f"falcor_ext*.so not found under: {falcor_python_path}/falcor"
        )

    # Pick extension matching current Python ABI.
    abi_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    match = [p for p in ext_candidates if abi_tag in Path(p).name]
    if not match:
        found = ", ".join(Path(p).name for p in ext_candidates)
        raise RuntimeError(
            "Falcor Python ABI mismatch: current interpreter is "
            f"{sys.version_info.major}.{sys.version_info.minor}, "
            f"but falcor_ext candidates are [{found}]. "
            "Please run with a matching Python interpreter (typically Falcor's "
            "pythondist python3.10) or use an environment where both Falcor and "
            "Torch are installed for the same Python version."
        )

    ext_path = match[0]

    module_name = "falcor.falcor_ext"
    if module_name in sys.modules:
        del sys.modules[module_name]

    loader = importlib.machinery.ExtensionFileLoader(module_name, ext_path)
    spec = importlib.util.spec_from_file_location(module_name, ext_path, loader=loader)
    if spec is None:
        raise RuntimeError("Failed to create module spec for falcor_ext")
    try:
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Falcor extension: {ext_path}. Original error: {exc}"
        ) from exc
    sys.modules[module_name] = module

    for name in dir(module):
        if not name.startswith("_"):
            setattr(fc, name, getattr(module, name))
    setattr(fc, "falcor_ext", module)

    if not hasattr(fc, "Testbed"):
        raise RuntimeError(
            "Falcor extension loaded, but Testbed is still missing. "
            "Check Falcor Python bindings compatibility."
        )

    return fc


def main() -> None:
    args = build_arg_parser().parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = _resolve_dataset_npz(args.dataset)
    scene_path = Path(args.scene)
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    routes: List[str]
    if args.route == "both":
        routes = ["gt", "model"]
    else:
        routes = [args.route]

    if "model" in routes and not args.checkpoint:
        raise ValueError("--checkpoint is required when route includes model")

    if args.split_runtime:
        _run_split_runtime(args, out_dir=out_dir, npz_path=npz_path, scene_path=scene_path, routes=routes)
        return

    if args.falcor_python_path and args.falcor_python_path not in sys.path:
        sys.path.insert(0, args.falcor_python_path)

    if "BISTRO_FBX" not in os.environ or not os.environ.get("BISTRO_FBX", ""):
        os.environ["BISTRO_FBX"] = str(
            ROOT / "1_data_generation" / "scenes" / "Bistro_v5_2" / "BistroExterior.fbx"
        )

    fc = _import_falcor_with_testbed(args.falcor_python_path)

    data = np.load(npz_path, allow_pickle=True)
    tensor = np.asarray(data["tensor"], dtype=np.float32)
    probes = np.asarray(data["probe_positions"], dtype=np.float32)
    light_configs = np.asarray(data["light_configs"], dtype=np.float32)
    light_mask = np.asarray(data["light_mask"], dtype=np.float32) if "light_mask" in data else np.ones(light_configs.shape[:2], dtype=np.float32)
    metadata = data["metadata"].item() if "metadata" in data else {}

    if "valid_mask" in data:
        valid_mask = np.asarray(data["valid_mask"], dtype=np.float32).reshape(-1)
        valid_idx = np.where(valid_mask > 0.5)[0]
        if valid_idx.size > 0 and valid_idx.size < probes.shape[0]:
            probes = probes[valid_idx]
            tensor = tensor[valid_idx]

    num_probes, num_frames_tensor, sh_dim = tensor.shape
    if sh_dim != 27:
        raise ValueError(f"Expected SH dim=27, got {sh_dim}")

    num_frames_light = int(light_configs.shape[0])
    num_frames = min(int(num_frames_tensor), num_frames_light)
    if num_frames < num_frames_tensor:
        tensor = tensor[:, :num_frames, :]
    if num_frames < num_frames_light:
        light_configs = light_configs[:num_frames]
        light_mask = light_mask[:num_frames]

    probe_positions_norm, _, _ = _normalize_positions(probes)

    bounds_min, bounds_max = _compute_bounds(metadata, probes)
    field_knn, field_w = _build_probe_field_weights(
        probes,
        bounds_min,
        bounds_max,
        grid_res=args.field_res,
        k=args.field_knn,
        chunk=args.grid_chunk,
        weight_eps=args.weight_eps,
    )

    model_provider = None
    if "model" in routes:
        model_provider = ModelProvider.create(
            checkpoint=Path(args.checkpoint),
            probe_positions_norm_np=probe_positions_norm,
            light_configs_np=light_configs,
            light_mask_np=light_mask,
            device_str=args.device,
            top_k_override=args.top_k,
            sync_cuda=args.sync_gpu,
        )

    render_scale = max(0.1, float(args.render_scale))
    out_w = int(round(args.width * render_scale))
    out_h = int(round(args.height * render_scale))

    testbed_cls = getattr(fc, "Testbed", None)
    if testbed_cls is None and hasattr(fc, "falcor_ext"):
        testbed_cls = getattr(fc.falcor_ext, "Testbed", None)
    if testbed_cls is None:
        raise RuntimeError("Falcor Testbed not found")

    testbed = testbed_cls(width=out_w, height=out_h, create_window=False)

    if hasattr(testbed, "create_render_graph"):
        graph = testbed.create_render_graph("BenchmarkProbeVolume")
    elif hasattr(testbed, "createRenderGraph"):
        graph = testbed.createRenderGraph("BenchmarkProbeVolume")
    else:
        graph = fc.RenderGraph("BenchmarkProbeVolume")

    if hasattr(graph, "create_pass"):
        graph.create_pass("GBufferRT", "GBufferRT", {"samplePattern": "Center", "sampleCount": 1})
    elif hasattr(graph, "createPass"):
        graph.createPass("GBufferRT", "GBufferRT", {"samplePattern": "Center", "sampleCount": 1})
    else:
        p = fc.createPass("GBufferRT", {"samplePattern": "Center", "sampleCount": 1})
        graph.addPass(p, "GBufferRT")

    for out_name in (
        "GBufferRT.posW",
        "GBufferRT.normW",
        "GBufferRT.diffuseOpacity",
        "GBufferRT.emissive",
        "GBufferRT.depth",
        "GBufferRT.linearZ",
    ):
        if hasattr(graph, "mark_output"):
            graph.mark_output(out_name)
        else:
            graph.markOutput(out_name)

    if hasattr(testbed, "render_graph"):
        testbed.render_graph = graph
    elif hasattr(testbed, "setRenderGraph"):
        testbed.setRenderGraph(graph)
    else:
        raise RuntimeError("Failed to attach render graph to Falcor Testbed")
    testbed.load_scene(str(scene_path))

    device = testbed.device
    field_buf = device.create_structured_buffer(
        struct_size=16,
        element_count=(args.field_res ** 3) * 9,
        bind_flags=fc.ResourceBindFlags.ShaderResource | fc.ResourceBindFlags.UnorderedAccess,
    )
    output_tex = device.create_texture(
        format=fc.ResourceFormat.RGBA32Float,
        width=out_w,
        height=out_h,
        mip_levels=1,
        bind_flags=fc.ResourceBindFlags.UnorderedAccess | fc.ResourceBindFlags.ShaderResource,
    )

    shader_path = Path(__file__).parent / "sh_probe_volume.cs.slang"
    compute = fc.ComputePass(device, file=shader_path, cs_entry="main")
    compute.globals.gFieldSH = field_buf
    compute.globals.gOutput = output_tex
    compute.globals.gBoundsMin = fc.float3(float(bounds_min[0]), float(bounds_min[1]), float(bounds_min[2]))
    compute.globals.gBoundsMax = fc.float3(float(bounds_max[0]), float(bounds_max[1]), float(bounds_max[2]))
    compute.globals.gFieldRes = fc.uint3(int(args.field_res), int(args.field_res), int(args.field_res))
    compute.globals.gFrameDim = fc.uint2(int(out_w), int(out_h))
    compute.globals.gExposure = 1.0
    compute.globals.gAOStrength = 0.0

    frame_indices = list(range(max(0, int(args.frame_start)), num_frames, max(1, int(args.frame_step))))
    if args.max_frames is not None:
        frame_indices = frame_indices[: max(0, int(args.max_frames))]
    if not frame_indices:
        raise ValueError("No frames selected. Check frame_start/frame_step/max_frames")

    warmup_n = min(max(0, int(args.warmup_frames)), len(frame_indices) - 1)
    warmup_frames = frame_indices[:warmup_n]
    bench_frames = frame_indices[warmup_n:]
    if args.benchmark_frames is not None:
        bench_frames = bench_frames[: max(1, int(args.benchmark_frames))]
    if not bench_frames:
        raise ValueError("No benchmark frames after warmup")

    phase_times: Dict[str, Dict[str, List[float]]] = {
        "gbuffer": {"ms": []},
        "gt": {"source": [], "field": [], "pack": [], "upload": [], "shade": [], "total": [], "with_gbuffer": []},
        "model": {"source": [], "field": [], "pack": [], "upload": [], "shade": [], "total": [], "with_gbuffer": []},
    }
    frame_rows: List[Dict[str, Any]] = []

    do_image_metrics = bool(args.compute_image_metrics and set(routes) == {"gt", "model"})
    metric_every = max(1, int(args.save_frame_metrics_every))
    image_metrics: List[Dict[str, Any]] = []
    if do_image_metrics:
        from utils.rendering_utils import compute_image_metrics

    all_frames = warmup_frames + bench_frames
    bench_set = set(bench_frames)
    bench_counter = 0

    print(
        f"Benchmark setup: probes={num_probes}, frames={num_frames}, "
        f"warmup={len(warmup_frames)}, benchmark={len(bench_frames)}, routes={routes}"
    )

    for local_idx, frame in enumerate(all_frames):
        t_frame = float(frame) / float(args.fps)
        try:
            testbed.clock.time = t_frame
        except Exception:
            try:
                testbed.clock.setTime(t_frame)
            except Exception:
                pass

        t0 = time.perf_counter()
        testbed.resize_frame_buffer(out_w, out_h)
        testbed.frame()
        t1 = time.perf_counter()
        gbuffer_ms = (t1 - t0) * 1000.0

        pos_tex = graph.get_output("GBufferRT.posW")
        norm_tex = graph.get_output("GBufferRT.normW")
        alb_tex = graph.get_output("GBufferRT.diffuseOpacity")
        emi_tex = graph.get_output("GBufferRT.emissive")
        dep_tex = graph.get_output("GBufferRT.depth")
        lin_tex = graph.get_output("GBufferRT.linearZ")

        compute.globals.gPosW = pos_tex
        compute.globals.gNormW = norm_tex
        compute.globals.gAlbedo = alb_tex
        compute.globals.gEmissive = emi_tex
        compute.globals.gDepth = dep_tex
        compute.globals.gLinearZ = lin_tex

        is_bench = frame in bench_set
        if is_bench:
            bench_counter += 1
        need_image_metrics_this_frame = bool(do_image_metrics and is_bench and (bench_counter % metric_every == 0))

        row: Dict[str, Any] = {
            "frame": int(frame),
            "is_warmup": not is_bench,
            "gbuffer_ms": gbuffer_ms,
        }

        rendered_images: Dict[str, np.ndarray] = {}

        for route in routes:
            t_src0 = time.perf_counter()
            if route == "gt":
                sh_probe = tensor[:, frame, :]
            elif route == "model":
                assert model_provider is not None
                sh_probe = model_provider.infer_frame(frame)
            else:
                raise RuntimeError(f"Unknown route: {route}")
            t_src1 = time.perf_counter()

            t_field0 = time.perf_counter()
            field = (field_w[..., None] * sh_probe[field_knn]).sum(axis=-2)
            t_field1 = time.perf_counter()

            t_pack0 = time.perf_counter()
            packed = _pack_field(field)
            t_pack1 = time.perf_counter()

            t_upload0 = time.perf_counter()
            field_buf.from_numpy(packed)
            t_upload1 = time.perf_counter()

            t_shade_submit0 = time.perf_counter()
            compute.execute(threads_x=out_w, threads_y=out_h)
            t_shade_submit1 = time.perf_counter()

            should_sync = bool(args.sync_gpu)
            if (not should_sync) and int(args.sync_every) > 0 and is_bench and (bench_counter % int(args.sync_every) == 0):
                should_sync = True
            if need_image_metrics_this_frame:
                should_sync = True

            shade_sync_ms = 0.0
            if should_sync:
                t_sync0 = time.perf_counter()
                img = np.asarray(output_tex.to_numpy(), dtype=np.float32)
                t_sync1 = time.perf_counter()
                shade_sync_ms = (t_sync1 - t_sync0) * 1000.0
                if img.shape[-1] > 3:
                    img = img[..., :3]
                if need_image_metrics_this_frame:
                    rendered_images[route] = img

            source_ms = (t_src1 - t_src0) * 1000.0
            field_ms = (t_field1 - t_field0) * 1000.0
            pack_ms = (t_pack1 - t_pack0) * 1000.0
            upload_ms = (t_upload1 - t_upload0) * 1000.0
            shade_submit_ms = (t_shade_submit1 - t_shade_submit0) * 1000.0
            shade_ms = shade_submit_ms + shade_sync_ms
            total_ms = source_ms + field_ms + pack_ms + upload_ms + shade_ms

            row[f"{route}_source_ms"] = source_ms
            row[f"{route}_field_ms"] = field_ms
            row[f"{route}_pack_ms"] = pack_ms
            row[f"{route}_upload_ms"] = upload_ms
            row[f"{route}_shade_submit_ms"] = shade_submit_ms
            row[f"{route}_shade_sync_ms"] = shade_sync_ms
            row[f"{route}_shade_ms"] = shade_ms
            row[f"{route}_sync"] = int(should_sync)
            row[f"{route}_total_ms"] = total_ms
            row[f"{route}_with_gbuffer_ms"] = total_ms + gbuffer_ms

            if is_bench:
                phase_times[route]["source"].append(source_ms)
                phase_times[route]["field"].append(field_ms)
                phase_times[route]["pack"].append(pack_ms)
                phase_times[route]["upload"].append(upload_ms)
                phase_times[route]["shade"].append(shade_ms)
                phase_times[route]["total"].append(total_ms)
                phase_times[route]["with_gbuffer"].append(total_ms + gbuffer_ms)

        if is_bench:
            phase_times["gbuffer"]["ms"].append(gbuffer_ms)
            frame_rows.append(row)

            if need_image_metrics_this_frame:
                gt_img = _srgb_encode(_tone_map_reinhard(rendered_images["gt"]))
                model_img = _srgb_encode(_tone_map_reinhard(rendered_images["model"]))
                m = compute_image_metrics(gt_img, model_img)
                image_metrics.append(
                    {
                        "frame": int(frame),
                        "psnr": float(m["psnr"]),
                        "ssim": float(m["ssim"]),
                    }
                )

        if local_idx % max(1, len(all_frames) // 20) == 0:
            tag = "warmup" if not is_bench else "bench"
            print(f"[{local_idx+1}/{len(all_frames)}] frame={frame} ({tag})")

    summary: Dict[str, Any] = {
        "dataset": str(npz_path),
        "scene": str(scene_path),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "routes": routes,
        "num_probes": int(num_probes),
        "num_frames_total": int(num_frames),
        "warmup_frames": int(len(warmup_frames)),
        "benchmark_frames": int(len(bench_frames)),
        "field_res": int(args.field_res),
        "field_knn": int(args.field_knn),
        "field_builder": str(args.field_builder),
        "gbuffer_mode": str(args.gbuffer_mode),
        "render_resolution": [int(out_w), int(out_h)],
        "sync_gpu_forced": bool(args.sync_gpu),
        "sync_every": int(max(0, int(args.sync_every))),
        "gbuffer": _summarize_ms(phase_times["gbuffer"]["ms"]),
        "results": {},
    }

    for route in routes:
        r = {
            "source_ms": _summarize_ms(phase_times[route]["source"]),
            "field_ms": _summarize_ms(phase_times[route]["field"]),
            "pack_ms": _summarize_ms(phase_times[route]["pack"]),
            "upload_ms": _summarize_ms(phase_times[route]["upload"]),
            "shade_ms": _summarize_ms(phase_times[route]["shade"]),
            "total_ms": _summarize_ms(phase_times[route]["total"]),
            "with_gbuffer_ms": _summarize_ms(phase_times[route]["with_gbuffer"]),
        }
        r["fps_route_only"] = _fps_from_ms(r["total_ms"]["mean"])
        r["fps_with_gbuffer"] = _fps_from_ms(r["with_gbuffer_ms"]["mean"])
        summary["results"][route] = r

    if set(routes) == {"gt", "model"}:
        gt_total = summary["results"]["gt"]["total_ms"]["mean"]
        model_total = summary["results"]["model"]["total_ms"]["mean"]
        gt_e2e = summary["results"]["gt"]["with_gbuffer_ms"]["mean"]
        model_e2e = summary["results"]["model"]["with_gbuffer_ms"]["mean"]
        summary["comparison"] = {
            "model_vs_gt_total_speed_ratio": float(gt_total / model_total) if model_total > 0 else float("nan"),
            "model_vs_gt_e2e_speed_ratio": float(gt_e2e / model_e2e) if model_e2e > 0 else float("nan"),
            "delta_total_ms": float(model_total - gt_total),
            "delta_e2e_ms": float(model_e2e - gt_e2e),
        }

    if image_metrics:
        summary["image_metrics"] = {
            "samples": int(len(image_metrics)),
            "mean_psnr": float(np.mean([x["psnr"] for x in image_metrics])),
            "mean_ssim": float(np.mean([x["ssim"] for x in image_metrics])),
            "per_sample": image_metrics,
        }

    json_path = out_dir / "benchmark_summary.json"
    csv_path = out_dir / "benchmark_frames.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if frame_rows:
        fieldnames = sorted({k for row in frame_rows for k in row.keys()})
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(frame_rows)

    print(f"Saved summary: {json_path}")
    print(f"Saved frame timings: {csv_path}")
    if "results" in summary:
        for route in routes:
            fps = summary["results"][route]["fps_with_gbuffer"]
            tms = summary["results"][route]["with_gbuffer_ms"]["mean"]
            print(f"[{route}] mean_with_gbuffer={tms:.3f} ms, fps={fps:.2f}")


if __name__ == "__main__":
    main()
