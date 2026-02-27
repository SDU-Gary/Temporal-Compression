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
import csv
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
    parser.add_argument("--normal-transform",
                        choices=["identity", "swap_yz", "swap_xz", "swap_xy", "flip_x", "flip_y", "flip_z"],
                        default="identity",
                        help="Debug SH normal orientation mismatch")
    parser.add_argument("--cosine-mode", choices=["irradiance", "radiance"], default="irradiance",
                        help="Irradiance applies cosine convolution A_l; radiance disables A_l")
    parser.add_argument("--sh-debug-coeff", type=int, default=-1,
                        help="If in [0,8], keep only one SH coefficient per RGB")
    parser.add_argument("--single-probe-index", type=int, default=-1,
                        help="If >=0, broadcast one probe SH to all probes")
    parser.add_argument("--axis-test", choices=["off", "x", "y", "z"], default="off",
                        help="Override SH with a single strong first-order axis component")
    parser.add_argument("--axis-test-sign", choices=["pos", "neg"], default="pos",
                        help="Sign for axis-test component")
    parser.add_argument("--axis-test-strength", type=float, default=1.0,
                        help="Strength for axis-test SH coefficient")
    parser.add_argument("--save-sampled-images-dir", default=None,
                        help="Optional dir to dump sampled metric frames (npz)")
    parser.add_argument("--roi", default=None,
                        help="Optional normalized ROI x0,y0,x1,y1 for extra image metrics")
    parser.add_argument("--output-scale", type=float, default=1.0,
                        help="Scale applied to SH shading output before metrics/output")
    parser.add_argument("--metric-align-scale-gt", type=float, default=1.0,
                        help="Scale GT route linear image before metric computation (metrics-only)")
    parser.add_argument("--metric-align-scale-model", type=float, default=1.0,
                        help="Scale Model route linear image before metric computation (metrics-only)")
    parser.add_argument("--pt-reference", action="store_true", default=False,
                        help="Also render PathTracer reference and report route-vs-PT metrics")
    parser.add_argument("--pt-spp", type=int, default=8,
                        help="PathTracer spp per frame for PT reference")
    parser.add_argument("--pt-bounces", type=int, default=4,
                        help="PathTracer max surface bounces for PT reference")
    parser.add_argument("--pt-use-nee", action="store_true", default=True,
                        help="Enable next-event estimation for PT reference")
    parser.add_argument("--pt-no-nee", action="store_false", dest="pt_use_nee")
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
        "--normal-transform", str(args.normal_transform),
        "--cosine-mode", str(args.cosine_mode),
        "--sh-debug-coeff", str(int(args.sh_debug_coeff)),
        "--single-probe-index", str(int(args.single_probe_index)),
        "--axis-test", str(args.axis_test),
        "--axis-test-sign", str(args.axis_test_sign),
        "--axis-test-strength", str(float(args.axis_test_strength)),
        "--output-scale", str(float(args.output_scale)),
        "--metric-align-scale-gt", str(float(args.metric_align_scale_gt)),
        "--metric-align-scale-model", str(float(args.metric_align_scale_model)),
        "--falcor-python-path", str(args.falcor_python_path),
    ]
    if model_sh_npz is not None:
        cmd.extend(["--model-sh-npz", str(model_sh_npz)])
    cmd.append("--sync-gpu" if args.sync_gpu else "--no-sync-gpu")
    cmd.extend(["--sync-every", str(max(0, int(args.sync_every)))])
    if args.compute_image_metrics:
        cmd.append("--compute-image-metrics")
    if args.compute_image_metrics or args.pt_reference:
        cmd.extend(["--save-frame-metrics-every", str(max(1, int(args.save_frame_metrics_every)))])
    if args.save_sampled_images_dir:
        cmd.extend(["--save-sampled-images-dir", str(args.save_sampled_images_dir)])
    if args.roi:
        cmd.extend(["--roi", str(args.roi)])
    if args.pt_reference:
        cmd.append("--pt-reference")
        cmd.extend([
            "--pt-spp", str(max(1, int(args.pt_spp))),
            "--pt-bounces", str(max(0, int(args.pt_bounces))),
        ])
        cmd.append("--pt-use-nee" if bool(args.pt_use_nee) else "--pt-no-nee")

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
    summary["measurement_mode"] = (
        "force_every_frame" if bool(args.sync_gpu)
        else ("sampled_every_n" if int(max(0, int(args.sync_every))) > 0 else "async_no_sync")
    )
    summary["split_runtime"] = {
        "falcor_python_bin": str(args.falcor_python_bin),
        "worker_script": str(worker_script),
        "worker_output_dir": str(worker_out),
        "falcor_python_path": str(args.falcor_python_path),
        "field_builder": str(args.field_builder),
        "gbuffer_mode": str(args.gbuffer_mode),
        "normal_transform": str(args.normal_transform),
        "cosine_mode": str(args.cosine_mode),
        "sh_debug_coeff": int(args.sh_debug_coeff),
        "single_probe_index": int(args.single_probe_index),
        "axis_test": str(args.axis_test),
        "axis_test_sign": str(args.axis_test_sign),
        "axis_test_strength": float(args.axis_test_strength),
        "output_scale": float(args.output_scale),
        "metric_align_scale_gt": float(args.metric_align_scale_gt),
        "metric_align_scale_model": float(args.metric_align_scale_model),
        "roi": str(args.roi) if args.roi else None,
        "pt_reference": bool(args.pt_reference),
        "pt_spp": int(max(1, int(args.pt_spp))),
        "pt_bounces": int(max(0, int(args.pt_bounces))),
        "pt_use_nee": bool(args.pt_use_nee),
        "sync_gpu_forced": bool(args.sync_gpu),
        "sync_every": int(max(0, int(args.sync_every))),
    }

    if rows:
        for route in routes:
            key = f"{route}_sync"
            if key in rows[0]:
                vals = []
                for row in rows:
                    try:
                        vals.append(float(row.get(key, 0.0)))
                    except Exception:
                        vals.append(0.0)
                ratio = float(np.mean(np.asarray(vals, dtype=np.float64))) if vals else float("nan")
                summary.setdefault("results", {}).setdefault(route, {})["sync_ratio"] = ratio

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


def main() -> None:
    args = build_arg_parser().parse_args()

    if not args.split_runtime:
        raise NotImplementedError(
            "Non-split runtime path has been removed. "
            "Please use split runtime (default) with Falcor worker."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = _resolve_dataset_npz(args.dataset)
    scene_path = Path(args.scene)
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    if args.route == "both":
        routes: List[str] = ["gt", "model"]
    else:
        routes = [args.route]

    if "model" in routes and not args.checkpoint:
        raise ValueError("--checkpoint is required when route includes model")

    _run_split_runtime(args, out_dir=out_dir, npz_path=npz_path, scene_path=scene_path, routes=routes)


if __name__ == "__main__":
    main()
