#!/usr/bin/env python3
"""Export Oracle SH predictions for benchmark worker and summarize SH gains.

Outputs an NPZ compatible with tools/benchmark_realtime_pipeline_worker.py:
  - frames: int64 [F]
  - sh_model: float32 [F, P, 27]

Also writes:
  - summary JSON (aggregate SH gains)
  - per-frame SH metrics CSV (baseline/oracle) for correlation analysis
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "2_src"
if str(_SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_SRC))
if str(_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ROOT))

from data.sh_scaler import AdaptiveSHScaler
from training.gaussian_physics_trainer import BatchAdapter
from utils.unified_metrics import compute_sh_band_metrics, compute_sh_metrics


def _load_diag_module():
    diag_path = _ROOT / "3_experiments" / "scripts" / "analysis" / "run_sh_oracle_diagnostics.py"
    spec = importlib.util.spec_from_file_location("run_sh_oracle_diagnostics", str(diag_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load diagnostics module from {diag_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _device_from_arg(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_frames(args: argparse.Namespace, num_frames: int) -> np.ndarray:
    if args.frames_json:
        fs = json.loads(Path(args.frames_json).read_text(encoding="utf-8"))
        frames = np.asarray(fs["all_frames"], dtype=np.int64)
    else:
        start = int(args.frame_start)
        step = max(1, int(args.frame_step))
        if args.max_frames is None:
            stop = num_frames
        else:
            stop = min(num_frames, start + step * int(args.max_frames))
        frames = np.arange(start, stop, step, dtype=np.int64)
    if frames.size == 0:
        raise ValueError("No frames selected")
    if int(frames.min()) < 0 or int(frames.max()) >= int(num_frames):
        raise ValueError(f"Frame index out of range [0, {num_frames-1}] -> min={frames.min()}, max={frames.max()}")
    return frames


def _build_model_provider_from_ckpt(diag_mod, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    meta = ckpt.get("meta") if isinstance(ckpt, dict) else {}
    if meta is None:
        meta = {}
    model, model_info = diag_mod._build_model_from_checkpoint(state, meta, device)
    return model, model_info, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Oracle SH NPZ for benchmark worker")
    parser.add_argument("--dataset", required=True, help="Dataset root or parametric_tensor.npz")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-npz", required=True, help="Output NPZ path with frames/sh_model")
    parser.add_argument("--summary-json", required=True, help="Output SH comparison summary JSON")
    parser.add_argument("--per-frame-sh-csv", default=None, help="Optional per-frame SH metrics CSV")
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--frames-json", default=None, help="Reuse benchmark _split_runtime/frames.json")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)

    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--sh-scaler", default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    dataset_npz = dataset_path / "parametric_tensor.npz" if dataset_path.is_dir() else dataset_path
    if not dataset_npz.exists():
        raise FileNotFoundError(f"Dataset npz not found: {dataset_npz}")

    data = np.load(dataset_npz, allow_pickle=True)
    tensor_full = np.asarray(data["tensor"], dtype=np.float32)  # [P, M, 27]
    probes_full = np.asarray(data["probe_positions"], dtype=np.float32)
    light_configs = np.asarray(data["light_configs"], dtype=np.float32)  # [M, N, 12]
    light_mask = np.asarray(data["light_mask"], dtype=np.float32) if "light_mask" in data else np.ones(light_configs.shape[:2], dtype=np.float32)

    valid_idx = None
    if "valid_mask" in data:
        valid_mask = np.asarray(data["valid_mask"], dtype=np.float32).reshape(-1)
        tmp = np.where(valid_mask > 0.5)[0]
        if 0 < tmp.size < tensor_full.shape[0]:
            valid_idx = tmp

    if valid_idx is not None:
        tensor = tensor_full[valid_idx]
        probes = probes_full[valid_idx]
    else:
        tensor = tensor_full
        probes = probes_full

    num_probes, num_frames, sh_dim = tensor.shape
    if sh_dim != 27:
        raise ValueError(f"Expected SH dim 27, got {sh_dim}")

    frames = _load_frames(args, num_frames)

    diag_mod = _load_diag_module()
    device = _device_from_arg(args.device)
    model, model_info, meta = _build_model_provider_from_ckpt(diag_mod, Path(args.checkpoint), device)
    top_k = int(args.top_k) if args.top_k is not None else int(model_info["top_k"])

    scaler = AdaptiveSHScaler.load(args.sh_scaler) if args.sh_scaler else diag_mod._build_scaler_from_meta(meta)
    adapter = BatchAdapter(params_key="light_params", mask_key="light_mask")
    if scaler is not None:
        adapter.target_transform = scaler.transform
        adapter.target_inverse = scaler.inverse

    sh_oracle = np.zeros((frames.shape[0], num_probes, 27), dtype=np.float32)
    sh_baseline = np.zeros((frames.shape[0], num_probes, 27), dtype=np.float32)
    sh_gt = np.zeros((frames.shape[0], num_probes, 27), dtype=np.float32)
    frame_rows = []

    with torch.no_grad():
        positions = torch.from_numpy(probes).to(device)
        for out_idx, fr in enumerate(frames.tolist()):
            gt_frame = torch.from_numpy(tensor[:, fr, :]).to(device)
            params_frame = torch.from_numpy(light_configs[fr]).to(device)
            mask_frame = torch.from_numpy(light_mask[fr]).to(device)

            params_b = params_frame.unsqueeze(0).expand(num_probes, -1, -1)
            mask_b = mask_frame.unsqueeze(0).expand(num_probes, -1)

            gt_scaled = adapter.target_transform(gt_frame) if adapter.target_transform is not None else gt_frame

            baseline_scaled = model(positions, params_b, top_k=top_k, light_mask=mask_b)
            comp = diag_mod._extract_effective_bases(model, positions, params_b, mask_b, top_k)
            oracle_scaled = diag_mod._oracle_timeweights_only(gt_scaled, comp)

            baseline_eval, gt_eval = adapter.inverse_targets(baseline_scaled, gt_scaled)
            oracle_eval, _ = adapter.inverse_targets(oracle_scaled, gt_scaled)

            base_np = baseline_eval.detach().cpu().numpy().astype(np.float32)
            orc_np = oracle_eval.detach().cpu().numpy().astype(np.float32)
            gt_np = gt_eval.detach().cpu().numpy().astype(np.float32)

            sh_baseline[out_idx] = base_np
            sh_oracle[out_idx] = orc_np
            sh_gt[out_idx] = gt_np

            base_m = compute_sh_metrics(gt_np, base_np, max_i=1.0)
            orc_m = compute_sh_metrics(gt_np, orc_np, max_i=1.0)
            frame_rows.append(
                {
                    "frame": int(fr),
                    "baseline_sh_psnr": float(base_m["sh_psnr"]),
                    "oracle_sh_psnr": float(orc_m["sh_psnr"]),
                    "delta_sh_psnr": float(orc_m["sh_psnr"] - base_m["sh_psnr"]),
                    "baseline_sh_ssim": float(base_m["sh_ssim"]),
                    "oracle_sh_ssim": float(orc_m["sh_ssim"]),
                    "delta_sh_ssim": float(orc_m["sh_ssim"] - base_m["sh_ssim"]),
                }
            )

            if (out_idx + 1) % max(1, len(frames) // 10) == 0:
                print(f"[oracle-export] {out_idx+1}/{len(frames)} frames")

    out_npz = Path(args.output_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, frames=frames.astype(np.int64), sh_model=sh_oracle.astype(np.float32))

    gt_flat = sh_gt.reshape(-1, 27)
    base_flat = sh_baseline.reshape(-1, 27)
    oracle_flat = sh_oracle.reshape(-1, 27)

    baseline_metrics = {
        **compute_sh_metrics(gt_flat, base_flat, max_i=1.0),
        **compute_sh_band_metrics(gt_flat, base_flat, max_i=1.0),
    }
    oracle_metrics = {
        **compute_sh_metrics(gt_flat, oracle_flat, max_i=1.0),
        **compute_sh_band_metrics(gt_flat, oracle_flat, max_i=1.0),
    }

    summary = {
        "dataset_npz": str(dataset_npz),
        "checkpoint": str(args.checkpoint),
        "frames_count": int(frames.shape[0]),
        "frames_min": int(frames.min()),
        "frames_max": int(frames.max()),
        "top_k": int(top_k),
        "scaler_enabled": bool(scaler is not None),
        "baseline": baseline_metrics,
        "oracle": oracle_metrics,
        "delta": {
            "sh_psnr_gain_db": float(oracle_metrics["sh_psnr"] - baseline_metrics["sh_psnr"]),
            "sh_ssim_gain": float(oracle_metrics["sh_ssim"] - baseline_metrics["sh_ssim"]),
            "sh_l1_psnr_gain_db": float(oracle_metrics["sh_l1_psnr"] - baseline_metrics["sh_l1_psnr"]),
            "sh_l2_psnr_gain_db": float(oracle_metrics["sh_l2_psnr"] - baseline_metrics["sh_l2_psnr"]),
        },
        "artifacts": {
            "oracle_model_sh_npz": str(out_npz),
        },
    }

    out_sum = Path(args.summary_json)
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    out_sum.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.per_frame_sh_csv:
        out_csv = Path(args.per_frame_sh_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
            w.writeheader()
            w.writerows(frame_rows)
        summary.setdefault("artifacts", {})["per_frame_sh_csv"] = str(out_csv)
        out_sum.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved oracle NPZ: {out_npz}")
    print(f"Saved SH summary: {out_sum}")
    if args.per_frame_sh_csv:
        print(f"Saved per-frame SH CSV: {args.per_frame_sh_csv}")


if __name__ == "__main__":
    main()
