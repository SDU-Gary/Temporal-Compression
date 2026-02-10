#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Case:
    name: str
    field_builder: str
    gbuffer_mode: str
    sync_gpu: bool
    sync_every: int


def _safe_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _build_cases(include_sync_baseline: bool, sync_every: int) -> List[Case]:
    cases = [
        Case("cpu_realtime_async", "cpu", "realtime", False, sync_every),
        Case("cpu_reuse_async", "cpu", "reuse_first", False, sync_every),
        Case("gpu_realtime_async", "gpu", "realtime", False, sync_every),
        Case("gpu_reuse_async", "gpu", "reuse_first", False, sync_every),
    ]
    if include_sync_baseline:
        cases.insert(0, Case("cpu_realtime_sync", "cpu", "realtime", True, 0))
    return cases


def _extract_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "execution_mode": summary.get("execution_mode"),
        "measurement_mode": summary.get("measurement_mode"),
        "field_builder": summary.get("field_builder"),
        "gbuffer_mode": summary.get("gbuffer_mode"),
        "sync_gpu_forced": summary.get("sync_gpu_forced"),
        "sync_every": summary.get("sync_every"),
        "gbuffer_frames_rendered": summary.get("gbuffer_frames_rendered"),
        "gbuffer_mean_ms": _safe_get(summary, "gbuffer", "mean"),
    }

    for route in ("gt", "model"):
        prefix = f"{route}_"
        row[prefix + "sync_ratio"] = _safe_get(summary, "results", route, "sync_ratio")
        row[prefix + "source_ms"] = _safe_get(summary, "results", route, "source_ms", "mean")
        row[prefix + "field_ms"] = _safe_get(summary, "results", route, "field_ms", "mean")
        row[prefix + "pack_ms"] = _safe_get(summary, "results", route, "pack_ms", "mean")
        row[prefix + "upload_ms"] = _safe_get(summary, "results", route, "upload_ms", "mean")
        row[prefix + "field_build_submit_ms"] = _safe_get(summary, "results", route, "field_build_submit_ms", "mean")
        row[prefix + "field_build_sync_ms"] = _safe_get(summary, "results", route, "field_build_sync_ms", "mean")
        row[prefix + "shade_submit_ms"] = _safe_get(summary, "results", route, "shade_submit_ms", "mean")
        row[prefix + "shade_sync_ms"] = _safe_get(summary, "results", route, "shade_sync_ms", "mean")
        row[prefix + "shade_ms"] = _safe_get(summary, "results", route, "shade_ms", "mean")
        row[prefix + "total_ms"] = _safe_get(summary, "results", route, "total_ms", "mean")
        row[prefix + "with_gbuffer_ms"] = _safe_get(summary, "results", route, "with_gbuffer_ms", "mean")
        row[prefix + "fps_with_gbuffer"] = _safe_get(summary, "results", route, "fps_with_gbuffer")

    row["model_inference_ms"] = _safe_get(summary, "results", "model", "inference_ms", "mean")
    row["model_with_gbuffer_full_ms"] = _safe_get(summary, "results", "model", "with_gbuffer_full_ms", "mean")
    row["model_fps_with_gbuffer_full"] = _safe_get(summary, "results", "model", "fps_with_gbuffer_full")
    row["delta_e2e_ms_model_vs_gt"] = _safe_get(summary, "comparison_full", "delta_e2e_ms")
    row["speed_ratio_e2e_model_vs_gt"] = _safe_get(summary, "comparison_full", "model_vs_gt_e2e_speed_ratio")
    row["image_samples"] = _safe_get(summary, "image_metrics", "samples")
    row["image_mean_psnr"] = _safe_get(summary, "image_metrics", "mean_psnr")
    row["image_mean_ssim"] = _safe_get(summary, "image_metrics", "mean_ssim")
    return row


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run benchmark matrix and aggregate CSV reports")
    p.add_argument("--dataset", required=True, help="Dataset root or path to parametric_tensor.npz")
    p.add_argument("--scene", required=True, help="Falcor scene path (.pyscene)")
    p.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    p.add_argument("--output-root", required=True, help="Output root for all matrix runs")

    p.add_argument("--benchmark-script", default="tools/benchmark_realtime_pipeline.py")
    p.add_argument("--route", default="both", choices=["gt", "model", "both"])

    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--render-scale", type=float, default=1.0)
    p.add_argument("--field-res", type=int, default=32)
    p.add_argument("--field-knn", type=int, default=8)
    p.add_argument("--weight-eps", type=float, default=0.1)
    p.add_argument("--grid-chunk", type=int, default=4096)

    p.add_argument("--warmup-frames", type=int, default=60)
    p.add_argument("--benchmark-frames", type=int, default=300)
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=None)

    p.add_argument("--sync-every", type=int, default=30, help="Sampling sync interval for async cases")
    p.add_argument("--include-sync-baseline", action="store_true", default=False)

    p.add_argument("--falcor-python-path", default=None)
    p.add_argument("--falcor-python-bin", default=None)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--top-k", type=int, default=None)

    p.add_argument("--continue-on-error", action="store_true", default=False)
    p.add_argument("--compute-image-metrics", action="store_true", default=False,
                   help="Enable PSNR/SSIM computation in split-runtime worker")
    p.add_argument("--save-frame-metrics-every", type=int, default=30,
                   help="Compute image metrics every N benchmark frames")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    bench_script = Path(args.benchmark_script)
    if not bench_script.exists():
        raise FileNotFoundError(f"Benchmark script not found: {bench_script}")

    cases = _build_cases(include_sync_baseline=bool(args.include_sync_baseline), sync_every=max(0, int(args.sync_every)))

    run_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        case_out = out_root / case.name
        case_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(bench_script),
            "--dataset", str(args.dataset),
            "--scene", str(args.scene),
            "--checkpoint", str(args.checkpoint),
            "--output-dir", str(case_out),
            "--route", str(args.route),
            "--width", str(args.width),
            "--height", str(args.height),
            "--fps", str(args.fps),
            "--render-scale", str(args.render_scale),
            "--field-res", str(args.field_res),
            "--field-knn", str(args.field_knn),
            "--weight-eps", str(args.weight_eps),
            "--grid-chunk", str(args.grid_chunk),
            "--warmup-frames", str(args.warmup_frames),
            "--benchmark-frames", str(args.benchmark_frames),
            "--frame-start", str(args.frame_start),
            "--frame-step", str(args.frame_step),
            "--field-builder", case.field_builder,
            "--gbuffer-mode", case.gbuffer_mode,
            "--device", str(args.device),
        ]
        if args.max_frames is not None:
            cmd.extend(["--max-frames", str(args.max_frames)])
        if args.falcor_python_path:
            cmd.extend(["--falcor-python-path", str(args.falcor_python_path)])
        if args.falcor_python_bin:
            cmd.extend(["--falcor-python-bin", str(args.falcor_python_bin)])
        if args.top_k is not None:
            cmd.extend(["--top-k", str(args.top_k)])

        if case.sync_gpu:
            cmd.append("--sync-gpu")
        else:
            cmd.append("--no-sync-gpu")
            cmd.extend(["--sync-every", str(case.sync_every)])

        if args.compute_image_metrics:
            cmd.append("--compute-image-metrics")
            cmd.extend(["--save-frame-metrics-every", str(max(1, int(args.save_frame_metrics_every)))])

        print(f"[{idx}/{len(cases)}] Running case: {case.name}")
        print(" ".join(cmd))

        t0 = time.perf_counter()
        proc = subprocess.run(cmd)
        elapsed = time.perf_counter() - t0

        run_row = {
            "case": case.name,
            "field_builder": case.field_builder,
            "gbuffer_mode": case.gbuffer_mode,
            "sync_gpu": int(case.sync_gpu),
            "sync_every": int(case.sync_every),
            "return_code": int(proc.returncode),
            "elapsed_sec": float(elapsed),
            "output_dir": str(case_out),
        }
        run_rows.append(run_row)

        summary_path = case_out / "benchmark_summary.json"
        if proc.returncode == 0 and summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            row = {
                "case": case.name,
                "summary_path": str(summary_path),
                "return_code": int(proc.returncode),
                "elapsed_sec": float(elapsed),
            }
            row.update(_extract_metrics(summary))
            metric_rows.append(row)
        else:
            print(f"[warn] Case failed or summary missing: {case.name}")
            if not args.continue_on_error:
                break

    run_csv = out_root / "matrix_runs.csv"
    metric_csv = out_root / "matrix_summary.csv"
    _write_csv(run_csv, run_rows)
    _write_csv(metric_csv, metric_rows)

    comparison_rows: List[Dict[str, Any]] = []
    baseline = next((r for r in metric_rows if r.get("return_code") == 0), None)
    if baseline is not None:
        base_case = str(baseline["case"])
        for row in metric_rows:
            out = {
                "case": row.get("case"),
                "baseline_case": base_case,
                "delta_gt_with_gbuffer_ms": None,
                "delta_model_with_gbuffer_full_ms": None,
                "delta_model_fps_with_gbuffer_full": None,
            }
            try:
                if row.get("gt_with_gbuffer_ms") is not None and baseline.get("gt_with_gbuffer_ms") is not None:
                    out["delta_gt_with_gbuffer_ms"] = float(row["gt_with_gbuffer_ms"]) - float(baseline["gt_with_gbuffer_ms"])
                if row.get("model_with_gbuffer_full_ms") is not None and baseline.get("model_with_gbuffer_full_ms") is not None:
                    out["delta_model_with_gbuffer_full_ms"] = float(row["model_with_gbuffer_full_ms"]) - float(baseline["model_with_gbuffer_full_ms"])
                if row.get("model_fps_with_gbuffer_full") is not None and baseline.get("model_fps_with_gbuffer_full") is not None:
                    out["delta_model_fps_with_gbuffer_full"] = float(row["model_fps_with_gbuffer_full"]) - float(baseline["model_fps_with_gbuffer_full"])
            except Exception:
                pass
            comparison_rows.append(out)

    comparison_csv = out_root / "matrix_comparison.csv"
    _write_csv(comparison_csv, comparison_rows)

    print(f"Saved runs: {run_csv}")
    print(f"Saved summary: {metric_csv}")
    print(f"Saved comparison: {comparison_csv}")


if __name__ == "__main__":
    main()
