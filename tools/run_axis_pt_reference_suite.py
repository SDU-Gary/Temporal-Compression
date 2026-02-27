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
class ExpCase:
    name: str
    route: str  # gt | both
    axis: str   # off|x|y|z
    sign: str   # pos|neg
    need_checkpoint: bool
    need_image_metrics: bool


def _safe_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_cases(run_baseline: bool) -> List[ExpCase]:
    cases: List[ExpCase] = []
    for axis in ("x", "y", "z"):
        for sign in ("pos", "neg"):
            cases.append(
                ExpCase(
                    name=f"axis_{axis}_{sign}",
                    route="gt",
                    axis=axis,
                    sign=sign,
                    need_checkpoint=False,
                    need_image_metrics=True,
                )
            )
    if run_baseline:
        cases.append(
            ExpCase(
                name="baseline_both_pt",
                route="both",
                axis="off",
                sign="pos",
                need_checkpoint=True,
                need_image_metrics=True,
            )
        )
    return cases


def _extract_summary_row(case: ExpCase, summary: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "case": case.name,
        "route": case.route,
        "axis": case.axis,
        "sign": case.sign,
        "measurement_mode": summary.get("measurement_mode"),
        "field_builder": summary.get("field_builder"),
        "gbuffer_mode": summary.get("gbuffer_mode"),
        "gt_fps_with_gbuffer": _safe_get(summary, "results", "gt", "fps_with_gbuffer"),
        "gt_ms_with_gbuffer": _safe_get(summary, "results", "gt", "with_gbuffer_ms", "mean"),
        "model_fps_with_gbuffer_full": _safe_get(summary, "results", "model", "fps_with_gbuffer_full"),
        "model_ms_with_gbuffer_full": _safe_get(summary, "results", "model", "with_gbuffer_full_ms", "mean"),
        "pt_ref_ms": _safe_get(summary, "pt_reference_ms", "mean"),
    }

    img = summary.get("image_metrics", {})
    row["image_samples"] = img.get("samples")
    for k, v in img.items():
        if isinstance(k, str) and k.startswith("mean_"):
            row[k] = v

    return row


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run axis-test + PT-reference benchmark suite in one command")
    p.add_argument("--dataset", required=True, help="Dataset root or path to parametric_tensor.npz")
    p.add_argument("--scene", required=True, help="Falcor scene .pyscene path")
    p.add_argument("--checkpoint", default=None, help="Checkpoint path (required when baseline is enabled)")
    p.add_argument("--output-root", required=True, help="Output root directory for suite runs")

    p.add_argument("--benchmark-script", default="tools/benchmark_realtime_pipeline.py")
    p.add_argument("--falcor-python-path", default=None)
    p.add_argument("--falcor-python-bin", default=None)

    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--render-scale", type=float, default=1.0)
    p.add_argument("--field-res", type=int, default=32)
    p.add_argument("--field-knn", type=int, default=8)
    p.add_argument("--weight-eps", type=float, default=0.1)
    p.add_argument("--grid-chunk", type=int, default=4096)

    p.add_argument("--warmup-frames", type=int, default=8)
    p.add_argument("--benchmark-frames", type=int, default=24)
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=None)

    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--save-frame-metrics-every", type=int, default=1)
    p.add_argument("--sync-every", type=int, default=1)

    p.add_argument("--axis-test-strength", type=float, default=1.0)

    p.add_argument("--pt-spp", type=int, default=4)
    p.add_argument("--pt-bounces", type=int, default=4)
    p.add_argument("--pt-use-nee", action="store_true", default=True)
    p.add_argument("--pt-no-nee", action="store_false", dest="pt_use_nee")

    p.add_argument("--run-baseline", action="store_true", default=True)
    p.add_argument("--no-baseline", action="store_false", dest="run_baseline")

    p.add_argument("--save-sampled-images", action="store_true", default=True)
    p.add_argument("--no-save-sampled-images", action="store_false", dest="save_sampled_images")

    p.add_argument("--continue-on-error", action="store_true", default=False)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    bench_script = Path(args.benchmark_script)
    if not bench_script.exists():
        raise FileNotFoundError(f"Benchmark script not found: {bench_script}")

    cases = _build_cases(run_baseline=bool(args.run_baseline))
    if any(c.need_checkpoint for c in cases) and not args.checkpoint:
        raise ValueError("--checkpoint is required when baseline case is enabled")

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
            "--route", str(case.route),
            "--output-dir", str(case_out),
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
            "--save-frame-metrics-every", str(max(1, int(args.save_frame_metrics_every))),
            "--device", str(args.device),
            "--axis-test", str(case.axis),
            "--axis-test-sign", str(case.sign),
            "--axis-test-strength", str(float(args.axis_test_strength)),
            "--pt-reference",
            "--pt-spp", str(max(1, int(args.pt_spp))),
            "--pt-bounces", str(max(0, int(args.pt_bounces))),
            "--sync-every", str(max(0, int(args.sync_every))),
            "--no-sync-gpu",
        ]

        if bool(args.pt_use_nee):
            cmd.append("--pt-use-nee")
        else:
            cmd.append("--pt-no-nee")

        if args.max_frames is not None:
            cmd.extend(["--max-frames", str(args.max_frames)])
        if args.falcor_python_path:
            cmd.extend(["--falcor-python-path", str(args.falcor_python_path)])
        if args.falcor_python_bin:
            cmd.extend(["--falcor-python-bin", str(args.falcor_python_bin)])
        if args.top_k is not None:
            cmd.extend(["--top-k", str(args.top_k)])

        if case.need_checkpoint:
            cmd.extend(["--checkpoint", str(args.checkpoint)])
        if case.need_image_metrics:
            cmd.append("--compute-image-metrics")
        if bool(args.save_sampled_images):
            cmd.extend(["--save-sampled-images-dir", str(case_out / "sampled")])

        print(f"[{idx}/{len(cases)}] Running {case.name}")
        print(" ".join(cmd))

        t0 = time.perf_counter()
        proc = subprocess.run(cmd)
        elapsed = time.perf_counter() - t0

        run_row: Dict[str, Any] = {
            "case": case.name,
            "route": case.route,
            "axis": case.axis,
            "sign": case.sign,
            "return_code": int(proc.returncode),
            "elapsed_sec": float(elapsed),
            "output_dir": str(case_out),
            "command": " ".join(cmd),
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
            row.update(_extract_summary_row(case, summary))
            metric_rows.append(row)
        else:
            print(f"[warn] failed or missing summary: {case.name}")
            if not args.continue_on_error:
                break

    runs_csv = out_root / "axis_pt_runs.csv"
    summary_csv = out_root / "axis_pt_summary.csv"
    _write_csv(runs_csv, run_rows)
    _write_csv(summary_csv, metric_rows)

    print(f"Saved runs: {runs_csv}")
    print(f"Saved summary: {summary_csv}")


if __name__ == "__main__":
    main()
