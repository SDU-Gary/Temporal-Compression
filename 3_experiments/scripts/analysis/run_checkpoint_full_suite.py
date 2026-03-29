#!/usr/bin/env python3
"""Run full benchmark + diagnostics suite for multiple checkpoints.

Features:
- Parallel benchmark runs for multiple checkpoints.
- Per-checkpoint expert utilization audit.
- Semantic drift audit against a reference checkpoint.
- Consolidated JSON + CSV report with PSNR/SSIM-focused metrics.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Case:
    name: str
    checkpoint: Path
    case_dir: Path


@dataclass
class RunResult:
    ok: bool
    return_code: int
    elapsed_sec: float
    log_path: Path
    summary_path: Optional[Path]
    error: Optional[str] = None


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_\-]+", "_", str(name).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "case"


def _resolve_existing_path(raw: str) -> Path:
    cand = Path(raw)
    candidates = [cand, ROOT / cand]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(f"Path not found: {raw}")


def _collect_checkpoint_paths(
    checkpoint_args: List[str],
    checkpoint_list: Optional[str],
    checkpoint_globs: List[str],
) -> List[Path]:
    out: List[Path] = []

    for item in checkpoint_args:
        out.append(_resolve_existing_path(item))

    if checkpoint_list:
        list_path = _resolve_existing_path(checkpoint_list)
        with list_path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                out.append(_resolve_existing_path(text))

    for pat in checkpoint_globs:
        hits = sorted(Path(ROOT).glob(pat))
        if not hits:
            continue
        for h in hits:
            if h.is_file():
                out.append(h.resolve())

    uniq: List[Path] = []
    seen: set[str] = set()
    for p in out:
        s = str(p)
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq


def _build_cases(checkpoints: List[Path], output_root: Path) -> List[Case]:
    counts: Dict[str, int] = {}
    cases: List[Case] = []
    for ckpt in checkpoints:
        base = _sanitize_name(f"{ckpt.parent.name}_{ckpt.stem}")
        idx = counts.get(base, 0)
        counts[base] = idx + 1
        name = base if idx == 0 else f"{base}_{idx+1}"
        case_dir = output_root / name
        case_dir.mkdir(parents=True, exist_ok=True)
        cases.append(Case(name=name, checkpoint=ckpt, case_dir=case_dir))
    return cases


def _ensure_profiles_json(raw: str) -> str:
    if str(raw).strip():
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            parsed = {"profiles": parsed}
        if not isinstance(parsed, dict):
            raise ValueError("--profiles-json must decode to list/dict")
        return json.dumps(parsed, ensure_ascii=False)

    default_profiles = {
        "profiles": [
            {
                "name": "hard3",
                "top_k": 3,
                "training_soft_routing": False,
            },
            {
                "name": "soft8_t020",
                "top_k": 3,
                "training_soft_routing": True,
                "routing_soft_topk": 8,
                "routing_temperature": 0.20,
            },
        ]
    }
    return json.dumps(default_profiles, ensure_ascii=False)


def _run_command(cmd: List[str], log_path: Path) -> RunResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - t0
    ok = int(proc.returncode) == 0
    return RunResult(
        ok=ok,
        return_code=int(proc.returncode),
        elapsed_sec=float(elapsed),
        log_path=log_path,
        summary_path=None,
        error=None if ok else f"return_code={proc.returncode}",
    )


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_numeric(d: Any, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_numeric(v, key))
    elif isinstance(d, (int, float)):
        out[prefix] = float(d)
    return out


def _extract_benchmark_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    flat = _flatten_numeric(summary)
    out: Dict[str, float] = {}
    for key, value in flat.items():
        low = key.lower()
        if any(tag in low for tag in ["psnr", "ssim", "rmse", "mae", "fps", "with_gbuffer", "comparison_full"]):
            out[key] = float(value)
    # Common key aliases for quick scanning.
    aliases = {
        "image.mean_psnr": "image_metrics.mean_psnr",
        "image.mean_ssim": "image_metrics.mean_ssim",
        "image.mean_psnr_linear": "image_metrics.mean_psnr_linear",
        "image.mean_psnr_tonemapped": "image_metrics.mean_psnr_tonemapped",
        "image.mean_psnr_hdr": "image_metrics.mean_psnr_hdr",
        "image.mean_ssim_linear": "image_metrics.mean_ssim_linear",
        "image.mean_ssim_tonemapped": "image_metrics.mean_ssim_tonemapped",
        "image.mean_ssim_hdr": "image_metrics.mean_ssim_hdr",
    }
    for alias, src in aliases.items():
        if src in flat:
            out[alias] = float(flat[src])
    return out


def _extract_expert_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    profiles = summary.get("profiles", {})
    if not isinstance(profiles, dict):
        return out
    for name, payload in profiles.items():
        if not isinstance(payload, dict):
            continue
        p = payload.get("summary", {})
        if not isinstance(p, dict):
            continue
        safe = _sanitize_name(name)
        dist = p.get("distribution", {}).get("mass_sum", {})
        nz = p.get("nonzero_ratio", {})
        if isinstance(dist, dict):
            if "entropy_norm" in dist:
                out[f"expert.{safe}.mass_entropy_norm"] = float(dist["entropy_norm"])
            if "effective_k" in dist:
                out[f"expert.{safe}.mass_effective_k"] = float(dist["effective_k"])
            if "hhi" in dist:
                out[f"expert.{safe}.mass_hhi"] = float(dist["hhi"])
        if isinstance(nz, dict):
            if "sample_count" in nz:
                out[f"expert.{safe}.nonzero_sample_ratio"] = float(nz["sample_count"])
            if "probe_coverage" in nz:
                out[f"expert.{safe}.nonzero_probe_ratio"] = float(nz["probe_coverage"])
    return out


def _extract_drift_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    param = summary.get("parameter_drift", {})
    if isinstance(param, dict) and "global_rel_l2_to_a" in param:
        out["drift.global_rel_l2_to_a"] = float(param["global_rel_l2_to_a"])
    profiles = summary.get("profiles", {})
    if isinstance(profiles, dict):
        for name, payload in profiles.items():
            if not isinstance(payload, dict):
                continue
            metrics = payload.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            safe = _sanitize_name(name)
            if "b_minus_a_psnr_vs_gt" in metrics:
                out[f"drift.{safe}.b_minus_a_psnr_vs_gt"] = float(metrics["b_minus_a_psnr_vs_gt"])
            if "compensation_index" in metrics:
                out[f"drift.{safe}.compensation_index"] = float(metrics["compensation_index"])
    return out


def _csv_write(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _build_benchmark_cmd(case: Case, args: argparse.Namespace) -> Tuple[List[str], Path]:
    bench_out = case.case_dir / "benchmark"
    bench_out.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        str(args.python_bin),
        str(_resolve_existing_path(args.benchmark_script)),
        "--dataset",
        str(args.dataset),
        "--scene",
        str(args.scene),
        "--checkpoint",
        str(case.checkpoint),
        "--output-dir",
        str(bench_out),
        "--route",
        str(args.route),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--fps",
        str(args.fps),
        "--render-scale",
        str(args.render_scale),
        "--field-res",
        str(args.field_res),
        "--field-knn",
        str(args.field_knn),
        "--weight-eps",
        str(args.weight_eps),
        "--grid-chunk",
        str(args.grid_chunk),
        "--warmup-frames",
        str(args.warmup_frames),
        "--benchmark-frames",
        str(args.benchmark_frames),
        "--frame-start",
        str(args.frame_start),
        "--frame-step",
        str(args.frame_step),
        "--device",
        str(args.device),
        "--save-frame-metrics-every",
        str(max(1, int(args.save_frame_metrics_every))),
    ]

    if args.max_frames is not None:
        cmd.extend(["--max-frames", str(args.max_frames)])
    if args.falcor_python_path:
        cmd.extend(["--falcor-python-path", str(args.falcor_python_path)])
    if args.falcor_python_bin:
        cmd.extend(["--falcor-python-bin", str(args.falcor_python_bin)])
    if args.top_k is not None:
        cmd.extend(["--top-k", str(args.top_k)])
    if args.benchmark_training_soft_routing is not None:
        cmd.append(
            "--training-soft-routing"
            if bool(args.benchmark_training_soft_routing)
            else "--no-training-soft-routing"
        )
    if args.benchmark_routing_soft_topk is not None:
        cmd.extend(["--routing-soft-topk", str(int(args.benchmark_routing_soft_topk))])
    if args.benchmark_routing_temperature is not None:
        cmd.extend(["--routing-temperature", str(float(args.benchmark_routing_temperature))])
    if str(args.benchmark_routing_profile_json or "").strip():
        cmd.extend(["--routing-profile-json", str(args.benchmark_routing_profile_json)])

    if bool(args.sync_gpu):
        cmd.append("--sync-gpu")
    else:
        cmd.append("--no-sync-gpu")
        cmd.extend(["--sync-every", str(max(0, int(args.sync_every)))])

    if bool(args.compute_image_metrics):
        cmd.append("--compute-image-metrics")
    if bool(args.pt_reference):
        cmd.append("--pt-reference")
        cmd.extend(["--pt-spp", str(max(1, int(args.pt_spp)))])
        cmd.extend(["--pt-bounces", str(max(0, int(args.pt_bounces)))])
        cmd.append("--pt-use-nee" if bool(args.pt_use_nee) else "--pt-no-nee")
    if bool(args.save_sampled_images):
        cmd.extend(["--save-sampled-images-dir", str(bench_out / "sampled_images")])
    if args.roi:
        cmd.extend(["--roi", str(args.roi)])
    if args.benchmark_extra:
        cmd.extend(list(args.benchmark_extra))

    summary_path = bench_out / "benchmark_summary.json"
    return cmd, summary_path


def _run_benchmark_case(case: Case, args: argparse.Namespace) -> Dict[str, Any]:
    cmd, summary_path = _build_benchmark_cmd(case, args)
    log_path = case.case_dir / "logs" / "benchmark.log"
    result = _run_command(cmd, log_path)
    payload: Dict[str, Any] = {
        "checkpoint": str(case.checkpoint),
        "case_name": case.name,
        "ok": bool(result.ok),
        "return_code": int(result.return_code),
        "elapsed_sec": float(result.elapsed_sec),
        "log_path": str(result.log_path),
        "summary_path": str(summary_path),
        "metrics": {},
        "error": result.error,
    }
    if result.ok and summary_path.exists():
        summary = _read_json(summary_path)
        payload["metrics"] = _extract_benchmark_metrics(summary)
    elif result.ok and not summary_path.exists():
        payload["ok"] = False
        payload["error"] = f"benchmark summary missing: {summary_path}"
    return payload


def _run_expert_util_for_case(case: Case, args: argparse.Namespace, profiles_json: str) -> Dict[str, Any]:
    script = _resolve_existing_path("3_experiments/scripts/analysis/run_expert_utilization_audit.py")
    out_path = case.case_dir / "diagnostics" / "expert_utilization_summary.json"
    cmd = [
        str(args.python_bin),
        str(script),
        "--data-root",
        str(args.dataset),
        "--checkpoint",
        str(case.checkpoint),
        "--output",
        str(out_path),
        "--split",
        str(args.audit_split),
        "--device",
        str(args.audit_device),
        "--batch-size",
        str(args.audit_batch_size),
        "--num-workers",
        str(args.audit_num_workers),
        "--seed",
        str(args.seed),
        "--top-k",
        str(args.top_k if args.top_k is not None else 3),
        "--max-samples",
        str(args.audit_max_samples),
        "--sample-seed",
        str(args.audit_sample_seed),
        "--profiles-json",
        profiles_json,
    ]
    if args.train_ratio is not None:
        cmd.extend(["--train-ratio", str(args.train_ratio)])
    if args.val_ratio is not None:
        cmd.extend(["--val-ratio", str(args.val_ratio)])

    log_path = case.case_dir / "logs" / "expert_utilization.log"
    result = _run_command(cmd, log_path)
    payload: Dict[str, Any] = {
        "ok": bool(result.ok),
        "return_code": int(result.return_code),
        "elapsed_sec": float(result.elapsed_sec),
        "log_path": str(log_path),
        "summary_path": str(out_path),
        "metrics": {},
        "error": result.error,
    }
    if result.ok and out_path.exists():
        payload["metrics"] = _extract_expert_metrics(_read_json(out_path))
    return payload


def _run_semantic_drift(
    *,
    name: str,
    checkpoint_a: Path,
    checkpoint_b: Path,
    output_path: Path,
    args: argparse.Namespace,
    profiles_json: str,
) -> Dict[str, Any]:
    script = _resolve_existing_path("3_experiments/scripts/analysis/run_semantic_drift_audit.py")
    cmd = [
        str(args.python_bin),
        str(script),
        "--data-root",
        str(args.dataset),
        "--checkpoint-a",
        str(checkpoint_a),
        "--checkpoint-b",
        str(checkpoint_b),
        "--output",
        str(output_path),
        "--name",
        str(name),
        "--split",
        str(args.audit_split),
        "--device",
        str(args.audit_device),
        "--batch-size",
        str(args.audit_batch_size),
        "--num-workers",
        str(args.audit_num_workers),
        "--seed",
        str(args.seed),
        "--top-k",
        str(args.top_k if args.top_k is not None else 3),
        "--max-samples",
        str(args.audit_max_samples),
        "--sample-seed",
        str(args.audit_sample_seed),
        "--profiles-json",
        profiles_json,
    ]
    if args.train_ratio is not None:
        cmd.extend(["--train-ratio", str(args.train_ratio)])
    if args.val_ratio is not None:
        cmd.extend(["--val-ratio", str(args.val_ratio)])

    log_path = output_path.with_suffix(".log")
    result = _run_command(cmd, log_path)
    payload: Dict[str, Any] = {
        "name": name,
        "checkpoint_a": str(checkpoint_a),
        "checkpoint_b": str(checkpoint_b),
        "ok": bool(result.ok),
        "return_code": int(result.return_code),
        "elapsed_sec": float(result.elapsed_sec),
        "log_path": str(log_path),
        "summary_path": str(output_path),
        "metrics": {},
        "error": result.error,
    }
    if result.ok and output_path.exists():
        payload["metrics"] = _extract_drift_metrics(_read_json(output_path))
    return payload


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="One-shot suite: parallel benchmark + expert utilization + semantic drift"
    )
    p.add_argument("--dataset", required=True, help="Dataset root (or parametric_tensor.npz)")
    p.add_argument("--scene", required=True, help="Falcor scene path (.pyscene)")
    p.add_argument("--output-root", required=True, help="Suite output root directory")

    p.add_argument("--checkpoint", action="append", default=[], help="Checkpoint path; can repeat")
    p.add_argument("--checkpoint-list", default=None, help="Text file with checkpoint paths, one per line")
    p.add_argument("--checkpoint-glob", action="append", default=[], help="Glob pattern under repo root")
    p.add_argument("--parallel-jobs", type=int, default=1, help="Parallel benchmark workers")

    p.add_argument("--python-bin", default=sys.executable, help="Python binary used to run child scripts")
    p.add_argument("--benchmark-script", default="tools/benchmark_realtime_pipeline.py")
    p.add_argument("--route", choices=["gt", "model", "both"], default="both")
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
    p.add_argument("--falcor-python-path", default=None)
    p.add_argument("--falcor-python-bin", default=None)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--top-k", type=int, default=None)
    bench_soft_group = p.add_mutually_exclusive_group()
    bench_soft_group.add_argument(
        "--benchmark-training-soft-routing",
        dest="benchmark_training_soft_routing",
        action="store_true",
        help="Enable soft routing in benchmark model inference",
    )
    bench_soft_group.add_argument(
        "--benchmark-no-training-soft-routing",
        dest="benchmark_training_soft_routing",
        action="store_false",
        help="Force hard routing in benchmark model inference",
    )
    p.set_defaults(benchmark_training_soft_routing=None)
    p.add_argument(
        "--benchmark-routing-soft-topk",
        type=int,
        default=None,
        help="Soft routing candidate count for benchmark model inference",
    )
    p.add_argument(
        "--benchmark-routing-temperature",
        type=float,
        default=None,
        help="Soft routing temperature for benchmark model inference",
    )
    p.add_argument(
        "--benchmark-routing-profile-json",
        default="",
        help=(
            "Routing profile JSON forwarded to benchmark; supports dict/list/"
            "{profiles:[...]} and benchmark uses the first profile"
        ),
    )
    p.add_argument("--sync-gpu", action="store_true", default=False)
    p.add_argument("--sync-every", type=int, default=0)
    p.add_argument("--compute-image-metrics", action="store_true", default=True)
    p.add_argument("--save-frame-metrics-every", type=int, default=1)
    p.add_argument("--pt-reference", action="store_true", default=False)
    p.add_argument("--pt-spp", type=int, default=8)
    p.add_argument("--pt-bounces", type=int, default=4)
    p.add_argument("--pt-use-nee", action="store_true", default=True)
    p.add_argument("--pt-no-nee", action="store_false", dest="pt_use_nee")
    p.add_argument("--save-sampled-images", action="store_true", default=False)
    p.add_argument("--roi", default=None)
    p.add_argument("--benchmark-extra", nargs=argparse.REMAINDER, default=None)

    p.add_argument("--skip-benchmark", action="store_true", default=False)
    p.add_argument("--skip-expert-utilization", action="store_true", default=False)
    p.add_argument("--skip-semantic-drift", action="store_true", default=False)

    p.add_argument("--audit-split", choices=["train", "val", "test"], default="test")
    p.add_argument("--audit-device", default="cpu")
    p.add_argument("--audit-batch-size", type=int, default=512)
    p.add_argument("--audit-num-workers", type=int, default=0)
    p.add_argument("--audit-max-samples", type=int, default=8192)
    p.add_argument("--audit-sample-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=None)
    p.add_argument("--val-ratio", type=float, default=None)
    p.add_argument(
        "--profiles-json",
        default="",
        help="Route profiles JSON for expert/drift audits. Empty => hard3 + soft8_t020",
    )

    p.add_argument(
        "--semantic-reference",
        default=None,
        help="Reference checkpoint for semantic drift. Default: first checkpoint in suite.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    checkpoint_paths = _collect_checkpoint_paths(
        checkpoint_args=list(args.checkpoint or []),
        checkpoint_list=args.checkpoint_list,
        checkpoint_globs=list(args.checkpoint_glob or []),
    )
    if not checkpoint_paths:
        raise ValueError("No checkpoints resolved. Use --checkpoint / --checkpoint-list / --checkpoint-glob")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = _resolve_existing_path(args.output_root) if Path(args.output_root).exists() else (ROOT / args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    suite_root = out_root / f"full_suite_{stamp}"
    suite_root.mkdir(parents=True, exist_ok=True)

    cases = _build_cases(checkpoint_paths, suite_root / "cases")
    profiles_json = _ensure_profiles_json(args.profiles_json)

    benchmark_results: Dict[str, Dict[str, Any]] = {}
    if not bool(args.skip_benchmark):
        workers = max(1, int(args.parallel_jobs))
        print(f"[suite] running benchmark for {len(cases)} checkpoints with parallel_jobs={workers}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            fut_map = {ex.submit(_run_benchmark_case, case, args): case for case in cases}
            for fut in concurrent.futures.as_completed(fut_map):
                case = fut_map[fut]
                try:
                    payload = fut.result()
                except Exception as e:
                    payload = {
                        "checkpoint": str(case.checkpoint),
                        "case_name": case.name,
                        "ok": False,
                        "return_code": -1,
                        "elapsed_sec": 0.0,
                        "log_path": str(case.case_dir / "logs" / "benchmark.log"),
                        "summary_path": str(case.case_dir / "benchmark" / "benchmark_summary.json"),
                        "metrics": {},
                        "error": str(e),
                    }
                benchmark_results[case.name] = payload
                status = "ok" if bool(payload.get("ok", False)) else "fail"
                print(f"[benchmark:{status}] {case.name}")

    expert_results: Dict[str, Dict[str, Any]] = {}
    if not bool(args.skip_expert_utilization):
        print(f"[suite] running expert utilization audit for {len(cases)} checkpoints")
        for case in cases:
            payload = _run_expert_util_for_case(case, args, profiles_json)
            expert_results[case.name] = payload
            status = "ok" if bool(payload.get("ok", False)) else "fail"
            print(f"[expert:{status}] {case.name}")

    drift_results: Dict[str, Dict[str, Any]] = {}
    if not bool(args.skip_semantic_drift):
        if args.semantic_reference is not None:
            ref_path = _resolve_existing_path(str(args.semantic_reference))
        else:
            ref_path = cases[0].checkpoint
        ref_name = _sanitize_name(ref_path.stem)
        drift_dir = suite_root / "semantic_drift"
        drift_dir.mkdir(parents=True, exist_ok=True)

        print(f"[suite] running semantic drift against reference: {ref_path}")
        for case in cases:
            if str(case.checkpoint) == str(ref_path):
                continue
            pair_name = f"{ref_name}_to_{_sanitize_name(case.name)}"
            out_path = drift_dir / f"{pair_name}.json"
            payload = _run_semantic_drift(
                name=pair_name,
                checkpoint_a=ref_path,
                checkpoint_b=case.checkpoint,
                output_path=out_path,
                args=args,
                profiles_json=profiles_json,
            )
            drift_results[case.name] = payload
            status = "ok" if bool(payload.get("ok", False)) else "fail"
            print(f"[drift:{status}] {case.name}")

    summary_cases: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    for case in cases:
        bench = benchmark_results.get(case.name, {})
        expert = expert_results.get(case.name, {})
        drift = drift_results.get(case.name, {})
        payload = {
            "case_name": case.name,
            "checkpoint": str(case.checkpoint),
            "benchmark": bench,
            "expert_utilization": expert,
            "semantic_drift_vs_reference": drift,
        }
        summary_cases.append(payload)

        row: Dict[str, Any] = {
            "case_name": case.name,
            "checkpoint": str(case.checkpoint),
            "benchmark_ok": bool(bench.get("ok", False)),
            "expert_ok": bool(expert.get("ok", False)),
            "drift_ok": bool(drift.get("ok", False)) if drift else None,
        }
        for k, v in (bench.get("metrics", {}) or {}).items():
            row[f"bench.{k}"] = v
        for k, v in (expert.get("metrics", {}) or {}).items():
            row[k] = v
        for k, v in (drift.get("metrics", {}) or {}).items():
            row[k] = v
        csv_rows.append(row)

    summary = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "suite_root": str(suite_root),
            "dataset": str(args.dataset),
            "scene": str(args.scene),
            "num_checkpoints": int(len(cases)),
            "parallel_jobs": int(max(1, int(args.parallel_jobs))),
            "profiles_json": json.loads(profiles_json),
            "benchmark_routing_request": {
                "top_k": int(args.top_k) if args.top_k is not None else None,
                "training_soft_routing": (
                    bool(args.benchmark_training_soft_routing)
                    if args.benchmark_training_soft_routing is not None
                    else None
                ),
                "routing_soft_topk": (
                    int(args.benchmark_routing_soft_topk)
                    if args.benchmark_routing_soft_topk is not None
                    else None
                ),
                "routing_temperature": (
                    float(args.benchmark_routing_temperature)
                    if args.benchmark_routing_temperature is not None
                    else None
                ),
                "routing_profile_json": (
                    json.loads(args.benchmark_routing_profile_json)
                    if str(args.benchmark_routing_profile_json or "").strip()
                    else None
                ),
            },
        },
        "cases": summary_cases,
    }

    summary_json = suite_root / "suite_summary.json"
    summary_csv = suite_root / "suite_summary.csv"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _csv_write(summary_csv, csv_rows)

    print(f"Saved suite summary: {summary_json}")
    print(f"Saved suite csv: {summary_csv}")


if __name__ == "__main__":
    main()
