#!/usr/bin/env python3
"""Build a unified experiment-vs-baseline report.

Report focus:
1) Whether experiment shrinks Oracle gap (all27 / non-L0)
2) How much PSNR/SSIM improves across unified benchmark domains
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def _read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be object: {p}")
    return data


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _extract_oracle_gap(oracle_summary: Dict[str, Any]) -> Dict[str, float]:
    delta = oracle_summary.get("delta", {})
    if not isinstance(delta, dict):
        raise ValueError("oracle summary missing delta block")

    gap_all27 = _safe_float(delta.get("all27_psnr_gain_db"))
    gap_non_l0 = _safe_float(delta.get("non_l0_psnr_gain_db"))
    if gap_all27 is None or gap_non_l0 is None:
        raise ValueError("oracle delta missing all27_psnr_gain_db/non_l0_psnr_gain_db")

    return {
        "all27_gap_db": float(gap_all27),
        "non_l0_gap_db": float(gap_non_l0),
    }


def _extract_metric_means(benchmark_summary: Dict[str, Any]) -> Dict[str, float]:
    image_metrics = benchmark_summary.get("image_metrics", {})
    if not isinstance(image_metrics, dict):
        image_metrics = {}

    metrics = {}
    for k, v in image_metrics.items():
        if not str(k).startswith("mean_"):
            continue
        val = _safe_float(v)
        if val is not None:
            metrics[str(k)[5:]] = float(val)
    return metrics


def _compute_delta_map(base: Dict[str, float], cur: Dict[str, float], suffix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    common = sorted(set(base.keys()) & set(cur.keys()))
    for key in common:
        if not key.endswith(suffix):
            continue
        out[key] = float(cur[key] - base[key])
    return out


def build_report(
    *,
    baseline_oracle_summary: Dict[str, Any],
    experiment_oracle_summary: Dict[str, Any],
    baseline_benchmark_summary: Dict[str, Any],
    experiment_benchmark_summary: Dict[str, Any],
    baseline_refs: Dict[str, str],
    experiment_refs: Dict[str, str],
) -> Dict[str, Any]:
    base_gap = _extract_oracle_gap(baseline_oracle_summary)
    exp_gap = _extract_oracle_gap(experiment_oracle_summary)

    base_all = float(base_gap["all27_gap_db"])
    exp_all = float(exp_gap["all27_gap_db"])
    base_non_l0 = float(base_gap["non_l0_gap_db"])
    exp_non_l0 = float(exp_gap["non_l0_gap_db"])

    shrink_all = float(base_all - exp_all)
    shrink_non_l0 = float(base_non_l0 - exp_non_l0)
    shrink_ratio_all = float(1.0 - (exp_all / base_all)) if abs(base_all) > 1e-12 else float("nan")
    shrink_ratio_non_l0 = float(1.0 - (exp_non_l0 / base_non_l0)) if abs(base_non_l0) > 1e-12 else float("nan")

    base_means = _extract_metric_means(baseline_benchmark_summary)
    exp_means = _extract_metric_means(experiment_benchmark_summary)

    psnr_delta = _compute_delta_map(base_means, exp_means, "_psnr")
    ssim_delta = _compute_delta_map(base_means, exp_means, "_ssim")

    improved_psnr_domains = [k for k, v in sorted(psnr_delta.items()) if v > 0.0]
    improved_ssim_domains = [k for k, v in sorted(ssim_delta.items()) if v > 0.0]

    report = {
        "baseline_refs": baseline_refs,
        "experiment_refs": experiment_refs,
        "oracle_gap": {
            "baseline_all27_gap_db": base_all,
            "experiment_all27_gap_db": exp_all,
            "gap_shrink_all27_db": shrink_all,
            "gap_shrink_ratio_all27": shrink_ratio_all,
            "baseline_non_l0_gap_db": base_non_l0,
            "experiment_non_l0_gap_db": exp_non_l0,
            "gap_shrink_non_l0_db": shrink_non_l0,
            "gap_shrink_ratio_non_l0": shrink_ratio_non_l0,
        },
        "psnr_gain_vs_baseline": psnr_delta,
        "ssim_gain_vs_baseline": ssim_delta,
        "final_diagnosis": {
            "oracle_gap_improved": bool(shrink_all > 0.0 or shrink_non_l0 > 0.0),
            "oracle_gap_improved_all27": bool(shrink_all > 0.0),
            "oracle_gap_improved_non_l0": bool(shrink_non_l0 > 0.0),
            "render_psnr_improved_domains": improved_psnr_domains,
            "render_ssim_improved_domains": improved_ssim_domains,
            "notes": "Positive gap_shrink means experiment is closer to Oracle upper bound.",
        },
    }
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Report experiment improvements vs baseline")
    p.add_argument("--baseline-oracle-summary", required=True)
    p.add_argument("--experiment-oracle-summary", required=True)
    p.add_argument("--baseline-benchmark-summary", required=True)
    p.add_argument("--experiment-benchmark-summary", required=True)
    p.add_argument("--output-json", required=True)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    baseline_oracle = _read_json(args.baseline_oracle_summary)
    experiment_oracle = _read_json(args.experiment_oracle_summary)
    baseline_benchmark = _read_json(args.baseline_benchmark_summary)
    experiment_benchmark = _read_json(args.experiment_benchmark_summary)

    report = build_report(
        baseline_oracle_summary=baseline_oracle,
        experiment_oracle_summary=experiment_oracle,
        baseline_benchmark_summary=baseline_benchmark,
        experiment_benchmark_summary=experiment_benchmark,
        baseline_refs={
            "oracle_summary": str(args.baseline_oracle_summary),
            "benchmark_summary": str(args.baseline_benchmark_summary),
        },
        experiment_refs={
            "oracle_summary": str(args.experiment_oracle_summary),
            "benchmark_summary": str(args.experiment_benchmark_summary),
        },
    )

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved experiment report: {out}")


if __name__ == "__main__":
    main()
