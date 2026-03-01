#!/usr/bin/env python3
"""Analyze correlation between SH metric gains and rendering metric gains.

Supports input as either:
- CSV with frame-wise metrics
- JSON summary containing image_metrics.per_sample (e.g. falcor_summary.json)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def _read_rows(path: Path) -> List[Dict[str, float]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        per = data.get("image_metrics", {}).get("per_sample", [])
        rows: List[Dict[str, float]] = []
        for r in per:
            out: Dict[str, float] = {}
            for k, v in r.items():
                try:
                    out[k] = float(v)
                except Exception:
                    continue
            rows.append(out)
        return rows

    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out: Dict[str, float] = {}
            for k, v in r.items():
                if v is None or v == "":
                    continue
                try:
                    out[k] = float(v)
                except ValueError:
                    continue
            rows.append(out)
    return rows


def _index_by_frame(rows: List[Dict[str, float]]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for r in rows:
        if "frame" not in r:
            continue
        out[int(r["frame"])] = r
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x0 = x - x.mean()
    y0 = y - y.mean()
    den = float(np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0)))
    if den <= 1e-12:
        return float("nan")
    return float(np.sum(x0 * y0) / den)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return _pearson(rx.astype(np.float64), ry.astype(np.float64))


def _metric_relation_label(pearson: float, mean_delta: float) -> str:
    if np.isnan(pearson):
        return "inconclusive"
    if pearson >= 0.5 and mean_delta > 0:
        return "positive-coupled"
    if pearson <= -0.3 and mean_delta > 0:
        return "trade-off"
    if mean_delta <= 0:
        return "no-improvement"
    return "weak-positive"


def _load_sh_delta_from_rows(
    baseline_map: Dict[int, Dict[str, float]],
    oracle_map: Dict[int, Dict[str, float]],
    frames: List[int],
) -> np.ndarray:
    sh_key = "sh_psnr"
    if all((sh_key in baseline_map[f] and sh_key in oracle_map[f]) for f in frames):
        return np.asarray([oracle_map[f][sh_key] - baseline_map[f][sh_key] for f in frames], dtype=np.float64)
    raise RuntimeError("sh_psnr not found in baseline/oracle rows")


def _load_sh_delta_from_external_csv(path: Path, frames: List[int]) -> np.ndarray:
    rows = _read_rows(path)
    m = _index_by_frame(rows)
    if not all(f in m for f in frames):
        missing = [f for f in frames if f not in m][:10]
        raise RuntimeError(f"External SH CSV missing frames, examples: {missing}")
    if not all("delta_sh_psnr" in m[f] for f in frames):
        raise RuntimeError("External SH CSV missing delta_sh_psnr column")
    return np.asarray([m[f]["delta_sh_psnr"] for f in frames], dtype=np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze SH gain vs rendering gain correlation")
    p.add_argument("--baseline-metrics", required=True, help="CSV or JSON summary (baseline)")
    p.add_argument("--oracle-metrics", required=True, help="CSV or JSON summary (oracle)")
    p.add_argument("--sh-delta-csv", default=None, help="Optional per-frame SH CSV with delta_sh_psnr")
    p.add_argument("--output-json", required=True)
    args = p.parse_args()

    base_rows = _read_rows(Path(args.baseline_metrics))
    oracle_rows = _read_rows(Path(args.oracle_metrics))
    base_map = _index_by_frame(base_rows)
    oracle_map = _index_by_frame(oracle_rows)

    common_frames = sorted(set(base_map.keys()) & set(oracle_map.keys()))
    if not common_frames:
        raise RuntimeError("No common frames found between baseline and oracle metrics")

    if args.sh_delta_csv:
        d_sh = _load_sh_delta_from_external_csv(Path(args.sh_delta_csv), common_frames)
        sh_source = str(args.sh_delta_csv)
    else:
        d_sh = _load_sh_delta_from_rows(base_map, oracle_map, common_frames)
        sh_source = "from_metrics_sh_psnr"

    target_keys = [
        "benchmark_psnr",
        "benchmark_linear_psnr",
        "hdr_psnr",
        "fixedtm_psnr",
        "real_render_hdr_psnr",
        "pt_psnr_model",
        "pt_linear_psnr_model",
    ]

    available_keys = [k for k in target_keys if all((k in base_map[f] and k in oracle_map[f]) for f in common_frames)]
    if not available_keys:
        raise RuntimeError("No rendering metric columns found in both inputs")

    relations: Dict[str, Dict[str, float | str]] = {}
    for k in available_keys:
        d_k = np.asarray([oracle_map[f][k] - base_map[f][k] for f in common_frames], dtype=np.float64)
        pear = _pearson(d_sh, d_k)
        spear = _spearman(d_sh, d_k)
        relations[k] = {
            "mean_delta": float(np.mean(d_k)),
            "median_delta": float(np.median(d_k)),
            "pearson_with_sh_delta": float(pear),
            "spearman_with_sh_delta": float(spear),
            "positive_frame_ratio": float(np.mean(d_k > 0.0)),
            "label": _metric_relation_label(float(pear), float(np.mean(d_k))),
        }

    overall = {
        "frames_compared": int(len(common_frames)),
        "sh_psnr_delta_mean": float(np.mean(d_sh)),
        "sh_psnr_delta_median": float(np.median(d_sh)),
        "sh_psnr_positive_frame_ratio": float(np.mean(d_sh > 0.0)),
        "sh_delta_source": sh_source,
    }

    report = {
        "baseline_metrics": str(args.baseline_metrics),
        "oracle_metrics": str(args.oracle_metrics),
        "overall": overall,
        "relations": relations,
        "interpretation": {
            "note": "positive-coupled means SH improvements align with rendering improvements; trade-off means SH up but rendering down.",
        },
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved correlation report: {out}")


if __name__ == "__main__":
    main()
