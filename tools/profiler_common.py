"""Shared utilities for training/inference profiling scripts."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np


@dataclass
class Thresholds:
    op_top3_recommend_pct: float = 60.0
    train_non_kernel_defer_pct: float = 40.0
    pipeline_model_share_min_pct: float = 35.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_repo_import_paths() -> tuple[Path, Path, Path]:
    root = repo_root()
    src = root / "2_src"
    scripts = root / "3_experiments" / "scripts"
    for path in (root, src, scripts):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return root, src, scripts


def build_run_dir(base_dir: str | Path, tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(base_dir) / f"{tag}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def export_json(path: str | Path, payload: Dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def export_csv(path: str | Path, rows: Iterable[Dict[str, Any]]) -> Path:
    rows = list(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return out
    fields = sorted({k for row in rows for k in row.keys()})
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return out


def summarize_latency(samples_ms: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(samples_ms, dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def build_nsys_command(
    target_cmd: Sequence[str],
    output_prefix: str | Path,
    trace: str = "cuda,nvtx,osrt",
    sample: str = "none",
) -> list[str]:
    return [
        "nsys",
        "profile",
        "-o",
        str(output_prefix),
        "--trace",
        str(trace),
        "--sample",
        str(sample),
        *list(target_cmd),
    ]


def build_ncu_command(
    target_cmd: Sequence[str],
    output_path: str | Path,
    set_name: str = "launchStats",
) -> list[str]:
    return [
        "ncu",
        "--set",
        str(set_name),
        "--target-processes",
        "all",
        "--export",
        str(output_path),
        *list(target_cmd),
    ]


def _safe_pct(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(100.0 * num / den)


def recommend_cuda(profile_bundle: Dict[str, Any], thresholds: Thresholds | None = None) -> Dict[str, Any]:
    thresholds = thresholds or Thresholds()

    train = profile_bundle.get("training", {}) if isinstance(profile_bundle, dict) else {}
    infer_model = profile_bundle.get("inference_model", {}) if isinstance(profile_bundle, dict) else {}
    pipeline = profile_bundle.get("pipeline", {}) if isinstance(profile_bundle, dict) else {}

    top3_pct = float(infer_model.get("top3_ops_pct", train.get("top3_ops_pct", 0.0)) or 0.0)
    train_non_kernel = float(train.get("dataloader_wait_pct", 0.0) or 0.0) + float(
        train.get("python_overhead_pct", 0.0) or 0.0
    )
    model_share_pct = float(pipeline.get("model_share_pct", 0.0) or 0.0)

    reasons: list[str] = []
    decision = "defer"

    if model_share_pct > 0.0 and model_share_pct < thresholds.pipeline_model_share_min_pct:
        decision = "not_needed"
        reasons.append(
            f"pipeline model share {model_share_pct:.2f}% < {thresholds.pipeline_model_share_min_pct:.2f}%"
        )
    elif train_non_kernel > thresholds.train_non_kernel_defer_pct:
        decision = "defer"
        reasons.append(
            f"training non-kernel overhead {train_non_kernel:.2f}% > {thresholds.train_non_kernel_defer_pct:.2f}%"
        )
    elif top3_pct >= thresholds.op_top3_recommend_pct:
        decision = "recommend"
        reasons.append(f"top-3 ops concentration {top3_pct:.2f}% >= {thresholds.op_top3_recommend_pct:.2f}%")
    else:
        decision = "defer"
        reasons.append(
            f"top-3 ops concentration {top3_pct:.2f}% < {thresholds.op_top3_recommend_pct:.2f}%"
        )

    return {
        "decision": decision,
        "reasons": reasons,
        "metrics": {
            "top3_ops_pct": top3_pct,
            "training_non_kernel_pct": train_non_kernel,
            "pipeline_model_share_pct": model_share_pct,
        },
        "thresholds": {
            "op_top3_recommend_pct": thresholds.op_top3_recommend_pct,
            "train_non_kernel_defer_pct": thresholds.train_non_kernel_defer_pct,
            "pipeline_model_share_min_pct": thresholds.pipeline_model_share_min_pct,
        },
    }


def topk_ops_by_cuda_time(key_averages: Any, limit: int = 20) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for event in key_averages:
        rows.append(
            {
                "name": str(getattr(event, "key", "")),
                "self_cpu_time_total": float(getattr(event, "self_cpu_time_total", 0.0)),
                "self_cuda_time_total": float(getattr(event, "self_cuda_time_total", 0.0)),
                "cpu_time_total": float(getattr(event, "cpu_time_total", 0.0)),
                "cuda_time_total": float(getattr(event, "cuda_time_total", 0.0)),
                "count": int(getattr(event, "count", 0)),
            }
        )
    rows.sort(key=lambda r: r.get("self_cuda_time_total", 0.0), reverse=True)
    return rows[: max(1, int(limit))]


def top3_concentration_pct(op_rows: Sequence[Dict[str, Any]]) -> float:
    if not op_rows:
        return 0.0
    total = sum(float(r.get("self_cuda_time_total", 0.0)) for r in op_rows)
    top3 = sum(float(r.get("self_cuda_time_total", 0.0)) for r in op_rows[:3])
    return _safe_pct(top3, total)

