#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _to_float_if_finite(value: Any) -> float | None:
    try:
        value_f = float(value)
    except Exception:
        return None
    if not np.isfinite(value_f):
        return None
    return float(value_f)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return _pearson(xr, yr)


def _is_loss_like_metric(metric_name: str) -> bool:
    k = metric_name.lower()
    if not k.startswith("train_") and k not in {"val_mae", "val_img_psnr"}:
        return False
    tokens = [
        "loss",
        "total",
        "recon",
        "image",
        "gbuffer",
        "linearity",
        "spatial",
        "routing_balance",
        "coeff_l1",
        "temporal",
        "lambda",
        "weight",
    ]
    return any(token in k for token in tokens)


def _build_phase_df(phase_path: Path) -> pd.DataFrame:
    rows = _load_jsonl(phase_path)
    data: List[Dict[str, float]] = []
    for row in rows:
        if str(row.get("kind", "")).strip() != "epoch_end":
            continue
        epoch = int(row.get("epoch", -1))
        metrics = row.get("metrics", {})
        if epoch <= 0 or not isinstance(metrics, dict):
            continue
        out: Dict[str, float] = {"epoch": float(epoch)}
        for k, v in metrics.items():
            v_f = _to_float_if_finite(v)
            if v_f is None:
                continue
            out[str(k)] = v_f
        data.append(out)
    if not data:
        return pd.DataFrame(columns=["epoch"])
    df = pd.DataFrame(data)
    df = df.sort_values("epoch").drop_duplicates(subset=["epoch"], keep="last")
    return df


def _build_falcor_df(
    falcor_path: Path,
    *,
    target_metric: str,
    include_stage_end_full: bool,
) -> pd.DataFrame:
    rows = _load_jsonl(falcor_path)
    data: List[Dict[str, float]] = []
    allowed_kinds = {"periodic"}
    if include_stage_end_full:
        allowed_kinds.add("stage_end_full")
    for row in rows:
        kind = str(row.get("kind", "")).strip()
        if kind not in allowed_kinds:
            continue
        epoch = int(row.get("epoch", -1))
        if epoch <= 0:
            continue
        metric_value = _to_float_if_finite(row.get("metric_value", None))
        metrics = row.get("metrics", {})
        if metric_value is None and isinstance(metrics, dict):
            metric_value = _to_float_if_finite(metrics.get(target_metric, None))
        if metric_value is None:
            continue
        data.append(
            {
                "epoch": float(epoch),
                "falcor_psnr": float(metric_value),
            }
        )
    if not data:
        return pd.DataFrame(columns=["epoch", "falcor_psnr"])
    df = pd.DataFrame(data)
    df = df.sort_values("epoch").drop_duplicates(subset=["epoch"], keep="last")
    return df


def _collect_candidate_metrics(
    phase_df: pd.DataFrame,
    *,
    metric_regex: str | None,
    include_all_numeric: bool,
) -> List[str]:
    candidates = [c for c in phase_df.columns if c != "epoch"]
    if include_all_numeric:
        pass
    else:
        candidates = [c for c in candidates if _is_loss_like_metric(c)]
    if metric_regex:
        pattern = re.compile(metric_regex)
        candidates = [c for c in candidates if pattern.search(c)]
    return sorted(candidates)


def _compute_sparse_pairs(
    phase_df: pd.DataFrame,
    falcor_df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    if metric not in phase_df.columns:
        return pd.DataFrame(columns=["epoch", metric, "falcor_psnr"])
    merged = phase_df[["epoch", metric]].merge(
        falcor_df[["epoch", "falcor_psnr"]],
        on="epoch",
        how="inner",
    )
    merged = merged.dropna(subset=[metric, "falcor_psnr"])
    return merged


def _compute_dense_pairs(
    phase_df: pd.DataFrame,
    falcor_df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    if metric not in phase_df.columns:
        return pd.DataFrame(columns=["epoch", metric, "falcor_psnr"])
    if len(falcor_df) < 2:
        return pd.DataFrame(columns=["epoch", metric, "falcor_psnr"])

    phase_epoch = phase_df["epoch"].to_numpy(dtype=float)
    falcor_epoch = falcor_df["epoch"].to_numpy(dtype=float)
    falcor_val = falcor_df["falcor_psnr"].to_numpy(dtype=float)

    e_min = int(max(np.min(phase_epoch), np.min(falcor_epoch)))
    e_max = int(min(np.max(phase_epoch), np.max(falcor_epoch)))
    if e_max - e_min < 1:
        return pd.DataFrame(columns=["epoch", metric, "falcor_psnr"])

    all_epoch = np.arange(e_min, e_max + 1, dtype=float)
    interp_falcor = np.interp(all_epoch, falcor_epoch, falcor_val)

    metric_series = phase_df.set_index("epoch")[metric]
    metric_val: List[float] = []
    use_epoch: List[float] = []
    use_falcor: List[float] = []
    for e, f in zip(all_epoch, interp_falcor):
        if e not in metric_series.index:
            continue
        m = _to_float_if_finite(metric_series.loc[e])
        if m is None:
            continue
        metric_val.append(m)
        use_epoch.append(float(e))
        use_falcor.append(float(f))

    if not use_epoch:
        return pd.DataFrame(columns=["epoch", metric, "falcor_psnr"])
    return pd.DataFrame(
        {
            "epoch": use_epoch,
            metric: metric_val,
            "falcor_psnr": use_falcor,
        }
    )


def _analyze_one_experiment(
    exp_dir: Path,
    *,
    target_metric: str,
    include_stage_end_full: bool,
    min_pairs: int,
    metric_regex: str | None,
    include_all_numeric: bool,
    output_dir: Path,
) -> Dict[str, Any]:
    phase_path = exp_dir / "runtime" / "phase0_metrics.jsonl"
    falcor_path = exp_dir / "runtime" / "falcor_periodic_eval.jsonl"
    if not phase_path.exists() or not falcor_path.exists():
        return {
            "experiment": exp_dir.name,
            "status": "skip_missing_logs",
            "phase0_path": str(phase_path),
            "falcor_path": str(falcor_path),
        }

    phase_df = _build_phase_df(phase_path)
    falcor_df = _build_falcor_df(
        falcor_path,
        target_metric=target_metric,
        include_stage_end_full=include_stage_end_full,
    )
    if phase_df.empty or falcor_df.empty:
        return {
            "experiment": exp_dir.name,
            "status": "skip_empty_data",
            "phase_rows": int(len(phase_df)),
            "falcor_rows": int(len(falcor_df)),
        }

    metric_names = _collect_candidate_metrics(
        phase_df,
        metric_regex=metric_regex,
        include_all_numeric=include_all_numeric,
    )

    rows: List[Dict[str, Any]] = []
    for metric in metric_names:
        sparse = _compute_sparse_pairs(phase_df, falcor_df, metric)
        dense = _compute_dense_pairs(phase_df, falcor_df, metric)

        out: Dict[str, Any] = {"metric": metric}
        if len(sparse) >= min_pairs:
            x = sparse[metric].to_numpy(dtype=float)
            y = sparse["falcor_psnr"].to_numpy(dtype=float)
            out["n_sparse"] = int(len(sparse))
            out["pearson_sparse"] = _pearson(x, y)
            out["spearman_sparse"] = _spearman(x, y)
        else:
            out["n_sparse"] = int(len(sparse))
            out["pearson_sparse"] = float("nan")
            out["spearman_sparse"] = float("nan")

        if len(dense) >= min_pairs:
            x = dense[metric].to_numpy(dtype=float)
            y = dense["falcor_psnr"].to_numpy(dtype=float)
            out["n_dense_interp"] = int(len(dense))
            out["pearson_dense_interp"] = _pearson(x, y)
            out["spearman_dense_interp"] = _spearman(x, y)
        else:
            out["n_dense_interp"] = int(len(dense))
            out["pearson_dense_interp"] = float("nan")
            out["spearman_dense_interp"] = float("nan")
        rows.append(out)

    corr_df = pd.DataFrame(rows)
    if corr_df.empty:
        return {
            "experiment": exp_dir.name,
            "status": "skip_no_metrics",
            "phase_rows": int(len(phase_df)),
            "falcor_rows": int(len(falcor_df)),
        }

    score = corr_df["pearson_dense_interp"].abs().fillna(corr_df["pearson_sparse"].abs())
    corr_df = corr_df.assign(abs_corr_score=score)
    corr_df = corr_df.sort_values("abs_corr_score", ascending=False)

    exp_out = output_dir / exp_dir.name
    exp_out.mkdir(parents=True, exist_ok=True)
    corr_csv = exp_out / "loss_falcor_correlation.csv"
    corr_df.to_csv(corr_csv, index=False)

    phase_df.to_csv(exp_out / "phase0_epoch_metrics.csv", index=False)
    falcor_df.to_csv(exp_out / "falcor_epoch_metrics.csv", index=False)

    pos_df = corr_df.dropna(subset=["pearson_dense_interp"]).sort_values("pearson_dense_interp", ascending=False)
    neg_df = corr_df.dropna(subset=["pearson_dense_interp"]).sort_values("pearson_dense_interp", ascending=True)
    top_pos = pos_df.head(10).to_dict(orient="records")
    top_neg = neg_df.head(10).to_dict(orient="records")

    summary = {
        "experiment": exp_dir.name,
        "status": "ok",
        "phase_rows": int(len(phase_df)),
        "falcor_rows": int(len(falcor_df)),
        "metric_count": int(len(corr_df)),
        "phase0_path": str(phase_path),
        "falcor_path": str(falcor_path),
        "target_metric": str(target_metric),
        "include_stage_end_full": bool(include_stage_end_full),
        "top_positive_dense": top_pos,
        "top_negative_dense": top_neg,
        "output_csv": str(corr_csv),
    }
    (exp_out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _resolve_experiment_dirs(args: argparse.Namespace) -> List[Path]:
    exp_dirs: List[Path] = []
    if args.experiment_dir:
        for raw in args.experiment_dir:
            p = Path(raw)
            if not p.is_absolute():
                p = ROOT / p
            exp_dirs.append(p)
    else:
        root = Path(args.experiments_root)
        if not root.is_absolute():
            root = ROOT / root
        for p in sorted(root.glob(args.experiment_glob)):
            if p.is_dir():
                exp_dirs.append(p)
    uniq: List[Path] = []
    seen: set[str] = set()
    for p in exp_dirs:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze optimization correlation between logged losses/metrics and Falcor HDR PSNR."
    )
    parser.add_argument(
        "--experiment-dir",
        action="append",
        default=[],
        help="Experiment directory (repeatable). If omitted, scan experiments-root with experiment-glob.",
    )
    parser.add_argument(
        "--experiments-root",
        default="3_experiments/results/bistro_clean_v2",
        help="Root dir used when experiment-dir is not provided.",
    )
    parser.add_argument(
        "--experiment-glob",
        default="exp_*",
        help="Glob pattern under experiments-root for auto-discovery.",
    )
    parser.add_argument(
        "--target-metric",
        default="mean_real_render_hdr_psnr",
        help="Falcor metric key.",
    )
    parser.add_argument(
        "--include-stage-end-full",
        action="store_true",
        help="Include stage_end_full points in falcor timeline.",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=4,
        help="Minimum aligned samples required before reporting a correlation.",
    )
    parser.add_argument(
        "--metric-regex",
        default=None,
        help="Optional regex to filter candidate metric names.",
    )
    parser.add_argument(
        "--include-all-numeric",
        action="store_true",
        help="Analyze all numeric phase0 metrics instead of loss-like subset.",
    )
    parser.add_argument(
        "--output-dir",
        default="3_experiments/results/loss_falcor_correlation/latest",
        help="Output directory for correlation reports.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_dirs = _resolve_experiment_dirs(args)
    if not exp_dirs:
        raise RuntimeError("No experiment directories found.")

    summaries: List[Dict[str, Any]] = []
    for exp_dir in exp_dirs:
        summaries.append(
            _analyze_one_experiment(
                exp_dir=exp_dir,
                target_metric=str(args.target_metric),
                include_stage_end_full=bool(args.include_stage_end_full),
                min_pairs=max(2, int(args.min_pairs)),
                metric_regex=str(args.metric_regex) if args.metric_regex else None,
                include_all_numeric=bool(args.include_all_numeric),
                output_dir=out_dir,
            )
        )

    summary_json = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_metric": str(args.target_metric),
        "include_stage_end_full": bool(args.include_stage_end_full),
        "min_pairs": int(args.min_pairs),
        "metric_regex": str(args.metric_regex) if args.metric_regex else None,
        "include_all_numeric": bool(args.include_all_numeric),
        "num_experiments": int(len(summaries)),
        "experiments": summaries,
    }
    (out_dir / "summary_all.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(out_dir / "summary_all.csv", index=False)

    print(f"[OK] correlation reports written to: {out_dir}")


if __name__ == "__main__":
    main()

