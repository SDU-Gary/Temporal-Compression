#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


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


def _residualize(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    if z.size == 0:
        return y - np.mean(y)
    x = np.column_stack([np.ones((y.shape[0],), dtype=float), z])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


def _partial_corr(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    rank_mode: bool,
) -> float:
    if x.size < 3 or y.size < 3:
        return float("nan")

    if rank_mode:
        x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
        y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
        if z.size > 0:
            z_rank_cols: List[np.ndarray] = []
            for i in range(z.shape[1]):
                z_rank_cols.append(pd.Series(z[:, i]).rank(method="average").to_numpy(dtype=float))
            z = np.column_stack(z_rank_cols) if z_rank_cols else np.empty((x.shape[0], 0), dtype=float)

    if z.size == 0:
        return _pearson(x, y)
    if x.shape[0] <= z.shape[1] + 2:
        return float("nan")

    rx = _residualize(x, z)
    ry = _residualize(y, z)
    return _pearson(rx, ry)


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
        "contrib",
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
    return df.sort_values("epoch").drop_duplicates(subset=["epoch"], keep="last")


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
        data.append({"epoch": float(epoch), "falcor_psnr": float(metric_value)})
    if not data:
        return pd.DataFrame(columns=["epoch", "falcor_psnr"])
    df = pd.DataFrame(data)
    return df.sort_values("epoch").drop_duplicates(subset=["epoch"], keep="last")


def _collect_candidate_metrics(
    phase_df: pd.DataFrame,
    *,
    metric_regex: str | None,
    include_all_numeric: bool,
) -> List[str]:
    candidates = [c for c in phase_df.columns if c != "epoch"]
    if not include_all_numeric:
        candidates = [c for c in candidates if _is_loss_like_metric(c)]
    if metric_regex:
        pattern = re.compile(metric_regex)
        candidates = [c for c in candidates if pattern.search(c)]
    return sorted(candidates)


def _build_dense_aligned_df(
    phase_df: pd.DataFrame,
    falcor_df: pd.DataFrame,
    *,
    needed_phase_cols: List[str],
) -> pd.DataFrame:
    if len(falcor_df) < 2 or len(phase_df) < 2:
        return pd.DataFrame(columns=["epoch", "falcor_psnr"])

    phase_epoch = phase_df["epoch"].to_numpy(dtype=float)
    falcor_epoch = falcor_df["epoch"].to_numpy(dtype=float)
    falcor_val = falcor_df["falcor_psnr"].to_numpy(dtype=float)

    e_min = int(max(np.min(phase_epoch), np.min(falcor_epoch)))
    e_max = int(min(np.max(phase_epoch), np.max(falcor_epoch)))
    if e_max - e_min < 1:
        return pd.DataFrame(columns=["epoch", "falcor_psnr"])

    dense_epoch = np.arange(e_min, e_max + 1, dtype=float)
    out = pd.DataFrame({"epoch": dense_epoch})
    out["falcor_psnr"] = np.interp(dense_epoch, falcor_epoch, falcor_val)

    phase_idx = phase_df.set_index("epoch")
    for col in needed_phase_cols:
        if col not in phase_idx.columns:
            continue
        vals: List[float] = []
        valid_rows = 0
        for e in dense_epoch:
            v = _to_float_if_finite(phase_idx.at[e, col]) if e in phase_idx.index else None
            vals.append(np.nan if v is None else float(v))
            if v is not None:
                valid_rows += 1
        if valid_rows > 0:
            out[col] = np.asarray(vals, dtype=float)
    return out


def _compute_lag_rows(
    aligned_df: pd.DataFrame,
    *,
    metric: str,
    lag_epochs: int,
    controls: List[str],
    min_pairs: int,
) -> Dict[str, Any]:
    # Positive lag means metric leads Falcor (x_t vs y_{t+lag}).
    df = aligned_df.copy()
    df["falcor_shifted"] = df["falcor_psnr"].shift(periods=-int(lag_epochs))
    # Deduplicate and avoid using the same metric both as target and control.
    controls_uniq = []
    seen_controls: set[str] = set()
    for c in controls:
        c_s = str(c)
        if c_s == metric:
            continue
        if c_s in seen_controls:
            continue
        seen_controls.add(c_s)
        controls_uniq.append(c_s)

    cols = [metric, "falcor_shifted"] + controls_uniq
    seen_cols: set[str] = set()
    cols = [c for c in cols if c in df.columns and not (c in seen_cols or seen_cols.add(c))]
    pair = df[cols].dropna(axis=0, how="any")

    row: Dict[str, Any] = {
        "metric": metric,
        "lag_epochs": int(lag_epochs),
        "n_pairs": int(len(pair)),
        "controls": "|".join(controls),
        "pearson": float("nan"),
        "spearman": float("nan"),
        "partial_pearson": float("nan"),
        "partial_spearman": float("nan"),
    }
    if len(pair) < min_pairs or metric not in pair.columns:
        return row

    x = pair[metric].to_numpy(dtype=float)
    y = pair["falcor_shifted"].to_numpy(dtype=float)

    z_cols = [c for c in controls_uniq if c in pair.columns]
    z = pair[z_cols].to_numpy(dtype=float) if z_cols else np.empty((x.shape[0], 0), dtype=float)

    row["controls"] = "|".join(z_cols)
    row["pearson"] = _pearson(x, y)
    row["spearman"] = _spearman(x, y)
    row["partial_pearson"] = _partial_corr(x, y, z, rank_mode=False)
    row["partial_spearman"] = _partial_corr(x, y, z, rank_mode=True)
    return row


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


def _analyze_one_experiment(
    exp_dir: Path,
    *,
    target_metric: str,
    include_stage_end_full: bool,
    min_pairs: int,
    metric_regex: str | None,
    include_all_numeric: bool,
    max_lag_epochs: int,
    controls: List[str],
    add_epoch_control: bool,
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
    if not metric_names:
        return {
            "experiment": exp_dir.name,
            "status": "skip_no_metrics",
            "phase_rows": int(len(phase_df)),
            "falcor_rows": int(len(falcor_df)),
        }

    controls_norm = [str(c).strip() for c in controls if str(c).strip()]
    if add_epoch_control and "epoch" not in controls_norm:
        controls_norm = ["epoch"] + controls_norm

    needed_cols = sorted(set(metric_names + [c for c in controls_norm if c != "epoch"]))
    aligned_df = _build_dense_aligned_df(
        phase_df=phase_df,
        falcor_df=falcor_df,
        needed_phase_cols=needed_cols,
    )
    if aligned_df.empty:
        return {
            "experiment": exp_dir.name,
            "status": "skip_no_overlap",
            "phase_rows": int(len(phase_df)),
            "falcor_rows": int(len(falcor_df)),
        }

    missing_controls = [c for c in controls_norm if c != "epoch" and c not in aligned_df.columns]
    available_controls = [c for c in controls_norm if c == "epoch" or c in aligned_df.columns]

    rows: List[Dict[str, Any]] = []
    for metric in metric_names:
        if metric not in aligned_df.columns:
            continue
        for lag in range(-int(max_lag_epochs), int(max_lag_epochs) + 1):
            rows.append(
                _compute_lag_rows(
                    aligned_df,
                    metric=metric,
                    lag_epochs=int(lag),
                    controls=available_controls,
                    min_pairs=min_pairs,
                )
            )

    corr_df = pd.DataFrame(rows)
    if corr_df.empty:
        return {
            "experiment": exp_dir.name,
            "status": "skip_no_valid_pairs",
            "phase_rows": int(len(phase_df)),
            "falcor_rows": int(len(falcor_df)),
        }

    exp_out = output_dir / exp_dir.name
    exp_out.mkdir(parents=True, exist_ok=True)
    corr_csv = exp_out / "loss_falcor_lag_partial_correlation.csv"
    corr_df.to_csv(corr_csv, index=False)

    lag0 = corr_df[corr_df["lag_epochs"] == 0].copy()
    lag0["abs_partial_pearson"] = lag0["partial_pearson"].abs()
    lag0 = lag0.sort_values("abs_partial_pearson", ascending=False)
    lag0_csv = exp_out / "loss_falcor_partial_lag0.csv"
    lag0.to_csv(lag0_csv, index=False)

    best_rows: List[Dict[str, Any]] = []
    for metric, g in corr_df.groupby("metric", sort=True):
        use = g.dropna(subset=["partial_pearson"]).copy()
        if use.empty:
            continue
        idx = use["partial_pearson"].abs().idxmax()
        best_rows.append(dict(corr_df.loc[idx]))
    best_df = pd.DataFrame(best_rows).sort_values("partial_pearson", key=lambda s: s.abs(), ascending=False)
    best_csv = exp_out / "loss_falcor_partial_best_lag_per_metric.csv"
    best_df.to_csv(best_csv, index=False)

    summary = {
        "experiment": exp_dir.name,
        "status": "ok",
        "phase_rows": int(len(phase_df)),
        "falcor_rows": int(len(falcor_df)),
        "aligned_rows_dense": int(len(aligned_df)),
        "metric_count": int(corr_df["metric"].nunique()),
        "max_lag_epochs": int(max_lag_epochs),
        "min_pairs": int(min_pairs),
        "target_metric": str(target_metric),
        "include_stage_end_full": bool(include_stage_end_full),
        "controls_requested": controls_norm,
        "controls_used": available_controls,
        "controls_missing": missing_controls,
        "output_csv": str(corr_csv),
        "lag0_top_abs_partial": lag0.head(10).to_dict(orient="records"),
        "best_lag_top_abs_partial": best_df.head(10).to_dict(orient="records"),
    }
    (exp_out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    aligned_df.to_csv(exp_out / "aligned_dense_epoch_metrics.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze lagged and partial correlations between training metrics and Falcor PSNR."
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
        default=8,
        help="Minimum aligned pairs required before reporting a correlation.",
    )
    parser.add_argument(
        "--max-lag-epochs",
        type=int,
        default=30,
        help="Lag search range in epochs, evaluated on [-max_lag_epochs, +max_lag_epochs].",
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
        "--control-metric",
        action="append",
        default=[],
        help=(
            "Control metric name from phase0 metrics (repeatable). "
            "Example: --control-metric train_routing_temp"
        ),
    )
    parser.add_argument(
        "--no-epoch-control",
        action="store_true",
        help="Disable using epoch as a default control variable.",
    )
    parser.add_argument(
        "--output-dir",
        default="3_experiments/results/loss_falcor_correlation_lag_partial/latest",
        help="Output directory for lag+partial reports.",
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
                min_pairs=max(3, int(args.min_pairs)),
                metric_regex=str(args.metric_regex) if args.metric_regex else None,
                include_all_numeric=bool(args.include_all_numeric),
                max_lag_epochs=max(0, int(args.max_lag_epochs)),
                controls=[str(x) for x in args.control_metric],
                add_epoch_control=not bool(args.no_epoch_control),
                output_dir=out_dir,
            )
        )

    summary_json = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_metric": str(args.target_metric),
        "include_stage_end_full": bool(args.include_stage_end_full),
        "min_pairs": int(args.min_pairs),
        "max_lag_epochs": int(args.max_lag_epochs),
        "metric_regex": str(args.metric_regex) if args.metric_regex else None,
        "include_all_numeric": bool(args.include_all_numeric),
        "controls": [str(x) for x in args.control_metric],
        "epoch_control_enabled": not bool(args.no_epoch_control),
        "num_experiments": int(len(summaries)),
        "experiments": summaries,
    }
    (out_dir / "summary_all.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(out_dir / "summary_all.csv", index=False)
    print(f"[OK] lag+partial correlation reports written to: {out_dir}")


if __name__ == "__main__":
    main()
