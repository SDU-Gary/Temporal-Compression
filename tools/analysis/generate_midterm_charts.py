#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _pick_key_experiments(base: Path) -> pd.DataFrame:
    mapping = [
        ("exp_A1_4_anchor_globalbest", "A1.4"),
        ("exp_A1_5_bypassA", "A1.5-bypassA"),
        ("exp_A1_5_bypassB", "A1.5-bypassB"),
        ("exp_A1_5_bypassB_noproxy", "A1.5-bypassB-noProxy"),
        ("exp_A1_6_P1_bypassB_proxy_l002", "A1.6-P1-l002"),
        ("exp_A1_6_P2_enc_normal", "A1.6-P2-normal"),
        ("exp_A1_7_P0_bypassB_seed19", "A1.7-P0-seed19"),
        ("exp_A1_7_P2_K30_seed19", "A1.7-P2-K30"),
        ("exp_A1_7_P2_K1_seed19", "A1.7-P2-K1"),
    ]

    rows: List[Dict] = []
    for exp, label in mapping:
        p = base / exp / "global_best_falcor_summary.json"
        if not p.exists():
            continue
        d = _load_json(p)
        rows.append(
            {
                "exp_id": exp,
                "label": label,
                "falcor_hdr_psnr": float(d.get("value", float("nan"))),
                "epoch": int(d.get("epoch", -1)),
                "stage_name": str(d.get("stage_name", "")),
                "summary_path": str(d.get("summary", "")),
            }
        )
    return pd.DataFrame(rows)


def _resolve_path_maybe_relative(path_text: str) -> Path:
    p = Path(path_text)
    if not p.is_absolute():
        p = ROOT / p
    return p


def _resolve_dataset_npz_from_ckpt_meta(ckpt_obj: Dict, fallback_npz: Path) -> Path:
    meta = ckpt_obj.get("meta", {}) if isinstance(ckpt_obj, dict) else {}
    data_root = None
    if isinstance(meta, dict):
        data_root = meta.get("data_root")
    if isinstance(data_root, str) and data_root.strip():
        p = _resolve_path_maybe_relative(data_root.strip())
        if p.is_dir():
            npz = p / "parametric_tensor.npz"
        else:
            npz = p
        if npz.exists():
            return npz
    return fallback_npz


def _collect_compression_rows(base: Path, fallback_dataset_npz: Path) -> pd.DataFrame:
    exp_dirs = sorted([p for p in base.glob("exp_*") if p.is_dir()])
    dataset_nbytes_cache: Dict[str, int] = {}
    rows: List[Dict] = []

    for exp_dir in exp_dirs:
        gbest = exp_dir / "global_best_falcor_summary.json"
        if not gbest.exists():
            continue
        g = _load_json(gbest)
        ckpt_path_text = str(g.get("checkpoint", "")).strip()
        if not ckpt_path_text:
            continue
        ckpt_path = _resolve_path_maybe_relative(ckpt_path_text)
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        if not isinstance(state, dict):
            continue

        param_count = 0
        param_bytes = 0
        for v in state.values():
            if torch.is_tensor(v):
                param_count += int(v.numel())
                param_bytes += int(v.numel()) * int(v.element_size())

        if param_count <= 0 or param_bytes <= 0:
            continue

        dataset_npz = _resolve_dataset_npz_from_ckpt_meta(ckpt, fallback_dataset_npz)
        ds_key = str(dataset_npz.resolve())
        if ds_key not in dataset_nbytes_cache:
            with np.load(dataset_npz, allow_pickle=True) as ds:
                if "tensor" not in ds:
                    raise KeyError(f"{dataset_npz} missing tensor")
                dataset_nbytes_cache[ds_key] = int(np.asarray(ds["tensor"]).nbytes)
        raw_bytes = int(dataset_nbytes_cache[ds_key])

        ckpt_file_bytes = int(ckpt_path.stat().st_size)
        param_ratio = float(raw_bytes / max(param_bytes, 1))
        file_ratio = float(raw_bytes / max(ckpt_file_bytes, 1))

        rows.append(
            {
                "exp_id": exp_dir.name,
                "dataset_npz": ds_key,
                "checkpoint_path": str(ckpt_path.resolve()),
                "raw_sh_bytes": raw_bytes,
                "param_count": int(param_count),
                "param_bytes": int(param_bytes),
                "checkpoint_bytes": int(ckpt_file_bytes),
                "compression_ratio_param": param_ratio,
                "compression_ratio_checkpoint": file_ratio,
                "falcor_hdr_psnr_best": float(g.get("value", float("nan"))),
            }
        )

    return pd.DataFrame(rows)


def _plot_compression_key(df_all: pd.DataFrame, key_df: pd.DataFrame, out_png: Path) -> None:
    if df_all.empty or key_df.empty:
        raise RuntimeError("Compression data is empty.")

    merged = key_df.merge(
        df_all[["exp_id", "compression_ratio_param", "compression_ratio_checkpoint"]],
        on="exp_id",
        how="left",
    )
    merged = merged.dropna(subset=["compression_ratio_param", "compression_ratio_checkpoint"]).copy()
    if merged.empty:
        raise RuntimeError("No overlapping compression rows for key experiments.")

    x = np.arange(len(merged))
    w = 0.36
    plt.figure(figsize=(12.8, 5.4))
    b1 = plt.bar(x - w / 2, merged["compression_ratio_param"], width=w, label="Param-based ratio")
    b2 = plt.bar(x + w / 2, merged["compression_ratio_checkpoint"], width=w, label="Checkpoint-size ratio")
    plt.xticks(x, merged["label"], rotation=20, ha="right")
    plt.ylabel("Compression Ratio (x)")
    plt.title("Compression Ratio Across Key Experiments")
    plt.grid(axis="y", linestyle="--", alpha=0.25)

    mean_param = float(np.nanmean(df_all["compression_ratio_param"].to_numpy(dtype=float)))
    mean_file = float(np.nanmean(df_all["compression_ratio_checkpoint"].to_numpy(dtype=float)))
    plt.axhline(mean_param, color="#1f77b4", linestyle="--", linewidth=1.2, alpha=0.8, label=f"Mean param ratio={mean_param:.1f}x")
    plt.axhline(mean_file, color="#ff7f0e", linestyle="--", linewidth=1.2, alpha=0.8, label=f"Mean file ratio={mean_file:.1f}x")

    for bars in [b1, b2]:
        for b in bars:
            h = float(b.get_height())
            plt.text(b.get_x() + b.get_width() / 2.0, h + 0.12, f"{h:.1f}", ha="center", va="bottom", fontsize=8)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def _plot_compression_all_sorted(df_all: pd.DataFrame, out_png: Path) -> None:
    if df_all.empty:
        raise RuntimeError("Compression data is empty.")
    d = df_all.sort_values("compression_ratio_param", ascending=False).copy()
    plt.figure(figsize=(13.5, 5.6))
    x = np.arange(len(d))
    plt.plot(x, d["compression_ratio_param"], marker="o", linewidth=1.6, label="Param-based ratio")
    plt.plot(x, d["compression_ratio_checkpoint"], marker="s", linewidth=1.6, label="Checkpoint-size ratio")
    plt.xticks(x, d["exp_id"], rotation=70, ha="right", fontsize=8)
    plt.ylabel("Compression Ratio (x)")
    plt.title("Compression Ratio (All Experiments, Sorted)")
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def _plot_experiment_overview(df: pd.DataFrame, out_png: Path) -> None:
    if df.empty:
        raise RuntimeError("No experiment summary found for overview chart.")
    plt.figure(figsize=(12, 5.2))
    x = np.arange(len(df))
    colors = ["#4e79a7" if v >= 34.0 else "#f28e2b" for v in df["falcor_hdr_psnr"]]
    bars = plt.bar(x, df["falcor_hdr_psnr"], color=colors, alpha=0.9)
    plt.xticks(x, df["label"], rotation=20, ha="right")
    plt.ylabel("Falcor HDR PSNR (dB)")
    plt.title("Midterm Key Experiment Comparison (Falcor HDR PSNR)")
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.ylim(min(18.0, float(df["falcor_hdr_psnr"].min()) - 1.0), float(df["falcor_hdr_psnr"].max()) + 1.5)
    for b, v in zip(bars, df["falcor_hdr_psnr"]):
        plt.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def _load_route_gap_data(base_video: Path) -> pd.DataFrame:
    runs = [
        ("Hard top-k", base_video / "diag_hard_20260407" / "benchmark_summary.json"),
        ("Soft8@0.18", base_video / "diag_soft_20260407" / "benchmark_summary.json"),
    ]
    rows: List[Dict] = []
    for name, path in runs:
        if not path.exists():
            continue
        d = _load_json(path)
        m = d.get("image_metrics", {})
        rows.append(
            {
                "route_mode": name,
                "mean_real_render_hdr_psnr": float(m.get("mean_real_render_hdr_psnr", float("nan"))),
                "mean_benchmark_psnr": float(m.get("mean_benchmark_psnr", float("nan"))),
                "mean_linear_scale": float(m.get("mean_linear_scale", float("nan"))),
                "mean_gt_p99": float(m.get("mean_gt_p99", float("nan"))),
                "mean_model_p99": float(m.get("mean_model_p99", float("nan"))),
            }
        )
    return pd.DataFrame(rows)


def _plot_route_gap(df: pd.DataFrame, out_png: Path) -> None:
    if df.empty:
        raise RuntimeError("No route gap diagnostics found.")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    x = np.arange(len(df))
    w = 0.34
    axes[0].bar(x - w / 2, df["mean_real_render_hdr_psnr"], width=w, label="Real HDR PSNR")
    axes[0].bar(x + w / 2, df["mean_benchmark_psnr"], width=w, label="sRGB PSNR")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["route_mode"])
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("Route Mode Quality Gap")
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].legend(frameon=False)

    ratio = df["mean_model_p99"] / np.maximum(df["mean_gt_p99"], 1e-8)
    axes[1].bar(x - w / 2, df["mean_linear_scale"], width=w, label="Linear Scale")
    axes[1].bar(x + w / 2, ratio, width=w, label="p99 ratio (model/gt)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["route_mode"])
    axes[1].set_ylabel("Scale / Ratio")
    axes[1].set_title("Brightness Mismatch Diagnostics")
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _load_drift_data(exp_dir: Path) -> pd.DataFrame:
    phase = _load_jsonl(exp_dir / "runtime" / "phase0_metrics.jsonl")
    falcor = _load_jsonl(exp_dir / "runtime" / "falcor_periodic_eval.jsonl")

    vmap: Dict[int, float] = {}
    for row in phase:
        ep = int(row.get("epoch", -1))
        metrics = row.get("metrics", {})
        if isinstance(metrics, dict) and metrics.get("val_profile_soft8_t018_img_psnr") is not None:
            vmap[ep] = float(metrics["val_profile_soft8_t018_img_psnr"])

    fmap: Dict[int, float] = {}
    for row in falcor:
        ep = int(row.get("epoch", -1))
        m = row.get("metric_value")
        if m is None:
            mm = row.get("metrics", {})
            if isinstance(mm, dict):
                m = mm.get("mean_real_render_hdr_psnr")
        if m is not None:
            fmap[ep] = float(m)

    common = sorted(set(vmap.keys()).intersection(fmap.keys()))
    data = [{"epoch": e, "val_img_psnr_soft8_t018": vmap[e], "falcor_hdr_psnr": fmap[e]} for e in common]
    return pd.DataFrame(data)


def _plot_drift_line(df: pd.DataFrame, out_png: Path, title_suffix: str) -> None:
    if df.empty:
        raise RuntimeError("No drift data available.")
    fig, ax1 = plt.subplots(figsize=(10.8, 4.6))
    ax2 = ax1.twinx()
    ax1.plot(df["epoch"], df["val_img_psnr_soft8_t018"], marker="o", linewidth=1.8, color="#4e79a7", label="Val img_psnr (soft8@0.18)")
    ax2.plot(df["epoch"], df["falcor_hdr_psnr"], marker="s", linewidth=1.8, color="#e15759", label="Falcor HDR PSNR")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val img_psnr (dB)", color="#4e79a7")
    ax2.set_ylabel("Falcor HDR PSNR (dB)", color="#e15759")
    ax1.grid(True, linestyle="--", alpha=0.25)
    ax1.set_title(f"Metric Drift Across Training ({title_suffix})")
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _plot_drift_scatter(df: pd.DataFrame, out_png: Path, title_suffix: str) -> float:
    if df.empty:
        raise RuntimeError("No drift data available.")
    x = df["val_img_psnr_soft8_t018"].to_numpy(dtype=float)
    y = df["falcor_hdr_psnr"].to_numpy(dtype=float)
    corr = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else float("nan")

    plt.figure(figsize=(6.3, 5.0))
    plt.scatter(x, y, s=48, alpha=0.85, color="#4e79a7", edgecolor="white", linewidth=0.7)
    if len(x) >= 2:
        k, b = np.polyfit(x, y, 1)
        xx = np.linspace(float(np.min(x)), float(np.max(x)), 80)
        plt.plot(xx, k * xx + b, color="#e15759", linewidth=1.8, label=f"fit: y={k:.2f}x+{b:.2f}")
    plt.xlabel("Val img_psnr soft8@0.18 (dB)")
    plt.ylabel("Falcor HDR PSNR (dB)")
    plt.title(f"Metric Correlation ({title_suffix})\nPearson r = {corr:.3f}")
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    return corr


def _latency_from_best_summary(exp_dir: Path) -> Dict[str, float]:
    best = _load_json(exp_dir / "global_best_falcor_summary.json")
    summary_path = ROOT / Path(best["summary"])
    s = _load_json(summary_path)
    model = s.get("results", {}).get("model", {})
    inf = model.get("inference_ms", {})
    return {
        "inference_mean_ms": float(inf.get("mean", float("nan"))),
        "inference_p95_ms": float(inf.get("p95", float("nan"))),
        "fps_route_only_full": float(model.get("fps_route_only_full", float("nan"))),
        "fps_with_gbuffer_full": float(model.get("fps_with_gbuffer_full", float("nan"))),
    }


def _plot_latency(lat: Dict[str, float], out_png: Path) -> None:
    vals = [lat["inference_mean_ms"], lat["inference_p95_ms"]]
    labels = ["Mean", "P95"]
    plt.figure(figsize=(5.6, 4.6))
    bars = plt.bar(labels, vals, color=["#59a14f", "#edc948"], width=0.55)
    plt.axhline(1.5, color="#e15759", linestyle="--", linewidth=1.4, label="1.5 ms target")
    plt.ylabel("Inference Time (ms)")
    plt.title("Best Model Decompression Latency")
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2.0, v + 0.03, f"{v:.2f}", ha="center", va="bottom")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison charts for midterm presentation.")
    parser.add_argument(
        "--output-dir",
        default="3_experiments/results/midterm_charts_20260408",
        help="Output directory for charts/csv.",
    )
    parser.add_argument(
        "--experiments-root",
        default="3_experiments/results/bistro_clean_v2",
        help="Root directory for experiment results.",
    )
    parser.add_argument(
        "--video-root",
        default="3_experiments/results/video_exports",
        help="Root directory for route-gap diagnostics.",
    )
    parser.add_argument(
        "--dataset-npz",
        default="1_data_generation/output/bistro_clean_v2/parametric_tensor.npz",
        help="Fallback dataset npz for compression-ratio computation.",
    )
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    exp_root = ROOT / args.experiments_root
    video_root = ROOT / args.video_root
    dataset_npz = ROOT / args.dataset_npz
    _ensure_dir(out_dir)

    # 1) Experiment overview
    exp_df = _pick_key_experiments(exp_root)
    exp_df.to_csv(out_dir / "table_experiment_overview.csv", index=False)
    _plot_experiment_overview(exp_df, out_dir / "fig01_experiment_overview_psnr.png")

    # 2) Soft vs hard deployment gap (diagnostic run generated earlier)
    route_df = _load_route_gap_data(video_root)
    route_df.to_csv(out_dir / "table_route_gap_diagnostics.csv", index=False)
    _plot_route_gap(route_df, out_dir / "fig02_soft_vs_hard_gap.png")

    # 3) Drift and correlation (A1.3 is a typical negative-correlation case)
    drift_exp = exp_root / "exp_A1_3_proxylog_balance_anneal"
    drift_df = _load_drift_data(drift_exp)
    drift_df.to_csv(out_dir / "table_a13_drift_pairs.csv", index=False)
    _plot_drift_line(drift_df, out_dir / "fig03a_a13_drift_line.png", "A1.3")
    corr = _plot_drift_scatter(drift_df, out_dir / "fig03b_a13_correlation_scatter.png", "A1.3")

    # 4) Best model inference latency (for progress section)
    best_exp = exp_root / "exp_A1_6_P1_bypassB_proxy_l002"
    latency = _latency_from_best_summary(best_exp)
    pd.DataFrame([latency]).to_csv(out_dir / "table_best_model_latency.csv", index=False)
    _plot_latency(latency, out_dir / "fig04_best_model_latency.png")

    # 5) Compression ratio (parameter-based + checkpoint-size based)
    comp_df = _collect_compression_rows(exp_root, dataset_npz)
    comp_df.to_csv(out_dir / "table_compression_all_experiments.csv", index=False)
    _plot_compression_key(comp_df, exp_df, out_dir / "fig05_compression_key_experiments.png")
    _plot_compression_all_sorted(comp_df, out_dir / "fig06_compression_all_sorted.png")

    comp_summary = pd.DataFrame(
        [
            {
                "num_experiments": int(len(comp_df)),
                "mean_param_ratio_x": float(np.nanmean(comp_df["compression_ratio_param"].to_numpy(dtype=float))),
                "mean_checkpoint_ratio_x": float(np.nanmean(comp_df["compression_ratio_checkpoint"].to_numpy(dtype=float))),
                "median_param_ratio_x": float(np.nanmedian(comp_df["compression_ratio_param"].to_numpy(dtype=float))),
                "median_checkpoint_ratio_x": float(np.nanmedian(comp_df["compression_ratio_checkpoint"].to_numpy(dtype=float))),
            }
        ]
    )
    comp_summary.to_csv(out_dir / "table_compression_summary.csv", index=False)

    # 6) Summary markdown
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    readme = out_dir / "README.md"
    with open(readme, "w", encoding="utf-8") as f:
        f.write("# Midterm Charts Pack\n\n")
        f.write(f"- Generated at: `{now}`\n")
        f.write(f"- Experiments root: `{exp_root}`\n")
        f.write(f"- Video diagnostics root: `{video_root}`\n\n")
        f.write("## Charts\n")
        f.write("- `fig01_experiment_overview_psnr.png`: 关键实验 Falcor HDR PSNR 对照\n")
        f.write("- `fig02_soft_vs_hard_gap.png`: soft/hard 路由质量与亮度偏差对照\n")
        f.write("- `fig03a_a13_drift_line.png`: A1.3 训练期 proxy 指标与 Falcor 曲线\n")
        f.write("- `fig03b_a13_correlation_scatter.png`: A1.3 指标相关性散点（Pearson）\n")
        f.write("- `fig04_best_model_latency.png`: 最佳模型解压时延（mean/p95）\n\n")
        f.write("- `fig05_compression_key_experiments.png`: 关键实验压缩率（参数量口径 + checkpoint口径）\n")
        f.write("- `fig06_compression_all_sorted.png`: 全部实验压缩率排序对比\n\n")
        f.write("## Tables\n")
        f.write("- `table_experiment_overview.csv`\n")
        f.write("- `table_route_gap_diagnostics.csv`\n")
        f.write("- `table_a13_drift_pairs.csv`\n")
        f.write("- `table_best_model_latency.csv`\n\n")
        f.write("- `table_compression_all_experiments.csv`\n")
        f.write("- `table_compression_summary.csv`\n\n")
        f.write("## Notes\n")
        f.write(f"- A1.3 correlation (val img_psnr vs Falcor HDR): `{corr:.3f}`\n")
        if not comp_summary.empty:
            c = comp_summary.iloc[0]
            f.write(
                "- Compression ratio mean (all experiments): "
                f"param `{float(c['mean_param_ratio_x']):.2f}x`, "
                f"checkpoint `{float(c['mean_checkpoint_ratio_x']):.2f}x`\n"
            )

    print(f"[OK] Charts generated in: {out_dir}")


if __name__ == "__main__":
    main()
