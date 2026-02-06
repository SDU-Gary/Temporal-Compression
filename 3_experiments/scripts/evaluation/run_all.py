"""Unified evaluation runner (dry-run friendly)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List
import yaml
import sys

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "2_src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.config import validate_with_schema


def build_eval_steps(output_dir: str | Path, args: argparse.Namespace) -> List[Dict[str, str]]:
    out_dir = Path(output_dir)
    interp_args: List[str] = []
    if args.data_root:
        interp_args += ["--data-root", str(args.data_root)]
    if args.manifest:
        interp_args += ["--manifest", str(args.manifest)]
    if args.checkpoint:
        interp_args += ["--checkpoint", str(args.checkpoint)]
    if args.device:
        interp_args += ["--device", str(args.device)]
    if args.rank is not None:
        interp_args += ["--rank", str(args.rank)]
    if args.train_mae is not None:
        interp_args += ["--train-mae", str(args.train_mae)]
    if args.n_samples is not None:
        interp_args += ["--n-samples", str(args.n_samples)]
    if args.seed is not None:
        interp_args += ["--seed", str(args.seed)]
    if args.probe_idx is not None:
        interp_args += ["--probe-idx", str(args.probe_idx)]

    latency_args: List[str] = []
    if args.checkpoint:
        latency_args += ["--checkpoint", str(args.checkpoint)]
    if args.device:
        latency_args += ["--device", str(args.device)]
    steps = [
        {
            "name": "interpolation",
            "script": "3_experiments/scripts/evaluation/test_interpolation_query_physics.py",
            "args": ["--output", str(out_dir / "interpolation"), *interp_args],
        },
        {
            "name": "extrapolation",
            "script": "3_experiments/scripts/evaluation/test_extrapolation_query_physics.py",
            "args": ["--output", str(out_dir / "extrapolation"), *interp_args],
        },
        {
            "name": "latency",
            "script": "3_experiments/scripts/evaluation/test_query_latency_physics.py",
            "args": ["--output", str(out_dir / "latency"), *latency_args],
        },
    ]
    return steps


def build_unified_steps(output_dir: str | Path, args: argparse.Namespace) -> List[Dict[str, str]]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ablation_args: List[str] = []
    if args.data_root:
        ablation_args += ["--data-root", str(args.data_root)]
    if args.checkpoint:
        ablation_args += ["--checkpoint", str(args.checkpoint)]
    if args.device:
        ablation_args += ["--device", str(args.device)]
    if args.batch_size is not None:
        ablation_args += ["--batch-size", str(args.batch_size)]
    if args.num_workers is not None:
        ablation_args += ["--num-workers", str(args.num_workers)]
    if args.train_ratio is not None:
        ablation_args += ["--train-ratio", str(args.train_ratio)]
    if args.val_ratio is not None:
        ablation_args += ["--val-ratio", str(args.val_ratio)]
    if args.seed is not None:
        ablation_args += ["--seed", str(args.seed)]
    if args.split is not None:
        ablation_args += ["--split", str(args.split)]
    if args.num_gaussians is not None:
        ablation_args += ["--num-gaussians", str(args.num_gaussians)]
    if args.rank is not None:
        ablation_args += ["--rank", str(args.rank)]
    if args.top_k is not None:
        ablation_args += ["--top-k", str(args.top_k)]
    if args.light_dim is not None:
        ablation_args += ["--light-dim", str(args.light_dim)]
    if args.embed_dim is not None:
        ablation_args += ["--embed-dim", str(args.embed_dim)]
    if args.intensity_dim is not None:
        ablation_args += ["--intensity-dim", str(args.intensity_dim)]
    if args.intensity_offset is not None:
        ablation_args += ["--intensity-offset", str(args.intensity_offset)]
    if args.disable_film:
        ablation_args += ["--disable-film"]
    if args.sh_scaler is not None:
        ablation_args += ["--sh-scaler", str(args.sh_scaler)]
    if args.near_threshold is not None:
        ablation_args += ["--near-threshold", str(args.near_threshold)]
    if args.k_neighbors is not None:
        ablation_args += ["--k-neighbors", str(args.k_neighbors)]
    if args.save_pred_config is not None:
        ablation_args += ["--save-pred-config", str(args.save_pred_config)]
    if args.save_pred_out is not None:
        ablation_args += ["--save-pred-out", str(args.save_pred_out)]

    steps = [
        {
            "name": "unified_ablation",
            "script": "3_experiments/scripts/analysis/run_ablation_analysis.py",
            "args": ["--output", str(out_dir / "ablation_results.json"), *ablation_args],
        }
    ]
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified evaluation suite")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--train-mae", type=float, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--probe-idx", type=int, default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--suite", choices=["physics", "unified"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--num-gaussians", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--light-dim", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--intensity-dim", type=int, default=None)
    parser.add_argument("--intensity-offset", type=int, default=None)
    parser.add_argument("--disable-film", action="store_true")
    parser.add_argument("--sh-scaler", type=str, default=None)
    parser.add_argument("--near-threshold", type=float, default=None)
    parser.add_argument("--k-neighbors", type=int, default=None)
    parser.add_argument("--save-pred-config", type=int, default=None)
    parser.add_argument("--save-pred-out", type=str, default=None)
    parser.add_argument("--config", default=None, help="YAML config containing evaluation section")
    parser.add_argument(
        "--schema",
        default=str(_ROOT / "metadata" / "schemas" / "train.schema.json"),
        help="Optional JSON schema for config validation",
    )
    parser.add_argument("--strict-schema", action="store_true", help="Fail if schema validation fails")
    parser.add_argument("--python", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        if args.schema:
            messages = validate_with_schema(cfg, args.schema, strict=args.strict_schema)
            for msg in messages:
                print(f"[schema] {msg}")
        eval_cfg = cfg.get("evaluation", cfg.get("eval", {})) if isinstance(cfg, dict) else {}
        experiment_cfg = cfg.get("experiment", {}) if isinstance(cfg, dict) else {}
        model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
        training_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
        if args.output_dir is None and eval_cfg.get("output_dir"):
            args.output_dir = eval_cfg.get("output_dir")
        if args.checkpoint is None:
            args.checkpoint = eval_cfg.get("checkpoint")
        if args.data_root is None:
            args.data_root = eval_cfg.get("data_root") or eval_cfg.get("dataset_path")
        if args.variant is None:
            args.variant = experiment_cfg.get("variant") or cfg.get("variant")
        if args.rank is None:
            args.rank = eval_cfg.get("rank")
        if args.n_samples is None:
            args.n_samples = eval_cfg.get("n_samples")
        if args.seed is None:
            args.seed = eval_cfg.get("seed")
        if args.probe_idx is None:
            args.probe_idx = eval_cfg.get("probe_idx")
        if args.train_mae is None:
            args.train_mae = eval_cfg.get("train_mae")
        if args.batch_size is None:
            args.batch_size = data_cfg.get("batch_size")
        if args.num_workers is None:
            args.num_workers = data_cfg.get("num_workers")
        if args.train_ratio is None:
            args.train_ratio = data_cfg.get("train_ratio")
        if args.val_ratio is None:
            args.val_ratio = data_cfg.get("val_ratio")
        if args.num_gaussians is None:
            args.num_gaussians = model_cfg.get("num_gaussians")
        if args.top_k is None:
            args.top_k = model_cfg.get("top_k")
        if args.light_dim is None:
            args.light_dim = model_cfg.get("light_dim")
        if args.embed_dim is None:
            args.embed_dim = model_cfg.get("embed_dim")
        if args.intensity_dim is None:
            args.intensity_dim = model_cfg.get("intensity_dim")
        if args.intensity_offset is None:
            args.intensity_offset = model_cfg.get("intensity_offset")
        if not args.disable_film:
            args.disable_film = bool(model_cfg.get("disable_film", False))
        if args.sh_scaler is None:
            args.sh_scaler = training_cfg.get("sh_scaler_path")

    if args.output_dir is None:
        raise ValueError("output_dir is required (via --output-dir or config)")

    suite = args.suite
    if suite is None:
        if args.variant and str(args.variant).startswith("unified"):
            suite = "unified"
        else:
            suite = "physics"

    if suite == "unified":
        steps = build_unified_steps(args.output_dir, args)
    else:
        steps = build_eval_steps(args.output_dir, args)
    root_dir = _ROOT
    src_dir = _ROOT / "2_src"
    base_env = os.environ.copy()
    extra_paths = [str(root_dir), str(src_dir)]
    if base_env.get("PYTHONPATH"):
        extra_paths.append(base_env["PYTHONPATH"])
    base_env["PYTHONPATH"] = os.pathsep.join(extra_paths)
    for step in steps:
        cmd = [step["script"], *step["args"]]
        if args.python:
            cmd = [args.python] + cmd
        elif cmd[0].endswith(".py"):
            cmd = [sys.executable] + cmd
        if args.dry_run:
            print(f"DRY-RUN [{step['name']}]", " ".join(cmd))
            continue
        import subprocess
        subprocess.run(cmd, check=True, env=base_env)


if __name__ == "__main__":
    main()
