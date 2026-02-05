"""Unified evaluation runner (dry-run friendly)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import yaml


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
    parser.add_argument("--config", default=None, help="YAML config containing evaluation section")
    parser.add_argument("--python", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        eval_cfg = cfg.get("evaluation", cfg.get("eval", {})) if isinstance(cfg, dict) else {}
        if args.output_dir is None and eval_cfg.get("output_dir"):
            args.output_dir = eval_cfg.get("output_dir")
        if args.checkpoint is None:
            args.checkpoint = eval_cfg.get("checkpoint")
        if args.data_root is None:
            args.data_root = eval_cfg.get("data_root") or eval_cfg.get("dataset_path")
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

    if args.output_dir is None:
        raise ValueError("output_dir is required (via --output-dir or config)")

    steps = build_eval_steps(args.output_dir, args)
    for step in steps:
        cmd = [step["script"], *step["args"]]
        if args.python:
            cmd = [args.python] + cmd
        if args.dry_run:
            print(f"DRY-RUN [{step['name']}]", " ".join(cmd))
            continue
        import subprocess
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
