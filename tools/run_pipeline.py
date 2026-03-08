"""Pipeline runner for dataset -> train -> eval.

This runner intentionally keeps the pipeline YAML simple while making execution
robust:

- Injects `data_root` from the dataset step when missing.
- Ensures `train.output_dir` is set when an eval step exists.
- Automatically resolves eval checkpoint from the train output directory:
  prefers `best_model.pt`, falls back to `last_model.pt`.
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from tools.run_dataset import build_plan
from tools.run_summary import write_run_summary
from tools.workflow_config import ensure_python, extend_cli_args, select_python


def _load_pipeline(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if yaml is None:
        raise ImportError("pyyaml is required for run_pipeline.py. Install via `uv pip install pyyaml`.")
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Pipeline YAML root must be a mapping")
    data["_path"] = str(path)
    return data


def _load_train_output_dir(config_path: str | Path) -> str | None:
    """Try to read output_dir from a training YAML config."""
    try:
        cfg_path = Path(config_path)
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            return None
        experiment = cfg.get("experiment", {})
        if isinstance(experiment, dict) and experiment.get("output_dir"):
            return str(experiment["output_dir"])
        if cfg.get("output_dir"):
            return str(cfg["output_dir"])
    except Exception:
        return None
    return None


def build_pipeline_steps(pipeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    run_dir = Path(pipeline.get("run_dir", "."))
    dataset_cfg = pipeline.get("dataset") or {}
    dataset_output_dir = None
    if dataset_cfg:
        output_dir, cmd, dataset_config = build_plan(dataset_cfg["config"], dataset_cfg.get("preset"))
        dataset_output_dir = str(output_dir)
        dataset_python = select_python(dataset_config)
        steps.append({"name": "dataset", "cmd": cmd, "output_dir": dataset_output_dir, "python": dataset_python})
    train_cfg = pipeline.get("train") or {}
    pipeline_python_train = pipeline.get("python_train")
    pipeline_python_eval = pipeline.get("python_eval")
    train_output_dir = None
    if train_cfg:
        args = train_cfg.get("args", {})
        if "config" in train_cfg and "config" not in args:
            args["config"] = train_cfg["config"]
        if dataset_output_dir and "data_root" not in args:
            args["data_root"] = dataset_output_dir

        # Backward-compat: convert num_steps -> epochs.
        if "num_steps" in args and "epochs" not in args:
            args["epochs"] = args.pop("num_steps")

        if "output_dir" in args:
            train_output_dir = str(args["output_dir"])
        elif "config" in args:
            train_output_dir = _load_train_output_dir(args["config"])

        # If an eval step exists and train output_dir is not set, force one so
        # eval can resolve checkpoint paths deterministically.
        if (pipeline.get("eval") or {}) and not train_output_dir:
            train_output_dir = str((run_dir / "out_train").resolve())
            args["output_dir"] = train_output_dir

        cmd = [train_cfg["entry"]]
        python_bin = train_cfg.get("python") or pipeline_python_train
        extend_cli_args(cmd, args)
        steps.append({"name": "train", "cmd": cmd, "python": python_bin, "output_dir": train_output_dir})
    eval_cfg = pipeline.get("eval") or {}
    if eval_cfg:
        args = eval_cfg.get("args", {})
        if "config" in eval_cfg and "config" not in args:
            args["config"] = eval_cfg["config"]
        if dataset_output_dir and "data_root" not in args:
            args["data_root"] = dataset_output_dir

        # Provide deterministic defaults for eval output + checkpoint.
        # - output: <train_output_dir>/eval/eval.json
        # - checkpoint: resolved at runtime after training
        if "output" not in args and train_output_dir:
            args["output"] = str(Path(train_output_dir) / "eval" / "eval.json")

        if "checkpoint" not in args and train_output_dir:
            args["checkpoint"] = str(Path(train_output_dir) / "best_model.pt")

        cmd = [eval_cfg["entry"]]
        python_bin = eval_cfg.get("python") or pipeline_python_eval
        extend_cli_args(cmd, args)
        steps.append(
            {
                "name": "eval",
                "cmd": cmd,
                "python": python_bin,
                "train_output_dir": train_output_dir,
            }
        )
    return steps


def _resolve_eval_checkpoint(step: Dict[str, Any]) -> None:
    """Mutate eval command to point at an existing checkpoint."""

    cmd = step.get("cmd")
    if not isinstance(cmd, list):
        return

    # Find current --checkpoint flag.
    checkpoint_idx = None
    for i, tok in enumerate(cmd):
        if tok == "--checkpoint" and i + 1 < len(cmd):
            checkpoint_idx = i + 1
            break
    if checkpoint_idx is None:
        return

    candidate = Path(cmd[checkpoint_idx])
    if candidate.exists():
        return

    train_output_dir = step.get("train_output_dir")
    if not train_output_dir:
        return

    best = Path(train_output_dir) / "best_model.pt"
    last = Path(train_output_dir) / "last_model.pt"
    if best.exists():
        cmd[checkpoint_idx] = str(best)
    elif last.exists():
        cmd[checkpoint_idx] = str(last)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pipeline YAML")
    parser.add_argument("--config", required=True)
    parser.add_argument("--python", default=None, help="Python executable to run steps")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pipeline = _load_pipeline(args.config)
    steps = build_pipeline_steps(pipeline)

    start_time = datetime.now().isoformat()
    for step in steps:
        if step.get("name") == "eval":
            _resolve_eval_checkpoint(step)
        python_bin = step.get("python") or args.python
        cmd = ensure_python(step["cmd"], python_bin)
        if args.dry_run:
            print(f"DRY-RUN [{step['name']}]", " ".join(cmd))
            continue
        subprocess.run(cmd, check=True)
    end_time = datetime.now().isoformat()

    run_dir = Path(pipeline.get("run_dir", "."))
    summary_items = {
        "pipeline_id": str(pipeline.get("id", "pipeline")),
        "config": str(pipeline.get("_path", "N/A")),
        "run_dir": str(run_dir),
        "started_at": start_time,
        "finished_at": end_time,
    }
    sections = []
    for step in steps:
        sections.append({
            "name": step.get("name", "step"),
            "python": str(step.get("python") or args.python or "default"),
            "command": " ".join(step.get("cmd", [])),
        })
    write_run_summary(run_dir / "run_summary.md", "Pipeline Run Summary", summary_items, sections=sections)


if __name__ == "__main__":  # pragma: no cover
    main()
