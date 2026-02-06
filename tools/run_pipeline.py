"""Pipeline runner for dataset -> train -> eval."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

from tools.run_dataset import build_plan
from tools.run_summary import write_run_summary
from tools.workflow_config import ensure_python, select_python


def _load_pipeline(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
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
        if "output_dir" in args:
            train_output_dir = str(args["output_dir"])
        elif "config" in args:
            train_output_dir = _load_train_output_dir(args["config"])
        cmd = [train_cfg["entry"]]
        python_bin = train_cfg.get("python") or pipeline_python_train
        for key, value in args.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
                continue
            cmd.extend([flag, str(value)])
        steps.append({"name": "train", "cmd": cmd, "python": python_bin})
    eval_cfg = pipeline.get("eval") or {}
    if eval_cfg:
        args = eval_cfg.get("args", {})
        if "config" in eval_cfg and "config" not in args:
            args["config"] = eval_cfg["config"]
        if dataset_output_dir and "data_root" not in args:
            args["data_root"] = dataset_output_dir
        if "output_dir" not in args and train_output_dir:
            args["output_dir"] = str(Path(train_output_dir) / "eval")
        cmd = [eval_cfg["entry"]]
        python_bin = eval_cfg.get("python") or pipeline_python_eval
        for key, value in args.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
                continue
            cmd.extend([flag, str(value)])
        steps.append({"name": "eval", "cmd": cmd, "python": python_bin})
    return steps


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
