"""Shared helpers for experiment driver scripts."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml


def set_nested_value(data: Dict[str, Any], key_path: str, value: Any) -> None:
    keys = key_path.split(".")
    target = data
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def modify_config(config_file: str, modifications: Dict[str, Any], output_file: str) -> Dict[str, Any]:
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for key_path, value in modifications.items():
        set_nested_value(config, key_path, value)

    output_dir = config.get("experiment", {}).get("output_dir")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config


def run_command(cmd: List[str], description: str, log_file: str | None = None) -> int:
    print(f"\n{'=' * 80}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")

    if log_file:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Started: {datetime.now()}\n\n")

        with open(log_file, "a", encoding="utf-8") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\nFinished: {datetime.now()}\n")
            f.write(f"Exit code: {result.returncode}\n")
    else:
        result = subprocess.run(cmd)

    return result.returncode


def read_json_if_exists(path: str | Path, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        return {} if default is None else default
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)
