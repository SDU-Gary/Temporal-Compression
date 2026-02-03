"""Helpers for loading and validating workflow YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import os
import sys

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping")
    data["_config_path"] = str(path)
    return data


def apply_preset(config: Dict[str, Any], preset: str | None) -> Dict[str, Any]:
    if not preset:
        return config
    presets = config.get("presets", {})
    if preset not in presets:
        raise ValueError(f"Preset '{preset}' not found in config")
    override = presets[preset].get("args", {})
    generator = config.setdefault("generator", {})
    args = generator.setdefault("args", {})
    args.update(override)
    config["_preset"] = preset
    return config


def validate_dataset_config(config: Dict[str, Any]) -> None:
    generator = config.get("generator")
    if not isinstance(generator, dict):
        raise ValueError("Missing generator section in dataset config")
    script = generator.get("script")
    if not script:
        raise ValueError("generator.script is required")
    output_dir = config.get("output_dir")
    if not output_dir:
        raise ValueError("output_dir is required")


def build_generator_command(config: Dict[str, Any]) -> Tuple[Path, list[str]]:
    generator = config["generator"]
    script = Path(generator["script"]).resolve()
    args = generator.get("args", {})
    output_dir = Path(config["output_dir"]).resolve()

    cmd = [str(script), "--output", str(output_dir)]
    for key, value in args.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                cmd.extend([flag, str(item)])
            continue
        cmd.extend([flag, str(value)])
    return output_dir, cmd


def ensure_python(cmd: list[str], python_bin: str | None = None) -> list[str]:
    """Ensure python executable is prepended when running .py scripts."""
    if not cmd:
        raise ValueError("Command is empty")
    if python_bin:
        return [python_bin] + cmd
    if cmd[0].endswith(".py"):
        return [sys.executable] + cmd
    return cmd


def select_python(config: Dict[str, Any], env_var: str = "FALCOR_PYTHON") -> str | None:
    """Select python executable from config or environment."""
    if not isinstance(config, dict):
        return None
    python_bin = config.get("python")
    if not python_bin:
        generator = config.get("generator", {}) if isinstance(config.get("generator"), dict) else {}
        python_bin = generator.get("python")
    if python_bin:
        return str(python_bin)
    env_val = os.environ.get(env_var)
    if env_val:
        return env_val
    return None
