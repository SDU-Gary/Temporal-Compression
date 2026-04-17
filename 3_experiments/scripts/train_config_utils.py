"""Config loading and argparse mapping for training script."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict
import warnings

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from train_typed_config import dedupe_alias_hits, parse_typed_train_config


def load_yaml(path: str | Path) -> Dict[str, Any]:
    import yaml

    yaml_path = Path(path)
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML root must be a mapping")
    return data


def apply_config(args: argparse.Namespace, defaults: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    typed = parse_typed_train_config(cfg)
    mapping = typed.to_arg_mapping()

    for alias in dedupe_alias_hits(list(typed.alias_hits)):
        warnings.warn(
            f"Deprecated config alias detected: {alias}. Please migrate to the canonical key.",
            DeprecationWarning,
            stacklevel=2,
        )

    for key, value in mapping.items():
        if value is None or not hasattr(args, key):
            continue
        current = getattr(args, key)
        default = getattr(defaults, key, None)
        if current == default:
            setattr(args, key, value)
