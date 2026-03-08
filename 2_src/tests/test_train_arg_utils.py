from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


train_arg_utils = _load_module(
    "train_arg_utils_test",
    ROOT / "3_experiments" / "scripts" / "train_arg_utils.py",
)


def test_normalize_namespace_fills_missing() -> None:
    args = Namespace(variant="unified_set")
    defaults = Namespace(val_image_metrics=False, val_superposition=False, seed=42)
    out = train_arg_utils.normalize_namespace(args, defaults)
    assert out is args
    assert args.val_image_metrics is False
    assert args.val_superposition is False
    assert args.seed == 42


def test_apply_variant_defaults_only_when_none() -> None:
    args = Namespace(variant="unified_set", num_gaussians=None, rank=99)
    train_arg_utils.apply_variant_defaults(args)
    assert args.num_gaussians == 20
    assert args.rank == 99

