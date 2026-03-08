from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


cfg_utils = _load_module(
    "train_config_utils_test",
    ROOT / "3_experiments" / "scripts" / "train_config_utils.py",
)


def test_load_yaml_mapping(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("experiment:\n  variant: unified_set\n", encoding="utf-8")
    loaded = cfg_utils.load_yaml(path)
    assert loaded["experiment"]["variant"] == "unified_set"


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("- not\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError):
        cfg_utils.load_yaml(path)


def test_apply_config_only_overrides_defaults() -> None:
    args = Namespace(variant=None, rank=None, lr=1e-3)
    defaults = Namespace(variant=None, rank=None, lr=1e-3)
    cfg_utils.apply_config(
        args,
        defaults,
        {
            "experiment": {"variant": "unified_set"},
            "model": {"rank": 8},
            "training": {"lr": 5e-4},
        },
    )
    assert args.variant == "unified_set"
    assert args.rank == 8
    assert args.lr == 5e-4

    # Existing non-default value should be preserved.
    args2 = Namespace(variant=None, rank=16, lr=1e-3)
    cfg_utils.apply_config(
        args2,
        defaults,
        {
            "model": {"rank": 8},
        },
    )
    assert args2.rank == 16



def test_apply_config_maps_routing_training_fields() -> None:
    args = Namespace(
        lambda_routing_balance=0.0,
        routing_soft_train=False,
        routing_soft_topk=0,
        routing_temp_start=1.0,
        routing_temp_end=1.0,
        routing_temp_anneal_epochs=0,
    )
    defaults = Namespace(
        lambda_routing_balance=0.0,
        routing_soft_train=False,
        routing_soft_topk=0,
        routing_temp_start=1.0,
        routing_temp_end=1.0,
        routing_temp_anneal_epochs=0,
    )
    cfg_utils.apply_config(
        args,
        defaults,
        {
            "training": {
                "lambda_routing_balance": 0.02,
                "routing_soft_train": True,
                "routing_soft_topk": 8,
                "routing_temp_start": 1.0,
                "routing_temp_end": 0.2,
                "routing_temp_anneal_epochs": 320,
            }
        },
    )
    assert args.lambda_routing_balance == 0.02
    assert args.routing_soft_train is True
    assert args.routing_soft_topk == 8
    assert args.routing_temp_start == 1.0
    assert args.routing_temp_end == 0.2
    assert args.routing_temp_anneal_epochs == 320
