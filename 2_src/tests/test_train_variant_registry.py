from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


variant_registry = _load_module(
    "train_variant_registry_test",
    ROOT / "3_experiments" / "scripts" / "train_variant_registry.py",
)
train_script = _load_module(
    "train_variant_registry_train_test",
    ROOT / "3_experiments" / "scripts" / "train.py",
)


def test_variant_registry_register_and_build() -> None:
    registry = variant_registry.VariantRegistry()

    def _builder(args: Namespace, device):
        return ("model", "adapter", ("train", "val", "test"), {"device": device, "v": args.variant})

    registry.register("dummy", _builder)
    model, adapter, loaders, helpers = registry.build("dummy", Namespace(variant="dummy"), "cpu")
    assert model == "model"
    assert adapter == "adapter"
    assert loaders == ("train", "val", "test")
    assert helpers["device"] == "cpu"
    assert helpers["v"] == "dummy"


def test_variant_registry_duplicate_and_unknown() -> None:
    registry = variant_registry.VariantRegistry()
    registry.register("dummy", lambda *_: (None, None, (None, None, None), {}))
    with pytest.raises(ValueError):
        registry.register("dummy", lambda *_: (None, None, (None, None, None), {}))
    with pytest.raises(ValueError):
        registry.build("missing", Namespace(), "cpu")


def test_train_parser_variant_choices_reflect_registry() -> None:
    train_script.register_variant(
        "dummy",
        lambda args, device: ("m", "a", ("t", "v", "x"), {"variant": "dummy", "device": device}),
        overwrite=True,
    )
    parser = train_script.build_arg_parser()
    parsed = parser.parse_args(["--variant", "dummy"])
    assert parsed.variant == "dummy"
