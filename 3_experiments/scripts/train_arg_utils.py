"""Argument normalization helpers for training entrypoints."""

from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
from typing import Dict, Any


VARIANT_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "unified_set": {
        "num_gaussians": 20,
        "rank": 5,
        "epochs": 2000,
        "batch_size": 512,
        "lr": 1e-3,
        "lambda_temporal": 0.001,
        "recon_loss": "charbonnier",
        "light_dim": 12,
        "embed_dim": 32,
        "intensity_dim": 3,
        "intensity_offset": 1,
        "lambda_linearity": 0.1,
        "linearity_aug_pairs": 2,
    },
}


def normalize_namespace(args: Namespace, defaults: Namespace) -> Namespace:
    for key, value in vars(defaults).items():
        if not hasattr(args, key):
            setattr(args, key, deepcopy(value))
    return args


def apply_variant_defaults(args: Namespace, variant_defaults: Dict[str, Dict[str, Any]] | None = None) -> None:
    defaults = variant_defaults or VARIANT_DEFAULTS
    selected = defaults.get(args.variant, {})
    for key, value in selected.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
