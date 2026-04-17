"""Variant registry for training entrypoint."""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, Tuple


VariantBuildResult = Tuple[Any, Any, Tuple[Any, Any, Any], Dict[str, Any]]
VariantBuilder = Callable[[argparse.Namespace, Any], VariantBuildResult]


class VariantRegistry:
    """Small registry to decouple variant extensions from train.py control flow."""

    def __init__(self) -> None:
        self._builders: Dict[str, VariantBuilder] = {}

    def register(self, name: str, builder: VariantBuilder, *, overwrite: bool = False) -> None:
        key = str(name).strip()
        if not key:
            raise ValueError("variant name cannot be empty")
        if (not overwrite) and key in self._builders:
            raise ValueError(f"variant already registered: {key}")
        self._builders[key] = builder

    def build(self, name: str, args: argparse.Namespace, device: Any) -> VariantBuildResult:
        builder = self._builders.get(name)
        if builder is None:
            known = ", ".join(sorted(self._builders.keys())) or "<none>"
            raise ValueError(f"Unknown variant: {name}. Available: {known}")
        return builder(args, device)

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._builders.keys()))
