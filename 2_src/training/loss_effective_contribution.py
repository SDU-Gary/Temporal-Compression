"""Utilities for balancing auxiliary loss effective contributions.

This module keeps auxiliary losses on a comparable effective scale with
minimal coupling to the trainer. It supports:
- EMA-based magnitude estimation per loss term
- optional frequency-aware normalization (for sparse terms like gbuffer)
- ratio-to-reconstruction targeting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import torch


def _to_non_negative_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if out < 0.0:
        return 0.0
    return out


def _to_unit_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if out < 0.0:
        return 0.0
    if out > 1.0:
        return 1.0
    return out


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    return bool(value)


@dataclass
class _TermState:
    ema_loss: Optional[float] = None
    last_scale: float = 1.0
    last_weight_base: float = 0.0
    last_weight_effective: float = 0.0
    last_contrib_base: float = 0.0
    last_contrib_effective: float = 0.0
    last_target: float = 0.0
    last_expected: float = 0.0
    last_frequency: float = 1.0


@dataclass
class LossEffectiveContributionController:
    """EMA controller for harmonizing auxiliary loss effective contributions."""

    enabled: bool = False
    ema_decay: float = 0.98
    warmup_steps: int = 0
    eps: float = 1e-8
    clamp_min_scale: float = 0.25
    clamp_max_scale: float = 4.0
    frequency_aware: bool = True
    min_target_contribution: float = 0.0
    max_target_contribution: float = 0.0
    target_ratio_default: float = 0.25
    target_ratios: Dict[str, float] = field(default_factory=dict)
    normalize_terms: Dict[str, bool] = field(default_factory=dict)
    step_count: int = 0
    recon_ema: Optional[float] = None
    _terms: Dict[str, _TermState] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: Optional[Mapping[str, Any]]) -> "LossEffectiveContributionController":
        raw = dict(cfg or {})
        ratios_raw = raw.get("target_ratios", {})
        if not isinstance(ratios_raw, dict):
            ratios_raw = {}
        normalize_raw = raw.get("normalize_terms", {})
        if not isinstance(normalize_raw, dict):
            normalize_raw = {}

        target_ratios = {
            str(k): _to_non_negative_float(v, 0.25)
            for k, v in ratios_raw.items()
        }
        normalize_terms = {
            str(k): _to_bool(v, True)
            for k, v in normalize_raw.items()
        }

        clamp_min = _to_non_negative_float(raw.get("clamp_min_scale", 0.25), 0.25)
        clamp_max = _to_non_negative_float(raw.get("clamp_max_scale", 4.0), 4.0)
        if clamp_max < clamp_min:
            clamp_max = clamp_min

        eps = _to_non_negative_float(raw.get("eps", 1e-8), 1e-8)
        eps = max(1e-12, eps)

        return cls(
            enabled=_to_bool(raw.get("enabled", False), False),
            ema_decay=_to_unit_float(raw.get("ema_decay", 0.98), 0.98),
            warmup_steps=max(0, int(raw.get("warmup_steps", 0))),
            eps=eps,
            clamp_min_scale=clamp_min,
            clamp_max_scale=clamp_max,
            frequency_aware=_to_bool(raw.get("frequency_aware", True), True),
            min_target_contribution=_to_non_negative_float(raw.get("min_target_contribution", 0.0), 0.0),
            max_target_contribution=_to_non_negative_float(raw.get("max_target_contribution", 0.0), 0.0),
            target_ratio_default=_to_non_negative_float(raw.get("target_ratio_default", 0.25), 0.25),
            target_ratios=target_ratios,
            normalize_terms=normalize_terms,
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step_count": int(self.step_count),
            "recon_ema": float(self.recon_ema) if self.recon_ema is not None else None,
            "terms": {
                name: {
                    "ema_loss": float(st.ema_loss) if st.ema_loss is not None else None,
                    "last_scale": float(st.last_scale),
                    "last_weight_base": float(st.last_weight_base),
                    "last_weight_effective": float(st.last_weight_effective),
                    "last_contrib_base": float(st.last_contrib_base),
                    "last_contrib_effective": float(st.last_contrib_effective),
                    "last_target": float(st.last_target),
                    "last_expected": float(st.last_expected),
                    "last_frequency": float(st.last_frequency),
                }
                for name, st in self._terms.items()
            },
        }

    def load_state_dict(self, state: Optional[Mapping[str, Any]]) -> None:
        if not isinstance(state, Mapping):
            return
        try:
            self.step_count = max(0, int(state.get("step_count", self.step_count)))
        except Exception:
            pass
        recon_ema = state.get("recon_ema", self.recon_ema)
        if recon_ema is None:
            self.recon_ema = None
        else:
            try:
                self.recon_ema = max(0.0, float(recon_ema))
            except Exception:
                pass
        terms_raw = state.get("terms", {})
        if not isinstance(terms_raw, Mapping):
            return
        for name, item in terms_raw.items():
            if not isinstance(item, Mapping):
                continue
            st = self._terms.setdefault(str(name), _TermState())
            ema_raw = item.get("ema_loss", None)
            if ema_raw is None:
                st.ema_loss = None
            else:
                try:
                    st.ema_loss = max(0.0, float(ema_raw))
                except Exception:
                    pass
            for field_name in (
                "last_scale",
                "last_weight_base",
                "last_weight_effective",
                "last_contrib_base",
                "last_contrib_effective",
                "last_target",
                "last_expected",
                "last_frequency",
            ):
                try:
                    setattr(st, field_name, float(item.get(field_name, getattr(st, field_name))))
                except Exception:
                    continue

    @staticmethod
    def _tensor_to_abs_scalar(loss_value: Any) -> float:
        if isinstance(loss_value, torch.Tensor):
            val = float(loss_value.detach().to(dtype=torch.float32).abs().mean().item())
        else:
            val = abs(float(loss_value))
        if not (val == val) or val == float("inf"):  # NaN or +inf
            return 0.0
        return max(0.0, val)

    def _ema_update(self, prev: Optional[float], value: float) -> float:
        if prev is None:
            return float(value)
        d = float(self.ema_decay)
        return float(d * prev + (1.0 - d) * value)

    def begin_step(self, recon_loss: Any) -> None:
        recon_abs = self._tensor_to_abs_scalar(recon_loss)
        self.recon_ema = self._ema_update(self.recon_ema, recon_abs)
        self.step_count += 1

    def _target_ratio_for(self, term: str) -> float:
        if term in self.target_ratios:
            return max(0.0, float(self.target_ratios[term]))
        return max(0.0, float(self.target_ratio_default))

    def _normalize_term_enabled(self, term: str) -> bool:
        if term in self.normalize_terms:
            return bool(self.normalize_terms[term])
        return True

    def compute_weight(
        self,
        *,
        term: str,
        base_weight: float,
        raw_loss: Any,
        frequency: float = 1.0,
    ) -> tuple[float, Dict[str, float]]:
        """Return effective weight and diagnostic stats for one term."""
        name = str(term)
        st = self._terms.setdefault(name, _TermState())

        base_w = max(0.0, float(base_weight))
        raw_abs = self._tensor_to_abs_scalar(raw_loss)
        st.ema_loss = self._ema_update(st.ema_loss, raw_abs)

        freq = max(self.eps, float(frequency)) if self.frequency_aware else 1.0
        expected = max(self.eps, base_w * max(self.eps, float(st.ema_loss or 0.0)) * freq)

        target = expected
        scale = 1.0
        if (
            self.enabled
            and self._normalize_term_enabled(name)
            and base_w > 0.0
            and self.recon_ema is not None
            and int(self.step_count) > int(self.warmup_steps)
        ):
            target = float(self.recon_ema) * self._target_ratio_for(name)
            target = max(self.min_target_contribution, target)
            if self.max_target_contribution > 0.0:
                target = min(target, self.max_target_contribution)
            raw_scale = target / expected
            scale = max(self.clamp_min_scale, min(self.clamp_max_scale, raw_scale))

        eff_w = base_w * scale
        contrib_base = base_w * raw_abs * freq
        contrib_eff = eff_w * raw_abs * freq

        st.last_scale = float(scale)
        st.last_weight_base = float(base_w)
        st.last_weight_effective = float(eff_w)
        st.last_contrib_base = float(contrib_base)
        st.last_contrib_effective = float(contrib_eff)
        st.last_target = float(target)
        st.last_expected = float(expected)
        st.last_frequency = float(freq)

        return float(eff_w), {
            "scale": float(st.last_scale),
            "weight_base": float(st.last_weight_base),
            "weight_effective": float(st.last_weight_effective),
            "contrib_base": float(st.last_contrib_base),
            "contrib_effective": float(st.last_contrib_effective),
            "target": float(st.last_target),
            "expected": float(st.last_expected),
            "ema_loss": float(st.ema_loss or 0.0),
            "frequency": float(st.last_frequency),
            "enabled": 1.0 if (self.enabled and self._normalize_term_enabled(name)) else 0.0,
        }
