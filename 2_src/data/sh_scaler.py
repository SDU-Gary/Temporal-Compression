"""Adaptive scaling for SH coefficients.

Scales L0 (per RGB channel) with mean/std and higher-order terms with RMS
to stabilize training while preserving directional details.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


L0_INDICES = (0, 9, 18)


def _ensure_numpy(array: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return array


def _ensure_torch(array: np.ndarray | torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array.to(device) if device is not None else array
    tensor = torch.from_numpy(array)
    return tensor.to(device) if device is not None else tensor


@dataclass
class AdaptiveSHScaler:
    """Scale SH coefficients with separate L0 and higher-order normalization."""

    l0_mean: np.ndarray  # [3]
    l0_std: np.ndarray  # [3]
    ho_rms: np.ndarray  # [3]
    eps: float = 1e-6

    @classmethod
    def fit(cls, sh_coeffs: np.ndarray | torch.Tensor, eps: float = 1e-6) -> "AdaptiveSHScaler":
        sh = _ensure_numpy(sh_coeffs)
        sh = sh.reshape(-1, 27)

        l0 = sh[:, list(L0_INDICES)]  # [N, 3]
        l0_mean = l0.mean(axis=0)
        l0_std = l0.std(axis=0)

        # Higher-order coefficients per channel [N, 3, 8]
        ho = np.zeros((sh.shape[0], 3, 8), dtype=sh.dtype)
        for c in range(3):
            base = c * 9
            ho[:, c, :] = sh[:, base + 1 : base + 9]

        ho_rms = np.sqrt(np.mean(ho * ho, axis=(0, 2)))

        # Prevent division by zero
        l0_std = np.maximum(l0_std, eps)
        ho_rms = np.maximum(ho_rms, eps)

        return cls(l0_mean=l0_mean, l0_std=l0_std, ho_rms=ho_rms, eps=eps)

    @classmethod
    def fit_from_dataset(
        cls, dataset, max_samples: Optional[int] = None, eps: float = 1e-6
    ) -> "AdaptiveSHScaler":
        """Fit scaler from a dataset object.

        Tries common attributes: tensor_subset, sh_coeffs_subset, tensor, sh_coeffs_all.
        Falls back to sampling dataset items if needed.
        """
        sh_array = None
        for attr in ("tensor_subset", "sh_coeffs_subset", "tensor", "sh_coeffs_all"):
            if hasattr(dataset, attr):
                sh_array = getattr(dataset, attr)
                break

        if sh_array is not None:
            sh = np.array(sh_array).reshape(-1, 27)
            if max_samples is not None and sh.shape[0] > max_samples:
                rng = np.random.default_rng(42)
                idx = rng.choice(sh.shape[0], size=max_samples, replace=False)
                sh = sh[idx]
            return cls.fit(sh, eps=eps)

        # Fallback: sample items from dataset
        if max_samples is None:
            max_samples = min(len(dataset), 50000)

        samples = []
        for i in range(min(len(dataset), max_samples)):
            item = dataset[i]
            if "sh_coeffs" not in item:
                raise KeyError("Dataset item does not contain 'sh_coeffs'")
            samples.append(item["sh_coeffs"].numpy())
        sh = np.stack(samples, axis=0)
        return cls.fit(sh, eps=eps)

    def transform(self, sh_coeffs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Scale SH coefficients to normalized space."""
        if isinstance(sh_coeffs, torch.Tensor):
            return self._transform_torch(sh_coeffs)
        return self._transform_numpy(sh_coeffs)

    def inverse(self, sh_scaled: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Invert scaling to recover physical SH coefficients."""
        if isinstance(sh_scaled, torch.Tensor):
            return self._inverse_torch(sh_scaled)
        return self._inverse_numpy(sh_scaled)

    def _transform_numpy(self, sh: np.ndarray) -> np.ndarray:
        out = sh.copy()
        for c in range(3):
            base = c * 9
            out[..., base] = (out[..., base] - self.l0_mean[c]) / (self.l0_std[c] + self.eps)
            out[..., base + 1 : base + 9] = out[..., base + 1 : base + 9] / (self.ho_rms[c] + self.eps)
        return out

    def _inverse_numpy(self, sh: np.ndarray) -> np.ndarray:
        out = sh.copy()
        for c in range(3):
            base = c * 9
            out[..., base] = out[..., base] * (self.l0_std[c] + self.eps) + self.l0_mean[c]
            out[..., base + 1 : base + 9] = out[..., base + 1 : base + 9] * (self.ho_rms[c] + self.eps)
        return out

    def _transform_torch(self, sh: torch.Tensor) -> torch.Tensor:
        out = sh.clone()
        l0_mean = _ensure_torch(self.l0_mean, device=sh.device)
        l0_std = _ensure_torch(self.l0_std, device=sh.device)
        ho_rms = _ensure_torch(self.ho_rms, device=sh.device)
        for c in range(3):
            base = c * 9
            out[..., base] = (out[..., base] - l0_mean[c]) / (l0_std[c] + self.eps)
            out[..., base + 1 : base + 9] = out[..., base + 1 : base + 9] / (ho_rms[c] + self.eps)
        return out

    def _inverse_torch(self, sh: torch.Tensor) -> torch.Tensor:
        out = sh.clone()
        l0_mean = _ensure_torch(self.l0_mean, device=sh.device)
        l0_std = _ensure_torch(self.l0_std, device=sh.device)
        ho_rms = _ensure_torch(self.ho_rms, device=sh.device)
        for c in range(3):
            base = c * 9
            out[..., base] = out[..., base] * (l0_std[c] + self.eps) + l0_mean[c]
            out[..., base + 1 : base + 9] = out[..., base + 1 : base + 9] * (ho_rms[c] + self.eps)
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            l0_mean=self.l0_mean,
            l0_std=self.l0_std,
            ho_rms=self.ho_rms,
            eps=np.array(self.eps, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "AdaptiveSHScaler":
        data = np.load(Path(path))
        eps = 1e-6
        if "eps" in data.files:
            eps = float(data["eps"])
        return cls(
            l0_mean=data["l0_mean"],
            l0_std=data["l0_std"],
            ho_rms=data["ho_rms"],
            eps=eps,
        )
