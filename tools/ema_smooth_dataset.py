#!/usr/bin/env python3
"""Apply temporal EMA smoothing to SH tensor in a dataset .npz."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np


def _ema_forward(tensor: np.ndarray, alpha: float) -> np.ndarray:
    """Forward EMA over time axis (axis=1)."""
    out = np.empty_like(tensor)
    out[:, 0, :] = tensor[:, 0, :]
    for i in range(1, tensor.shape[1]):
        out[:, i, :] = alpha * tensor[:, i, :] + (1.0 - alpha) * out[:, i - 1, :]
    return out


def _ema_backward(tensor: np.ndarray, alpha: float) -> np.ndarray:
    """Backward EMA over time axis (axis=1)."""
    out = np.empty_like(tensor)
    out[:, -1, :] = tensor[:, -1, :]
    for i in range(tensor.shape[1] - 2, -1, -1):
        out[:, i, :] = alpha * tensor[:, i, :] + (1.0 - alpha) * out[:, i + 1, :]
    return out


def _apply_ema(tensor: np.ndarray, alpha: float, bidirectional: bool) -> np.ndarray:
    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor with shape [P, M, C], got {tensor.shape}")
    out = _ema_forward(tensor, alpha)
    if bidirectional:
        out = _ema_backward(out, alpha)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="EMA smooth SH tensor along time axis")
    parser.add_argument("--input", required=True, help="Input .npz dataset")
    parser.add_argument("--output", required=True, help="Output .npz dataset")
    parser.add_argument("--alpha", type=float, default=0.8, help="EMA alpha (0-1), higher = smoother")
    parser.add_argument("--bidirectional", action="store_true", default=True, help="Apply EMA forward+backward to avoid lag")
    parser.add_argument("--no-bidirectional", action="store_false", dest="bidirectional")
    args = parser.parse_args()

    if not (0.0 < args.alpha <= 1.0):
        raise ValueError("--alpha must be in (0, 1].")

    in_path = Path(args.input)
    out_path = Path(args.output)
    data = np.load(in_path, allow_pickle=True)

    if "tensor" not in data:
        raise ValueError("Input dataset missing 'tensor' array.")

    tensor = data["tensor"].astype(np.float32, copy=False)
    smoothed = _apply_ema(tensor, float(args.alpha), bool(args.bidirectional))

    payload: Dict[str, Any] = {}
    for key in data.files:
        if key == "tensor":
            payload[key] = smoothed
        else:
            payload[key] = data[key]

    if "metadata" in payload:
        try:
            md = payload["metadata"].item()
            if isinstance(md, dict):
                md = dict(md)
                md["ema_alpha"] = float(args.alpha)
                md["ema_bidirectional"] = bool(args.bidirectional)
                md["ema_source"] = str(in_path)
                payload["metadata"] = md
        except Exception:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"Saved EMA-smoothed dataset to: {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
