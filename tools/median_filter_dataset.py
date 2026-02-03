#!/usr/bin/env python3
"""Apply temporal median filter to SH tensor along time axis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np


def _median_filter_time(
    tensor: np.ndarray,
    kernel: int,
    pad_mode: str = "edge",
) -> np.ndarray:
    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor shape [P, M, C], got {tensor.shape}")
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("kernel size must be an odd positive integer")

    pad = kernel // 2
    padded = np.pad(tensor, ((0, 0), (pad, pad), (0, 0)), mode=pad_mode)
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=kernel, axis=1)
    # windows: [P, M, C, K]
    return np.median(windows, axis=-1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Median filter SH tensor along time axis")
    parser.add_argument("--input", required=True, help="Input .npz dataset")
    parser.add_argument("--output", required=True, help="Output .npz dataset")
    parser.add_argument("--kernel", type=int, default=3, help="Median filter window size (odd, e.g. 3 or 5)")
    parser.add_argument("--pad-mode", type=str, default="edge", choices=["edge", "reflect", "symmetric"], help="Padding mode")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    data = np.load(in_path, allow_pickle=True)

    if "tensor" not in data:
        raise ValueError("Input dataset missing 'tensor' array.")

    tensor = data["tensor"].astype(np.float32, copy=False)
    smoothed = _median_filter_time(tensor, int(args.kernel), pad_mode=str(args.pad_mode))

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
                md["median_kernel"] = int(args.kernel)
                md["median_pad_mode"] = str(args.pad_mode)
                md["median_source"] = str(in_path)
                payload["metadata"] = md
        except Exception:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"Saved median-filtered dataset to: {out_path}")


if __name__ == "__main__":
    main()
