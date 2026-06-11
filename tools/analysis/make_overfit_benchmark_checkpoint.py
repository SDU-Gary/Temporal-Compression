#!/usr/bin/env python3
"""Wrap an overfit_lightset_probe checkpoint with benchmark model metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _resolve_npz(dataset: Path) -> Path:
    return dataset / "parametric_tensor.npz" if dataset.is_dir() else dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input", required=True, help="overfit model.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--light-dim", type=int, default=12)
    parser.add_argument("--intensity-dim", type=int, default=3)
    parser.add_argument("--intensity-offset", type=int, default=1)
    parser.add_argument("--light-encoder-mode", default="bypass_linear")
    parser.add_argument("--bypass-feature-pairs", default="[[0,7],[0,8],[0,9]]")
    args = parser.parse_args()

    dataset_npz = _resolve_npz(Path(args.dataset))
    if not dataset_npz.exists():
        raise FileNotFoundError(dataset_npz)
    ckpt = torch.load(str(args.input), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    summary = ckpt.get("summary", {}) if isinstance(ckpt, dict) else {}

    with np.load(dataset_npz, allow_pickle=True) as data:
        light_configs = np.asarray(data["light_configs"], dtype=np.float32)
    if light_configs.ndim != 3 or light_configs.shape[2] < 10:
        raise ValueError(f"Expected light_configs [M,N,>=10], got {light_configs.shape}")

    # Keep parity with overfit_lightset_probe: bypass uses first light direction.
    feats = light_configs[:, 0, [7, 8, 9]].astype(np.float32)
    mean = feats.mean(axis=0).astype(float).tolist()
    std = np.maximum(feats.std(axis=0), 1e-6).astype(float).tolist()

    pairs = json.loads(args.bypass_feature_pairs)
    model_meta = {
        "num_gaussians": int(state["mu"].shape[0]),
        "rank": int(state["U"].shape[-1]),
        "sh_dim": int(state["U"].shape[1]),
        "top_k": int(args.top_k),
        "light_dim": int(args.light_dim),
        "embed_dim": int(state["coeffs"].shape[-1]),
        "intensity_dim": int(args.intensity_dim),
        "intensity_offset": int(args.intensity_offset),
        "enable_film": True,
        "light_encoder_mode": str(args.light_encoder_mode),
        "bypass_feature_pairs": pairs,
        "bypass_feature_norm_mean": mean,
        "bypass_feature_norm_std": std,
    }
    payload = {
        "model_state_dict": state,
        "meta": {
            "model": model_meta,
            "source_overfit_summary": summary,
            "dataset": str(dataset_npz),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out))
    print(f"Wrote {out}")
    print(json.dumps(model_meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
