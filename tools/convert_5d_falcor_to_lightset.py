#!/usr/bin/env python3
"""Convert raw 5D Falcor light configs to LightSetDataset descriptors.

The Falcor 5D generator stores light_configs as raw parameters:
  [zenith, azimuth, intensity, color_temperature, cloud]

The current training path expects:
  light_configs: [M, N, 12]
  light_mask: [M, N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "2_src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.light_descriptor import build_descriptor_5d  # noqa: E402


PARAM_MIN = torch.tensor([0.0, 0.0, 0.1, 2500.0, 0.0], dtype=torch.float32)
PARAM_MAX = torch.tensor([90.0, 360.0, 2.0, 10000.0, 0.9], dtype=torch.float32)


def _metadata_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        else:
            return {"source_metadata_array": value.tolist()}
    if isinstance(value, dict):
        return dict(value)
    return {"source_metadata": str(value)}


def convert(src: Path, dst_dir: Path) -> Path:
    src_npz = src / "parametric_tensor.npz" if src.is_dir() else src
    if not src_npz.exists():
        raise FileNotFoundError(src_npz)

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_npz = dst_dir / "parametric_tensor.npz"

    with np.load(src_npz, allow_pickle=True) as data:
        tensor = np.asarray(data["tensor"], dtype=np.float32)
        probe_positions = np.asarray(data["probe_positions"], dtype=np.float32)
        raw_light_configs = np.asarray(data["light_configs"], dtype=np.float32)
        valid_mask = np.asarray(data["valid_mask"], dtype=np.float32) if "valid_mask" in data.files else None
        metadata = _metadata_to_dict(data["metadata"] if "metadata" in data.files else None)

    if raw_light_configs.ndim == 3 and raw_light_configs.shape[1] == 1:
        raw_light_configs = raw_light_configs[:, 0, :]
    if raw_light_configs.ndim != 2 or raw_light_configs.shape[1] != 5:
        raise ValueError(
            "Expected raw light_configs shape [M,5] or [M,1,5], "
            f"got {tuple(raw_light_configs.shape)}"
        )
    if tensor.ndim != 3 or tensor.shape[1] != raw_light_configs.shape[0]:
        raise ValueError(
            "tensor shape must be [P,M,27] and match light config count; "
            f"tensor={tuple(tensor.shape)}, light_configs={tuple(raw_light_configs.shape)}"
        )

    raw = torch.from_numpy(raw_light_configs)
    raw_norm = 2.0 * (raw - PARAM_MIN) / (PARAM_MAX - PARAM_MIN) - 1.0
    desc = build_descriptor_5d(raw_norm, PARAM_MIN, PARAM_MAX).cpu().numpy().astype(np.float32)
    desc = desc[:, None, :]
    light_mask = np.ones(desc.shape[:2], dtype=np.float32)

    metadata.update(
        {
            "converted_to_lightset_descriptor": True,
            "source_parametric_tensor": str(src_npz),
            "num_lights": 1,
            "light_descriptor_dim": 12,
            "raw_5d_param_min": PARAM_MIN.cpu().numpy().tolist(),
            "raw_5d_param_max": PARAM_MAX.cpu().numpy().tolist(),
        }
    )

    payload: dict[str, Any] = {
        "tensor": tensor,
        "probe_positions": probe_positions,
        "light_configs": desc,
        "light_mask": light_mask,
        "metadata": metadata,
    }
    if valid_mask is not None:
        payload["valid_mask"] = valid_mask

    np.savez_compressed(dst_npz, **payload)
    return dst_npz


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input directory or parametric_tensor.npz")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    out = convert(Path(args.input), Path(args.output))
    with np.load(out, allow_pickle=True) as data:
        print(f"Wrote {out}")
        print(f"tensor={data['tensor'].shape}")
        print(f"probe_positions={data['probe_positions'].shape}")
        print(f"light_configs={data['light_configs'].shape}")
        print(f"light_mask={data['light_mask'].shape}")
        print(f"tensor_rms={float(np.sqrt(np.mean(np.asarray(data['tensor'], dtype=np.float32) ** 2))):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
