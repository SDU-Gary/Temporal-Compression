#!/usr/bin/env python3
"""Slice probe field and save a heatmap for SH-derived values."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
_CORE = _ROOT / "2_src"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from utils.probe_field_slicer import slice_probe_field, SliceResult


def _load_sh_from_npz(npz_path: Path, config_idx: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    probes = data["probe_positions"]
    tensor = data["tensor"]
    if config_idx < 0 or config_idx >= tensor.shape[1]:
        raise ValueError(f"config_idx out of range: {config_idx} (0..{tensor.shape[1]-1})")
    sh = tensor[:, config_idx, :]
    return probes, sh


def _load_pred_sh(pred_path: Path) -> np.ndarray:
    if pred_path.suffix == ".npy":
        return np.load(pred_path)
    if pred_path.suffix == ".npz":
        data = np.load(pred_path)
        if "sh_coeffs" in data:
            return data["sh_coeffs"]
        # fallback: first array
        for key in data.files:
            return data[key]
    raise ValueError("Unsupported pred_sh format (use .npy or .npz)")


def _save_heatmap_png(heatmap: np.ndarray, output_path: Path) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None

    plt.figure(figsize=(6, 5))
    plt.imshow(heatmap, origin="lower", cmap="inferno")
    plt.colorbar(label="Intensity")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def _slice_coeff(
    probe_positions: np.ndarray,
    sh_coeffs: np.ndarray,
    axis: str,
    value: float,
    tolerance: float,
    grid_size: int,
    coeff_idx: int,
) -> SliceResult:
    axis = axis.lower()
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be one of: x, y, z")
    if coeff_idx < 0 or coeff_idx >= sh_coeffs.shape[1]:
        raise ValueError(f"coeff_idx out of range: {coeff_idx} (0..{sh_coeffs.shape[1]-1})")

    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    other_axes = [i for i in range(3) if i != axis_idx]

    mask = np.abs(probe_positions[:, axis_idx] - value) <= tolerance
    if not np.any(mask):
        raise ValueError("No probes found within the slice tolerance.")

    positions_2d = probe_positions[mask][:, other_axes]
    values = sh_coeffs[mask][:, coeff_idx]

    u = positions_2d[:, 0]
    v = positions_2d[:, 1]
    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())

    eps = 1e-8
    u_norm = (u - u_min) / (u_max - u_min + eps)
    v_norm = (v - v_min) / (v_max - v_min + eps)

    H = grid_size
    W = grid_size
    heatmap = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.int32)

    uu = np.clip((u_norm * (W - 1)).astype(np.int32), 0, W - 1)
    vv = np.clip((v_norm * (H - 1)).astype(np.int32), 0, H - 1)

    for idx in range(len(values)):
        heatmap[vv[idx], uu[idx]] += values[idx]
        counts[vv[idx], uu[idx]] += 1

    valid = counts > 0
    heatmap[valid] /= counts[valid]

    return SliceResult(
        heatmap=heatmap,
        counts=counts,
        extent=(u_min, u_max, v_min, v_max),
        axis=axis,
        value=value,
        tolerance=tolerance,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe field slicer")
    parser.add_argument("--data-root", required=True, help="Dataset root containing parametric_tensor.npz")
    parser.add_argument("--config-idx", type=int, default=0, help="Light config index")
    parser.add_argument("--pred-sh", type=str, default=None, help="Optional predicted SH file (.npy/.npz)")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="y")
    parser.add_argument("--value", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--mode", choices=["l0", "dir", "coeff"], default="l0")
    parser.add_argument("--direction", type=float, nargs=3, default=None,
                        help="Direction vector for mode=dir (x y z)")
    parser.add_argument("--coeff-idx", type=int, default=0,
                        help="SH coefficient index when mode=coeff (0..26)")
    parser.add_argument("--output", type=str, default=None, help="Output npz path")
    parser.add_argument("--save-png", action="store_true", help="Also save PNG heatmap if matplotlib is available")

    args = parser.parse_args()

    data_root = Path(args.data_root)
    npz_path = data_root / "parametric_tensor.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"parametric_tensor.npz not found at {npz_path}")

    probe_positions, sh = _load_sh_from_npz(npz_path, args.config_idx)

    if args.pred_sh:
        pred = _load_pred_sh(Path(args.pred_sh))
        if pred.shape != sh.shape:
            raise ValueError(f"pred_sh shape {pred.shape} does not match gt shape {sh.shape}")
        sh = pred

    direction = np.array(args.direction, dtype=np.float32) if args.direction else None
    if args.mode == "coeff":
        result = _slice_coeff(
            probe_positions=probe_positions,
            sh_coeffs=sh,
            axis=args.axis,
            value=args.value,
            tolerance=args.tolerance,
            grid_size=args.grid_size,
            coeff_idx=args.coeff_idx,
        )
    else:
        result = slice_probe_field(
            probe_positions=probe_positions,
            sh_coeffs=sh,
            axis=args.axis,
            value=args.value,
            tolerance=args.tolerance,
            grid_size=args.grid_size,
            mode=args.mode,
            direction=direction,
        )

    if args.output:
        output_path = Path(args.output)
    else:
        suffix = f"{args.axis}_{args.value:.2f}"
        if args.mode == "coeff":
            suffix = f"{suffix}_c{args.coeff_idx}"
        output_path = data_root / f"probe_slice_{suffix}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        heatmap=result.heatmap,
        counts=result.counts,
        extent=np.array(result.extent, dtype=np.float32),
        axis=result.axis,
        value=result.value,
        tolerance=result.tolerance,
    )
    print(f"Saved slice heatmap to {output_path}")

    if args.save_png:
        png_path = output_path.with_suffix(".png")
        saved = _save_heatmap_png(result.heatmap, png_path)
        if saved is None:
            print("matplotlib not available; PNG not saved.")
        else:
            print(f"Saved PNG heatmap to {saved}")


if __name__ == "__main__":
    main()
