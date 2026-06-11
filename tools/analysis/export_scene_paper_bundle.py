#!/usr/bin/env python3
"""Export paper-facing metrics, curves, timing CSVs, and PNG frames for one scene."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _resolve_npz(path: Path) -> Path:
    return path / "parametric_tensor.npz" if path.is_dir() else path


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_u8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        if image.ndim == 3 and image.shape[-1] > 3:
            image = image[..., :3]
        return image
    img = np.clip(image.astype(np.float32), 0.0, 1.0)
    if img.ndim == 3 and img.shape[-1] > 3:
        img = img[..., :3]
    return (img * 255.0 + 0.5).astype(np.uint8)


def _save_png(path: Path, image_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image  # type: ignore

        Image.fromarray(image_u8).save(str(path))
    except Exception:
        import binascii
        import struct
        import zlib

        img = np.asarray(image_u8, dtype=np.uint8)
        height, width, _ = img.shape
        raw = b"".join(b"\x00" + img[y].tobytes() for y in range(height))

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack("!I", len(data))
                + tag
                + data
                + struct.pack("!I", binascii.crc32(tag + data) & 0xFFFFFFFF)
            )

        png = b"".join(
            [
                b"\x89PNG\r\n\x1a\n",
                chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)),
                chunk(b"IDAT", zlib.compress(raw, level=6)),
                chunk(b"IEND", b""),
            ]
        )
        path.write_bytes(png)


def _extract_frame_id(path: Path) -> int:
    stem = path.stem
    if stem.startswith("frame_"):
        return int(stem.split("_", 1)[1])
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 0


def _export_png_frames(sampled_dir: Path, png_dir: Path, scene_name: str, err_gain: float) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    npz_files = sorted(sampled_dir.glob("frame_*.npz"), key=_extract_frame_id)
    if not npz_files:
        return records
    for seq, npz_path in enumerate(npz_files, start=1):
        data = np.load(npz_path, allow_pickle=False)
        if "gt_srgb_u8" in data:
            gt = _to_u8(data["gt_srgb_u8"])
        elif "gt_srgb" in data:
            gt = _to_u8(data["gt_srgb"])
        else:
            raise KeyError(f"{npz_path} missing gt_srgb/gt_srgb_u8")

        if "model_srgb_u8" in data:
            pred = _to_u8(data["model_srgb_u8"])
        elif "model_srgb" in data:
            pred = _to_u8(data["model_srgb"])
        else:
            raise KeyError(f"{npz_path} missing model_srgb/model_srgb_u8")

        if "err_srgb" in data:
            err = _to_u8(np.asarray(data["err_srgb"], dtype=np.float32) * float(err_gain))
        else:
            err = _to_u8(np.abs(gt.astype(np.float32) - pred.astype(np.float32)) / 255.0 * float(err_gain))

        compare = np.concatenate([gt, pred, err], axis=1)
        stem = f"{scene_name}_frame{seq:04d}"
        paths = {
            "gt_png": png_dir / f"{stem}_gt.png",
            "pred_png": png_dir / f"{stem}_pred.png",
            "err_png": png_dir / f"{stem}_err.png",
            "compare_png": png_dir / f"{stem}_compare.png",
        }
        _save_png(paths["gt_png"], gt)
        _save_png(paths["pred_png"], pred)
        _save_png(paths["err_png"], err)
        _save_png(paths["compare_png"], compare)
        records.append(
            {
                "source_npz": str(npz_path),
                "source_frame": _extract_frame_id(npz_path),
                **{k: str(v) for k, v in paths.items()},
            }
        )
    return records


def _param_stats(checkpoint: Path) -> dict[str, Any]:
    if not checkpoint.exists():
        return {}
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    if not isinstance(state, dict):
        return {}
    elems = 0
    bytes_ = 0
    for value in state.values():
        if torch.is_tensor(value):
            elems += int(value.numel())
            bytes_ += int(value.numel() * value.element_size())
    return {
        "parameter_count": elems,
        "state_tensor_bytes": bytes_,
        "checkpoint_bytes": checkpoint.stat().st_size,
    }


def _clean_number(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _clean_number(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_number(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--scene-type", default="")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--benchmark-summary", default=None)
    parser.add_argument("--timing-summary", default=None)
    parser.add_argument("--sampled-images-dir", default=None)
    parser.add_argument("--png-dir", default=None)
    parser.add_argument("--err-gain", type=float, default=6.0)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    dataset_npz = _resolve_npz(Path(args.dataset))
    summary_path = result_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    overfit = _json(summary_path)
    benchmark_path = Path(args.benchmark_summary) if args.benchmark_summary else result_dir / "benchmark_1080p_images" / "benchmark_summary.json"
    timing_path = Path(args.timing_summary) if args.timing_summary else result_dir / "pipeline_stage_profile_1080p" / "pipeline_stage_summary.json"
    checkpoint = Path(args.checkpoint) if args.checkpoint else result_dir / "benchmark_checkpoint.pt"

    with np.load(dataset_npz, allow_pickle=True) as data:
        tensor = np.asarray(data["tensor"], dtype=np.float32)
        probes = np.asarray(data["probe_positions"], dtype=np.float32)
        light_configs = np.asarray(data["light_configs"], dtype=np.float32)
        metadata = data["metadata"].item() if "metadata" in data.files else {}
    raw_bytes = int(tensor.size * tensor.itemsize)

    benchmark = _json(benchmark_path) if benchmark_path.exists() else {}
    timing = _json(timing_path) if timing_path.exists() else {}
    params = _param_stats(checkpoint)
    if params.get("state_tensor_bytes"):
        params["raw_sh_to_state_tensor_ratio"] = float(raw_bytes / params["state_tensor_bytes"])
    if params.get("checkpoint_bytes"):
        params["raw_sh_to_checkpoint_ratio"] = float(raw_bytes / params["checkpoint_bytes"])

    image_records: list[dict[str, Any]] = []
    sampled_dir = Path(args.sampled_images_dir) if args.sampled_images_dir else result_dir / "render_exports_npz"
    png_dir = Path(args.png_dir) if args.png_dir else result_dir / "render_exports_png"
    if sampled_dir.exists():
        image_records = _export_png_frames(sampled_dir, png_dir, args.scene_name, args.err_gain)

    hist = overfit.get("history", [])
    curve_path = result_dir / "training_curve_for_paper.csv"
    if hist:
        keys = ["step", "loss", "mae", "rmse", "mse", "range_psnr", "target_rms", "pred_rms", "target_p99_abs", "pred_p99_abs"]
        with curve_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in hist:
                writer.writerow({k: row.get(k) for k in keys})

    timing_csv = result_dir / "timing_for_paper.csv"
    mapping = timing.get("requested_stage_mapping_ms", {}) if isinstance(timing, dict) else {}
    if mapping:
        with timing_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["stage", "mean_ms", "p50_ms", "p95_ms", "count"])
            writer.writeheader()
            for stage, payload in mapping.items():
                if isinstance(payload, dict):
                    writer.writerow(
                        {
                            "stage": stage,
                            "mean_ms": payload.get("mean"),
                            "p50_ms": payload.get("p50"),
                            "p95_ms": payload.get("p95"),
                            "count": payload.get("count"),
                        }
                    )

    paper = {
        "scene": {
            "name": args.scene_name,
            "type": args.scene_type,
            "dataset_npz": str(dataset_npz),
            "num_probes": int(tensor.shape[0]),
            "num_light_configs": int(tensor.shape[1]),
            "sh_dim": int(tensor.shape[2]),
            "num_samples": int(tensor.shape[0] * tensor.shape[1]),
            "split": "overfit: all samples used for training and evaluation",
            "raw_sh_float32_bytes": raw_bytes,
            "raw_sh_float32_kib": float(raw_bytes / 1024.0),
            "dataset_npz_bytes": int(dataset_npz.stat().st_size),
            "metadata": metadata if isinstance(metadata, dict) else {},
        },
        "model": {
            "K": overfit.get("num_gaussians"),
            "rank": overfit.get("rank"),
            "embed_dim": overfit.get("embed_dim"),
            "film": "enabled",
            "soft_routing": overfit.get("soft_routing"),
            "soft_topk": overfit.get("routing_soft_topk"),
            "temperature": overfit.get("routing_temperature"),
            **params,
        },
        "training": {
            "steps": int(hist[-1]["step"]) if hist else None,
            "eval_interval_steps": int(hist[1]["step"] - hist[0]["step"]) if len(hist) > 2 else None,
            "standardized_sh_mse": overfit.get("standardize_loss"),
            "best": overfit.get("best"),
            "final": overfit.get("final"),
        },
        "render_metrics_1080p": benchmark.get("image_metrics", {}),
        "timing_1080p": mapping,
        "render_exports": {
            "npz_dir": str(sampled_dir),
            "png_dir": str(png_dir),
            "frames": image_records,
        },
        "source_files": {
            "overfit_summary": str(summary_path),
            "benchmark_summary": str(benchmark_path),
            "timing_summary": str(timing_path),
            "checkpoint": str(checkpoint),
        },
    }
    out_path = result_dir / "paper_metrics_summary.json"
    out_path.write_text(json.dumps(_clean_number(paper), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {curve_path}")
    if mapping:
        print(f"Wrote {timing_csv}")
    if image_records:
        print(f"Wrote {len(image_records)} PNG frame groups to {png_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
