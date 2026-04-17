#!/usr/bin/env python3
"""Unified GT/Pred video exporter based on Falcor benchmark pipeline.

This tool runs one consistent render pass (same scene/camera/time axis) and
exports:
1) gt.mp4
2) pred.mp4
3) compare.mp4 (GT|Pred side-by-side)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = ROOT / "tools" / "benchmark_realtime_pipeline.py"


def _resolve_dataset_npz(dataset_arg: str) -> Path:
    p = Path(dataset_arg)
    if p.is_dir():
        p = p / "parametric_tensor.npz"
    if not p.exists():
        raise FileNotFoundError(f"Dataset npz not found: {p}")
    return p.resolve()


def _read_dataset_meta(npz_path: Path) -> Tuple[Dict[str, Any], int]:
    data = np.load(npz_path, allow_pickle=True)
    metadata = data["metadata"].item() if "metadata" in data else {}
    tensor = np.asarray(data["tensor"])
    light_configs = np.asarray(data["light_configs"]) if "light_configs" in data else None
    num_frames_tensor = int(tensor.shape[1])
    if light_configs is not None:
        num_frames = min(num_frames_tensor, int(light_configs.shape[0]))
    else:
        num_frames = num_frames_tensor
    return metadata if isinstance(metadata, dict) else {}, num_frames


def _resolve_scene_path(scene_override: Optional[str], metadata: Dict[str, Any]) -> Path:
    if scene_override:
        scene_path = Path(scene_override).expanduser().resolve()
    else:
        scene_text = str(metadata.get("scene", "")).strip()
        if not scene_text:
            raise ValueError("Scene path missing. Provide --scene or dataset metadata['scene'].")
        scene_path = Path(scene_text).expanduser().resolve()
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")
    return scene_path


def _resolve_fps(fps_override: Optional[float], metadata: Dict[str, Any]) -> float:
    if fps_override is not None:
        fps = float(fps_override)
    else:
        fps = float(metadata.get("fps", 30.0))
    if fps <= 0.0:
        raise ValueError(f"fps must be > 0, got {fps}")
    return fps


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        t = value.strip().lower()
        if t in {"1", "true", "yes", "y", "on"}:
            return True
        if t in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _extract_routing_profile_from_summary(summary_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, None

    profile = data.get("profile")
    if not isinstance(profile, dict):
        return None, None

    out: Dict[str, Any] = {}
    if "top_k" in profile:
        try:
            out["top_k"] = int(profile["top_k"])
        except Exception:
            pass
    if "training_soft_routing" in profile:
        b = _coerce_bool(profile.get("training_soft_routing"))
        if b is not None:
            out["training_soft_routing"] = b
    if "routing_soft_topk" in profile:
        v = profile.get("routing_soft_topk")
        if v is None:
            out["routing_soft_topk"] = None
        else:
            try:
                out["routing_soft_topk"] = int(v)
            except Exception:
                pass
    if "routing_temperature" in profile:
        try:
            out["routing_temperature"] = float(profile["routing_temperature"])
        except Exception:
            pass

    summary_ckpt = data.get("checkpoint")
    summary_ckpt_text = str(summary_ckpt).strip() if summary_ckpt is not None else None
    if summary_ckpt_text == "":
        summary_ckpt_text = None

    return out or None, summary_ckpt_text


def _is_same_checkpoint(summary_ckpt_text: Optional[str], checkpoint_path: Path) -> bool:
    if not summary_ckpt_text:
        return False
    try:
        p = Path(summary_ckpt_text).expanduser()
        # Use string-compare fallback when path doesn't exist.
        if p.exists():
            return p.resolve() == checkpoint_path.resolve()
        return str(p) == str(checkpoint_path)
    except Exception:
        return False


def _resolve_routing_profile_json(
    checkpoint_path: Path,
    routing_profile_json: str,
) -> Tuple[str, Optional[str], bool]:
    explicit = str(routing_profile_json or "").strip()
    if explicit:
        return explicit, None, True

    candidates: List[Path] = []
    seen = set()
    for p in [checkpoint_path.parent, *list(checkpoint_path.parents[:4])]:
        cand = p / "global_best_falcor_summary.json"
        key = str(cand.resolve()) if cand.exists() else str(cand)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(cand)

    fallback_profile_json: str = ""
    fallback_source: Optional[str] = None
    for cand in candidates:
        if not cand.exists():
            continue
        profile, summary_ckpt_text = _extract_routing_profile_from_summary(cand)
        if profile:
            encoded = json.dumps(profile, ensure_ascii=False)
            source = str(cand.resolve())
            if _is_same_checkpoint(summary_ckpt_text, checkpoint_path):
                return encoded, source, True
            if not fallback_profile_json:
                fallback_profile_json = encoded
                fallback_source = source

    if fallback_profile_json:
        return fallback_profile_json, fallback_source, False
    return "", None, False


def _select_frames(
    num_frames_total: int,
    frame_start: int,
    frame_step: int,
    max_frames: Optional[int],
) -> List[int]:
    step = max(1, int(frame_step))
    start = max(0, int(frame_start))
    frame_ids = list(range(start, int(num_frames_total), step))
    if max_frames is not None:
        frame_ids = frame_ids[: max(0, int(max_frames))]
    if not frame_ids:
        raise ValueError("No frames selected. Check frame_start/frame_step/max_frames.")
    return frame_ids


def _tone_map_reinhard(image: np.ndarray) -> np.ndarray:
    img = np.maximum(image, 0.0)
    return img / (1.0 + img)


def _srgb_encode(image: np.ndarray) -> np.ndarray:
    img = np.maximum(image, 0.0)
    threshold = 0.0031308
    low = 12.92 * img
    high = 1.055 * np.power(img, 1.0 / 2.4) - 0.055
    return np.where(img <= threshold, low, high)


def _to_u8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected HxWxC image, got shape={arr.shape}")
    if arr.shape[2] > 3:
        arr = arr[..., :3]
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _save_png(path: Path, image_u8: np.ndarray) -> None:
    try:
        from PIL import Image  # type: ignore

        Image.fromarray(image_u8).save(str(path))
        return
    except Exception:
        import zlib
        import struct
        import binascii

        h, w, _ = image_u8.shape
        raw = b"".join(b"\x00" + image_u8[y].tobytes() for y in range(h))
        compressed = zlib.compress(raw, level=6)

        def _chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack("!I", len(data))
                + tag
                + data
                + struct.pack("!I", binascii.crc32(tag + data) & 0xFFFFFFFF)
            )

        png = b"".join(
            [
                b"\x89PNG\r\n\x1a\n",
                _chunk(b"IHDR", struct.pack("!IIBBBBB", w, h, 8, 2, 0, 0, 0)),
                _chunk(b"IDAT", compressed),
                _chunk(b"IEND", b""),
            ]
        )
        with open(path, "wb") as f:
            f.write(png)


def _extract_frame_id(npz_path: Path) -> int:
    stem = npz_path.stem
    if not stem.startswith("frame_"):
        raise ValueError(f"Unexpected sampled file name: {npz_path.name}")
    return int(stem.split("_", 1)[1])


@dataclass
class FrameRecord:
    seq_idx: int
    source_frame: int
    sampled_npz: str
    gt_png: str
    pred_png: str
    compare_png: str


def _build_frames(sampled_dir: Path, frames_root: Path) -> List[FrameRecord]:
    gt_dir = frames_root / "gt"
    pred_dir = frames_root / "pred"
    cmp_dir = frames_root / "compare"
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    cmp_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(sampled_dir.glob("frame_*.npz"), key=_extract_frame_id)
    if not npz_files:
        raise FileNotFoundError(f"No sampled frame npz found in: {sampled_dir}")

    records: List[FrameRecord] = []
    for seq_idx, npz_path in enumerate(npz_files):
        source_frame = _extract_frame_id(npz_path)
        data = np.load(npz_path, allow_pickle=False)

        if "gt_srgb_u8" in data:
            gt_u8 = np.asarray(data["gt_srgb_u8"], dtype=np.uint8)
        elif "gt_srgb" in data:
            gt_srgb = np.asarray(data["gt_srgb"], dtype=np.float32)
            gt_u8 = _to_u8_rgb(gt_srgb)
        elif "gt_linear" in data:
            gt_srgb = _srgb_encode(_tone_map_reinhard(np.asarray(data["gt_linear"], dtype=np.float32)))
            gt_u8 = _to_u8_rgb(gt_srgb)
        else:
            raise KeyError(f"{npz_path.name} missing gt_srgb_u8/gt_srgb/gt_linear")

        if "model_srgb_u8" in data:
            pred_u8 = np.asarray(data["model_srgb_u8"], dtype=np.uint8)
        elif "model_srgb" in data:
            pred_srgb = np.asarray(data["model_srgb"], dtype=np.float32)
            pred_u8 = _to_u8_rgb(pred_srgb)
        elif "model_linear" in data:
            pred_srgb = _srgb_encode(_tone_map_reinhard(np.asarray(data["model_linear"], dtype=np.float32)))
            pred_u8 = _to_u8_rgb(pred_srgb)
        else:
            raise KeyError(f"{npz_path.name} missing model_srgb_u8/model_srgb/model_linear")

        if gt_u8.shape != pred_u8.shape:
            raise ValueError(
                f"GT/Pred shape mismatch in {npz_path.name}: {gt_u8.shape} vs {pred_u8.shape}"
            )

        cmp_u8 = np.concatenate([gt_u8, pred_u8], axis=1)
        name = f"frame_{seq_idx:06d}.png"

        gt_png = gt_dir / name
        pred_png = pred_dir / name
        cmp_png = cmp_dir / name
        _save_png(gt_png, gt_u8)
        _save_png(pred_png, pred_u8)
        _save_png(cmp_png, cmp_u8)

        records.append(
            FrameRecord(
                seq_idx=seq_idx,
                source_frame=source_frame,
                sampled_npz=str(npz_path),
                gt_png=str(gt_png),
                pred_png=str(pred_png),
                compare_png=str(cmp_png),
            )
        )

    return records


def _run_cmd(cmd: Sequence[str]) -> None:
    print("[run]", " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], check=True)


def _encode_video(
    frames_dir: Path,
    output_mp4: Path,
    fps: float,
    codec: str,
    crf: int,
    preset: str,
) -> None:
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        str(frames_dir / "frame_%06d.png"),
        "-c:v",
        codec,
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(int(crf)),
        "-preset",
        preset,
        str(output_mp4),
    ]
    _run_cmd(cmd)


def _assert_ffmpeg_available() -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        raise RuntimeError("ffmpeg is required but not available in PATH.") from exc


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified GT/Pred/Compare video exporter")
    p.add_argument("--dataset", required=True, help="Dataset root or parametric_tensor.npz")
    p.add_argument("--checkpoint", required=True, help="Model checkpoint for prediction route")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--scene", default=None, help="Override scene path (otherwise from metadata.scene)")
    p.add_argument("--fps", type=float, default=None, help="Override FPS (otherwise from metadata.fps)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--render-scale", type=float, default=1.0)
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument(
        "--routing-profile-json",
        default="",
        help=(
            "Passed to benchmark --routing-profile-json. "
            "If empty, auto-load from checkpoint-adjacent global_best_falcor_summary.json profile."
        ),
    )
    p.add_argument(
        "--field-builder",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Probe field builder backend for export benchmark.",
    )
    p.add_argument(
        "--gbuffer-mode",
        choices=["realtime", "reuse_first"],
        default="reuse_first",
        help="GBuffer mode for export benchmark.",
    )
    p.add_argument(
        "--dump-sampled-images-every",
        type=int,
        default=1,
        help="Dump sampled images every N benchmark frames.",
    )
    p.add_argument(
        "--benchmark-compute-metrics",
        action="store_true",
        default=False,
        help="Also compute heavy benchmark metrics during export (slower).",
    )
    p.add_argument("--falcor-python-path", default=None)
    p.add_argument("--falcor-python-bin", default=None)
    p.add_argument("--worker-script", default=None)
    p.add_argument("--codec", default="libx264")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--preset", default="slow")
    p.add_argument("--skip-benchmark", action="store_true", default=False)
    p.add_argument("--keep-intermediate", action="store_true", default=False)
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    if (not args.benchmark_compute_metrics) and int(max(0, int(args.dump_sampled_images_every))) <= 0:
        raise ValueError(
            "No sampled frames will be produced. Set --dump-sampled-images-every > 0 "
            "or enable --benchmark-compute-metrics."
        )

    dataset_npz = _resolve_dataset_npz(args.dataset)
    metadata, num_frames_total = _read_dataset_meta(dataset_npz)
    scene_path = _resolve_scene_path(args.scene, metadata)
    fps = _resolve_fps(args.fps, metadata)
    selected_frames = _select_frames(
        num_frames_total=num_frames_total,
        frame_start=args.frame_start,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
    )
    if str(args.gbuffer_mode) == "reuse_first" and len(selected_frames) > 1:
        print(
            "Warning: gbuffer_mode=reuse_first with multi-frame export reuses first-frame "
            "GBuffer for all frames. If camera/geometry changes over time, use --gbuffer-mode realtime."
        )

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).resolve()
    (
        effective_routing_profile_json,
        auto_profile_source,
        auto_profile_checkpoint_match,
    ) = _resolve_routing_profile_json(
        checkpoint_path=checkpoint_path,
        routing_profile_json=args.routing_profile_json,
    )
    if effective_routing_profile_json:
        if auto_profile_source:
            if auto_profile_checkpoint_match:
                print(
                    "Auto-loaded routing profile from "
                    f"{auto_profile_source}: {effective_routing_profile_json}"
                )
            else:
                print(
                    "Auto-loaded routing profile from "
                    f"{auto_profile_source} (checkpoint mismatch fallback): "
                    f"{effective_routing_profile_json}"
                )
        else:
            print(f"Using explicit routing profile: {effective_routing_profile_json}")
    else:
        print("No routing profile provided or auto-discovered; benchmark default routing will be used.")

    benchmark_dir = out_dir / "benchmark"
    sampled_dir = out_dir / "sampled_npz"
    frames_dir = out_dir / "frames"
    videos_dir = out_dir / "videos"

    if not args.skip_benchmark:
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        sampled_dir.mkdir(parents=True, exist_ok=True)

        benchmark_cmd = [
            sys.executable,
            str(BENCHMARK_SCRIPT),
            "--dataset",
            str(dataset_npz),
            "--scene",
            str(scene_path),
            "--checkpoint",
            str(checkpoint_path),
            "--route",
            "both",
            "--output-dir",
            str(benchmark_dir),
            "--width",
            str(int(args.width)),
            "--height",
            str(int(args.height)),
            "--fps",
            f"{fps:.6f}",
            "--render-scale",
            str(float(args.render_scale)),
            "--frame-start",
            str(int(args.frame_start)),
            "--frame-step",
            str(max(1, int(args.frame_step))),
            "--max-frames",
            str(len(selected_frames)),
            "--warmup-frames",
            "0",
            "--benchmark-frames",
            str(len(selected_frames)),
            "--device",
            args.device,
            "--field-builder",
            str(args.field_builder),
            "--gbuffer-mode",
            str(args.gbuffer_mode),
            "--dump-sampled-images-every",
            str(int(max(0, int(args.dump_sampled_images_every)))),
            "--save-sampled-images-dir",
            str(sampled_dir),
        ]
        if args.benchmark_compute_metrics:
            benchmark_cmd.extend(["--compute-image-metrics", "--save-frame-metrics-every", "1"])

        if effective_routing_profile_json:
            benchmark_cmd.extend(["--routing-profile-json", str(effective_routing_profile_json)])
        if args.falcor_python_path:
            benchmark_cmd.extend(["--falcor-python-path", str(args.falcor_python_path)])
        if args.falcor_python_bin:
            benchmark_cmd.extend(["--falcor-python-bin", str(args.falcor_python_bin)])
        if args.worker_script:
            benchmark_cmd.extend(["--worker-script", str(args.worker_script)])

        _run_cmd(benchmark_cmd)
    else:
        if not sampled_dir.exists():
            raise FileNotFoundError(
                f"--skip-benchmark requires existing sampled npz dir: {sampled_dir}"
            )

    records = _build_frames(sampled_dir=sampled_dir, frames_root=frames_dir)

    _assert_ffmpeg_available()
    _encode_video(frames_dir / "gt", videos_dir / "gt.mp4", fps, args.codec, args.crf, args.preset)
    _encode_video(frames_dir / "pred", videos_dir / "pred.mp4", fps, args.codec, args.crf, args.preset)
    _encode_video(
        frames_dir / "compare", videos_dir / "compare.mp4", fps, args.codec, args.crf, args.preset
    )

    manifest = {
        "dataset_npz": str(dataset_npz),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "scene": str(scene_path),
        "fps": float(fps),
        "resolution": {
            "width": int(args.width),
            "height": int(args.height),
            "render_scale": float(args.render_scale),
        },
        "frame_selector": {
            "frame_start": int(args.frame_start),
            "frame_step": int(max(1, int(args.frame_step))),
            "max_frames": None if args.max_frames is None else int(args.max_frames),
            "selected_count": int(len(selected_frames)),
            "selected_source_frames": [int(x) for x in selected_frames],
        },
        "benchmark_output_dir": str(benchmark_dir),
        "sampled_npz_dir": str(sampled_dir),
        "frames_dir": str(frames_dir),
        "videos": {
            "gt": str(videos_dir / "gt.mp4"),
            "pred": str(videos_dir / "pred.mp4"),
            "compare": str(videos_dir / "compare.mp4"),
        },
        "encoder": {
            "codec": str(args.codec),
            "crf": int(args.crf),
            "preset": str(args.preset),
        },
        "export_benchmark_profile": {
            "field_builder": str(args.field_builder),
            "gbuffer_mode": str(args.gbuffer_mode),
            "dump_sampled_images_every": int(max(0, int(args.dump_sampled_images_every))),
            "benchmark_compute_metrics": bool(args.benchmark_compute_metrics),
            "routing_profile_json_effective": str(effective_routing_profile_json),
            "routing_profile_auto_source": auto_profile_source,
            "routing_profile_auto_exact_checkpoint_match": bool(auto_profile_checkpoint_match),
        },
        "records": [r.__dict__ for r in records],
    }

    manifest_path = out_dir / "video_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if not args.keep_intermediate:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        if sampled_dir.exists():
            shutil.rmtree(sampled_dir)

    print(f"Done. Videos saved under: {videos_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
