#!/usr/bin/env python3
"""Render selected frames from offline G-Buffer samples using checkpoint SH model.

This follows the same gbuffer-image-loss rendering path used in training:
1) world pos -> normalized probe range
2) model forward (supports soft routing profile)
3) SH -> irradiance(normal) with band weights
4) softplus positivity + albedo modulation
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "2_src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from models.gaussian_physics_unified import GaussianPhysicsCompressionUnified  # noqa: E402


def _save_png(path: Path, image_u8: np.ndarray) -> None:
    try:
        from PIL import Image  # type: ignore

        Image.fromarray(image_u8).save(str(path))
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
        path.write_bytes(png)


def _tone_map_reinhard(x: np.ndarray) -> np.ndarray:
    arr = np.maximum(x, 0.0)
    return arr / (1.0 + arr)


def _compute_luminance(image: np.ndarray) -> np.ndarray:
    return (
        0.2126 * image[..., 0]
        + 0.7152 * image[..., 1]
        + 0.0722 * image[..., 2]
    )


def _compute_auto_exposure(
    image: np.ndarray,
    *,
    percentile: float = 95.0,
    target: float = 0.6,
    eps: float = 1e-6,
    min_exposure: float = 0.05,
    max_exposure: float = 2000.0,
) -> float:
    lum = _compute_luminance(np.maximum(image, 0.0))
    key = float(np.percentile(lum, percentile))
    exp = float(target / (key + eps))
    return float(np.clip(exp, min_exposure, max_exposure))


def _srgb_encode(linear: np.ndarray) -> np.ndarray:
    x = np.maximum(linear, 0.0)
    thr = 0.0031308
    low = 12.92 * x
    high = 1.055 * np.power(np.maximum(x, 0.0), 1.0 / 2.4) - 0.055
    return np.where(x <= thr, low, high)


def _to_u8(image: np.ndarray) -> np.ndarray:
    img = np.clip(image, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)


def _interpolate_sh_from_probes(
    pixel_pos: np.ndarray,
    probe_pos: np.ndarray,
    probe_sh: np.ndarray,
    *,
    knn: int = 8,
    weight_eps: float = 0.1,
    chunk: int = 32768,
) -> np.ndarray:
    p = np.asarray(pixel_pos, dtype=np.float32).reshape(-1, 3)
    q = np.asarray(probe_pos, dtype=np.float32).reshape(-1, 3)
    s = np.asarray(probe_sh, dtype=np.float32).reshape(q.shape[0], -1)
    if p.shape[0] == 0:
        return np.zeros((0, s.shape[1]), dtype=np.float32)
    if q.shape[0] == 0:
        raise ValueError("probe_pos is empty")
    if s.shape[0] != q.shape[0]:
        raise ValueError(f"probe_sh/probe_pos size mismatch: {s.shape[0]} vs {q.shape[0]}")

    k_eff = max(1, min(int(knn), int(q.shape[0])))
    eps2 = float(weight_eps) * float(weight_eps)
    out = np.zeros((p.shape[0], s.shape[1]), dtype=np.float32)
    for i in range(0, p.shape[0], int(max(1, chunk))):
        c = p[i : i + int(max(1, chunk))]
        d2 = ((c[:, None, :] - q[None, :, :]) ** 2).sum(axis=2)  # [C, P]
        idx = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]  # [C, K]
        d2_sel = np.take_along_axis(d2, idx, axis=1)
        w = 1.0 / (d2_sel + eps2)
        w = w / (w.sum(axis=1, keepdims=True) + 1e-8)
        sh_sel = s[idx]  # [C, K, 27]
        out[i : i + c.shape[0]] = np.sum(w[..., None] * sh_sel, axis=1)
    return out


def _build_sh_basis_from_dirs(dirs: torch.Tensor) -> torch.Tensor:
    x = dirs[:, 0]
    y = dirs[:, 1]
    z = dirs[:, 2]
    return torch.stack(
        [
            torch.full_like(x, 0.282095),
            0.488603 * y,
            0.488603 * z,
            0.488603 * x,
            1.092548 * x * y,
            1.092548 * y * z,
            0.315392 * (3.0 * z * z - 1.0),
            1.092548 * x * z,
            0.546274 * (x * x - y * y),
        ],
        dim=-1,
    )


def _render_irradiance_to_normals(pred_sh: torch.Tensor, normal: torch.Tensor, albedo: torch.Tensor) -> torch.Tensor:
    n = torch.nn.functional.normalize(normal, dim=-1, eps=1e-8)
    basis = _build_sh_basis_from_dirs(n)
    sh_rgb = torch.stack([pred_sh[..., 0:9], pred_sh[..., 9:18], pred_sh[..., 18:27]], dim=-2)
    w = torch.tensor(
        [
            float(np.pi),
            float(2.0 * np.pi / 3.0),
            float(2.0 * np.pi / 3.0),
            float(2.0 * np.pi / 3.0),
            float(np.pi / 4.0),
            float(np.pi / 4.0),
            float(np.pi / 4.0),
            float(np.pi / 4.0),
            float(np.pi / 4.0),
        ],
        dtype=pred_sh.dtype,
        device=pred_sh.device,
    )
    sh_rgb_weighted = sh_rgb * w.view(1, 1, 9)
    irradiance = torch.einsum("...cn,...n->...c", sh_rgb_weighted, basis)
    irradiance = torch.nn.functional.softplus(irradiance, beta=8.0)
    return albedo * irradiance / torch.pi


def _load_checkpoint_model(checkpoint: Path, device: torch.device) -> tuple[GaussianPhysicsCompressionUnified, Dict[str, Any]]:
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}

    K = int(state["mu"].shape[0])
    rank = int(state["U"].shape[-1])
    embed_dim = int(state["coeffs"].shape[-1])
    sh_dim = int(state["U"].shape[1])
    model_meta = meta.get("model", {}) if isinstance(meta, dict) else {}

    model = GaussianPhysicsCompressionUnified(
        num_gaussians=K,
        rank=rank,
        sh_dim=sh_dim,
        light_dim=int(model_meta.get("light_dim", 12)),
        embed_dim=embed_dim,
        intensity_dim=int(model_meta.get("intensity_dim", 3)),
        intensity_offset=int(model_meta.get("intensity_offset", 1)),
        enable_film=bool(model_meta.get("enable_film", True)),
        light_encoder_mode=str(model_meta.get("light_encoder_mode", "normal")),
        bypass_feature_pairs=model_meta.get("bypass_feature_pairs", None),
        bypass_feature_norm_mean=model_meta.get("bypass_feature_norm_mean", None),
        bypass_feature_norm_std=model_meta.get("bypass_feature_norm_std", None),
    ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, meta


def _load_routing_profile(
    checkpoint: Path,
    meta: Dict[str, Any],
    profile_name: Optional[str],
) -> Dict[str, Any]:
    model_meta = meta.get("model", {}) if isinstance(meta, dict) else {}
    training_meta = meta.get("training", {}) if isinstance(meta, dict) else {}

    profile = {
        "name": "fallback",
        "top_k": int(model_meta.get("top_k", 3)),
        "training_soft_routing": bool(training_meta.get("routing_soft_train", False)),
        "routing_soft_topk": training_meta.get("routing_soft_topk", None),
        "routing_temperature": float(training_meta.get("routing_temp_end", 1.0)),
    }
    if not profile["training_soft_routing"]:
        profile["routing_soft_topk"] = None
        profile["routing_temperature"] = 1.0

    summary_path = checkpoint.parent / "global_best_falcor_summary.json"
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            p = data.get("profile", {})
            if isinstance(p, dict):
                if (profile_name is None) or (str(p.get("name", "")).strip() == str(profile_name).strip()):
                    profile.update(
                        {
                            "name": str(p.get("name", "summary_profile")),
                            "top_k": int(p.get("top_k", profile["top_k"])),
                            "training_soft_routing": bool(
                                p.get("training_soft_routing", profile["training_soft_routing"])
                            ),
                            "routing_soft_topk": p.get("routing_soft_topk", profile["routing_soft_topk"]),
                            "routing_temperature": float(
                                p.get("routing_temperature", profile["routing_temperature"])
                            ),
                        }
                    )
        except Exception:
            pass

    if profile["training_soft_routing"]:
        if profile["routing_soft_topk"] is not None:
            profile["routing_soft_topk"] = int(profile["routing_soft_topk"])
        profile["routing_temperature"] = max(1e-6, float(profile["routing_temperature"]))
    else:
        profile["routing_soft_topk"] = None
        profile["routing_temperature"] = 1.0
    return profile


def _discover_frame_file(root: Path, frame_idx: int, view_idx: int) -> Path:
    p_npz = root / f"sample_f{int(frame_idx):06d}_v{int(view_idx):03d}.npz"
    if p_npz.exists():
        return p_npz
    p_pt = root / f"sample_f{int(frame_idx):06d}_v{int(view_idx):03d}.pt"
    if p_pt.exists():
        return p_pt
    raise FileNotFoundError(f"Sample file not found for frame={frame_idx}, view={view_idx}")


def _load_sample(path: Path) -> Dict[str, np.ndarray]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as obj:
            return {k: obj[k] for k in obj.files}
    if path.suffix.lower() == ".pt":
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid .pt sample payload: {path}")
        out: Dict[str, np.ndarray] = {}
        for k, v in raw.items():
            if torch.is_tensor(v):
                out[k] = v.detach().cpu().numpy()
            else:
                out[k] = np.asarray(v)
        return out
    raise ValueError(f"Unsupported sample format: {path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render selected G-Buffer frames with checkpoint (diff pipeline).")
    p.add_argument("--gbuffer-root", required=True)
    p.add_argument("--dataset-root", required=True, help="Root containing parametric_tensor.npz for probe bounds")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-step", type=int, default=60)
    p.add_argument("--num-frames", type=int, default=10)
    p.add_argument("--view-idx", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--routing-profile-name", default="soft8_t018")
    p.add_argument(
        "--gt-source",
        choices=["dataset_sh", "gbuffer_linear"],
        default="dataset_sh",
        help="Target source for comparison/loss diagnostics",
    )
    p.add_argument(
        "--light-source",
        choices=["dataset_parametric", "gbuffer_sample"],
        default="dataset_parametric",
        help="Lighting source for pred rendering path.",
    )
    p.add_argument("--gt-knn", type=int, default=8, help="K for probe SH interpolation when --gt-source=dataset_sh")
    p.add_argument(
        "--gt-weight-eps",
        type=float,
        default=0.1,
        help="Distance epsilon for inverse-distance SH interpolation",
    )
    p.add_argument("--gt-knn-chunk", type=int, default=32768)
    p.add_argument("--save-linear-npz", action="store_true")
    p.add_argument("--auto-exposure", action="store_true", default=True)
    p.add_argument("--no-auto-exposure", action="store_false", dest="auto_exposure")
    p.add_argument("--exposure", type=float, default=None, help="Manual exposure scale; overrides auto-exposure")
    p.add_argument(
        "--exposure-mode",
        choices=["first_frame", "per_frame"],
        default="first_frame",
        help="Auto-exposure strategy when --auto-exposure is enabled",
    )
    p.add_argument(
        "--exposure-coupling",
        choices=["shared", "separate"],
        default="shared",
        help="Exposure coupling for GT/Pred visualization when auto-exposure is enabled",
    )
    p.add_argument(
        "--exposure-percentile",
        type=float,
        default=99.9,
        help="Luminance percentile for auto-exposure key (higher is more robust in dark sparse-light scenes)",
    )
    p.add_argument("--exposure-target", type=float, default=0.6)
    p.add_argument("--exposure-min", type=float, default=0.05)
    p.add_argument(
        "--exposure-max",
        type=float,
        default=50.0,
        help="Upper clamp for auto-exposure to avoid amplifying Monte Carlo noise",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    gbuf_root = Path(args.gbuffer_root).resolve()
    data_root = Path(args.dataset_root).resolve()
    ckpt_path = Path(args.checkpoint).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not gbuf_root.exists():
        raise FileNotFoundError(f"gbuffer root not found: {gbuf_root}")
    if not (data_root / "parametric_tensor.npz").exists():
        raise FileNotFoundError(f"parametric_tensor.npz not found under dataset_root: {data_root}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[render-gbuffer] device={device}")

    with np.load(data_root / "parametric_tensor.npz", allow_pickle=True) as d:
        probes = np.asarray(d["probe_positions"], dtype=np.float32)
        light_configs = np.asarray(d["light_configs"], dtype=np.float32)
        if "light_mask" in d.files:
            light_mask_cfg = np.asarray(d["light_mask"], dtype=np.float32)
        else:
            light_mask_cfg = np.ones(light_configs.shape[:2], dtype=np.float32)
        tensor_gt = np.asarray(d["tensor"], dtype=np.float32) if str(args.gt_source) == "dataset_sh" else None
        valid_probe_mask = np.asarray(d["valid_mask"], dtype=np.float32).reshape(-1) if "valid_mask" in d.files else None
        frame_indices_raw = None
        if "frame_indices" in d.files:
            frame_indices_raw = np.asarray(d["frame_indices"], dtype=np.int64).reshape(-1)
        elif "metadata" in d.files:
            try:
                md_raw = d["metadata"]
                md_obj = md_raw.item() if isinstance(md_raw, np.ndarray) and md_raw.dtype == object else md_raw
                if isinstance(md_obj, dict) and "frame_indices" in md_obj:
                    frame_indices_raw = np.asarray(md_obj["frame_indices"], dtype=np.int64).reshape(-1)
            except Exception:
                frame_indices_raw = None
    probe_min = probes.min(axis=0).astype(np.float32)
    probe_max = probes.max(axis=0).astype(np.float32)
    frame_to_cfg_idx: Dict[int, int] = {}
    if tensor_gt is not None:
        if valid_probe_mask is not None and valid_probe_mask.shape[0] == probes.shape[0]:
            keep = valid_probe_mask > 0.5
            probes = probes[keep]
            tensor_gt = tensor_gt[keep]
        if tensor_gt.ndim != 3 or tensor_gt.shape[2] != 27:
            raise ValueError(f"Expected tensor shape [P,M,27], got {tensor_gt.shape}")
    cfg_count = int(light_configs.shape[0])
    if frame_indices_raw is not None and frame_indices_raw.shape[0] == cfg_count:
        frame_to_cfg_idx = {int(fid): i for i, fid in enumerate(frame_indices_raw.tolist())}
    else:
        frame_to_cfg_idx = {i: i for i in range(cfg_count)}

    model, meta = _load_checkpoint_model(ckpt_path, device=device)
    profile = _load_routing_profile(ckpt_path, meta, profile_name=args.routing_profile_name)
    print(
        "[render-gbuffer] routing "
        f"name={profile.get('name')} top_k={profile['top_k']} "
        f"soft={profile['training_soft_routing']} soft_topk={profile['routing_soft_topk']} "
        f"temp={profile['routing_temperature']}"
    )

    frame_ids = [int(args.frame_start) + i * int(args.frame_step) for i in range(int(args.num_frames))]
    frame_rows = []
    shared_auto_exposure: Optional[float] = None
    for frame_idx in frame_ids:
        sample_path = _discover_frame_file(gbuf_root, frame_idx=frame_idx, view_idx=int(args.view_idx))
        sample = _load_sample(sample_path)

        pos = np.asarray(sample["posW"], dtype=np.float32)
        norm = np.asarray(sample["normW"], dtype=np.float32)
        albedo = np.asarray(sample["albedo"], dtype=np.float32)
        gt_falcor = np.asarray(sample["gt_linear"], dtype=np.float32)
        valid = np.asarray(sample.get("valid_mask", np.ones(pos.shape[:2], dtype=np.float32)), dtype=np.float32) > 0.5
        if str(args.light_source) == "dataset_parametric":
            cfg_idx_for_light = frame_to_cfg_idx.get(int(frame_idx), None)
            if cfg_idx_for_light is None:
                raise KeyError(
                    f"frame_idx={frame_idx} not found in dataset frame mapping for light configs"
                )
            light_params_np = np.asarray(light_configs[int(cfg_idx_for_light)], dtype=np.float32)
            light_mask_np = np.asarray(light_mask_cfg[int(cfg_idx_for_light)], dtype=np.float32)
        else:
            light_params_np = np.asarray(sample["light_params"], dtype=np.float32)
            light_mask_np = np.asarray(
                sample.get("light_mask", np.ones((light_params_np.shape[0],), dtype=np.float32)),
                dtype=np.float32,
            )

        h, w = pos.shape[:2]
        pos_f = pos.reshape(-1, 3)
        norm_f = norm.reshape(-1, 3)
        alb_f = albedo.reshape(-1, 3)
        gt_f = gt_falcor.reshape(-1, 3)
        vm_f = valid.reshape(-1)

        pos_v = pos_f[vm_f]
        norm_v = norm_f[vm_f]
        alb_v = alb_f[vm_f]
        gt_v = gt_f[vm_f]

        pos_norm = 2.0 * (pos_v - probe_min[None, :]) / (probe_max[None, :] - probe_min[None, :] + 1e-8) - 1.0

        pos_t = torch.from_numpy(pos_norm).to(device=device, dtype=torch.float32)
        norm_t = torch.from_numpy(norm_v).to(device=device, dtype=torch.float32)
        alb_t = torch.from_numpy(alb_v).to(device=device, dtype=torch.float32)
        lp_t = torch.from_numpy(light_params_np).to(device=device, dtype=torch.float32)
        lm_t = torch.from_numpy(light_mask_np).to(device=device, dtype=torch.float32)
        if lp_t.ndim == 1:
            lp_t = lp_t.unsqueeze(0)
        if lm_t.ndim == 0:
            lm_t = lm_t.reshape(1)
        lp_t = lp_t.unsqueeze(0).expand(pos_t.shape[0], -1, -1)
        lm_t = lm_t.unsqueeze(0).expand(pos_t.shape[0], -1)

        with torch.no_grad():
            pred_sh = model(
                pos_t,
                lp_t,
                top_k=int(profile["top_k"]),
                light_mask=lm_t,
                training_soft_routing=bool(profile["training_soft_routing"]),
                routing_soft_topk=profile["routing_soft_topk"],
                routing_temperature=float(profile["routing_temperature"]),
            )
            pred_rgb = _render_irradiance_to_normals(pred_sh=pred_sh, normal=norm_t, albedo=alb_t)

        if str(args.gt_source) == "dataset_sh":
            cfg_idx = frame_to_cfg_idx.get(int(frame_idx), None)
            if cfg_idx is None:
                raise KeyError(
                    f"frame_idx={frame_idx} not found in dataset frame mapping; "
                    "ensure gbuffer and parametric_tensor use the same frame timeline"
                )
            sh_frame = tensor_gt[:, int(cfg_idx), :]  # [P, 27]
            gt_sh_v = _interpolate_sh_from_probes(
                pos_v,
                probes,
                sh_frame,
                knn=int(args.gt_knn),
                weight_eps=float(args.gt_weight_eps),
                chunk=int(args.gt_knn_chunk),
            )
            gt_sh_t = torch.from_numpy(gt_sh_v).to(device=device, dtype=torch.float32)
            gt_target_t = _render_irradiance_to_normals(pred_sh=gt_sh_t, normal=norm_t, albedo=alb_t)
            gt_full = np.zeros((h * w, 3), dtype=np.float32)
            gt_full[vm_f] = gt_target_t.detach().cpu().numpy().astype(np.float32)
            gt_full = gt_full.reshape(h, w, 3)
        else:
            gt_target_t = torch.from_numpy(gt_v).to(device=device, dtype=torch.float32)
            gt_full = gt_falcor.copy()

        diff = pred_rgb - gt_target_t
        mse = float(torch.mean(diff * diff).item())
        mae = float(torch.mean(torch.abs(diff)).item())
        rmse = float(torch.sqrt(torch.clamp(torch.tensor(mse), min=1e-12)).item())
        psnr = float(10.0 * math.log10(1.0 / max(mse, 1e-12)))

        pred_v = pred_rgb.detach().cpu().numpy().astype(np.float32)
        pred_full = np.zeros((h * w, 3), dtype=np.float32)
        pred_full[vm_f] = pred_v
        pred_full = pred_full.reshape(h, w, 3)

        if args.exposure is not None:
            exp_gt = float(args.exposure)
            exp_pred = float(args.exposure)
        elif bool(args.auto_exposure):
            if str(args.exposure_mode) == "first_frame":
                if shared_auto_exposure is None:
                    shared_auto_exposure = _compute_auto_exposure(
                        gt_full,
                        percentile=float(args.exposure_percentile),
                        target=float(args.exposure_target),
                        min_exposure=float(args.exposure_min),
                        max_exposure=float(args.exposure_max),
                    )
                exp_gt = float(shared_auto_exposure)
                exp_pred = float(shared_auto_exposure)
            else:
                exp_gt = _compute_auto_exposure(
                    gt_full,
                    percentile=float(args.exposure_percentile),
                    target=float(args.exposure_target),
                    min_exposure=float(args.exposure_min),
                    max_exposure=float(args.exposure_max),
                )
                if str(args.exposure_coupling) == "separate":
                    exp_pred = _compute_auto_exposure(
                        pred_full,
                        percentile=float(args.exposure_percentile),
                        target=float(args.exposure_target),
                        min_exposure=float(args.exposure_min),
                        max_exposure=float(args.exposure_max),
                    )
                else:
                    exp_pred = float(exp_gt)
        else:
            exp_gt = 1.0
            exp_pred = 1.0

        pred_vis = _to_u8(_srgb_encode(_tone_map_reinhard(pred_full * exp_pred)))
        gt_vis = _to_u8(_srgb_encode(_tone_map_reinhard(gt_full * exp_gt)))
        err_map = np.mean(np.abs(pred_full - gt_full), axis=2, keepdims=True)
        err_map = np.clip(err_map / (np.percentile(err_map, 95.0) + 1e-6), 0.0, 1.0)
        err_vis = _to_u8(np.repeat(err_map, 3, axis=2))
        compare = np.concatenate([gt_vis, pred_vis, err_vis], axis=1)

        stem = f"frame_{frame_idx:06d}_v{int(args.view_idx):03d}"
        _save_png(out_dir / f"{stem}_gt.png", gt_vis)
        _save_png(out_dir / f"{stem}_pred.png", pred_vis)
        _save_png(out_dir / f"{stem}_compare.png", compare)
        if args.save_linear_npz:
            np.savez_compressed(
                out_dir / f"{stem}_linear.npz",
                gt_linear=gt_full.astype(np.float32),
                pred_linear=pred_full.astype(np.float32),
                gt_linear_falcor=gt_falcor.astype(np.float32),
                valid_mask=valid.astype(np.float32),
                exposure_gt=np.float32(exp_gt),
                exposure_pred=np.float32(exp_pred),
            )

        frame_rows.append(
            {
                "frame_idx": int(frame_idx),
                "view_idx": int(args.view_idx),
                "mae": mae,
                "rmse": rmse,
                "psnr": psnr,
                "mse": mse,
                "exposure_gt": float(exp_gt),
                "exposure_pred": float(exp_pred),
                "gt_source": str(args.gt_source),
                "light_source": str(args.light_source),
                "sample_path": str(sample_path),
            }
        )
        print(
            f"[render-gbuffer] frame={frame_idx:04d} "
            f"psnr={psnr:.4f} mae={mae:.6f} rmse={rmse:.6f} "
            f"exp_gt={exp_gt:.4f} exp_pred={exp_pred:.4f} "
            f"saved={stem}_*.png"
        )

    summary = {
        "gbuffer_root": str(gbuf_root),
        "dataset_root": str(data_root),
        "checkpoint": str(ckpt_path),
        "device": str(device),
        "gt_source": str(args.gt_source),
        "light_source": str(args.light_source),
        "routing_profile": profile,
        "exposure_mode": str(args.exposure_mode),
        "exposure_coupling": str(args.exposure_coupling),
        "frames": frame_rows,
        "avg_psnr": float(np.mean([r["psnr"] for r in frame_rows])) if frame_rows else float("nan"),
        "avg_mae": float(np.mean([r["mae"] for r in frame_rows])) if frame_rows else float("nan"),
        "avg_rmse": float(np.mean([r["rmse"] for r in frame_rows])) if frame_rows else float("nan"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[render-gbuffer] done -> {out_dir}")
    print(f"[render-gbuffer] summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
