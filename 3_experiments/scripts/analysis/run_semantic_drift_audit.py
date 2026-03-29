#!/usr/bin/env python3
"""Audit semantic drift between two checkpoints.

The goal is not to compare raw parameter values only, but to detect whether:
- parameters change a lot while functional behavior stays similar (compensation drift)
- performance under a target route profile regresses despite small parameter movement
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "2_src"
if str(_SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_SRC))

from data.lightset_dataset import LightSetDataset
from data.sh_scaler import AdaptiveSHScaler
from training.gaussian_physics_trainer import BatchAdapter
from utils.unified_metrics import SH_L0_INDICES, compute_sh_band_metrics, compute_sh_metrics


def _load_diag_module():
    diag_path = _ROOT / "3_experiments" / "scripts" / "analysis" / "run_sh_oracle_diagnostics.py"
    spec = importlib.util.spec_from_file_location("run_sh_oracle_diagnostics", str(diag_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load diagnostics module from {diag_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _device_from_arg(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _norm_profiles(raw: Any, default_top_k: int) -> List[Dict[str, Any]]:
    if not raw:
        return [
            {
                "name": f"hard{int(default_top_k)}",
                "top_k": int(default_top_k),
                "training_soft_routing": False,
                "routing_soft_topk": None,
                "routing_temperature": None,
            }
        ]

    if isinstance(raw, dict):
        items = raw.get("profiles", [])
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("profiles-json must decode to list or mapping with key 'profiles'")

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"profiles[{i}] must be mapping")
        cfg = {
            "name": str(item.get("name", f"profile_{i+1}")),
            "top_k": int(item.get("top_k", default_top_k)),
            "training_soft_routing": bool(item.get("training_soft_routing", False)),
            "routing_soft_topk": item.get("routing_soft_topk", None),
            "routing_temperature": item.get("routing_temperature", None),
        }
        if cfg["routing_soft_topk"] is not None:
            cfg["routing_soft_topk"] = int(cfg["routing_soft_topk"])
        if cfg["routing_temperature"] is not None:
            cfg["routing_temperature"] = float(cfg["routing_temperature"])
        out.append(cfg)
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run semantic drift audit between two checkpoints")
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint-a", required=True)
    p.add_argument("--checkpoint-b", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--name", default="drift")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=None)
    p.add_argument("--val-ratio", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=8192)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--profiles-json", default="")
    return p


def _safe_ratio(num: float, den: float) -> float:
    return float(num / max(1e-12, den))


def _param_group(name: str) -> str:
    if name == "mu" or name.startswith("mu."):
        return "routing"
    if name == "log_scale" or name.startswith("log_scale."):
        return "routing"
    if name == "U" or name.startswith("U.") or name == "U_l0" or name.startswith("U_l0."):
        return "basis"
    if name == "coeffs" or name.startswith("coeffs.") or name == "coeffs_l0" or name.startswith("coeffs_l0."):
        return "coeff"
    if name.startswith("light_encoder."):
        return "encoder"
    if name.startswith("gamma.") or name.startswith("beta."):
        return "film"
    return "other"


def _state_drift_metrics(state_a: Dict[str, torch.Tensor], state_b: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    keys = sorted(k for k in state_a.keys() if k in state_b and torch.is_floating_point(state_a[k]) and torch.is_floating_point(state_b[k]))
    per_tensor: Dict[str, Dict[str, float]] = {}

    total_diff_sq = 0.0
    total_base_sq = 0.0
    group_acc: Dict[str, Dict[str, float]] = {}

    for k in keys:
        a = state_a[k].detach().cpu().to(dtype=torch.float64).reshape(-1).numpy()
        b = state_b[k].detach().cpu().to(dtype=torch.float64).reshape(-1).numpy()
        if a.shape != b.shape or a.size == 0:
            continue
        d = b - a
        diff_norm = float(np.linalg.norm(d))
        a_norm = float(np.linalg.norm(a))
        b_norm = float(np.linalg.norm(b))
        rel_to_a = _safe_ratio(diff_norm, a_norm)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        cos = float(np.dot(a, b) / max(1e-12, denom))
        cos = float(np.clip(cos, -1.0, 1.0))

        per_tensor[k] = {
            "numel": int(a.size),
            "diff_l2": diff_norm,
            "a_l2": a_norm,
            "b_l2": b_norm,
            "rel_l2_to_a": rel_to_a,
            "cosine": cos,
        }

        total_diff_sq += float(np.dot(d, d))
        total_base_sq += float(np.dot(a, a))

        g = _param_group(k)
        slot = group_acc.setdefault(g, {"diff_sq": 0.0, "base_sq": 0.0, "numel": 0.0})
        slot["diff_sq"] += float(np.dot(d, d))
        slot["base_sq"] += float(np.dot(a, a))
        slot["numel"] += float(a.size)

    groups: Dict[str, Dict[str, float]] = {}
    for g, acc in group_acc.items():
        groups[g] = {
            "numel": int(acc["numel"]),
            "rel_l2_to_a": _safe_ratio(np.sqrt(acc["diff_sq"]), np.sqrt(acc["base_sq"])),
        }

    return {
        "global_rel_l2_to_a": _safe_ratio(np.sqrt(total_diff_sq), np.sqrt(total_base_sq)),
        "per_group": groups,
        "per_tensor": per_tensor,
    }


def _subspace_drift(state_a: Dict[str, torch.Tensor], state_b: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    if "U" not in state_a or "U" not in state_b:
        return {}
    ua = state_a["U"].detach().cpu().to(dtype=torch.float64).numpy()  # [K,27,r]
    ub = state_b["U"].detach().cpu().to(dtype=torch.float64).numpy()
    if ua.shape != ub.shape:
        return {"shape_mismatch": {"U_a": list(ua.shape), "U_b": list(ub.shape)}}

    K = ua.shape[0]
    ang_means: List[float] = []
    ang_maxes: List[float] = []
    for g in range(K):
        qa, _ = np.linalg.qr(ua[g], mode="reduced")
        qb, _ = np.linalg.qr(ub[g], mode="reduced")
        m = qa.T @ qb
        s = np.linalg.svd(m, compute_uv=False)
        s = np.clip(s, -1.0, 1.0)
        ang = np.degrees(np.arccos(s))
        ang_means.append(float(np.mean(ang)))
        ang_maxes.append(float(np.max(ang)))

    return {
        "U_indexwise": {
            "mean_principal_angle_deg_mean": float(np.mean(ang_means)),
            "mean_principal_angle_deg_p90": float(np.quantile(ang_means, 0.90)),
            "max_principal_angle_deg_mean": float(np.mean(ang_maxes)),
            "max_principal_angle_deg_p90": float(np.quantile(ang_maxes, 0.90)),
        }
    }


def _non_l0_idx() -> List[int]:
    s = set(SH_L0_INDICES)
    return [i for i in range(27) if i not in s]


def _sample_dataset(dataset, max_samples: int, sample_seed: int):
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset, None
    rng = np.random.default_rng(int(sample_seed))
    indices = np.sort(rng.choice(len(dataset), size=int(max_samples), replace=False).astype(np.int64))
    return Subset(dataset, indices.tolist()), indices


def _build_adapter(meta: Dict[str, Any], sh_scaler_path: Optional[str]) -> BatchAdapter:
    adapter = BatchAdapter(params_key="light_params", mask_key="light_mask")
    scaler = None
    if sh_scaler_path:
        scaler = AdaptiveSHScaler.load(sh_scaler_path)
    else:
        diag_mod = _load_diag_module()
        scaler = diag_mod._build_scaler_from_meta(meta)
    if scaler is not None:
        adapter.target_transform = scaler.transform
        adapter.target_inverse = scaler.inverse
    return adapter


def _eval_profile_metrics(
    *,
    model_a,
    model_b,
    adapter_a: BatchAdapter,
    adapter_b: BatchAdapter,
    loader: DataLoader,
    profile: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    non_l0 = _non_l0_idx()
    gt_rows: List[np.ndarray] = []
    a_rows: List[np.ndarray] = []
    b_rows: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            unpacked_a = adapter_a.unpack(batch, device)
            unpacked_b = adapter_b.unpack(batch, device)

            if len(unpacked_a) == 4:
                pos_a, params_a, targets_a, mask_a = unpacked_a
            else:
                pos_a, params_a, targets_a = unpacked_a
                mask_a = None
            if len(unpacked_b) == 4:
                pos_b, params_b, targets_b, mask_b = unpacked_b
            else:
                pos_b, params_b, targets_b = unpacked_b
                mask_b = None

            kwargs = {
                "top_k": int(profile.get("top_k", 3)),
                "training_soft_routing": bool(profile.get("training_soft_routing", False)),
                "routing_soft_topk": profile.get("routing_soft_topk", None),
                "routing_temperature": float(profile.get("routing_temperature", 1.0))
                if profile.get("routing_temperature", None) is not None
                else 1.0,
            }
            pred_a_raw = model_a(pos_a, params_a, light_mask=mask_a, **kwargs)
            pred_b_raw = model_b(pos_b, params_b, light_mask=mask_b, **kwargs)

            pred_a_eval, gt_eval = adapter_a.inverse_targets(pred_a_raw, targets_a)
            pred_b_eval, _ = adapter_b.inverse_targets(pred_b_raw, targets_b)

            gt_rows.append(gt_eval.detach().cpu().numpy().astype(np.float32))
            a_rows.append(pred_a_eval.detach().cpu().numpy().astype(np.float32))
            b_rows.append(pred_b_eval.detach().cpu().numpy().astype(np.float32))

    gt = np.concatenate(gt_rows, axis=0)
    pa = np.concatenate(a_rows, axis=0)
    pb = np.concatenate(b_rows, axis=0)

    a_gt_all = compute_sh_metrics(gt, pa, max_i=1.0)
    b_gt_all = compute_sh_metrics(gt, pb, max_i=1.0)
    a_gt_band = compute_sh_band_metrics(gt, pa, max_i=1.0)
    b_gt_band = compute_sh_band_metrics(gt, pb, max_i=1.0)

    ab_all = compute_sh_metrics(pa, pb, max_i=1.0)
    ab_band = compute_sh_band_metrics(pa, pb, max_i=1.0)
    ab_non_l0 = compute_sh_metrics(pa[:, non_l0], pb[:, non_l0], max_i=1.0)

    perf_delta = float(b_gt_all["sh_psnr"] - a_gt_all["sh_psnr"])
    return {
        "a_vs_gt": {
            "all27": a_gt_all,
            "per_band": a_gt_band,
        },
        "b_vs_gt": {
            "all27": b_gt_all,
            "per_band": b_gt_band,
        },
        "a_vs_b": {
            "all27": ab_all,
            "per_band": ab_band,
            "non_l0": {
                "sh_non_l0_psnr": float(ab_non_l0["sh_psnr"]),
                "sh_non_l0_rmse": float(ab_non_l0["sh_rmse"]),
                "sh_non_l0_mae": float(ab_non_l0["sh_mae"]),
                "sh_non_l0_ssim": float(ab_non_l0["sh_ssim"]),
            },
        },
        "b_minus_a_psnr_vs_gt": perf_delta,
    }


def main() -> None:
    args = _build_parser().parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    diag_mod = _load_diag_module()
    device = _device_from_arg(args.device)

    ckpt_a = torch.load(args.checkpoint_a, map_location=device, weights_only=False)
    ckpt_b = torch.load(args.checkpoint_b, map_location=device, weights_only=False)
    state_a = ckpt_a.get("model_state_dict", ckpt_a)
    state_b = ckpt_b.get("model_state_dict", ckpt_b)
    meta_a = ckpt_a.get("meta") if isinstance(ckpt_a, dict) else {}
    meta_b = ckpt_b.get("meta") if isinstance(ckpt_b, dict) else {}
    meta_a = meta_a or {}
    meta_b = meta_b or {}

    model_a, info_a = diag_mod._build_model_from_checkpoint(state_a, meta_a, device)
    model_b, info_b = diag_mod._build_model_from_checkpoint(state_b, meta_b, device)

    default_top_k = int(args.top_k) if args.top_k is not None else int(info_a["top_k"])
    profiles = _norm_profiles(json.loads(args.profiles_json) if str(args.profiles_json).strip() else None, default_top_k)
    train_ratio, val_ratio = diag_mod._resolve_split_ratios(args, meta_a)

    dataset = LightSetDataset(
        data_root=args.data_root,
        split=args.split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=args.seed,
        normalize_probes=True,
    )
    sampled_dataset, sampled_indices = _sample_dataset(dataset, int(args.max_samples), int(args.sample_seed))
    loader = DataLoader(
        sampled_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )

    adapter_a = _build_adapter(meta_a, None)
    adapter_b = _build_adapter(meta_b, None)

    drift = _state_drift_metrics(state_a, state_b)
    subspace = _subspace_drift(state_a, state_b)
    per_profile: Dict[str, Any] = {}
    for profile in profiles:
        name = str(profile["name"])
        metrics = _eval_profile_metrics(
            model_a=model_a,
            model_b=model_b,
            adapter_a=adapter_a,
            adapter_b=adapter_b,
            loader=loader,
            profile=profile,
            device=device,
        )
        compensation_index = _safe_ratio(
            float(drift.get("global_rel_l2_to_a", 0.0)),
            float(metrics["a_vs_b"]["all27"]["sh_rmse"]),
        )
        metrics["compensation_index"] = compensation_index
        per_profile[name] = {
            "profile": profile,
            "metrics": metrics,
        }

    summary = {
        "meta": {
            "name": str(args.name),
            "checkpoint_a": str(args.checkpoint_a),
            "checkpoint_b": str(args.checkpoint_b),
            "data_root": str(args.data_root),
            "split": str(args.split),
            "device": str(device),
            "seed": int(args.seed),
            "model_a": info_a,
            "model_b": info_b,
        },
        "counts": {
            "dataset_size_before_sampling": int(len(dataset)),
            "dataset_size_after_sampling": int(len(sampled_dataset)),
            "sampled": bool(sampled_indices is not None),
            "num_profiles": int(len(profiles)),
        },
        "parameter_drift": drift,
        "subspace_drift": subspace,
        "profiles": per_profile,
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved semantic drift summary: {output_path}")


if __name__ == "__main__":
    main()
