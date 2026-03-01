#!/usr/bin/env python3
"""Run SH-domain oracle diagnostics (oracle coeff, band split, scale sensitivity)."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "2_src"
if str(_SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_SRC))

from data.lightset_dataset import LightSetDataset
from data.sh_scaler import AdaptiveSHScaler
from models.gaussian_physics_unified import GaussianPhysicsCompressionUnified
from training.gaussian_physics_trainer import BatchAdapter
from utils.unified_metrics import (
    SH_L0_INDICES,
    compute_pair_metrics,
    compute_sh_band_metrics,
    compute_sh_metrics,
    compute_sh_metrics_with_rms_normalization,
)


def _device_from_arg(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _inverse_softplus(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xc = torch.clamp(x, min=eps)
    return torch.where(xc > 20.0, xc, torch.log(torch.expm1(xc) + eps))


def _batched_lstsq(a: torch.Tensor, b: torch.Tensor, rcond: float = 1e-5) -> torch.Tensor:
    """Stable batched least-squares via SVD pseudo-inverse.

    `torch.linalg.lstsq()` is numerically unstable on this workload (near-singular
    24x24 systems). We use pinv(SVD) in float64 for deterministic/stable Oracle.
    """
    a64 = a.to(dtype=torch.float64)
    b64 = b.to(dtype=torch.float64)
    try:
        pinv = torch.linalg.pinv(a64, rcond=rcond)
    except RuntimeError:
        pinv = torch.linalg.pinv(a64.detach().cpu(), rcond=rcond).to(device=a.device, dtype=torch.float64)
    x = torch.bmm(pinv, b64)

    if not torch.isfinite(x).all():
        # Last-resort fallback: ridge-stabilized normal equation.
        at = a64.transpose(-1, -2)
        eye = torch.eye(a64.shape[-1], device=a.device, dtype=torch.float64).unsqueeze(0).expand(a64.shape[0], -1, -1)
        ridge = 1e-6
        lhs = torch.bmm(at, a64) + ridge * eye
        rhs = torch.bmm(at, b64)
        x = torch.linalg.solve(lhs, rhs)

    return x


def _build_model_from_checkpoint(
    state: Dict[str, torch.Tensor],
    meta: Dict,
    device: torch.device,
) -> Tuple[GaussianPhysicsCompressionUnified, Dict[str, int | bool]]:
    model_meta = meta.get("model", {}) if isinstance(meta, dict) else {}
    num_gaussians = int(state["mu"].shape[0])
    rank = int(state["U"].shape[-1])
    sh_dim = int(state["U"].shape[1])
    embed_dim = int(state["coeffs"].shape[-1])

    light_dim = int(model_meta.get("light_dim", 12))
    intensity_dim = int(model_meta.get("intensity_dim", 3))
    intensity_offset = int(model_meta.get("intensity_offset", 1))
    top_k = int(model_meta.get("top_k", 3))
    enable_film = bool(model_meta.get("enable_film", True))

    model = GaussianPhysicsCompressionUnified(
        num_gaussians=num_gaussians,
        rank=rank,
        sh_dim=sh_dim,
        light_dim=light_dim,
        embed_dim=embed_dim,
        intensity_dim=intensity_dim,
        intensity_offset=intensity_offset,
        enable_film=enable_film,
    ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. missing={missing}, unexpected={unexpected}")
    model.eval()

    info = {
        "num_gaussians": num_gaussians,
        "rank": rank,
        "sh_dim": sh_dim,
        "embed_dim": embed_dim,
        "light_dim": light_dim,
        "intensity_dim": intensity_dim,
        "intensity_offset": intensity_offset,
        "top_k": top_k,
        "enable_film": enable_film,
    }
    return model, info


def _build_scaler_from_meta(meta: Dict) -> Optional[AdaptiveSHScaler]:
    scaler_meta = meta.get("sh_scaler") if isinstance(meta, dict) else None
    if not isinstance(scaler_meta, dict):
        return None
    try:
        return AdaptiveSHScaler(
            l0_mean=np.array(scaler_meta.get("l0_mean", [0.0, 0.0, 0.0]), dtype=np.float32),
            l0_std=np.array(scaler_meta.get("l0_std", [1.0, 1.0, 1.0]), dtype=np.float32),
            ho_rms=np.array(scaler_meta.get("ho_rms", [1.0, 1.0, 1.0]), dtype=np.float32),
            eps=float(scaler_meta.get("eps", 1e-6)),
        )
    except Exception:
        return None


def _resolve_split_ratios(args: argparse.Namespace, meta: Dict) -> Tuple[float, float]:
    split_meta = meta.get("split") if isinstance(meta, dict) else None
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    if train_ratio is None:
        train_ratio = float(split_meta.get("train_ratio", 0.7)) if isinstance(split_meta, dict) else 0.7
    if val_ratio is None:
        val_ratio = float(split_meta.get("val_ratio", 0.15)) if isinstance(split_meta, dict) else 0.15
    return float(train_ratio), float(val_ratio)


def _extract_effective_bases(
    model: GaussianPhysicsCompressionUnified,
    positions: torch.Tensor,
    light_params: torch.Tensor,
    light_mask: Optional[torch.Tensor],
    top_k: int,
) -> Dict[str, torch.Tensor]:
    z = model.light_encoder(light_params, light_mask)
    gaussian_weights, topk_indices = model.compute_gaussian_weights(positions, top_k)

    selected_u = model.U[topk_indices]
    selected_coeffs = model.coeffs[topk_indices]
    selected_u_l0 = model.U_l0[topk_indices]
    selected_coeffs_l0 = model.coeffs_l0[topk_indices]

    if model.enable_film:
        gamma = model.gamma(z).view(-1, 1, 1, model.rank)
        beta = model.beta(z).view(-1, 1, 1, model.rank)
        selected_u = selected_u * (1.0 + gamma) + beta
        selected_u_l0 = selected_u_l0 * (1.0 + gamma) + beta

    time_weights = torch.einsum("bkrl,bl->bkr", selected_coeffs, z)
    time_weights_l0 = torch.einsum("bkrl,bl->bkr", selected_coeffs_l0, z)

    return {
        "weights": gaussian_weights,
        "selected_u": selected_u,
        "selected_u_l0": selected_u_l0,
        "time_weights": time_weights,
        "time_weights_l0": time_weights_l0,
    }


def _compose_prediction_from_components(comp: Dict[str, torch.Tensor], sh_dim: int) -> torch.Tensor:
    sh_contrib = torch.einsum("bkdr,bkr->bkd", comp["selected_u"], comp["time_weights"])
    sh_pred = torch.einsum("bk,bkd->bd", comp["weights"], sh_contrib)

    l0_contrib = torch.einsum("bkdr,bkr->bkd", comp["selected_u_l0"], comp["time_weights_l0"])
    l0_pred = torch.einsum("bk,bkd->bd", comp["weights"], l0_contrib)
    l0_pred = F.softplus(l0_pred)

    if sh_dim >= 27:
        sh_pred = sh_pred.clone()
        sh_pred[:, 0] = l0_pred[:, 0]
        sh_pred[:, 9] = l0_pred[:, 1]
        sh_pred[:, 18] = l0_pred[:, 2]
    return sh_pred


def _oracle_timeweights_only(target_sh: torch.Tensor, comp: Dict[str, torch.Tensor]) -> torch.Tensor:
    weights = comp["weights"]
    selected_u = comp["selected_u"]
    selected_u_l0 = comp["selected_u_l0"]
    bsz, top_k, _, rank = selected_u.shape
    n_coeff = top_k * rank

    # Main branch is only used for non-L0 channels in model forward. Solving with
    # L0 rows included would bias least-squares to fit channels that are later
    # overwritten by the dedicated L0 branch.
    non_l0_cols = _non_l0_indices()
    a_main = (selected_u * weights[:, :, None, None]).permute(0, 2, 1, 3).reshape(bsz, 27, n_coeff)
    a_main_non_l0 = a_main[:, non_l0_cols, :]
    x_main = _batched_lstsq(a_main_non_l0, target_sh[:, non_l0_cols].unsqueeze(-1)).squeeze(-1)
    x_main = x_main.to(dtype=a_main.dtype)
    pred_main = torch.bmm(a_main, x_main.unsqueeze(-1)).squeeze(-1)

    l0_cols = list(SH_L0_INDICES)
    a_l0 = (selected_u_l0 * weights[:, :, None, None]).permute(0, 2, 1, 3).reshape(bsz, 3, n_coeff)
    target_l0_pre = _inverse_softplus(target_sh[:, l0_cols])
    x_l0 = _batched_lstsq(a_l0, target_l0_pre.unsqueeze(-1)).squeeze(-1)
    x_l0 = x_l0.to(dtype=a_l0.dtype)
    pred_l0 = F.softplus(torch.bmm(a_l0, x_l0.unsqueeze(-1)).squeeze(-1))

    pred_oracle = pred_main.clone()
    pred_oracle[:, l0_cols] = pred_l0
    return pred_oracle.to(dtype=target_sh.dtype)


def _non_l0_indices() -> list[int]:
    l0 = set(SH_L0_INDICES)
    return [i for i in range(27) if i not in l0]


def run_diagnostics(args: argparse.Namespace) -> Dict:
    device = _device_from_arg(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    meta = ckpt.get("meta") if isinstance(ckpt, dict) else {}
    if meta is None:
        meta = {}

    model, model_info = _build_model_from_checkpoint(state, meta, device)
    top_k = int(args.top_k) if args.top_k is not None else int(model_info["top_k"])
    train_ratio, val_ratio = _resolve_split_ratios(args, meta)

    dataset = LightSetDataset(
        data_root=args.data_root,
        split=args.split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=args.seed,
        normalize_probes=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    scaler = AdaptiveSHScaler.load(args.sh_scaler) if args.sh_scaler else _build_scaler_from_meta(meta)
    adapter = BatchAdapter(params_key="light_params", mask_key="light_mask")
    if scaler is not None:
        adapter.target_transform = scaler.transform
        adapter.target_inverse = scaler.inverse

    gt_list: list[np.ndarray] = []
    pred_list: list[np.ndarray] = []
    oracle_list: list[np.ndarray] = []
    probe_idx_list: list[np.ndarray] = []
    config_idx_list: list[np.ndarray] = []
    sample_rows: list[Tuple[int, int, float, float, float, float]] = []
    non_l0_idx = _non_l0_indices()

    with torch.no_grad():
        for batch in loader:
            unpacked = adapter.unpack(batch, device)
            if len(unpacked) == 4:
                positions, params, targets, mask = unpacked
            else:
                positions, params, targets = unpacked
                mask = None

            comp = _extract_effective_bases(model, positions, params, mask, top_k)
            pred_raw = _compose_prediction_from_components(comp, sh_dim=int(model_info["sh_dim"]))
            oracle_raw = _oracle_timeweights_only(targets, comp)

            pred_eval, gt_eval = adapter.inverse_targets(pred_raw, targets)
            oracle_eval, _ = adapter.inverse_targets(oracle_raw, targets)

            pred_np = pred_eval.detach().cpu().numpy().astype(np.float32)
            oracle_np = oracle_eval.detach().cpu().numpy().astype(np.float32)
            gt_np = gt_eval.detach().cpu().numpy().astype(np.float32)

            gt_list.append(gt_np)
            pred_list.append(pred_np)
            oracle_list.append(oracle_np)

            probe_idx = batch["probe_idx"].detach().cpu().numpy().astype(np.int32)
            config_idx = batch["config_idx"].detach().cpu().numpy().astype(np.int32)
            probe_idx_list.append(probe_idx)
            config_idx_list.append(config_idx)

            if args.save_per_sample_csv:
                mse_base_all = np.mean((pred_np - gt_np) ** 2, axis=-1)
                mse_oracle_all = np.mean((oracle_np - gt_np) ** 2, axis=-1)
                mse_base_non_l0 = np.mean((pred_np[:, non_l0_idx] - gt_np[:, non_l0_idx]) ** 2, axis=-1)
                mse_oracle_non_l0 = np.mean((oracle_np[:, non_l0_idx] - gt_np[:, non_l0_idx]) ** 2, axis=-1)
                for i in range(len(probe_idx)):
                    sample_rows.append(
                        (
                            int(probe_idx[i]),
                            int(config_idx[i]),
                            float(mse_base_all[i]),
                            float(mse_oracle_all[i]),
                            float(mse_base_non_l0[i]),
                            float(mse_oracle_non_l0[i]),
                        )
                    )

    gt_all = np.concatenate(gt_list, axis=0)
    pred_all = np.concatenate(pred_list, axis=0)
    oracle_all = np.concatenate(oracle_list, axis=0)
    probe_idx_all = np.concatenate(probe_idx_list, axis=0)
    config_idx_all = np.concatenate(config_idx_list, axis=0)

    baseline_mse_all27 = np.mean((pred_all - gt_all) ** 2, axis=-1)
    oracle_mse_all27 = np.mean((oracle_all - gt_all) ** 2, axis=-1)
    baseline_mse_non_l0 = np.mean((pred_all[:, non_l0_idx] - gt_all[:, non_l0_idx]) ** 2, axis=-1)
    oracle_mse_non_l0 = np.mean((oracle_all[:, non_l0_idx] - gt_all[:, non_l0_idx]) ** 2, axis=-1)

    base_all = compute_sh_metrics(gt_all, pred_all, max_i=1.0)
    oracle_all_m = compute_sh_metrics(gt_all, oracle_all, max_i=1.0)

    base_non_l0 = compute_pair_metrics(
        gt_all[:, non_l0_idx],
        pred_all[:, non_l0_idx],
        max_i=1.0,
        clip_unit=False,
        ssim_mode="global",
    )
    oracle_non_l0 = compute_pair_metrics(
        gt_all[:, non_l0_idx],
        oracle_all[:, non_l0_idx],
        max_i=1.0,
        clip_unit=False,
        ssim_mode="global",
    )

    base_bands = compute_sh_band_metrics(gt_all, pred_all, max_i=1.0)
    oracle_bands = compute_sh_band_metrics(gt_all, oracle_all, max_i=1.0)

    base_rmsnorm = compute_sh_metrics_with_rms_normalization(gt_all, pred_all, max_i=1.0)
    oracle_rmsnorm = compute_sh_metrics_with_rms_normalization(gt_all, oracle_all, max_i=1.0)

    gain_all_psnr = float(oracle_all_m["sh_psnr"] - base_all["sh_psnr"])
    gain_non_l0_psnr = float(oracle_non_l0["psnr"] - base_non_l0["psnr"])
    if gain_all_psnr >= 3.0 and gain_non_l0_psnr >= 3.0:
        diagnosis = "coeff-limited"
    elif gain_all_psnr <= 1.0 and gain_non_l0_psnr <= 1.0:
        diagnosis = "basis-limited"
    else:
        diagnosis = "mixed"

    results = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint": str(args.checkpoint),
            "data_root": str(args.data_root),
            "split": args.split,
            "seed": int(args.seed),
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "batch_size": int(args.batch_size),
            "top_k": int(top_k),
            "device": str(device),
            "model": model_info,
            "scaler_enabled": bool(scaler is not None),
        },
        "counts": {
            "num_samples": int(gt_all.shape[0]),
            "num_unique_probes": int(np.unique(probe_idx_all).size),
            "num_unique_configs": int(np.unique(config_idx_all).size),
        },
        "baseline": {
            "all27": base_all,
            "non_l0": {
                "sh_non_l0_psnr": float(base_non_l0["psnr"]),
                "sh_non_l0_ssim": float(base_non_l0["ssim"]),
                "sh_non_l0_mae": float(base_non_l0["mae"]),
                "sh_non_l0_rmse": float(base_non_l0["rmse"]),
            },
            "per_band": base_bands,
            "rmsnorm": base_rmsnorm,
        },
        "oracle_timeweights": {
            "all27": oracle_all_m,
            "non_l0": {
                "sh_non_l0_psnr": float(oracle_non_l0["psnr"]),
                "sh_non_l0_ssim": float(oracle_non_l0["ssim"]),
                "sh_non_l0_mae": float(oracle_non_l0["mae"]),
                "sh_non_l0_rmse": float(oracle_non_l0["rmse"]),
            },
            "per_band": oracle_bands,
            "rmsnorm": oracle_rmsnorm,
        },
        "delta": {
            "all27_psnr_gain_db": gain_all_psnr,
            "non_l0_psnr_gain_db": gain_non_l0_psnr,
            "all27_ssim_gain": float(oracle_all_m["sh_ssim"] - base_all["sh_ssim"]),
            "non_l0_ssim_gain": float(oracle_non_l0["ssim"] - base_non_l0["ssim"]),
            "baseline_rmsnorm_gap_db": float(base_rmsnorm["sh_rmsnorm_psnr"] - base_all["sh_psnr"]),
            "oracle_rmsnorm_gap_db": float(oracle_rmsnorm["sh_rmsnorm_psnr"] - oracle_all_m["sh_psnr"]),
        },
        "oracle_consistency": {
            "mean_mse_all27_baseline": float(np.mean(baseline_mse_all27)),
            "mean_mse_all27_oracle": float(np.mean(oracle_mse_all27)),
            "mean_mse_non_l0_baseline": float(np.mean(baseline_mse_non_l0)),
            "mean_mse_non_l0_oracle": float(np.mean(oracle_mse_non_l0)),
            "improved_ratio_all27": float(np.mean(oracle_mse_all27 <= baseline_mse_all27)),
            "improved_ratio_non_l0": float(np.mean(oracle_mse_non_l0 <= baseline_mse_non_l0)),
        },
        "diagnosis": {
            "label": diagnosis,
            "rule": {
                "coeff_limited_if": "all27_gain>=3dB and non_l0_gain>=3dB",
                "basis_limited_if": "all27_gain<=1dB and non_l0_gain<=1dB",
                "otherwise": "mixed",
            },
        },
    }

    if args.save_per_sample_csv:
        csv_path = Path(args.save_per_sample_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "probe_idx",
                    "config_idx",
                    "baseline_mse_all27",
                    "oracle_mse_all27",
                    "baseline_mse_non_l0",
                    "oracle_mse_non_l0",
                ]
            )
            writer.writerows(sample_rows)
        results["artifacts"] = {"per_sample_csv": str(csv_path)}

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SH oracle diagnostics")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None, help="Summary JSON output path")
    parser.add_argument("--save-per-sample-csv", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--sh-scaler", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    results = run_diagnostics(args)

    if args.output:
        out_path = Path(args.output)
    else:
        ckpt_path = Path(args.checkpoint)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = ckpt_path.parent / f"sh_oracle_diag_{stamp}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved diagnostics: {out_path}")


if __name__ == "__main__":
    main()
