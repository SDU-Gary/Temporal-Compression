#!/usr/bin/env python3
"""Run Phase-1 coeff learnability diagnostics in one script.

This suite complements `run_sh_oracle_diagnostics.py` by exporting Oracle-paired
coeff targets, fitting lightweight probes (linear / MLP), and comparing
SH-domain reconstruction quality for:

- model-pred coeff (`w_pred`)
- probe-pred coeff (`w_probe`)
- Oracle coeff (`w_oracle`)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "2_src"
if str(_SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_SRC))

from data.lightset_dataset import LightSetDataset
from data.sh_scaler import AdaptiveSHScaler
from training.gaussian_physics_trainer import BatchAdapter
from utils.unified_metrics import (
    SH_L0_INDICES,
    compute_pair_metrics,
    compute_sh_band_metrics,
    compute_sh_metrics,
    compute_sh_metrics_with_rms_normalization,
)


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


def _softplus_np(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    y = np.where(x64 > 20.0, x64, np.log1p(np.exp(x64)))
    return y.astype(np.float32)


def _split_indices(num_samples: int, train_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(num_samples)
    train_n = max(1, min(num_samples - 2, int(round(num_samples * train_ratio))))
    remain = num_samples - train_n
    val_n = max(1, remain // 2)
    test_n = max(1, remain - val_n)
    if train_n + val_n + test_n > num_samples:
        overflow = train_n + val_n + test_n - num_samples
        test_n = max(1, test_n - overflow)
    train_idx = perm[:train_n]
    val_idx = perm[train_n : train_n + val_n]
    test_idx = perm[train_n + val_n : train_n + val_n + test_n]
    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


def _vector_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    diff = yp - yt
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))

    yt_norm = np.linalg.norm(yt, axis=1)
    yp_norm = np.linalg.norm(yp, axis=1)
    denom = np.maximum(1e-12, yt_norm * yp_norm)
    cosine = np.sum(yt * yp, axis=1) / denom
    cosine = np.clip(cosine, -1.0, 1.0)

    var = float(np.mean((yt - np.mean(yt, axis=0, keepdims=True)) ** 2))
    r2 = float(1.0 - mse / max(var, 1e-12))

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "cosine_mean": float(np.mean(cosine)),
        "cosine_median": float(np.median(cosine)),
        "cosine_p10": float(np.quantile(cosine, 0.10)),
        "cosine_p90": float(np.quantile(cosine, 0.90)),
        "r2": r2,
    }


def _fit_linear_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1e-4) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    ones = np.ones((x64.shape[0], 1), dtype=np.float64)
    x_aug = np.concatenate([x64, ones], axis=1)
    xtx = x_aug.T @ x_aug
    reg = np.eye(xtx.shape[0], dtype=np.float64) * float(alpha)
    reg[-1, -1] = 0.0
    xty = x_aug.T @ y64
    w = np.linalg.solve(xtx + reg, xty)
    return w.astype(np.float32)


def _predict_linear_ridge(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x32 = np.asarray(x, dtype=np.float32)
    ones = np.ones((x32.shape[0], 1), dtype=np.float32)
    x_aug = np.concatenate([x32, ones], axis=1)
    return x_aug @ w


class _MLPProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_mlp_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> Tuple[_MLPProbe, Dict[str, float]]:
    torch.manual_seed(int(seed))
    model = _MLPProbe(in_dim=x_train.shape[1], out_dim=y_train.shape[1], hidden_dim=int(hidden_dim)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=1e-6)
    criterion = nn.MSELoss()

    ds = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=True, num_workers=0)

    x_val_t = torch.from_numpy(x_val.astype(np.float32)).to(device)
    y_val_t = torch.from_numpy(y_val.astype(np.float32)).to(device)

    best_state: Dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    for _ in range(int(epochs)):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t)
            val_loss = float(criterion(val_pred, y_val_t).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_val_mse": float(best_val)}


def _cond_stats(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def _non_l0_metrics(gt: np.ndarray, pred: np.ndarray, non_l0_idx: list[int]) -> Dict[str, float]:
    m = compute_pair_metrics(gt[:, non_l0_idx], pred[:, non_l0_idx], max_i=1.0, clip_unit=False, ssim_mode="global")
    return {
        "sh_non_l0_psnr": float(m["psnr"]),
        "sh_non_l0_ssim": float(m["ssim"]),
        "sh_non_l0_mae": float(m["mae"]),
        "sh_non_l0_rmse": float(m["rmse"]),
    }


def _evaluate_sh(gt: np.ndarray, pred: np.ndarray, non_l0_idx: list[int]) -> Dict[str, Any]:
    return {
        "all27": compute_sh_metrics(gt, pred, max_i=1.0),
        "non_l0": _non_l0_metrics(gt, pred, non_l0_idx),
        "per_band": compute_sh_band_metrics(gt, pred, max_i=1.0),
        "rmsnorm": compute_sh_metrics_with_rms_normalization(gt, pred, max_i=1.0),
    }


def _fit_structured_coeffs(
    *,
    z_all: np.ndarray,
    topk_indices_all: np.ndarray,
    tw_target_all: np.ndarray,
    coeff_init: np.ndarray,
    train_sample_mask: np.ndarray,
    ridge_alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit Gaussian-level coeff matrix with architecture-consistent form.

    For each gaussian g, solve (ridge):
        tw_target[g] ~= z @ coeff[g]^T
    using only train samples where g appears in top-k routing.
    """

    coeff_fit = np.asarray(coeff_init, dtype=np.float32).copy()
    K, rank, embed_dim = coeff_fit.shape
    fit_counts = np.zeros((K,), dtype=np.int64)
    eye = np.eye(embed_dim, dtype=np.float64)

    for g in range(K):
        sample_ids, slot_ids = np.where(topk_indices_all == g)
        if sample_ids.size == 0:
            continue
        keep = train_sample_mask[sample_ids]
        if not np.any(keep):
            continue

        sample_ids = sample_ids[keep]
        slot_ids = slot_ids[keep]
        fit_counts[g] = int(sample_ids.size)

        if sample_ids.size < max(8, embed_dim // 2):
            continue

        z_g = z_all[sample_ids].astype(np.float64)  # [N, L]
        y_g = tw_target_all[sample_ids, slot_ids, :].astype(np.float64)  # [N, R]

        lhs = z_g.T @ z_g + float(ridge_alpha) * eye
        rhs = z_g.T @ y_g
        try:
            beta = np.linalg.solve(lhs, rhs)  # [L, R]
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(lhs) @ rhs

        coeff_fit[g] = beta.T.astype(np.float32)  # [R, L]

    return coeff_fit, fit_counts


def _predict_timeweights_from_coeff(
    z_all: np.ndarray,
    topk_indices_all: np.ndarray,
    coeff_matrix: np.ndarray,
) -> np.ndarray:
    selected = coeff_matrix[topk_indices_all]  # [B, K, R, L]
    return np.einsum("bkrl,bl->bkr", selected, z_all, optimize=True).astype(np.float32)


def _oracle_with_timeweights(
    *,
    target_sh: torch.Tensor,
    comp: Dict[str, torch.Tensor],
    diag_mod,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = comp["weights"]
    selected_u = comp["selected_u"]
    selected_u_l0 = comp["selected_u_l0"]
    bsz, top_k, _, rank = selected_u.shape
    n_coeff = top_k * rank
    non_l0_cols = diag_mod._non_l0_indices()

    a_main = (selected_u * weights[:, :, None, None]).permute(0, 2, 1, 3).reshape(bsz, 27, n_coeff)
    a_main_non_l0 = a_main[:, non_l0_cols, :]

    # Condition number of A^T A per-sample via singular values of A.
    svals = torch.linalg.svdvals(a_main_non_l0.to(dtype=torch.float64))
    cond = (svals[:, 0] / torch.clamp(svals[:, -1], min=1e-12)).to(dtype=torch.float32)

    x_main = diag_mod._batched_lstsq(a_main_non_l0, target_sh[:, non_l0_cols].unsqueeze(-1)).squeeze(-1)
    x_main = x_main.to(dtype=a_main.dtype)
    pred_main = torch.bmm(a_main, x_main.unsqueeze(-1)).squeeze(-1)

    l0_cols = list(SH_L0_INDICES)
    a_l0 = (selected_u_l0 * weights[:, :, None, None]).permute(0, 2, 1, 3).reshape(bsz, 3, n_coeff)
    target_l0_pre = diag_mod._inverse_softplus(target_sh[:, l0_cols])
    x_l0 = diag_mod._batched_lstsq(a_l0, target_l0_pre.unsqueeze(-1)).squeeze(-1)
    x_l0 = x_l0.to(dtype=a_l0.dtype)
    pred_l0 = torch.nn.functional.softplus(torch.bmm(a_l0, x_l0.unsqueeze(-1)).squeeze(-1))

    pred_oracle = pred_main.clone()
    pred_oracle[:, l0_cols] = pred_l0
    return pred_oracle.to(dtype=target_sh.dtype), x_main, x_l0, a_main, a_l0, cond


def _reconstruct_from_weights(
    a_main: np.ndarray,
    a_l0: np.ndarray,
    weights_concat: np.ndarray,
    n_coeff: int,
) -> np.ndarray:
    w_main = weights_concat[:, :n_coeff]
    w_l0 = weights_concat[:, n_coeff:]
    pred_main = np.einsum("bij,bj->bi", a_main, w_main, optimize=True)
    pred_l0_pre = np.einsum("bij,bj->bi", a_l0, w_l0, optimize=True)
    pred_l0 = _softplus_np(pred_l0_pre)

    pred = pred_main.astype(np.float32)
    pred[:, SH_L0_INDICES[0]] = pred_l0[:, 0]
    pred[:, SH_L0_INDICES[1]] = pred_l0[:, 1]
    pred[:, SH_L0_INDICES[2]] = pred_l0[:, 2]
    return pred


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run coeff learnability diagnostics suite")
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=None)
    p.add_argument("--val-ratio", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--sh-scaler", default=None)
    p.add_argument("--max-samples", type=int, default=8192)
    p.add_argument("--sample-seed", type=int, default=42)

    p.add_argument("--probe-train-ratio", type=float, default=0.8)
    p.add_argument("--ridge-alpha", type=float, default=1e-4)
    p.add_argument("--mlp-hidden", type=int, default=128)
    p.add_argument("--mlp-epochs", type=int, default=30)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-batch-size", type=int, default=256)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diag_mod = _load_diag_module()
    device = _device_from_arg(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    meta = ckpt.get("meta") if isinstance(ckpt, dict) else {}
    if meta is None:
        meta = {}

    model, model_info = diag_mod._build_model_from_checkpoint(state, meta, device)
    top_k = int(args.top_k) if args.top_k is not None else int(model_info["top_k"])
    train_ratio, val_ratio = diag_mod._resolve_split_ratios(args, meta)

    dataset = LightSetDataset(
        data_root=args.data_root,
        split=args.split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=args.seed,
        normalize_probes=True,
    )

    sampled_dataset = dataset
    sample_indices = None
    max_samples = int(getattr(args, "max_samples", 0) or 0)
    if max_samples > 0 and max_samples < len(dataset):
        rng = np.random.default_rng(int(getattr(args, "sample_seed", args.seed)))
        sample_indices = np.sort(rng.choice(len(dataset), size=max_samples, replace=False).astype(np.int64))
        sampled_dataset = Subset(dataset, sample_indices.tolist())

    loader = DataLoader(
        sampled_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )

    scaler = AdaptiveSHScaler.load(args.sh_scaler) if args.sh_scaler else diag_mod._build_scaler_from_meta(meta)
    adapter = BatchAdapter(params_key="light_params", mask_key="light_mask")
    if scaler is not None:
        adapter.target_transform = scaler.transform
        adapter.target_inverse = scaler.inverse

    z_rows: list[np.ndarray] = []
    w_pred_rows: list[np.ndarray] = []
    w_oracle_rows: list[np.ndarray] = []
    cond_rows: list[np.ndarray] = []
    a_main_rows: list[np.ndarray] = []
    a_l0_rows: list[np.ndarray] = []
    topk_idx_rows: list[np.ndarray] = []
    tw_pred_main_rows: list[np.ndarray] = []
    tw_pred_l0_rows: list[np.ndarray] = []
    tw_oracle_main_rows: list[np.ndarray] = []
    tw_oracle_l0_rows: list[np.ndarray] = []
    gt_rows: list[np.ndarray] = []
    pred_rows: list[np.ndarray] = []
    oracle_rows: list[np.ndarray] = []
    probe_idx_rows: list[np.ndarray] = []
    config_idx_rows: list[np.ndarray] = []

    non_l0_idx = diag_mod._non_l0_indices()
    n_coeff = int(top_k * int(model_info["rank"]))

    with torch.no_grad():
        for batch in loader:
            unpacked = adapter.unpack(batch, device)
            if len(unpacked) == 4:
                positions, params, targets, mask = unpacked
            else:
                positions, params, targets = unpacked
                mask = None

            z = model.light_encoder(params, mask)
            gaussian_weights, topk_indices = model.compute_gaussian_weights(positions, top_k)
            comp = diag_mod._extract_effective_bases(model, positions, params, mask, top_k)

            pred_raw = diag_mod._compose_prediction_from_components(comp, sh_dim=int(model_info["sh_dim"]))
            oracle_raw, x_main, x_l0, a_main, a_l0, cond = _oracle_with_timeweights(
                target_sh=targets,
                comp=comp,
                diag_mod=diag_mod,
            )

            pred_eval, gt_eval = adapter.inverse_targets(pred_raw, targets)
            oracle_eval, _ = adapter.inverse_targets(oracle_raw, targets)

            w_pred = torch.cat(
                [
                    comp["time_weights"].reshape(comp["time_weights"].shape[0], -1),
                    comp["time_weights_l0"].reshape(comp["time_weights_l0"].shape[0], -1),
                ],
                dim=1,
            )
            w_oracle = torch.cat([x_main.reshape(x_main.shape[0], -1), x_l0.reshape(x_l0.shape[0], -1)], dim=1)

            z_rows.append(z.detach().cpu().numpy().astype(np.float32))
            w_pred_rows.append(w_pred.detach().cpu().numpy().astype(np.float32))
            w_oracle_rows.append(w_oracle.detach().cpu().numpy().astype(np.float32))
            cond_rows.append(np.log10(np.clip(cond.detach().cpu().numpy().astype(np.float64), 1.0, 1e12)).astype(np.float32))
            a_main_rows.append(a_main.detach().cpu().numpy().astype(np.float32))
            a_l0_rows.append(a_l0.detach().cpu().numpy().astype(np.float32))
            topk_idx_rows.append(topk_indices.detach().cpu().numpy().astype(np.int64))
            tw_pred_main_rows.append(comp["time_weights"].detach().cpu().numpy().astype(np.float32))
            tw_pred_l0_rows.append(comp["time_weights_l0"].detach().cpu().numpy().astype(np.float32))
            tw_oracle_main_rows.append(x_main.reshape(-1, top_k, model_info["rank"]).detach().cpu().numpy().astype(np.float32))
            tw_oracle_l0_rows.append(x_l0.reshape(-1, top_k, model_info["rank"]).detach().cpu().numpy().astype(np.float32))
            gt_rows.append(gt_eval.detach().cpu().numpy().astype(np.float32))
            pred_rows.append(pred_eval.detach().cpu().numpy().astype(np.float32))
            oracle_rows.append(oracle_eval.detach().cpu().numpy().astype(np.float32))

            probe_idx_rows.append(batch["probe_idx"].detach().cpu().numpy().astype(np.int32))
            config_idx_rows.append(batch["config_idx"].detach().cpu().numpy().astype(np.int32))

    z_all = np.concatenate(z_rows, axis=0)
    w_pred_all = np.concatenate(w_pred_rows, axis=0)
    w_oracle_all = np.concatenate(w_oracle_rows, axis=0)
    cond_log10_all = np.concatenate(cond_rows, axis=0)
    a_main_all = np.concatenate(a_main_rows, axis=0)
    a_l0_all = np.concatenate(a_l0_rows, axis=0)
    topk_idx_all = np.concatenate(topk_idx_rows, axis=0)
    tw_pred_main_all = np.concatenate(tw_pred_main_rows, axis=0)
    tw_pred_l0_all = np.concatenate(tw_pred_l0_rows, axis=0)
    tw_oracle_main_all = np.concatenate(tw_oracle_main_rows, axis=0)
    tw_oracle_l0_all = np.concatenate(tw_oracle_l0_rows, axis=0)
    gt_all = np.concatenate(gt_rows, axis=0)
    pred_all = np.concatenate(pred_rows, axis=0)
    oracle_all = np.concatenate(oracle_rows, axis=0)
    probe_idx_all = np.concatenate(probe_idx_rows, axis=0)
    config_idx_all = np.concatenate(config_idx_rows, axis=0)

    train_idx, val_idx, test_idx = _split_indices(
        num_samples=int(z_all.shape[0]),
        train_ratio=float(args.probe_train_ratio),
        seed=int(args.sample_seed),
    )
    train_sample_mask = np.zeros((z_all.shape[0],), dtype=bool)
    train_sample_mask[train_idx] = True

    ridge_w = _fit_linear_ridge(z_all[train_idx], w_oracle_all[train_idx], alpha=float(args.ridge_alpha))
    w_linear_all = _predict_linear_ridge(z_all, ridge_w).astype(np.float32)

    mlp_model, mlp_train_stats = _train_mlp_probe(
        z_all[train_idx],
        w_oracle_all[train_idx],
        z_all[val_idx],
        w_oracle_all[val_idx],
        hidden_dim=int(args.mlp_hidden),
        epochs=int(args.mlp_epochs),
        lr=float(args.mlp_lr),
        batch_size=int(args.mlp_batch_size),
        device=device,
        seed=int(args.seed),
    )
    mlp_model.eval()
    with torch.no_grad():
        z_t = torch.from_numpy(z_all.astype(np.float32)).to(device)
        w_mlp_all = mlp_model(z_t).detach().cpu().numpy().astype(np.float32)

    coeff_main_init = np.asarray(model.coeffs.detach().cpu().numpy(), dtype=np.float32)
    coeff_l0_init = np.asarray(model.coeffs_l0.detach().cpu().numpy(), dtype=np.float32)
    coeff_main_fit, coeff_main_fit_counts = _fit_structured_coeffs(
        z_all=z_all,
        topk_indices_all=topk_idx_all,
        tw_target_all=tw_oracle_main_all,
        coeff_init=coeff_main_init,
        train_sample_mask=train_sample_mask,
        ridge_alpha=float(args.ridge_alpha),
    )
    coeff_l0_fit, coeff_l0_fit_counts = _fit_structured_coeffs(
        z_all=z_all,
        topk_indices_all=topk_idx_all,
        tw_target_all=tw_oracle_l0_all,
        coeff_init=coeff_l0_init,
        train_sample_mask=train_sample_mask,
        ridge_alpha=float(args.ridge_alpha),
    )

    tw_struct_main_all = _predict_timeweights_from_coeff(z_all, topk_idx_all, coeff_main_fit)
    tw_struct_l0_all = _predict_timeweights_from_coeff(z_all, topk_idx_all, coeff_l0_fit)

    pred_main = _reconstruct_from_weights(a_main_all, a_l0_all, w_pred_all, n_coeff=n_coeff)
    linear_main = _reconstruct_from_weights(a_main_all, a_l0_all, w_linear_all, n_coeff=n_coeff)
    mlp_main = _reconstruct_from_weights(a_main_all, a_l0_all, w_mlp_all, n_coeff=n_coeff)
    w_struct_all = np.concatenate(
        [
            tw_struct_main_all.reshape(tw_struct_main_all.shape[0], -1),
            tw_struct_l0_all.reshape(tw_struct_l0_all.shape[0], -1),
        ],
        axis=1,
    ).astype(np.float32)
    structured_main = _reconstruct_from_weights(a_main_all, a_l0_all, w_struct_all, n_coeff=n_coeff)
    oracle_main = _reconstruct_from_weights(a_main_all, a_l0_all, w_oracle_all, n_coeff=n_coeff)

    pred_eval, gt_eval = adapter.inverse_targets(torch.from_numpy(pred_main), torch.from_numpy(gt_all))
    linear_eval, _ = adapter.inverse_targets(torch.from_numpy(linear_main), torch.from_numpy(gt_all))
    mlp_eval, _ = adapter.inverse_targets(torch.from_numpy(mlp_main), torch.from_numpy(gt_all))
    structured_eval, _ = adapter.inverse_targets(torch.from_numpy(structured_main), torch.from_numpy(gt_all))
    oracle_eval, _ = adapter.inverse_targets(torch.from_numpy(oracle_main), torch.from_numpy(gt_all))
    gt_eval_np = gt_eval.numpy().astype(np.float32)
    pred_eval_np = pred_eval.numpy().astype(np.float32)
    linear_eval_np = linear_eval.numpy().astype(np.float32)
    mlp_eval_np = mlp_eval.numpy().astype(np.float32)
    structured_eval_np = structured_eval.numpy().astype(np.float32)
    oracle_eval_np = oracle_eval.numpy().astype(np.float32)

    coeff_pred_vs_oracle = _vector_metrics(w_oracle_all, w_pred_all)

    tw_oracle_concat_all = np.concatenate(
        [
            tw_oracle_main_all.reshape(tw_oracle_main_all.shape[0], -1),
            tw_oracle_l0_all.reshape(tw_oracle_l0_all.shape[0], -1),
        ],
        axis=1,
    )
    tw_pred_concat_all = np.concatenate(
        [
            tw_pred_main_all.reshape(tw_pred_main_all.shape[0], -1),
            tw_pred_l0_all.reshape(tw_pred_l0_all.shape[0], -1),
        ],
        axis=1,
    )
    tw_struct_concat_all = np.concatenate(
        [
            tw_struct_main_all.reshape(tw_struct_main_all.shape[0], -1),
            tw_struct_l0_all.reshape(tw_struct_l0_all.shape[0], -1),
        ],
        axis=1,
    )

    structured_tw_metrics = {
        "pred_vs_oracle": _vector_metrics(tw_oracle_concat_all, tw_pred_concat_all),
        "structured_vs_oracle": _vector_metrics(tw_oracle_concat_all, tw_struct_concat_all),
        "test_pred_vs_oracle": _vector_metrics(tw_oracle_concat_all[test_idx], tw_pred_concat_all[test_idx]),
        "test_structured_vs_oracle": _vector_metrics(tw_oracle_concat_all[test_idx], tw_struct_concat_all[test_idx]),
    }

    fitted_mask_main = coeff_main_fit_counts > 0
    fitted_mask_l0 = coeff_l0_fit_counts > 0
    if np.any(fitted_mask_main):
        coeff_main_similarity = _vector_metrics(
            coeff_main_init[fitted_mask_main].reshape(-1, coeff_main_init.shape[-1]),
            coeff_main_fit[fitted_mask_main].reshape(-1, coeff_main_fit.shape[-1]),
        )
    else:
        coeff_main_similarity = _vector_metrics(coeff_main_init.reshape(-1, coeff_main_init.shape[-1]), coeff_main_init.reshape(-1, coeff_main_init.shape[-1]))

    if np.any(fitted_mask_l0):
        coeff_l0_similarity = _vector_metrics(
            coeff_l0_init[fitted_mask_l0].reshape(-1, coeff_l0_init.shape[-1]),
            coeff_l0_fit[fitted_mask_l0].reshape(-1, coeff_l0_fit.shape[-1]),
        )
    else:
        coeff_l0_similarity = _vector_metrics(coeff_l0_init.reshape(-1, coeff_l0_init.shape[-1]), coeff_l0_init.reshape(-1, coeff_l0_init.shape[-1]))
    linear_probe_metrics = {
        "train": _vector_metrics(w_oracle_all[train_idx], w_linear_all[train_idx]),
        "val": _vector_metrics(w_oracle_all[val_idx], w_linear_all[val_idx]),
        "test": _vector_metrics(w_oracle_all[test_idx], w_linear_all[test_idx]),
    }
    mlp_probe_metrics = {
        "train": _vector_metrics(w_oracle_all[train_idx], w_mlp_all[train_idx]),
        "val": _vector_metrics(w_oracle_all[val_idx], w_mlp_all[val_idx]),
        "test": _vector_metrics(w_oracle_all[test_idx], w_mlp_all[test_idx]),
        "train_stats": mlp_train_stats,
    }

    sh_pred = _evaluate_sh(gt_eval_np, pred_eval_np, non_l0_idx)
    sh_linear = _evaluate_sh(gt_eval_np, linear_eval_np, non_l0_idx)
    sh_mlp = _evaluate_sh(gt_eval_np, mlp_eval_np, non_l0_idx)
    sh_structured = _evaluate_sh(gt_eval_np, structured_eval_np, non_l0_idx)
    sh_oracle = _evaluate_sh(gt_eval_np, oracle_eval_np, non_l0_idx)

    denom = max(1e-6, float(sh_oracle["all27"]["sh_psnr"] - sh_pred["all27"]["sh_psnr"]))
    linear_recovery = float((sh_linear["all27"]["sh_psnr"] - sh_pred["all27"]["sh_psnr"]) / denom)
    mlp_recovery = float((sh_mlp["all27"]["sh_psnr"] - sh_pred["all27"]["sh_psnr"]) / denom)
    structured_recovery = float((sh_structured["all27"]["sh_psnr"] - sh_pred["all27"]["sh_psnr"]) / denom)

    cond_stats = _cond_stats(cond_log10_all)
    diagnosis = "mixed"
    if float(cond_stats["p90"]) >= 7.0:
        diagnosis = "ill-conditioned"
    elif structured_recovery >= 0.5:
        diagnosis = "optimization-limited"
    elif mlp_recovery >= 0.6 and linear_recovery < 0.3:
        diagnosis = "mapping-nonlinear"
    elif structured_recovery < 0.2 and mlp_recovery < 0.2 and linear_recovery < 0.2:
        diagnosis = "z-information-limited"

    pairs_npz = out_dir / "oracle_pairs.npz"
    np.savez_compressed(
        pairs_npz,
        z=z_all.astype(np.float32),
        w_pred=w_pred_all.astype(np.float32),
        w_oracle=w_oracle_all.astype(np.float32),
        cond_log10=cond_log10_all.astype(np.float32),
        probe_idx=probe_idx_all.astype(np.int32),
        config_idx=config_idx_all.astype(np.int32),
    )

    coeff_csv = out_dir / "coeff_stats.csv"
    with coeff_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample",
                "probe_idx",
                "config_idx",
                "cond_log10",
                "mse_pred_vs_oracle",
                "cos_pred_vs_oracle",
                "mse_linear_vs_oracle",
                "cos_linear_vs_oracle",
                "mse_mlp_vs_oracle",
                "cos_mlp_vs_oracle",
                "mse_structured_vs_oracle",
                "cos_structured_vs_oracle",
            ],
        )
        writer.writeheader()
        for i in range(int(z_all.shape[0])):
            wt = w_oracle_all[i : i + 1]
            wp = w_pred_all[i : i + 1]
            wl = w_linear_all[i : i + 1]
            wm = w_mlp_all[i : i + 1]
            ws = w_struct_all[i : i + 1]
            m_pred = _vector_metrics(wt, wp)
            m_lin = _vector_metrics(wt, wl)
            m_mlp = _vector_metrics(wt, wm)
            m_struct = _vector_metrics(wt, ws)
            writer.writerow(
                {
                    "sample": i,
                    "probe_idx": int(probe_idx_all[i]),
                    "config_idx": int(config_idx_all[i]),
                    "cond_log10": float(cond_log10_all[i]),
                    "mse_pred_vs_oracle": float(m_pred["mse"]),
                    "cos_pred_vs_oracle": float(m_pred["cosine_mean"]),
                    "mse_linear_vs_oracle": float(m_lin["mse"]),
                    "cos_linear_vs_oracle": float(m_lin["cosine_mean"]),
                    "mse_mlp_vs_oracle": float(m_mlp["mse"]),
                    "cos_mlp_vs_oracle": float(m_mlp["cosine_mean"]),
                    "mse_structured_vs_oracle": float(m_struct["mse"]),
                    "cos_structured_vs_oracle": float(m_struct["cosine_mean"]),
                }
            )

    probe_metrics_json = out_dir / "probe_metrics.json"
    probe_metrics_payload = {
        "pred_vs_oracle": coeff_pred_vs_oracle,
        "sample_probe": {
            "linear": linear_probe_metrics,
            "mlp": mlp_probe_metrics,
        },
        "structured_probe": {
            "timeweights": structured_tw_metrics,
            "coeff_similarity_main": coeff_main_similarity,
            "coeff_similarity_l0": coeff_l0_similarity,
            "fit_counts_main": {
                "mean": float(np.mean(coeff_main_fit_counts)),
                "median": float(np.median(coeff_main_fit_counts)),
                "min": int(np.min(coeff_main_fit_counts)),
                "max": int(np.max(coeff_main_fit_counts)),
                "nonzero_ratio": float(np.mean(coeff_main_fit_counts > 0)),
            },
            "fit_counts_l0": {
                "mean": float(np.mean(coeff_l0_fit_counts)),
                "median": float(np.median(coeff_l0_fit_counts)),
                "min": int(np.min(coeff_l0_fit_counts)),
                "max": int(np.max(coeff_l0_fit_counts)),
                "nonzero_ratio": float(np.mean(coeff_l0_fit_counts > 0)),
            },
        },
    }
    probe_metrics_json.write_text(json.dumps(probe_metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    recon_json = out_dir / "recon_metrics.json"
    recon_payload = {
        "pred": sh_pred,
        "linear_probe": sh_linear,
        "mlp_probe": sh_mlp,
        "structured_probe": sh_structured,
        "oracle": sh_oracle,
    }
    recon_json.write_text(json.dumps(recon_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "meta": {
            "checkpoint": str(args.checkpoint),
            "data_root": str(args.data_root),
            "split": str(args.split),
            "device": str(device),
            "seed": int(args.seed),
            "top_k": int(top_k),
            "model": model_info,
            "scaler_enabled": bool(scaler is not None),
        },
        "counts": {
            "num_samples": int(z_all.shape[0]),
            "num_unique_probes": int(np.unique(probe_idx_all).size),
            "num_unique_configs": int(np.unique(config_idx_all).size),
            "dataset_size_before_sampling": int(len(dataset)),
            "dataset_size_after_sampling": int(len(sampled_dataset)),
            "sampled": bool(sample_indices is not None),
        },
        "phase1": {
            "coeff_alignment": {
                "pred_vs_oracle": coeff_pred_vs_oracle,
                "condition_log10": cond_stats,
            },
            "probe": {
                "split_counts": {
                    "train": int(train_idx.size),
                    "val": int(val_idx.size),
                    "test": int(test_idx.size),
                },
                "sample_probe": {
                    "linear": linear_probe_metrics,
                    "mlp": mlp_probe_metrics,
                },
                "structured_probe": {
                    "timeweights": structured_tw_metrics,
                    "coeff_similarity_main": coeff_main_similarity,
                    "coeff_similarity_l0": coeff_l0_similarity,
                    "fit_counts_main": {
                        "mean": float(np.mean(coeff_main_fit_counts)),
                        "median": float(np.median(coeff_main_fit_counts)),
                        "min": int(np.min(coeff_main_fit_counts)),
                        "max": int(np.max(coeff_main_fit_counts)),
                        "nonzero_ratio": float(np.mean(coeff_main_fit_counts > 0)),
                    },
                    "fit_counts_l0": {
                        "mean": float(np.mean(coeff_l0_fit_counts)),
                        "median": float(np.median(coeff_l0_fit_counts)),
                        "min": int(np.min(coeff_l0_fit_counts)),
                        "max": int(np.max(coeff_l0_fit_counts)),
                        "nonzero_ratio": float(np.mean(coeff_l0_fit_counts > 0)),
                    },
                },
            },
            "sh_reconstruction": {
                "pred": sh_pred,
                "linear_probe": sh_linear,
                "mlp_probe": sh_mlp,
                "structured_probe": sh_structured,
                "oracle": sh_oracle,
                "gains_over_pred_db": {
                    "linear_all27": float(sh_linear["all27"]["sh_psnr"] - sh_pred["all27"]["sh_psnr"]),
                    "mlp_all27": float(sh_mlp["all27"]["sh_psnr"] - sh_pred["all27"]["sh_psnr"]),
                    "structured_all27": float(sh_structured["all27"]["sh_psnr"] - sh_pred["all27"]["sh_psnr"]),
                    "oracle_all27": float(sh_oracle["all27"]["sh_psnr"] - sh_pred["all27"]["sh_psnr"]),
                    "linear_non_l0": float(sh_linear["non_l0"]["sh_non_l0_psnr"] - sh_pred["non_l0"]["sh_non_l0_psnr"]),
                    "mlp_non_l0": float(sh_mlp["non_l0"]["sh_non_l0_psnr"] - sh_pred["non_l0"]["sh_non_l0_psnr"]),
                    "structured_non_l0": float(sh_structured["non_l0"]["sh_non_l0_psnr"] - sh_pred["non_l0"]["sh_non_l0_psnr"]),
                    "oracle_non_l0": float(sh_oracle["non_l0"]["sh_non_l0_psnr"] - sh_pred["non_l0"]["sh_non_l0_psnr"]),
                },
                "recovery_ratio": {
                    "linear": linear_recovery,
                    "mlp": mlp_recovery,
                    "structured": structured_recovery,
                },
            },
            "diagnosis": {
                "label": diagnosis,
                "rules": {
                    "ill-conditioned": "cond_log10_p90>=7",
                    "optimization-limited": "structured_recovery>=0.5",
                    "mapping-nonlinear": "mlp_recovery>=0.6 and linear_recovery<0.3",
                    "z-information-limited": "structured_recovery<0.2 and mlp/linear recovery<0.2",
                    "mixed": "otherwise",
                },
            },
        },
        "artifacts": {
            "oracle_pairs_npz": str(pairs_npz),
            "probe_metrics_json": str(probe_metrics_json),
            "recon_metrics_json": str(recon_json),
            "coeff_stats_csv": str(coeff_csv),
        },
    }

    summary_path = out_dir / "phase1_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved coeff suite summary: {summary_path}")


if __name__ == "__main__":
    main()
