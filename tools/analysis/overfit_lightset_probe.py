#!/usr/bin/env python3
"""Small overfit probe for LightSetDataset-format SH tensors.

This intentionally bypasses the full experiment trainer so that diagnostics are
not affected by EMA, profile checkpoint selection, periodic Falcor evaluation,
or train/val/test splitting policy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "2_src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.gaussian_physics_unified import GaussianPhysicsCompressionUnified  # noqa: E402


def _normalize_positions(positions: np.ndarray) -> np.ndarray:
    pmin = positions.min(axis=0)
    pmax = positions.max(axis=0)
    return 2.0 * (positions - pmin) / (pmax - pmin + 1e-8) - 1.0


def _make_model(
    *,
    num_gaussians: int,
    rank: int,
    embed_dim: int,
    light_configs: np.ndarray,
    device: torch.device,
    init_positions: np.ndarray,
    init_tensor: torch.Tensor,
    light_encoder_mode: str,
) -> GaussianPhysicsCompressionUnified:
    # Use direction xyz for the 12D descriptor layout:
    # [type, rgb(3), pos(3), dir(3), meta1, meta2].
    pairs = [[0, 7], [0, 8], [0, 9]]
    feats = light_configs[:, 0, [7, 8, 9]].astype(np.float32)
    mean = feats.mean(axis=0)
    std = np.maximum(feats.std(axis=0), 1e-6)

    model = GaussianPhysicsCompressionUnified(
        num_gaussians=num_gaussians,
        rank=rank,
        sh_dim=27,
        light_dim=12,
        embed_dim=embed_dim,
        intensity_dim=3,
        intensity_offset=1,
        enable_film=True,
        light_encoder_mode=light_encoder_mode,
        bypass_feature_pairs=pairs,
        bypass_feature_norm_mean=mean.tolist(),
        bypass_feature_norm_std=std.tolist(),
    ).to(device)
    model.init_from_kmeans(init_positions, light_configs, init_tensor.cpu())
    return model


def _metrics(pred: torch.Tensor, target: torch.Tensor, prefix: str = "") -> dict[str, float]:
    err = pred - target
    mse = float(torch.mean(err * err).item())
    rmse = math.sqrt(max(mse, 0.0))
    mae = float(torch.mean(torch.abs(err)).item())
    tgt = target.detach()
    data_range = float((tgt.max() - tgt.min()).item())
    range_psnr = 20.0 * math.log10(max(data_range, 1e-12) / max(rmse, 1e-12))
    return {
        f"{prefix}mae": mae,
        f"{prefix}rmse": rmse,
        f"{prefix}mse": mse,
        f"{prefix}range_psnr": range_psnr,
        f"{prefix}target_rms": float(torch.sqrt(torch.mean(tgt * tgt)).item()),
        f"{prefix}pred_rms": float(torch.sqrt(torch.mean(pred.detach() * pred.detach())).item()),
        f"{prefix}target_p99_abs": float(torch.quantile(torch.abs(tgt).flatten(), 0.99).item()),
        f"{prefix}pred_p99_abs": float(torch.quantile(torch.abs(pred.detach()).flatten(), 0.99).item()),
    }


def _select_configs(m_count: int, mode: str, config_index: int) -> np.ndarray:
    if mode == "single":
        if config_index < 0 or config_index >= m_count:
            raise ValueError(f"config_index out of range: {config_index}, M={m_count}")
        return np.asarray([config_index], dtype=np.int64)
    if mode == "all":
        return np.arange(m_count, dtype=np.int64)
    raise ValueError(f"Unsupported mode: {mode}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(args.device)

    data_path = Path(args.dataset)
    if data_path.is_dir():
        data_path = data_path / "parametric_tensor.npz"
    with np.load(data_path, allow_pickle=True) as data:
        tensor_np = np.asarray(data["tensor"], dtype=np.float32)
        positions_np_raw = np.asarray(data["probe_positions"], dtype=np.float32)
        light_configs_np_all = np.asarray(data["light_configs"], dtype=np.float32)
        light_mask_np_all = np.asarray(data["light_mask"], dtype=np.float32)

    if tensor_np.ndim != 3 or tensor_np.shape[2] != 27:
        raise ValueError(f"Expected tensor [P,M,27], got {tensor_np.shape}")
    if light_configs_np_all.ndim != 3 or light_configs_np_all.shape[2] != 12:
        raise ValueError(f"Expected light_configs [M,N,12], got {light_configs_np_all.shape}")

    positions_np = _normalize_positions(positions_np_raw).astype(np.float32)
    config_indices = _select_configs(tensor_np.shape[1], args.mode, int(args.config_index))
    tensor_sel = tensor_np[:, config_indices, :]
    light_configs_sel = light_configs_np_all[config_indices]
    light_mask_sel = light_mask_np_all[config_indices]

    p_count, m_sel, _ = tensor_sel.shape
    pos = torch.from_numpy(np.repeat(positions_np[:, None, :], m_sel, axis=1).reshape(-1, 3)).to(device)
    target = torch.from_numpy(tensor_sel.reshape(-1, 27)).to(device)
    params = torch.from_numpy(np.repeat(light_configs_sel[None, :, :, :], p_count, axis=0).reshape(-1, *light_configs_sel.shape[1:])).to(device)
    mask = torch.from_numpy(np.repeat(light_mask_sel[None, :, :], p_count, axis=0).reshape(-1, light_mask_sel.shape[1])).to(device)

    model = _make_model(
        num_gaussians=int(args.num_gaussians),
        rank=int(args.rank),
        embed_dim=int(args.embed_dim),
        light_configs=light_configs_np_all,
        device=device,
        init_positions=positions_np,
        init_tensor=torch.from_numpy(tensor_np),
        light_encoder_mode=str(args.light_encoder_mode),
    )

    target_mean = target.mean(dim=0, keepdim=True)
    target_std = target.std(dim=0, keepdim=True).clamp_min(1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    n = int(target.shape[0])
    batch_size = min(int(args.batch_size), n)
    history: list[dict[str, float]] = []
    best = {"mae": float("inf"), "step": 0}

    for step in range(1, int(args.steps) + 1):
        perm = torch.randperm(n, device=device)[:batch_size]
        pred = model(
            pos[perm],
            params[perm],
            top_k=int(args.top_k),
            light_mask=mask[perm],
            training_soft_routing=bool(args.soft_routing),
            routing_soft_topk=int(args.routing_soft_topk),
            routing_temperature=float(args.routing_temperature),
        )
        tgt = target[perm]
        if bool(args.standardize_loss):
            loss = F.mse_loss((pred - target_mean) / target_std, (tgt - target_mean) / target_std)
        else:
            loss = F.mse_loss(pred, tgt)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if float(args.grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()

        if step == 1 or step % int(args.log_every) == 0 or step == int(args.steps):
            with torch.no_grad():
                full_pred = model(
                    pos,
                    params,
                    top_k=int(args.top_k),
                    light_mask=mask,
                    training_soft_routing=bool(args.soft_routing),
                    routing_soft_topk=int(args.routing_soft_topk),
                    routing_temperature=float(args.routing_temperature),
                )
                m = _metrics(full_pred, target)
                m["step"] = float(step)
                m["loss"] = float(loss.item())
                history.append(m)
                if m["mae"] < best["mae"]:
                    best = {"mae": m["mae"], "step": step, **m}
                print(
                    f"step {step:05d} loss={loss.item():.6g} "
                    f"mae={m['mae']:.6g} rmse={m['rmse']:.6g} "
                    f"range_psnr={m['range_psnr']:.3f} "
                    f"pred_rms={m['pred_rms']:.3f}/{m['target_rms']:.3f}",
                    flush=True,
                )

    with torch.no_grad():
        final_pred = model(
            pos,
            params,
            top_k=int(args.top_k),
            light_mask=mask,
            training_soft_routing=bool(args.soft_routing),
            routing_soft_topk=int(args.routing_soft_topk),
            routing_temperature=float(args.routing_temperature),
        )
    final_metrics = _metrics(final_pred, target)
    out = {
        "dataset": str(data_path),
        "mode": str(args.mode),
        "config_indices": config_indices.tolist(),
        "num_samples": int(n),
        "num_probes": int(p_count),
        "num_configs": int(m_sel),
        "num_gaussians": int(args.num_gaussians),
        "rank": int(args.rank),
        "embed_dim": int(args.embed_dim),
        "standardize_loss": bool(args.standardize_loss),
        "soft_routing": bool(args.soft_routing),
        "routing_soft_topk": int(args.routing_soft_topk),
        "routing_temperature": float(args.routing_temperature),
        "final": final_metrics,
        "best": best,
        "history": history,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "summary": out,
        },
        output_dir / "model.pt",
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        pred=final_pred.detach().cpu().numpy().reshape(p_count, m_sel, 27),
        target=target.detach().cpu().numpy().reshape(p_count, m_sel, 27),
        config_indices=config_indices,
    )
    print(f"Wrote {summary_path}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["single", "all"], default="single")
    parser.add_argument("--config-index", type=int, default=0)
    parser.add_argument("--num-gaussians", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--light-encoder-mode", choices=["normal", "bypass_linear", "bypass_mlp_16_32"], default="bypass_linear")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--standardize-loss", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--soft-routing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--routing-soft-topk", type=int, default=8)
    parser.add_argument("--routing-temperature", type=float, default=0.18)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
