#!/usr/bin/env python3
"""Minimal evaluation entrypoint for PG-GCPL mainline.

Computes MAE/RMSE on a LightSetDataset split and writes a small JSON report.

This script intentionally stays minimal and does not depend on matplotlib.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "2_src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PG-GCPL unified_set checkpoint")
    parser.add_argument("--data-root", required=True, help="Dataset root containing parametric_tensor.npz")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt/last_model.pt")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    # Default to 0 for portability (some environments disallow multiprocessing semaphores).
    # TODO(perf): 在可用环境中增大 num_workers 通常可显著提升评估吞吐。
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", default=None, help="Output JSON path (default: <checkpoint_dir>/eval.json)")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    import torch

    from data.lightset_dataset import LightSetDataset
    from models.gaussian_physics_unified import GaussianPhysicsCompressionUnified
    from training.gaussian_physics_trainer import BatchAdapter
    from torch.utils.data import DataLoader
    from utils.unified_metrics import compute_sh_metrics

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    meta = ckpt.get("meta") if isinstance(ckpt, dict) else None

    # Infer model dims from checkpoint tensors.
    # Expected keys: mu [K,3], U [K,27,rank], coeffs [K,rank,embed]
    K = int(state["mu"].shape[0])
    rank = int(state["U"].shape[-1])
    embed_dim = int(state["coeffs"].shape[-1])
    sh_dim = int(state["U"].shape[1])

    # Prefer training metadata stored in checkpoint for exact reproducibility.
    train_ratio = 0.7
    val_ratio = 0.15
    top_k = args.top_k
    light_dim = 12
    intensity_dim = 3
    intensity_offset = 1
    enable_film = True
    scaler = None

    if isinstance(meta, dict):
        split_meta = meta.get("split")
        if isinstance(split_meta, dict):
            train_ratio = float(split_meta.get("train_ratio", train_ratio))
            val_ratio = float(split_meta.get("val_ratio", val_ratio))
        model_meta = meta.get("model")
        if isinstance(model_meta, dict):
            top_k = int(model_meta.get("top_k", top_k))
            light_dim = int(model_meta.get("light_dim", light_dim))
            intensity_dim = int(model_meta.get("intensity_dim", intensity_dim))
            intensity_offset = int(model_meta.get("intensity_offset", intensity_offset))
            enable_film = bool(model_meta.get("enable_film", enable_film))

        scaler_meta = meta.get("sh_scaler")
        if isinstance(scaler_meta, dict):
            try:
                from data.sh_scaler import AdaptiveSHScaler
                import numpy as np

                scaler = AdaptiveSHScaler(
                    l0_mean=np.array(scaler_meta.get("l0_mean", [0.0, 0.0, 0.0]), dtype=np.float32),
                    l0_std=np.array(scaler_meta.get("l0_std", [1.0, 1.0, 1.0]), dtype=np.float32),
                    ho_rms=np.array(scaler_meta.get("ho_rms", [1.0, 1.0, 1.0]), dtype=np.float32),
                    eps=float(scaler_meta.get("eps", 1e-6)),
                )
            except Exception as e:
                print(f"Warning: failed to build SH scaler from checkpoint meta: {e}")

    model = GaussianPhysicsCompressionUnified(
        num_gaussians=K,
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
        raise RuntimeError(
            f"Checkpoint state_dict mismatch. missing={missing}, unexpected={unexpected}"
        )
    model.eval()

    def _build_loader(num_workers: int):
        dataset = LightSetDataset(
            data_root=args.data_root,
            split=args.split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            normalize_probes=True,
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    loader = _build_loader(args.num_workers)

    adapter = BatchAdapter(params_key="light_params", mask_key="light_mask")
    if scaler is not None:
        adapter.target_transform = scaler.transform
        adapter.target_inverse = scaler.inverse

    sum_abs = torch.zeros((), device=device, dtype=torch.float64)
    sum_sq = torch.zeros((), device=device, dtype=torch.float64)
    count = 0
    sh_psnr_sum = 0.0
    sh_ssim_sum = 0.0
    sh_batches = 0

    def run_eval(active_loader) -> None:
        nonlocal sum_abs, sum_sq, count, sh_psnr_sum, sh_ssim_sum, sh_batches
        # TODO(perf): 评估阶段可考虑 torch.inference_mode()，相较 no_grad 有更低的 autograd 开销。
        with torch.inference_mode():
            for batch in active_loader:
                positions, params, targets, mask = adapter.unpack(batch, device)
                preds = model(positions, params, top_k=top_k, light_mask=mask)
                preds_eval, targets_eval = adapter.inverse_targets(preds, targets)
                diff = preds_eval - targets_eval
                sum_abs += torch.sum(torch.abs(diff), dtype=torch.float64)
                sum_sq += torch.sum(diff * diff, dtype=torch.float64)
                count += int(diff.numel())

                sh_metrics = compute_sh_metrics(targets_eval, preds_eval, max_i=1.0)
                sh_psnr_sum += float(sh_metrics["sh_psnr"])
                sh_ssim_sum += float(sh_metrics["sh_ssim"])
                sh_batches += 1

    try:
        run_eval(loader)
    except PermissionError:
        if args.num_workers == 0:
            raise
        print("Warning: DataLoader multiprocessing failed; retrying with --num-workers 0")
        loader = _build_loader(0)
        run_eval(loader)

    denom = float(max(1, count))
    mae = float((sum_abs / denom).item())
    rmse = float(torch.sqrt(sum_sq / denom).item())
    sh_psnr = sh_psnr_sum / max(1, sh_batches)
    sh_ssim = sh_ssim_sum / max(1, sh_batches)

    out_path = Path(args.output) if args.output else ckpt_path.parent / "eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": args.split,
        "checkpoint": str(ckpt_path),
        "data_root": str(Path(args.data_root)),
        "device": str(device),
        "mae": mae,
        "rmse": rmse,
        "sh_psnr": sh_psnr,
        "sh_ssim": sh_ssim,
        "numel": count,
        "K": K,
        "rank": rank,
        "embed_dim": embed_dim,
        "top_k": top_k,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "light_dim": light_dim,
        "intensity_dim": intensity_dim,
        "intensity_offset": intensity_offset,
        "enable_film": enable_film,
        "has_meta": bool(isinstance(meta, dict)),
        "has_sh_scaler": bool(scaler is not None),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(f"MAE={mae:.6f} RMSE={rmse:.6f} SH-PSNR={sh_psnr:.4f} SH-SSIM={sh_ssim:.6f} (numel={count})")


if __name__ == "__main__":  # pragma: no cover
    main()
