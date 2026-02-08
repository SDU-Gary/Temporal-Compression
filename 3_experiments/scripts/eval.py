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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", default=None, help="Output JSON path (default: <checkpoint_dir>/eval.json)")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    import torch

    from data.lightset_dataset import create_dataloaders_lightset
    from models.gaussian_physics_unified import GaussianPhysicsCompressionUnified
    from training.gaussian_physics_trainer import BatchAdapter

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    # Infer model dims from checkpoint tensors.
    # Expected keys: mu [K,3], U [K,27,rank], coeffs [K,rank,embed]
    K = int(state["mu"].shape[0])
    rank = int(state["U"].shape[-1])
    embed_dim = int(state["coeffs"].shape[-1])
    sh_dim = int(state["U"].shape[1])

    model = GaussianPhysicsCompressionUnified(
        num_gaussians=K,
        rank=rank,
        sh_dim=sh_dim,
        light_dim=12,
        embed_dim=embed_dim,
        intensity_dim=3,
        intensity_offset=1,
        enable_film=True,
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    train_loader, val_loader, test_loader = create_dataloaders_lightset(
        data_root=args.data_root,
        batch_size=args.batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        num_workers=args.num_workers,
        normalize_probes=True,
    )
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    adapter = BatchAdapter(params_key="light_params", mask_key="light_mask")

    sum_abs = 0.0
    sum_sq = 0.0
    count = 0
    def run_eval(active_loader) -> None:
        nonlocal sum_abs, sum_sq, count
        with torch.no_grad():
            for batch in active_loader:
                positions, params, targets, mask = adapter.unpack(batch, device)
                preds = model(positions, params, top_k=args.top_k, light_mask=mask)
                diff = preds - targets
                sum_abs += torch.sum(torch.abs(diff)).item()
                sum_sq += torch.sum(diff * diff).item()
                count += int(diff.numel())

    try:
        run_eval(loader)
    except PermissionError:
        if args.num_workers == 0:
            raise
        print("Warning: DataLoader multiprocessing failed; retrying with --num-workers 0")
        train_loader, val_loader, test_loader = create_dataloaders_lightset(
            data_root=args.data_root,
            batch_size=args.batch_size,
            train_ratio=0.7,
            val_ratio=0.15,
            num_workers=0,
            normalize_probes=True,
        )
        loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]
        run_eval(loader)

    mae = sum_abs / max(1, count)
    rmse = (sum_sq / max(1, count)) ** 0.5

    out_path = Path(args.output) if args.output else ckpt_path.parent / "eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": args.split,
        "checkpoint": str(ckpt_path),
        "data_root": str(Path(args.data_root)),
        "device": str(device),
        "mae": mae,
        "rmse": rmse,
        "numel": count,
        "K": K,
        "rank": rank,
        "embed_dim": embed_dim,
        "top_k": args.top_k,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(f"MAE={mae:.6f} RMSE={rmse:.6f} (numel={count})")


if __name__ == "__main__":  # pragma: no cover
    main()
