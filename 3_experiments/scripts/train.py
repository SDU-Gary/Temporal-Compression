#!/usr/bin/env python3
"""Unified training entrypoint for Gaussian-Physics variants."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch

from training import BatchAdapter, GaussianPhysicsTrainer
from data.intensity_modulation_dataset import create_dataloaders as create_intensity_dataloaders
from data.transfer_tensor_dataset import create_dataloaders_5D
from models.gaussian_physics_1D import GaussianPhysicsCompression1D
from models.gaussian_physics_5D import GaussianPhysicsCompression5D


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _init_1d(model: GaussianPhysicsCompression1D, train_loader) -> None:
    dataset = train_loader.dataset
    probe_positions = dataset.get_full_probe_positions(normalized=True)
    model.initialize_from_probes(probe_positions, method="kmeans", device=str(next(model.parameters()).device))


def _init_5d(model: GaussianPhysicsCompression5D, train_loader) -> None:
    dataset = train_loader.dataset
    sh_tensor = torch.from_numpy(dataset.tensor).float()
    model.init_from_kmeans(
        probe_positions=dataset.probe_positions,
        light_configs=dataset.light_configs_subset,
        sh_tensor=sh_tensor,
    )


def build_variant(
    variant: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.nn.Module, BatchAdapter, Tuple, Dict]:
    if variant == "1d":
        train_loader, val_loader, test_loader = create_intensity_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            random_seed=args.seed,
        )
        model = GaussianPhysicsCompression1D(
            num_gaussians=args.num_gaussians,
            rank=args.rank,
            sh_dim=27,
        ).to(device)
        adapter = BatchAdapter(params_key="intensity")
        init_fn = _init_1d
        temporal_loss_fn = lambda _model: torch.tensor(0.0, device=device)
    elif variant == "5d":
        train_loader, val_loader, test_loader = create_dataloaders_5D(
            data_root=args.data_root,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            num_workers=args.num_workers,
            normalize_probes=True,
            normalize_params=True,
        )
        model = GaussianPhysicsCompression5D(
            num_gaussians=args.num_gaussians,
            rank=args.rank,
            sh_dim=27,
        ).to(device)
        adapter = BatchAdapter(params_key="light_params")
        init_fn = _init_5d
        temporal_loss_fn = None
    else:
        raise ValueError(f"Unknown variant: {variant}")

    loaders = (train_loader, val_loader, test_loader)
    return model, adapter, loaders, {"init_fn": init_fn, "temporal_loss_fn": temporal_loss_fn}


def run_training(args: argparse.Namespace) -> None:
    device = _device_from_arg(args.device)
    output_dir = Path(args.output_dir) if args.output_dir else None

    model, adapter, loaders, helpers = build_variant(args.variant, args, device)
    train_loader, val_loader, test_loader = loaders

    if not args.no_init:
        helpers["init_fn"](model, train_loader)
        model.to(device)

    trainer = GaussianPhysicsTrainer(
        model=model,
        device=device,
        adapter=adapter,
        lr=args.lr,
        recon_loss=args.recon_loss,
        charbonnier_eps=args.charbonnier_eps,
        temporal_weight=args.lambda_temporal,
        temporal_loss_fn=helpers.get("temporal_loss_fn"),
        top_k=args.top_k,
        grad_clip=args.grad_clip,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        output_dir=output_dir,
        save_best=output_dir is not None,
        best_metric="mae",
    )

    if test_loader is not None:
        test_metrics = trainer.validate_epoch(test_loader)
        print("Test metrics:", test_metrics)

    if args.print_history:
        print("Train history (last epoch):", {k: v[-1] for k, v in history["train"].items()})
        if history["val"]["mae"]:
            print("Val history (last epoch):", {k: v[-1] for k, v in history["val"].items()})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Gaussian-Physics trainer")

    parser.add_argument("--variant", choices=["1d", "5d"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)

    parser.add_argument("--num-gaussians", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--recon-loss", choices=["mse", "l1", "charbonnier"], default=None)
    parser.add_argument("--charbonnier-eps", type=float, default=1e-3)
    parser.add_argument("--lambda-temporal", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--no-init", action="store_true", help="Skip K-Means initialization")
    parser.add_argument("--print-history", action="store_true")

    return parser


def apply_variant_defaults(args: argparse.Namespace) -> None:
    defaults = {
        "1d": {
            "num_gaussians": 30,
            "rank": 8,
            "epochs": 2000,
            "batch_size": 256,
            "lr": 1e-3,
            "lambda_temporal": 0.001,
            "recon_loss": "charbonnier",
        },
        "5d": {
            "num_gaussians": 20,
            "rank": 5,
            "epochs": 2000,
            "batch_size": 512,
            "lr": 1e-3,
            "lambda_temporal": 0.001,
            "recon_loss": "charbonnier",
        },
    }

    variant_defaults = defaults.get(args.variant, {})
    for key, value in variant_defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    apply_variant_defaults(args)
    run_training(args)
