#!/usr/bin/env python3
"""Unified training entrypoint for Gaussian-Physics variants."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Tuple, Any

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "2_src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from training import BatchAdapter, GaussianPhysicsTrainer
from data.intensity_modulation_dataset import create_dataloaders as create_intensity_dataloaders
from data.transfer_tensor_dataset import create_dataloaders_5D
from data.lightset_dataset import create_dataloaders_lightset
from models.gaussian_physics_1D import GaussianPhysicsCompression1D
from models.gaussian_physics_5D import GaussianPhysicsCompression5D
from models.gaussian_physics_unified import GaussianPhysicsCompressionUnified
from utils.light_descriptor import build_descriptor_5d, build_descriptor_1d
from tools.manifest_utils import load_manifest
from tools.manifest_utils import get_git_commit
from tools.logexp import log_experiment
from utils.config import merge_configs, validate_with_schema


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    import yaml
    path = Path(path)
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML root must be a mapping")
    return data


def _apply_config(args: argparse.Namespace, defaults: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    # Support both flat and sectioned configs.
    experiment = cfg.get("experiment", {})
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    training = cfg.get("training", {})
    evaluation = cfg.get("evaluation", cfg.get("eval", {}))

    # Map config keys to argparse args.
    mapping = {
        "variant": experiment.get("variant") or cfg.get("variant"),
        "output_dir": experiment.get("output_dir") or cfg.get("output_dir"),
        "device": experiment.get("device") or cfg.get("device"),
        "seed": experiment.get("seed") or cfg.get("seed"),
        "data_root": data.get("data_root") or data.get("dataset_path") or cfg.get("data_root"),
        "manifest": data.get("manifest") or cfg.get("manifest"),
        "train_ratio": data.get("train_ratio"),
        "val_ratio": data.get("val_ratio"),
        "batch_size": data.get("batch_size"),
        "num_workers": data.get("num_workers"),
        "num_gaussians": model.get("num_gaussians"),
        "rank": model.get("rank"),
        "top_k": model.get("top_k"),
        "light_dim": model.get("light_dim"),
        "embed_dim": model.get("embed_dim"),
        "intensity_dim": model.get("intensity_dim"),
        "intensity_offset": model.get("intensity_offset"),
        "disable_film": model.get("disable_film"),
        "epochs": training.get("epochs") or training.get("num_epochs"),
        "lr": training.get("lr"),
        "weight_decay": training.get("weight_decay"),
        "lr_scheduler": training.get("lr_scheduler"),
        "lr_min": training.get("lr_min"),
        "recon_loss": training.get("recon_loss"),
        "charbonnier_eps": training.get("charbonnier_eps"),
        "lambda_temporal": training.get("lambda_temporal"),
        "grad_clip": training.get("grad_clip"),
        "lambda_linearity": training.get("lambda_linearity"),
        "linearity_aug_pairs": training.get("linearity_aug_pairs"),
        "lambda_spatial": training.get("lambda_spatial"),
        "spatial_k": training.get("spatial_k"),
        "enable_sh_scaler": training.get("enable_sh_scaler"),
        "sh_scaler_path": training.get("sh_scaler_path"),
        "sh_scaler_max_samples": training.get("sh_scaler_max_samples"),
        "no_init": training.get("no_init"),
        "load_model": training.get("load_model"),
        "enable_rerun": training.get("enable_rerun") or training.get("rerun"),
        "rerun_log_freq": training.get("rerun_log_freq"),
        "rerun_save_path": training.get("rerun_save_path"),
        "show_progress": training.get("show_progress") or training.get("progress"),
        # Eval defaults to keep in same config file for run_all.py consumption.
        "eval_output_dir": evaluation.get("output_dir") or evaluation.get("out_dir"),
        "eval_checkpoint": evaluation.get("checkpoint"),
    }

    for key, value in mapping.items():
        if value is None or not hasattr(args, key):
            continue
        current = getattr(args, key)
        default = getattr(defaults, key, None)
        if current == default:
            setattr(args, key, value)


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_data_root(data_root: str | None, manifest_path: str | None) -> str:
    if data_root:
        return data_root
    if not manifest_path:
        raise ValueError("data_root is required when manifest is not provided")
    manifest = load_manifest(manifest_path)
    output_dir = manifest.get("output_dir")
    if not output_dir:
        raise ValueError("manifest missing output_dir")
    return str(output_dir)


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
    elif variant == "unified_5d":
        train_loader, val_loader, test_loader = create_dataloaders_5D(
            data_root=args.data_root,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            num_workers=args.num_workers,
            normalize_probes=True,
            normalize_params=True,
        )
        dataset = train_loader.dataset
        param_min = torch.tensor(dataset.param_min, dtype=torch.float32)
        param_max = torch.tensor(dataset.param_max, dtype=torch.float32)

        def to_descriptor(params: torch.Tensor) -> torch.Tensor:
            return build_descriptor_5d(params, param_min.to(params.device), param_max.to(params.device))

        model = GaussianPhysicsCompressionUnified(
            num_gaussians=args.num_gaussians,
            rank=args.rank,
            sh_dim=27,
            light_dim=args.light_dim,
            embed_dim=args.embed_dim,
            intensity_dim=args.intensity_dim,
            intensity_offset=args.intensity_offset,
            enable_film=not args.disable_film,
        ).to(device)
        adapter = BatchAdapter(params_key="light_params", params_transform=to_descriptor)
        init_fn = _init_5d
        temporal_loss_fn = None
    elif variant == "unified_set":
        train_loader, val_loader, test_loader = create_dataloaders_lightset(
            data_root=args.data_root,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            num_workers=args.num_workers,
            normalize_probes=True,
        )
        model = GaussianPhysicsCompressionUnified(
            num_gaussians=args.num_gaussians,
            rank=args.rank,
            sh_dim=27,
            light_dim=args.light_dim,
            embed_dim=args.embed_dim,
            intensity_dim=args.intensity_dim,
            intensity_offset=args.intensity_offset,
            enable_film=not args.disable_film,
        ).to(device)
        adapter = BatchAdapter(params_key="light_params", mask_key="light_mask")
        init_fn = _init_5d
        temporal_loss_fn = None
    elif variant == "unified_1d":
        train_loader, val_loader, test_loader = create_intensity_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            random_seed=args.seed,
        )

        def to_descriptor(params: torch.Tensor) -> torch.Tensor:
            return build_descriptor_1d(params)

        model = GaussianPhysicsCompressionUnified(
            num_gaussians=args.num_gaussians,
            rank=args.rank,
            sh_dim=27,
            light_dim=args.light_dim,
            embed_dim=args.embed_dim,
            intensity_dim=args.intensity_dim,
            intensity_offset=args.intensity_offset,
            enable_film=not args.disable_film,
        ).to(device)
        adapter = BatchAdapter(params_key="intensity", params_transform=to_descriptor)
        init_fn = _init_1d
        temporal_loss_fn = lambda _model: torch.tensor(0.0, device=device)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    loaders = (train_loader, val_loader, test_loader)
    return model, adapter, loaders, {"init_fn": init_fn, "temporal_loss_fn": temporal_loss_fn}


def run_training(args: argparse.Namespace) -> None:
    device = _device_from_arg(args.device)
    output_dir = Path(args.output_dir) if args.output_dir else None

    if output_dir is None:
        from datetime import datetime
        run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = _ROOT / "3_experiments" / "results" / args.variant / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.show_progress:
        print("[init] loading dataset and building model...")
    model, adapter, loaders, helpers = build_variant(args.variant, args, device)
    train_loader, val_loader, test_loader = loaders

    if not args.no_init:
        if args.show_progress:
            print("[init] running K-Means/SVD initialization...")
        helpers["init_fn"](model, train_loader)
        model.to(device)
        if args.show_progress:
            print("[init] initialization done.")

    # Optional SH scaling
    if args.enable_sh_scaler:
        from data.sh_scaler import AdaptiveSHScaler

        scaler_path = Path(args.sh_scaler_path) if args.sh_scaler_path else None
        if scaler_path is None and output_dir is not None:
            scaler_path = output_dir / "sh_scaler.npz"

        if scaler_path is not None and scaler_path.exists():
            scaler = AdaptiveSHScaler.load(scaler_path)
            print(f"Loaded SH scaler from {scaler_path}")
        else:
            scaler = AdaptiveSHScaler.fit_from_dataset(
                train_loader.dataset,
                max_samples=args.sh_scaler_max_samples,
            )
            if scaler_path is not None:
                scaler.save(scaler_path)
                print(f"Saved SH scaler to {scaler_path}")

        adapter.target_transform = scaler.transform
        adapter.target_inverse = scaler.inverse
        l0_mean = [float(x) for x in scaler.l0_mean]
        l0_std = [float(x) for x in scaler.l0_std]
        ho_rms = [float(x) for x in scaler.ho_rms]
        print(f"SH scaler stats: L0 mean={l0_mean}, L0 std={l0_std}, HO rms={ho_rms}")

    if args.load_model:
        ckpt = torch.load(args.load_model, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"Warning: Missing keys when loading model: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys when loading model: {unexpected}")

    # Initialize Rerun logger if enabled
    if args.enable_rerun and not args.rerun_save_path and output_dir is not None:
        args.rerun_save_path = str(output_dir / "train.rrd")

    rerun_logger = None
    if args.enable_rerun:
        try:
            from utils.rerun_logger import RerunLogger
            # Disable spawn if saving to file (avoids GUI blocking)
            spawn_viewer = not args.rerun_save_path
            rerun_logger = RerunLogger(
                app_id=f"gaussian_physics_{args.variant}",
                spawn=spawn_viewer,
                save_path=Path(args.rerun_save_path) if args.rerun_save_path else None,
                log_frequency=args.rerun_log_freq
            )
            print(f"Rerun visualization enabled (log every {args.rerun_log_freq} epochs)")
        except ImportError as e:
            print(f"Warning: Could not initialize Rerun logger: {e}")
            print("Install with: pip install rerun-sdk")
            rerun_logger = None

    trainer = GaussianPhysicsTrainer(
        model=model,
        device=device,
        adapter=adapter,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        lr_min=args.lr_min,
        recon_loss=args.recon_loss,
        charbonnier_eps=args.charbonnier_eps,
        temporal_weight=args.lambda_temporal,
        temporal_loss_fn=helpers.get("temporal_loss_fn"),
        top_k=args.top_k,
        grad_clip=args.grad_clip,
        linearity_weight=args.lambda_linearity,
        linearity_aug_pairs=args.linearity_aug_pairs,
        spatial_weight=args.lambda_spatial,
        spatial_k=args.spatial_k,
        rerun_logger=rerun_logger,  # Pass logger to trainer
        show_progress=args.show_progress,
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

    if not args.no_auto_log:
        # Summarize metrics for logging.
        train_last = {k: (v[-1] if v else None) for k, v in history["train"].items()}
        val_last = {k: (v[-1] if v else None) for k, v in history["val"].items()}
        result_metrics: Dict[str, Any] = {}
        result_metrics.update({f"train_{k}": v for k, v in train_last.items() if v is not None})
        result_metrics.update({f"val_{k}": v for k, v in val_last.items() if v is not None})
        if test_loader is not None:
            result_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
        result_metrics["param_count"] = sum(p.numel() for p in model.parameters())
        result_metrics["git_commit"] = get_git_commit()

        dataset_id = args.log_dataset_id
        if dataset_id is None and args.manifest:
            try:
                manifest = load_manifest(args.manifest)
                dataset_id = manifest.get("dataset_id")
            except Exception:
                dataset_id = None
        if dataset_id is None:
            dataset_id = Path(args.data_root).name

        hyperparams = {
            "variant": args.variant,
            "num_gaussians": args.num_gaussians,
            "rank": args.rank,
            "top_k": args.top_k,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lr_scheduler": args.lr_scheduler,
            "lr_min": args.lr_min,
            "recon_loss": args.recon_loss,
            "charbonnier_eps": args.charbonnier_eps,
            "lambda_temporal": args.lambda_temporal,
            "lambda_linearity": args.lambda_linearity,
            "linearity_aug_pairs": args.linearity_aug_pairs,
            "lambda_spatial": args.lambda_spatial,
            "spatial_k": args.spatial_k,
            "light_dim": args.light_dim,
            "embed_dim": args.embed_dim,
            "intensity_dim": args.intensity_dim,
            "intensity_offset": args.intensity_offset,
        }

        try:
            exp_id = log_experiment(
                phase=args.log_phase,
                stage=args.log_stage,
                script_id=args.log_script_id,
                dataset_id=dataset_id,
                hyperparams=hyperparams,
                results=result_metrics,
                checkpoint_path=str(output_dir),
                notes=args.log_notes,
            )
            print(f"✅ Logged experiment: {exp_id}")
        except Exception as e:
            print(f"Warning: Auto-log failed: {e}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Gaussian-Physics trainer")

    parser.add_argument("--variant", choices=["1d", "5d", "unified_1d", "unified_5d", "unified_set"], required=False)
    parser.add_argument("--config", default=None, help="YAML config for training")
    parser.add_argument(
        "--schema",
        default=str(_ROOT / "metadata" / "schemas" / "train.schema.json"),
        help="Optional JSON schema for config validation",
    )
    parser.add_argument("--strict-schema", action="store_true", help="Fail if schema validation fails")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--manifest", default=None, help="Path to dataset manifest.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-id", default=None, help="Optional run id for output directory naming")

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
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine", "plateau"], default="none")
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--recon-loss", choices=["mse", "l1", "charbonnier"], default=None)
    parser.add_argument("--charbonnier-eps", type=float, default=1e-3)
    parser.add_argument("--lambda-temporal", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-linearity", type=float, default=0.0,
                        help="Weight for superposition linearity loss (requires light mask).")
    parser.add_argument("--linearity-aug-pairs", type=int, default=0,
                        help="Number of on-the-fly linearity augmentation pairs per batch.")
    parser.add_argument("--lambda-spatial", type=float, default=0.0,
                        help="Weight for spatial smoothness (KNN-TV) loss.")
    parser.add_argument("--spatial-k", type=int, default=1,
                        help="Number of nearest neighbors for spatial loss.")

    parser.add_argument("--light-dim", type=int, default=None,
                        help="Light descriptor dimension for unified model.")
    parser.add_argument("--embed-dim", type=int, default=None,
                        help="Light embedding dimension for unified model.")
    parser.add_argument("--intensity-dim", type=int, default=None,
                        help="Number of intensity channels at descriptor start (1 or 3).")
    parser.add_argument("--intensity-offset", type=int, default=None,
                        help="Start index for intensity channels in descriptor.")
    parser.add_argument("--disable-film", action="store_true",
                        help="Disable FiLM dynamic basis modulation (ablation).")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to checkpoint (.pt) to load model weights from.")

    parser.add_argument("--enable-sh-scaler", action="store_true",
                        help="Enable adaptive SH scaling (L0 mean/std + higher-order RMS).")
    parser.add_argument("--sh-scaler-path", type=str, default=None,
                        help="Path to load/save SH scaler (npz).")
    parser.add_argument("--sh-scaler-max-samples", type=int, default=200000,
                        help="Max samples to estimate SH scaler statistics.")

    parser.add_argument("--no-init", action="store_true", help="Skip K-Means initialization")
    parser.add_argument("--print-history", action="store_true")

    # Rerun visualization arguments
    parser.add_argument("--enable-rerun", action="store_true", dest="enable_rerun",
                       help="Enable Rerun visualization (default on)")
    parser.add_argument("--no-rerun", action="store_false", dest="enable_rerun",
                       help="Disable Rerun visualization")
    parser.set_defaults(enable_rerun=True)
    parser.add_argument("--rerun-log-freq", type=int, default=10,
                       help="Log visualizations every N epochs")
    parser.add_argument("--rerun-save-path", type=str, default=None,
                       help="Save .rrd file to path (default: output_dir/train.rrd)")

    # Progress display
    parser.add_argument("--no-progress", action="store_false", dest="show_progress",
                       help="Disable progress bars and init status logs")
    parser.set_defaults(show_progress=True)

    # Auto logging
    parser.add_argument("--no-auto-log", action="store_true", help="Disable auto experiment logging")
    parser.add_argument("--log-phase", default="Phase2_PGCPL")
    parser.add_argument("--log-stage", default="training")
    parser.add_argument("--log-script-id", default="train")
    parser.add_argument("--log-dataset-id", default=None)
    parser.add_argument("--log-notes", default=None)

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
        "unified_1d": {
            "num_gaussians": 30,
            "rank": 8,
            "epochs": 2000,
            "batch_size": 256,
            "lr": 1e-3,
            "lambda_temporal": 0.001,
            "recon_loss": "charbonnier",
            "light_dim": 12,
            "embed_dim": 32,
            "intensity_dim": 3,
            "intensity_offset": 1,
        },
        "unified_5d": {
            "num_gaussians": 20,
            "rank": 5,
            "epochs": 2000,
            "batch_size": 512,
            "lr": 1e-3,
            "lambda_temporal": 0.001,
            "recon_loss": "charbonnier",
            "light_dim": 12,
            "embed_dim": 32,
            "intensity_dim": 3,
            "intensity_offset": 1,
        },
        "unified_set": {
            "num_gaussians": 20,
            "rank": 5,
            "epochs": 2000,
            "batch_size": 512,
            "lr": 1e-3,
            "lambda_temporal": 0.001,
            "recon_loss": "charbonnier",
            "light_dim": 12,
            "embed_dim": 32,
            "intensity_dim": 3,
            "intensity_offset": 1,
            "lambda_linearity": 0.1,
            "linearity_aug_pairs": 2,
        },
    }

    variant_defaults = defaults.get(args.variant, {})
    for key, value in variant_defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


if __name__ == "__main__":  # pragma: no cover
    parser = build_arg_parser()
    defaults = parser.parse_args([])
    args = parser.parse_args()

    if args.config:
        cfg = _load_yaml(args.config)
        if args.schema:
            messages = validate_with_schema(cfg, args.schema, strict=args.strict_schema)
            for msg in messages:
                print(f"[schema] {msg}")
        # Allow legacy "train" section at top.
        if "train" in cfg and isinstance(cfg["train"], dict):
            cfg = merge_configs(cfg, cfg["train"])
        _apply_config(args, defaults, cfg)

    if args.variant is None:
        raise ValueError("variant must be provided via --variant or config")
    args.data_root = resolve_data_root(args.data_root, args.manifest)
    apply_variant_defaults(args)
    run_training(args)
