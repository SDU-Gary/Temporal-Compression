"""Config loading and argparse mapping for training script."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict


def load_yaml(path: str | Path) -> Dict[str, Any]:
    import yaml

    yaml_path = Path(path)
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML root must be a mapping")
    return data


def apply_config(args: argparse.Namespace, defaults: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    experiment = cfg.get("experiment", {})
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    training = cfg.get("training", {})
    evaluation = cfg.get("evaluation", cfg.get("eval", {}))

    def pick(*values):
        for value in values:
            if value is not None:
                return value
        return None

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
        "dataloader_timeout_seconds": pick(training.get("dataloader_timeout_seconds"), data.get("dataloader_timeout_seconds")),
        "dataloader_prefetch_factor": pick(training.get("dataloader_prefetch_factor"), data.get("dataloader_prefetch_factor")),
        "dataloader_persistent_workers": pick(training.get("dataloader_persistent_workers"), data.get("dataloader_persistent_workers")),
        "dataloader_multiprocessing_context": pick(training.get("dataloader_multiprocessing_context"), data.get("dataloader_multiprocessing_context")),
        "heartbeat_enabled": training.get("heartbeat_enabled"),
        "heartbeat_history": training.get("heartbeat_history"),
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
        "warmup_epochs": training.get("warmup_epochs"),
        "recon_loss": training.get("recon_loss"),
        "charbonnier_eps": training.get("charbonnier_eps"),
        "lambda_temporal": training.get("lambda_temporal"),
        "grad_clip": training.get("grad_clip"),
        "lambda_linearity": training.get("lambda_linearity"),
        "linearity_aug_pairs": training.get("linearity_aug_pairs"),
        "lambda_spatial": training.get("lambda_spatial"),
        "spatial_k": training.get("spatial_k"),
        "lambda_image": training.get("lambda_image"),
        "lambda_routing_balance": training.get("lambda_routing_balance"),
        "routing_soft_train": training.get("routing_soft_train"),
        "routing_soft_topk": training.get("routing_soft_topk"),
        "routing_temp_start": training.get("routing_temp_start"),
        "routing_temp_end": training.get("routing_temp_end"),
        "routing_temp_anneal_epochs": training.get("routing_temp_anneal_epochs"),
        "image_loss_type": training.get("image_loss_type"),
        "image_samples": training.get("image_samples"),
        "image_sample_seed": training.get("image_sample_seed"),
        "image_loss_space": training.get("image_loss_space"),
        "enable_weighted_sh_loss": training.get("enable_weighted_sh_loss"),
        "sh_loss_weights": training.get("sh_loss_weights"),
        "sh_weight_mode": training.get("sh_weight_mode"),
        "val_image_metrics": training.get("val_image_metrics"),
        "val_superposition": training.get("val_superposition"),
        "enable_sh_scaler": training.get("enable_sh_scaler"),
        "sh_scaler_path": training.get("sh_scaler_path"),
        "sh_scaler_max_samples": training.get("sh_scaler_max_samples"),
        "no_init": training.get("no_init"),
        "load_model": training.get("load_model"),
        "enable_rerun": training.get("enable_rerun") or training.get("rerun"),
        "rerun_log_freq": training.get("rerun_log_freq"),
        "rerun_save_path": training.get("rerun_save_path"),
        "show_progress": training.get("show_progress") or training.get("progress"),
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

