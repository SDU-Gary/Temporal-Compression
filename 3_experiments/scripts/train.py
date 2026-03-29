#!/usr/bin/env python3
"""Unified training entrypoint for Gaussian-Physics variants."""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import Future, ThreadPoolExecutor
import json
import numpy as np
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from copy import deepcopy
from typing import Dict, Tuple, Any, List

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _path_setup import ensure_repo_paths
from train_arg_utils import (
    apply_variant_defaults as apply_variant_defaults_from_profile,
    normalize_namespace,
)
from train_config_utils import (
    apply_config as apply_config_from_config,
    load_yaml as load_yaml_config,
)

_ROOT, _SRC = ensure_repo_paths(__file__, root_levels=2)

from tools.manifest_utils import load_manifest
from tools.manifest_utils import get_git_commit
from tools.logexp import log_experiment
from utils.config import merge_configs, validate_with_schema


# Heavy deps (torch + training + model) are imported lazily so that
# `python3 3_experiments/scripts/train.py --help` works even when the
# runtime environment hasn't been activated yet.
torch = None
BatchAdapter = None
GaussianPhysicsTrainer = None
create_dataloaders_lightset = None
GaussianPhysicsCompressionUnified = None
_TRAIN_ARG_DEFAULTS = None


def _lazy_imports() -> None:
    global torch
    global BatchAdapter
    global GaussianPhysicsTrainer
    global create_dataloaders_lightset
    global GaussianPhysicsCompressionUnified

    if torch is None:
        import torch as _torch

        torch = _torch
    if BatchAdapter is None or GaussianPhysicsTrainer is None:
        from training import BatchAdapter as _BatchAdapter
        from training import GaussianPhysicsTrainer as _GaussianPhysicsTrainer

        BatchAdapter = _BatchAdapter
        GaussianPhysicsTrainer = _GaussianPhysicsTrainer
    if create_dataloaders_lightset is None:
        from data.lightset_dataset import create_dataloaders_lightset as _create_dataloaders_lightset

        create_dataloaders_lightset = _create_dataloaders_lightset
    if GaussianPhysicsCompressionUnified is None:
        from models.gaussian_physics_unified import (
            GaussianPhysicsCompressionUnified as _GaussianPhysicsCompressionUnified,
        )

        GaussianPhysicsCompressionUnified = _GaussianPhysicsCompressionUnified


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    return load_yaml_config(path)


def _apply_config(args: argparse.Namespace, defaults: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    apply_config_from_config(args, defaults, cfg)


def _device_from_arg(device_arg: str):
    _lazy_imports()
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


def _init_5d(model, train_loader) -> None:
    _lazy_imports()
    dataset = train_loader.dataset
    sh_tensor = torch.from_numpy(dataset.tensor).float()
    model.init_from_kmeans(
        probe_positions=dataset.probe_positions,
        light_configs=dataset.light_configs_subset,
        sh_tensor=sh_tensor,
    )


def _resolve_bypass_feature_pairs(raw: Any) -> List[List[int]] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError("bypass_feature_pairs must be a list/tuple")
    if len(raw) == 0:
        return None

    # CLI may pass a flattened int list; config typically uses nested pairs.
    if all(not isinstance(item, (list, tuple)) for item in raw):
        if len(raw) % 2 != 0:
            raise ValueError("bypass_feature_pairs flattened list must contain an even number of integers")
        out: List[List[int]] = []
        for i in range(0, len(raw), 2):
            out.append([int(raw[i]), int(raw[i + 1])])
        return out

    out = []
    for idx, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ValueError(
                f"bypass_feature_pairs[{idx}] must be [light_index, feature_index]"
            )
        out.append([int(item[0]), int(item[1])])
    return out


def build_variant(
    variant: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.nn.Module, BatchAdapter, Tuple, Dict]:
    _lazy_imports()
    if variant == "unified_set":
        train_loader, val_loader, test_loader = create_dataloaders_lightset(
            data_root=args.data_root,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            num_workers=args.num_workers,
            normalize_probes=True,
            dataloader_timeout_seconds=getattr(args, "dataloader_timeout_seconds", 0),
            dataloader_prefetch_factor=getattr(args, "dataloader_prefetch_factor", None),
            dataloader_persistent_workers=getattr(args, "dataloader_persistent_workers", None),
            dataloader_multiprocessing_context=getattr(args, "dataloader_multiprocessing_context", "none"),
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
            light_encoder_mode=str(getattr(args, "light_encoder_mode", "normal")),
            bypass_feature_pairs=_resolve_bypass_feature_pairs(
                getattr(args, "bypass_feature_pairs", None)
            ),
            bypass_feature_norm_mean=getattr(args, "bypass_feature_norm_mean", None),
            bypass_feature_norm_std=getattr(args, "bypass_feature_norm_std", None),
        ).to(device)
        adapter = BatchAdapter(params_key="light_params", mask_key="light_mask")
        init_fn = _init_5d
        temporal_loss_fn = None
    else:
        raise ValueError(f"Unknown variant: {variant}")

    loaders = (train_loader, val_loader, test_loader)
    return model, adapter, loaders, {"init_fn": init_fn, "temporal_loss_fn": temporal_loss_fn}


_TRAINABLE_GROUPS = {"all", "routing", "basis", "coeff", "encoder", "film"}


def _empty_history() -> Dict[str, Dict[str, List[float]]]:
    return {
        "train": {"total": [], "recon": [], "image": [], "coeff_l1": [], "linearity": [], "spatial": []},
        "val": {"mae": [], "rmse": [], "charbonnier": [], "superposition": [], "img_mae": [], "img_rmse": [], "img_psnr": []},
    }


def _merge_history(dst: Dict[str, Dict[str, List[float]]], src: Dict[str, Dict[str, List[float]]]) -> None:
    for split, keys in dst.items():
        src_split = src.get(split, {})
        for key in keys:
            values = src_split.get(key, [])
            if values:
                dst[split][key].extend(values)


def _sanitize_stage_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_\-]+", "_", str(name).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "stage"


def _profile_checkpoint_filename(profile_name: str) -> str:
    safe = _sanitize_stage_name(profile_name)
    return f"best_model_val_{safe}.pt"


def _resolve_resume_checkpoint_path(resume_arg: str, output_dir: Path) -> Path:
    raw = Path(str(resume_arg))
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(output_dir / raw)
        candidates.append(_ROOT / raw)
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError(f"resume checkpoint not found: {resume_arg}")


def _infer_stage_index_from_checkpoint_path(path: Path) -> int | None:
    for part in [path.parent.name, path.parent.parent.name]:
        m = re.match(r"^stage(\d+)_", part)
        if m:
            return int(m.group(1))
    return None


def _param_group_from_name(name: str) -> str:
    if name == "mu" or name.startswith("mu."):
        return "routing"
    if name == "log_scale" or name.startswith("log_scale."):
        return "routing"
    if name == "U" or name.startswith("U."):
        return "basis"
    if name == "U_l0" or name.startswith("U_l0."):
        return "basis"
    if name == "coeffs" or name.startswith("coeffs."):
        return "coeff"
    if name == "coeffs_l0" or name.startswith("coeffs_l0."):
        return "coeff"
    if name.startswith("light_encoder."):
        return "encoder"
    if name.startswith("gamma.") or name.startswith("beta."):
        return "film"
    return "other"


def _set_trainable_groups(model, trainable_groups: List[str]) -> Dict[str, Any]:
    groups = set(trainable_groups)
    if not groups:
        raise ValueError("trainable_groups cannot be empty")
    unknown = sorted(g for g in groups if g not in _TRAINABLE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown trainable groups: {unknown}. Allowed: {sorted(_TRAINABLE_GROUPS)}")

    all_params = list(model.named_parameters())
    if "all" in groups:
        for _, param in all_params:
            param.requires_grad = True
    else:
        for _, param in all_params:
            param.requires_grad = False
        for name, param in all_params:
            if _param_group_from_name(name) in groups:
                param.requires_grad = True

    total_param_count = sum(int(p.numel()) for _, p in all_params)
    trainable_param_count = sum(int(p.numel()) for _, p in all_params if p.requires_grad)
    trainable_names = [name for name, p in all_params if p.requires_grad]
    if trainable_param_count <= 0:
        raise ValueError(f"No trainable parameters selected for groups={sorted(groups)}")

    group_counts: Dict[str, int] = {}
    for name, param in all_params:
        grp = _param_group_from_name(name)
        group_counts.setdefault(grp, 0)
        if param.requires_grad:
            group_counts[grp] += int(param.numel())

    return {
        "trainable_groups": sorted(groups),
        "total_param_count": int(total_param_count),
        "trainable_param_count": int(trainable_param_count),
        "trainable_ratio": float(trainable_param_count / max(1, total_param_count)),
        "trainable_param_names": trainable_names,
        "trainable_param_counts_by_group": group_counts,
    }


def _resolve_staged_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    cfg = getattr(args, "_config_obj", None)
    if not isinstance(cfg, dict):
        return []

    training_cfg = cfg.get("training")
    if not isinstance(training_cfg, dict):
        return []

    staged_cfg = training_cfg.get("staged")
    if not isinstance(staged_cfg, dict) or not bool(staged_cfg.get("enabled", False)):
        return []

    raw_stages = staged_cfg.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("training.staged.enabled=true but training.staged.stages is missing or empty")

    specs: List[Dict[str, Any]] = []
    for idx, raw in enumerate(raw_stages, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"training.staged.stages[{idx-1}] must be a mapping")

        name = str(raw.get("name") or f"stage{idx}")
        epochs = int(raw.get("epochs", 0))
        if epochs <= 0:
            raise ValueError(f"training.staged.stages[{idx-1}].epochs must be > 0")

        trainable_groups = raw.get("trainable_groups", ["all"])
        if isinstance(trainable_groups, str):
            trainable_groups = [x.strip() for x in trainable_groups.split(",") if x.strip()]
        if not isinstance(trainable_groups, list) or not trainable_groups:
            raise ValueError(f"training.staged.stages[{idx-1}].trainable_groups must be a non-empty list")

        raw_group_lrs = raw.get("group_lrs", {})
        if raw_group_lrs is None:
            raw_group_lrs = {}
        if not isinstance(raw_group_lrs, dict):
            raise ValueError(f"training.staged.stages[{idx-1}].group_lrs must be a mapping when provided")

        group_lrs: Dict[str, float] = {}
        for key, value in raw_group_lrs.items():
            group_lrs[str(key)] = float(value)

        spec = {
            "name": name,
            "epochs": epochs,
            "lr": float(raw.get("lr", args.lr)),
            "weight_decay": float(raw.get("weight_decay", args.weight_decay)),
            "lr_scheduler": str(raw.get("lr_scheduler", args.lr_scheduler)),
            "lr_min": float(raw.get("lr_min", args.lr_min)),
            "warmup_epochs": int(raw.get("warmup_epochs", int(getattr(args, "warmup_epochs", 0)))),
            "best_metric": str(raw.get("best_metric", "mae")),
            "trainable_groups": [str(x) for x in trainable_groups],
            "group_lrs": group_lrs,
        }
        specs.append(spec)

    return specs


def _resolve_ema_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = getattr(args, "_config_obj", None)
    raw_eval_default = max(1, int(getattr(args, "ema_raw_eval_every_epochs", 1)))
    if not isinstance(cfg, dict):
        return {
            "enabled": False,
            "decay": 0.999,
            "eval_on_ema": False,
            "save_best_with_ema": False,
            "raw_eval_every_epochs": raw_eval_default,
        }

    training_cfg = cfg.get("training")
    if not isinstance(training_cfg, dict):
        return {
            "enabled": False,
            "decay": 0.999,
            "eval_on_ema": False,
            "save_best_with_ema": False,
            "raw_eval_every_epochs": raw_eval_default,
        }

    ema_cfg = training_cfg.get("ema")
    if not isinstance(ema_cfg, dict):
        return {
            "enabled": False,
            "decay": 0.999,
            "eval_on_ema": False,
            "save_best_with_ema": False,
            "raw_eval_every_epochs": raw_eval_default,
        }

    enabled = bool(ema_cfg.get("enabled", False))
    decay = float(ema_cfg.get("decay", 0.999))
    eval_on_ema = bool(ema_cfg.get("eval_on_ema", True if enabled else False))
    save_best_with_ema = bool(ema_cfg.get("save_best_with_ema", True if enabled else False))
    raw_eval_every_epochs = int(ema_cfg.get("raw_eval_every_epochs", raw_eval_default))
    return {
        "enabled": enabled,
        "decay": decay,
        "eval_on_ema": eval_on_ema,
        "save_best_with_ema": save_best_with_ema,
        "raw_eval_every_epochs": max(1, raw_eval_every_epochs),
    }


def _resolve_oracle_monitor_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = getattr(args, "_config_obj", None)
    if not isinstance(cfg, dict):
        return {"enabled": False}

    training_cfg = cfg.get("training")
    if not isinstance(training_cfg, dict):
        return {"enabled": False}

    oracle_cfg = training_cfg.get("oracle_monitor")
    if not isinstance(oracle_cfg, dict):
        return {"enabled": False}

    enabled = bool(oracle_cfg.get("enabled", False))
    if not enabled:
        return {"enabled": False}

    return {
        "enabled": True,
        "split": str(oracle_cfg.get("split", "test")),
        "period_epochs": int(oracle_cfg.get("period_epochs", 0)),
        "lightweight_max_samples": int(oracle_cfg.get("lightweight_max_samples", 4096)),
        "lightweight_sample_seed": int(oracle_cfg.get("lightweight_sample_seed", args.seed)),
        "stage_end_full": bool(oracle_cfg.get("stage_end_full", True)),
        "full_max_samples": int(oracle_cfg.get("full_max_samples", 0)),
        "batch_size": int(oracle_cfg.get("batch_size", args.batch_size)),
        "num_workers": int(oracle_cfg.get("num_workers", 0)),
        "top_k": int(oracle_cfg.get("top_k", args.top_k)),
        "timeout_seconds": int(oracle_cfg.get("timeout_seconds", 180)),
        "fail_on_timeout": bool(oracle_cfg.get("fail_on_timeout", False)),
    }


def _resolve_coeff_suite_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = getattr(args, "_config_obj", None)
    if not isinstance(cfg, dict):
        return {"enabled": False}

    training_cfg = cfg.get("training")
    if not isinstance(training_cfg, dict):
        return {"enabled": False}

    suite_cfg = training_cfg.get("coeff_suite")
    if not isinstance(suite_cfg, dict):
        return {"enabled": False}

    enabled = bool(suite_cfg.get("enabled", False))
    if not enabled:
        return {"enabled": False}

    checkpoints_raw = suite_cfg.get("checkpoints", None)
    checkpoints: List[str] = []
    if checkpoints_raw is not None:
        if not isinstance(checkpoints_raw, list):
            raise ValueError("training.coeff_suite.checkpoints must be a list when provided")
        for item in checkpoints_raw:
            name = str(item).strip()
            if name:
                checkpoints.append(name)

    return {
        "enabled": True,
        "run_after_training": bool(suite_cfg.get("run_after_training", True)),
        "split": str(suite_cfg.get("split", "test")),
        "device": str(suite_cfg.get("device", "cpu")),
        "max_samples": int(suite_cfg.get("max_samples", 8192)),
        "sample_seed": int(suite_cfg.get("sample_seed", args.seed)),
        "batch_size": int(suite_cfg.get("batch_size", args.batch_size)),
        "num_workers": int(suite_cfg.get("num_workers", 0)),
        "probe_train_ratio": float(suite_cfg.get("probe_train_ratio", 0.8)),
        "ridge_alpha": float(suite_cfg.get("ridge_alpha", 1e-4)),
        "mlp_hidden": int(suite_cfg.get("mlp_hidden", 128)),
        "mlp_epochs": int(suite_cfg.get("mlp_epochs", 30)),
        "mlp_lr": float(suite_cfg.get("mlp_lr", 1e-3)),
        "mlp_batch_size": int(suite_cfg.get("mlp_batch_size", 256)),
        "timeout_seconds": int(suite_cfg.get("timeout_seconds", 1800)),
        "fail_on_timeout": bool(suite_cfg.get("fail_on_timeout", False)),
        "checkpoints": checkpoints,
    }


def _resolve_global_group_lrs(args: argparse.Namespace) -> Dict[str, float]:
    cfg = getattr(args, "_config_obj", None)
    if not isinstance(cfg, dict):
        return {}

    training_cfg = cfg.get("training")
    if not isinstance(training_cfg, dict):
        return {}

    raw = training_cfg.get("group_lrs")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("training.group_lrs must be a mapping")
    return {str(k): float(v) for k, v in raw.items()}


def _resolve_global_best_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = getattr(args, "_config_obj", None)
    defaults = {
        "enabled": True,
        "track_falcor": True,
        "track_soft_profile": True,
        "soft_profile_name": "soft8_t018",
        "soft_metric": "img_psnr",
        "soft_metric_maximize": True,
    }
    if not isinstance(cfg, dict):
        return defaults

    training_cfg = cfg.get("training")
    if not isinstance(training_cfg, dict):
        return defaults

    raw = training_cfg.get("global_best")
    if raw is None:
        return defaults
    if not isinstance(raw, dict):
        raise ValueError("training.global_best must be a mapping when provided")

    out = dict(defaults)
    out["enabled"] = bool(raw.get("enabled", out["enabled"]))
    out["track_falcor"] = bool(raw.get("track_falcor", out["track_falcor"]))
    out["track_soft_profile"] = bool(raw.get("track_soft_profile", out["track_soft_profile"]))
    out["soft_profile_name"] = str(raw.get("soft_profile_name", out["soft_profile_name"])).strip() or str(
        out["soft_profile_name"]
    )
    out["soft_metric"] = str(raw.get("soft_metric", out["soft_metric"])).strip() or str(out["soft_metric"])
    out["soft_metric_maximize"] = bool(raw.get("soft_metric_maximize", out["soft_metric_maximize"]))
    return out


def _resolve_proxy_image_loss_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = getattr(args, "_config_obj", None)
    lambda_default = float(getattr(args, "lambda_image", 0.0))
    warmup_default = int(getattr(args, "image_loss_warmup_epochs", 0))
    mix_mode_default = str(getattr(args, "proxy_image_mix_mode", "linear_log_mix")).strip().lower()
    log_mix_weight_default = float(getattr(args, "proxy_image_log_mix_weight", 0.3))

    def _normalize_mix_mode(raw: Any) -> str:
        mode = str(raw if raw is not None else mix_mode_default).strip().lower()
        if mode not in {"linear_log_mix", "linear_only", "log_only"}:
            raise ValueError(
                "training.proxy_image_loss.mix_mode must be one of: "
                "linear_log_mix, linear_only, log_only"
            )
        return mode

    def _normalize_weight(raw: Any) -> float:
        value = float(raw if raw is not None else log_mix_weight_default)
        return float(np.clip(value, 0.0, 1.0))

    if not isinstance(cfg, dict):
        return {
            "enabled": bool(lambda_default > 0.0),
            "lambda": lambda_default,
            "warmup_epochs": warmup_default,
            "enable_val_image_metrics": bool(lambda_default > 0.0),
            "mix_mode": _normalize_mix_mode(mix_mode_default),
            "log_mix_weight": _normalize_weight(log_mix_weight_default),
        }

    training_cfg = cfg.get("training")
    if not isinstance(training_cfg, dict):
        return {
            "enabled": bool(lambda_default > 0.0),
            "lambda": lambda_default,
            "warmup_epochs": warmup_default,
            "enable_val_image_metrics": bool(lambda_default > 0.0),
            "mix_mode": _normalize_mix_mode(mix_mode_default),
            "log_mix_weight": _normalize_weight(log_mix_weight_default),
        }

    raw = training_cfg.get("proxy_image_loss")
    if not isinstance(raw, dict):
        return {
            "enabled": bool(lambda_default > 0.0),
            "lambda": lambda_default,
            "warmup_epochs": warmup_default,
            "enable_val_image_metrics": bool(lambda_default > 0.0),
            "mix_mode": _normalize_mix_mode(mix_mode_default),
            "log_mix_weight": _normalize_weight(log_mix_weight_default),
        }

    enabled = bool(raw.get("enabled", lambda_default > 0.0))
    return {
        "enabled": enabled,
        "lambda": float(raw.get("lambda", lambda_default)),
        "warmup_epochs": int(raw.get("warmup_epochs", warmup_default)),
        "enable_val_image_metrics": bool(raw.get("enable_val_image_metrics", True if enabled else False)),
        "mix_mode": _normalize_mix_mode(raw.get("mix_mode", raw.get("mode", mix_mode_default))),
        "log_mix_weight": _normalize_weight(raw.get("log_mix_weight", raw.get("mix_weight", log_mix_weight_default))),
    }


def _resolve_routing_balance_anneal_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = getattr(args, "_config_obj", None)
    default_lambda = max(0.0, float(getattr(args, "lambda_routing_balance", 0.0)))
    default_ratio = 0.6

    if not isinstance(cfg, dict):
        return {
            "enabled": False,
            "start": float(default_lambda),
            "end": 0.0,
            "decay_end_ratio": float(default_ratio),
        }

    training_cfg = cfg.get("training")
    if not isinstance(training_cfg, dict):
        return {
            "enabled": False,
            "start": float(default_lambda),
            "end": 0.0,
            "decay_end_ratio": float(default_ratio),
        }

    raw = training_cfg.get("routing_balance_anneal")
    if not isinstance(raw, dict):
        return {
            "enabled": False,
            "start": float(default_lambda),
            "end": 0.0,
            "decay_end_ratio": float(default_ratio),
        }

    ratio = float(raw.get("decay_end_ratio", default_ratio))
    ratio = float(np.clip(ratio, 0.0, 1.0))
    return {
        "enabled": bool(raw.get("enabled", False)),
        "start": float(max(0.0, float(raw.get("start", default_lambda)))),
        "end": float(max(0.0, float(raw.get("end", 0.0)))),
        "decay_end_ratio": ratio,
    }


def _resolve_performance_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = getattr(args, "_config_obj", None)

    def _normalize_amp_mode(raw_mode: Any) -> str:
        if isinstance(raw_mode, bool):
            return "off" if raw_mode is False else "bf16"
        mode = str(raw_mode).strip().lower()
        if mode in {"off", "false", "0", "none", ""}:
            return "off"
        if mode in {"bf16", "bfloat16"}:
            return "bf16"
        if mode in {"fp16", "float16", "half"}:
            return "fp16"
        return "off"

    defaults = {
        "amp_mode": _normalize_amp_mode(getattr(args, "amp_mode", "off")),
        "torch_compile_enabled": bool(getattr(args, "torch_compile", False)),
        "torch_compile_mode": str(getattr(args, "torch_compile_mode", "reduce-overhead")),
        "torch_compile_dynamic": bool(getattr(args, "torch_compile_dynamic", False)),
        "grad_norm_log_every_steps": int(getattr(args, "grad_norm_log_every_steps", 1)),
        "enable_soft_profile_sharing": bool(getattr(args, "enable_soft_profile_sharing", False)),
        "soft_profile_equiv_check_batches": int(getattr(args, "soft_profile_equiv_check_batches", 2)),
        "soft_profile_mae_tolerance": float(getattr(args, "soft_profile_mae_tolerance", 1e-6)),
        "soft_profile_img_psnr_tolerance": float(getattr(args, "soft_profile_img_psnr_tolerance", 5e-4)),
    }
    if not isinstance(cfg, dict):
        return defaults
    training_cfg = cfg.get("training")
    if not isinstance(training_cfg, dict):
        return defaults
    raw = training_cfg.get("performance", {})
    if not isinstance(raw, dict):
        return defaults

    compile_raw = raw.get("torch_compile", {})
    if compile_raw is None:
        compile_raw = {}
    if not isinstance(compile_raw, dict):
        compile_raw = {}

    soft_profile_raw = raw.get("soft_profile_sharing", {})
    if soft_profile_raw is None:
        soft_profile_raw = {}
    if not isinstance(soft_profile_raw, dict):
        soft_profile_raw = {}

    out = dict(defaults)
    out["amp_mode"] = _normalize_amp_mode(raw.get("amp_mode", out["amp_mode"]))
    out["torch_compile_enabled"] = bool(compile_raw.get("enabled", out["torch_compile_enabled"]))
    out["torch_compile_mode"] = str(compile_raw.get("mode", out["torch_compile_mode"]))
    out["torch_compile_dynamic"] = bool(compile_raw.get("dynamic", out["torch_compile_dynamic"]))
    out["grad_norm_log_every_steps"] = max(0, int(raw.get("grad_norm_log_every_steps", out["grad_norm_log_every_steps"])))
    out["enable_soft_profile_sharing"] = bool(soft_profile_raw.get("enabled", out["enable_soft_profile_sharing"]))
    out["soft_profile_equiv_check_batches"] = max(
        1, int(soft_profile_raw.get("equiv_check_batches", out["soft_profile_equiv_check_batches"]))
    )
    out["soft_profile_mae_tolerance"] = float(
        soft_profile_raw.get("mae_tolerance", out["soft_profile_mae_tolerance"])
    )
    out["soft_profile_img_psnr_tolerance"] = float(
        soft_profile_raw.get("img_psnr_tolerance", out["soft_profile_img_psnr_tolerance"])
    )
    return out


def _resolve_falcor_periodic_eval_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = getattr(args, "_config_obj", None)
    if not isinstance(cfg, dict):
        return {"enabled": False}

    training_cfg = cfg.get("training")
    if not isinstance(training_cfg, dict):
        return {"enabled": False}

    raw = training_cfg.get("falcor_periodic_eval")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return {"enabled": False}

    scene = raw.get("scene", None)
    if scene is None or not str(scene).strip():
        raise ValueError("training.falcor_periodic_eval.scene is required when enabled=true")

    profile_raw = raw.get("profile", {})
    if not isinstance(profile_raw, dict):
        raise ValueError("training.falcor_periodic_eval.profile must be a mapping")
    profile = {
        "name": str(profile_raw.get("name", "soft8_t018")).strip() or "soft8_t018",
        "top_k": int(profile_raw.get("top_k", args.top_k)),
        "training_soft_routing": bool(profile_raw.get("training_soft_routing", True)),
        "routing_soft_topk": profile_raw.get("routing_soft_topk", 8),
        "routing_temperature": profile_raw.get("routing_temperature", 0.18),
    }
    if profile["routing_soft_topk"] is not None:
        profile["routing_soft_topk"] = int(profile["routing_soft_topk"])
    if profile["routing_temperature"] is not None:
        profile["routing_temperature"] = float(profile["routing_temperature"])

    return {
        "enabled": True,
        "scene": str(scene),
        "every_n_epochs": int(raw.get("every_n_epochs", 0)),
        "stage_end_full": bool(raw.get("stage_end_full", True)),
        "profile": profile,
        "best_metric": str(raw.get("best_metric", "mean_real_render_hdr_psnr")),
        "maximize": bool(raw.get("maximize", True)),
        "timeout_seconds": int(raw.get("timeout_seconds", 1800)),
        "fail_on_timeout": bool(raw.get("fail_on_timeout", False)),
        "fail_on_error": bool(raw.get("fail_on_error", False)),
        "max_frames": int(raw.get("max_frames", 24)),
        "warmup_frames": int(raw.get("warmup_frames", 8)),
        "benchmark_frames": int(raw.get("benchmark_frames", 24)),
        "save_frame_metrics_every": int(raw.get("save_frame_metrics_every", 8)),
        "width": int(raw.get("width", 1280)),
        "height": int(raw.get("height", 720)),
        "fps": float(raw.get("fps", 30.0)),
        "route": str(raw.get("route", "both")),
        "compute_image_metrics": bool(raw.get("compute_image_metrics", True)),
        "split_runtime": bool(raw.get("split_runtime", True)),
        "async_mode": bool(raw.get("async_mode", True)),
        "falcor_python_path": str(raw.get("falcor_python_path", "")),
        "falcor_python_bin": str(raw.get("falcor_python_bin", "")),
        "worker_script": str(raw.get("worker_script", "")),
        "device": str(raw.get("device", args.device or "cuda")),
    }


def _resolve_val_profiles_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = getattr(args, "_config_obj", None)
    if not isinstance(cfg, dict):
        return {"enabled": False, "best_metric": "mae", "profiles": [], "run_test_compare": False}

    training_cfg = cfg.get("training")
    if not isinstance(training_cfg, dict):
        return {"enabled": False, "best_metric": "mae", "profiles": [], "run_test_compare": False}

    raw = training_cfg.get("val_profiles")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return {"enabled": False, "best_metric": "mae", "profiles": [], "run_test_compare": False}

    raw_profiles = raw.get("profiles", [])
    if not isinstance(raw_profiles, list):
        raise ValueError("training.val_profiles.profiles must be a list")

    profiles: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_profiles):
        if not isinstance(item, dict):
            raise ValueError(f"training.val_profiles.profiles[{idx}] must be a mapping")
        name = str(item.get("name") or f"profile_{idx+1}").strip()
        if not name:
            raise ValueError(f"training.val_profiles.profiles[{idx}] has empty name")
        profile_cfg = {
            "name": name,
            "top_k": int(item.get("top_k", args.top_k)),
            "training_soft_routing": bool(item.get("training_soft_routing", False)),
            "routing_soft_topk": item.get("routing_soft_topk", None),
            "routing_temperature": item.get("routing_temperature", None),
        }
        if profile_cfg["routing_soft_topk"] is not None:
            profile_cfg["routing_soft_topk"] = int(profile_cfg["routing_soft_topk"])
        if profile_cfg["routing_temperature"] is not None:
            profile_cfg["routing_temperature"] = float(profile_cfg["routing_temperature"])
        profiles.append(profile_cfg)

    best_metric = str(raw.get("best_metric", "mae"))
    allowed_metrics = {"mae", "rmse", "charbonnier", "img_mae", "img_rmse", "img_psnr"}
    if best_metric not in allowed_metrics:
        raise ValueError(
            "training.val_profiles.best_metric must be one of: "
            + ", ".join(sorted(allowed_metrics))
        )

    raw_test_compare = training_cfg.get("test_compare_profiles", {})
    if isinstance(raw_test_compare, dict):
        run_test_compare = bool(raw_test_compare.get("enabled", False))
    elif isinstance(raw_test_compare, bool):
        run_test_compare = bool(raw_test_compare)
    else:
        run_test_compare = False

    return {
        "enabled": True,
        "best_metric": best_metric,
        "profiles": profiles,
        "run_test_compare": run_test_compare,
    }


def _normalize_route_profiles(raw_profiles: Any, default_top_k: int) -> List[Dict[str, Any]]:
    if not raw_profiles:
        return []
    if not isinstance(raw_profiles, list):
        raise ValueError("route profiles must be a list")
    profiles: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_profiles):
        if not isinstance(item, dict):
            raise ValueError(f"route profiles[{idx}] must be a mapping")
        name = str(item.get("name") or f"profile_{idx+1}").strip()
        if not name:
            raise ValueError(f"route profiles[{idx}] has empty name")
        profile_cfg = {
            "name": name,
            "top_k": int(item.get("top_k", default_top_k)),
            "training_soft_routing": bool(item.get("training_soft_routing", False)),
            "routing_soft_topk": item.get("routing_soft_topk", None),
            "routing_temperature": item.get("routing_temperature", None),
        }
        if profile_cfg["routing_soft_topk"] is not None:
            profile_cfg["routing_soft_topk"] = int(profile_cfg["routing_soft_topk"])
        if profile_cfg["routing_temperature"] is not None:
            profile_cfg["routing_temperature"] = float(profile_cfg["routing_temperature"])
        profiles.append(profile_cfg)
    return profiles


def _resolve_expert_utilization_audit_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = getattr(args, "_config_obj", None)
    if not isinstance(cfg, dict):
        return {"enabled": False}
    training_cfg = cfg.get("training")
    if not isinstance(training_cfg, dict):
        return {"enabled": False}
    raw = training_cfg.get("expert_utilization_audit")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return {"enabled": False}
    profiles = _normalize_route_profiles(raw.get("profiles", []), int(args.top_k))
    checkpoint = raw.get("checkpoint", None)
    checkpoint_str = str(checkpoint) if checkpoint is not None else None
    return {
        "enabled": True,
        "run_after_training": bool(raw.get("run_after_training", True)),
        "split": str(raw.get("split", "test")),
        "device": str(raw.get("device", "cpu")),
        "max_samples": int(raw.get("max_samples", 8192)),
        "sample_seed": int(raw.get("sample_seed", args.seed)),
        "batch_size": int(raw.get("batch_size", args.batch_size)),
        "num_workers": int(raw.get("num_workers", 0)),
        "checkpoint": checkpoint_str,
        "profiles": profiles,
        "timeout_seconds": int(raw.get("timeout_seconds", 900)),
        "fail_on_timeout": bool(raw.get("fail_on_timeout", False)),
    }


def _resolve_semantic_drift_audit_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = getattr(args, "_config_obj", None)
    if not isinstance(cfg, dict):
        return {"enabled": False}
    training_cfg = cfg.get("training")
    if not isinstance(training_cfg, dict):
        return {"enabled": False}
    raw = training_cfg.get("semantic_drift_audit")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return {"enabled": False}
    profiles = _normalize_route_profiles(raw.get("profiles", []), int(args.top_k))
    pairs_raw = raw.get("pairs", [])
    if pairs_raw is None:
        pairs_raw = []
    if not isinstance(pairs_raw, list):
        raise ValueError("training.semantic_drift_audit.pairs must be a list")
    pairs: List[Dict[str, str]] = []
    for idx, item in enumerate(pairs_raw):
        if not isinstance(item, dict):
            raise ValueError(f"semantic_drift_audit.pairs[{idx}] must be a mapping")
        name = str(item.get("name") or f"pair_{idx+1}")
        ckpt_a = item.get("checkpoint_a")
        ckpt_b = item.get("checkpoint_b")
        if ckpt_a is None or ckpt_b is None:
            raise ValueError(f"semantic_drift_audit.pairs[{idx}] requires checkpoint_a/checkpoint_b")
        pairs.append(
            {
                "name": name,
                "checkpoint_a": str(ckpt_a),
                "checkpoint_b": str(ckpt_b),
            }
        )
    return {
        "enabled": True,
        "run_after_training": bool(raw.get("run_after_training", True)),
        "split": str(raw.get("split", "test")),
        "device": str(raw.get("device", "cpu")),
        "max_samples": int(raw.get("max_samples", 8192)),
        "sample_seed": int(raw.get("sample_seed", args.seed)),
        "batch_size": int(raw.get("batch_size", args.batch_size)),
        "num_workers": int(raw.get("num_workers", 0)),
        "profiles": profiles,
        "pairs": pairs,
        "timeout_seconds": int(raw.get("timeout_seconds", 1200)),
        "fail_on_timeout": bool(raw.get("fail_on_timeout", False)),
    }


def _run_profile_test_compare(
    *,
    trainer,
    model,
    test_loader,
    output_dir: Path,
    profile_cfg: Dict[str, Any],
) -> Dict[str, Any] | None:
    if test_loader is None:
        return None
    if not bool(profile_cfg.get("enabled", False)):
        return None

    profiles = list(profile_cfg.get("profiles", []))
    if not profiles:
        return None

    compare: Dict[str, Any] = {
        "best_metric": str(profile_cfg.get("best_metric", "mae")),
        "results": {},
    }
    original_state = trainer._model_state_dict_clone()

    try:
        for profile in profiles:
            profile_name = str(profile.get("name", "profile"))
            ckpt_path = output_dir / _profile_checkpoint_filename(profile_name)
            if not ckpt_path.exists():
                compare["results"][profile_name] = {
                    "checkpoint": str(ckpt_path),
                    "exists": False,
                    "metrics": None,
                    "profile": profile,
                }
                continue

            ckpt = torch.load(ckpt_path, map_location=trainer.device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state, strict=False)

            metrics = trainer.validate_epoch(
                test_loader,
                top_k=int(profile.get("top_k", trainer.top_k)),
                training_soft_routing=bool(profile.get("training_soft_routing", False)),
                routing_soft_topk=profile.get("routing_soft_topk", None),
                routing_temperature=profile.get("routing_temperature", None),
            )

            compare["results"][profile_name] = {
                "checkpoint": str(ckpt_path),
                "exists": True,
                "metrics": metrics,
                "profile": profile,
                "epoch": int(ckpt.get("epoch", 0)),
            }
    finally:
        model.load_state_dict(original_state, strict=False)

    output_path = output_dir / "val_profile_test_compare.json"
    output_path.write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")
    return compare


def _run_oracle_monitor(
    *,
    checkpoint_path: Path,
    output_json_path: Path,
    args: argparse.Namespace,
    oracle_cfg: Dict[str, Any],
    max_samples: int,
    sample_seed: int,
    timeout_seconds: int,
) -> Dict[str, Any] | None:
    script_path = _ROOT / "3_experiments" / "scripts" / "analysis" / "run_sh_oracle_diagnostics.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--data-root",
        str(args.data_root),
        "--checkpoint",
        str(checkpoint_path),
        "--output",
        str(output_json_path),
        "--split",
        str(oracle_cfg.get("split", "test")),
        "--batch-size",
        str(int(oracle_cfg.get("batch_size", args.batch_size))),
        "--num-workers",
        str(int(oracle_cfg.get("num_workers", 0))),
        "--seed",
        str(int(args.seed)),
        "--top-k",
        str(int(oracle_cfg.get("top_k", args.top_k))),
        "--max-samples",
        str(int(max_samples)),
        "--sample-seed",
        str(int(sample_seed)),
    ]
    if args.device is not None:
        cmd.extend(["--device", str(args.device)])

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    timeout = int(timeout_seconds)
    subprocess.run(cmd, check=True, timeout=timeout if timeout > 0 else None)
    return json.loads(output_json_path.read_text(encoding="utf-8"))


def _run_coeff_suite(
    *,
    checkpoint_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    suite_cfg: Dict[str, Any],
    timeout_seconds: int,
) -> Dict[str, Any] | None:
    script_path = _ROOT / "3_experiments" / "scripts" / "analysis" / "run_coeff_learnability_suite.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--data-root",
        str(args.data_root),
        "--checkpoint",
        str(checkpoint_path),
        "--output-dir",
        str(output_dir),
        "--split",
        str(suite_cfg.get("split", "test")),
        "--device",
        str(suite_cfg.get("device", "cpu")),
        "--batch-size",
        str(int(suite_cfg.get("batch_size", args.batch_size))),
        "--num-workers",
        str(int(suite_cfg.get("num_workers", 0))),
        "--seed",
        str(int(args.seed)),
        "--train-ratio",
        str(float(args.train_ratio)),
        "--val-ratio",
        str(float(args.val_ratio)),
        "--top-k",
        str(int(args.top_k)),
        "--max-samples",
        str(int(suite_cfg.get("max_samples", 8192))),
        "--sample-seed",
        str(int(suite_cfg.get("sample_seed", args.seed))),
        "--probe-train-ratio",
        str(float(suite_cfg.get("probe_train_ratio", 0.8))),
        "--ridge-alpha",
        str(float(suite_cfg.get("ridge_alpha", 1e-4))),
        "--mlp-hidden",
        str(int(suite_cfg.get("mlp_hidden", 128))),
        "--mlp-epochs",
        str(int(suite_cfg.get("mlp_epochs", 30))),
        "--mlp-lr",
        str(float(suite_cfg.get("mlp_lr", 1e-3))),
        "--mlp-batch-size",
        str(int(suite_cfg.get("mlp_batch_size", 256))),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    timeout = int(timeout_seconds)
    subprocess.run(cmd, check=True, timeout=timeout if timeout > 0 else None)

    summary_path = output_dir / "phase1_summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _run_expert_utilization_audit(
    *,
    checkpoint_path: Path,
    output_json_path: Path,
    args: argparse.Namespace,
    audit_cfg: Dict[str, Any],
    profiles: List[Dict[str, Any]],
    timeout_seconds: int,
) -> Dict[str, Any] | None:
    script_path = _ROOT / "3_experiments" / "scripts" / "analysis" / "run_expert_utilization_audit.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--data-root",
        str(args.data_root),
        "--checkpoint",
        str(checkpoint_path),
        "--output",
        str(output_json_path),
        "--split",
        str(audit_cfg.get("split", "test")),
        "--device",
        str(audit_cfg.get("device", "cpu")),
        "--batch-size",
        str(int(audit_cfg.get("batch_size", args.batch_size))),
        "--num-workers",
        str(int(audit_cfg.get("num_workers", 0))),
        "--seed",
        str(int(args.seed)),
        "--train-ratio",
        str(float(args.train_ratio)),
        "--val-ratio",
        str(float(args.val_ratio)),
        "--top-k",
        str(int(args.top_k)),
        "--max-samples",
        str(int(audit_cfg.get("max_samples", 8192))),
        "--sample-seed",
        str(int(audit_cfg.get("sample_seed", args.seed))),
        "--profiles-json",
        json.dumps({"profiles": profiles}, ensure_ascii=False),
    ]
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    timeout = int(timeout_seconds)
    subprocess.run(cmd, check=True, timeout=timeout if timeout > 0 else None)
    if not output_json_path.exists():
        return None
    return json.loads(output_json_path.read_text(encoding="utf-8"))


def _run_semantic_drift_audit(
    *,
    checkpoint_a: Path,
    checkpoint_b: Path,
    output_json_path: Path,
    pair_name: str,
    args: argparse.Namespace,
    audit_cfg: Dict[str, Any],
    profiles: List[Dict[str, Any]],
    timeout_seconds: int,
) -> Dict[str, Any] | None:
    script_path = _ROOT / "3_experiments" / "scripts" / "analysis" / "run_semantic_drift_audit.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--data-root",
        str(args.data_root),
        "--checkpoint-a",
        str(checkpoint_a),
        "--checkpoint-b",
        str(checkpoint_b),
        "--output",
        str(output_json_path),
        "--name",
        str(pair_name),
        "--split",
        str(audit_cfg.get("split", "test")),
        "--device",
        str(audit_cfg.get("device", "cpu")),
        "--batch-size",
        str(int(audit_cfg.get("batch_size", args.batch_size))),
        "--num-workers",
        str(int(audit_cfg.get("num_workers", 0))),
        "--seed",
        str(int(args.seed)),
        "--train-ratio",
        str(float(args.train_ratio)),
        "--val-ratio",
        str(float(args.val_ratio)),
        "--top-k",
        str(int(args.top_k)),
        "--max-samples",
        str(int(audit_cfg.get("max_samples", 8192))),
        "--sample-seed",
        str(int(audit_cfg.get("sample_seed", args.seed))),
        "--profiles-json",
        json.dumps({"profiles": profiles}, ensure_ascii=False),
    ]
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    timeout = int(timeout_seconds)
    subprocess.run(cmd, check=True, timeout=timeout if timeout > 0 else None)
    if not output_json_path.exists():
        return None
    return json.loads(output_json_path.read_text(encoding="utf-8"))


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _update_heartbeat(
    *,
    heartbeat_path: Path,
    history_path: Path,
    enabled: bool,
    save_history: bool,
    run_context: Dict[str, Any],
    state: str,
    last_event: str,
    stage_index: int | None = None,
    stage_name: str | None = None,
    epoch: int | None = None,
    num_epochs: int | None = None,
    metrics: Dict[str, Any] | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    if not enabled:
        return

    payload: Dict[str, Any] = {
        "timestamp": _now_iso(),
        "state": state,
        "last_event": last_event,
        "run_context": run_context,
        "stage_index": stage_index,
        "stage_name": stage_name,
        "epoch": epoch,
        "num_epochs": num_epochs,
    }
    if metrics:
        payload["metrics"] = metrics
    if extra:
        payload["extra"] = extra

    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    heartbeat_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if save_history:
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _oracle_metrics_from_summary(summary: Dict[str, Any]) -> Dict[str, float]:
    baseline = summary.get("baseline", {}) if isinstance(summary, dict) else {}
    oracle = summary.get("oracle_timeweights", {}) if isinstance(summary, dict) else {}
    delta = summary.get("delta", {}) if isinstance(summary, dict) else {}
    base_all = baseline.get("all27", {}) if isinstance(baseline, dict) else {}
    base_non_l0 = baseline.get("non_l0", {}) if isinstance(baseline, dict) else {}
    ora_all = oracle.get("all27", {}) if isinstance(oracle, dict) else {}
    ora_non_l0 = oracle.get("non_l0", {}) if isinstance(oracle, dict) else {}

    return {
        "baseline_all27_psnr": float(base_all.get("sh_psnr", 0.0)),
        "baseline_non_l0_psnr": float(base_non_l0.get("sh_non_l0_psnr", 0.0)),
        "oracle_all27_psnr": float(ora_all.get("sh_psnr", 0.0)),
        "oracle_non_l0_psnr": float(ora_non_l0.get("sh_non_l0_psnr", 0.0)),
        "gap_all27_db": float(delta.get("all27_psnr_gain_db", 0.0)),
        "gap_non_l0_db": float(delta.get("non_l0_psnr_gain_db", 0.0)),
    }


def _extract_falcor_metrics_from_summary(summary: Dict[str, Any]) -> Dict[str, float]:
    image = summary.get("image_metrics", {}) if isinstance(summary, dict) else {}
    if not isinstance(image, dict):
        image = {}
    out: Dict[str, float] = {}
    for key in [
        "mean_real_render_hdr_psnr",
        "mean_real_render_hdr_ssim",
        "mean_hdr_psnr",
        "mean_hdr_ssim",
        "mean_benchmark_psnr",
        "mean_benchmark_ssim",
        "mean_benchmark_linear_psnr",
        "mean_benchmark_linear_ssim",
        "mean_psnr",
        "mean_ssim",
        "mean_linear_psnr",
        "mean_linear_ssim",
    ]:
        if key in image:
            try:
                out[key] = float(image[key])
            except Exception:
                pass
    return out


def _run_falcor_periodic_eval(
    *,
    checkpoint_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    eval_cfg: Dict[str, Any],
) -> Dict[str, Any] | None:
    script_path = _ROOT / "tools" / "benchmark_realtime_pipeline.py"
    profile = eval_cfg.get("profile", {}) if isinstance(eval_cfg, dict) else {}
    cmd = [
        sys.executable,
        str(script_path),
        "--dataset",
        str(args.data_root),
        "--scene",
        str(eval_cfg.get("scene")),
        "--output-dir",
        str(output_dir),
        "--checkpoint",
        str(checkpoint_path),
        "--route",
        str(eval_cfg.get("route", "both")),
        "--max-frames",
        str(int(eval_cfg.get("max_frames", 24))),
        "--warmup-frames",
        str(int(eval_cfg.get("warmup_frames", 8))),
        "--benchmark-frames",
        str(int(eval_cfg.get("benchmark_frames", 24))),
        "--save-frame-metrics-every",
        str(int(eval_cfg.get("save_frame_metrics_every", 8))),
        "--width",
        str(int(eval_cfg.get("width", 1280))),
        "--height",
        str(int(eval_cfg.get("height", 720))),
        "--fps",
        str(float(eval_cfg.get("fps", 30.0))),
        "--device",
        str(eval_cfg.get("device", args.device or "cuda")),
        "--top-k",
        str(int(profile.get("top_k", args.top_k))),
    ]
    if bool(eval_cfg.get("compute_image_metrics", True)):
        cmd.append("--compute-image-metrics")
    if bool(eval_cfg.get("split_runtime", True)):
        cmd.append("--split-runtime")
    else:
        cmd.append("--no-split-runtime")

    if bool(profile.get("training_soft_routing", False)):
        cmd.append("--training-soft-routing")
    else:
        cmd.append("--no-training-soft-routing")
    if profile.get("routing_soft_topk", None) is not None:
        cmd.extend(["--routing-soft-topk", str(int(profile.get("routing_soft_topk")))])
    if profile.get("routing_temperature", None) is not None:
        cmd.extend(["--routing-temperature", str(float(profile.get("routing_temperature")))])

    falcor_python_path = str(eval_cfg.get("falcor_python_path", "")).strip()
    falcor_python_bin = str(eval_cfg.get("falcor_python_bin", "")).strip()
    worker_script = str(eval_cfg.get("worker_script", "")).strip()
    if falcor_python_path:
        cmd.extend(["--falcor-python-path", falcor_python_path])
    if falcor_python_bin:
        cmd.extend(["--falcor-python-bin", falcor_python_bin])
    if worker_script:
        cmd.extend(["--worker-script", worker_script])

    output_dir.mkdir(parents=True, exist_ok=True)
    timeout_seconds = int(eval_cfg.get("timeout_seconds", 1800))
    subprocess.run(cmd, check=True, timeout=timeout_seconds if timeout_seconds > 0 else None)
    summary_path = output_dir / "benchmark_summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _phase0_metrics_from_payload(payload: Dict[str, Any]) -> Dict[str, float]:
    train_m = payload.get("train_metrics", {}) if isinstance(payload, dict) else {}
    val_m = payload.get("val_metrics", {}) if isinstance(payload, dict) else {}
    val_profiles_m = payload.get("val_profiles_metrics", {}) if isinstance(payload, dict) else {}

    grad_coeff = float(train_m.get("grad_post_clip/coeff", 0.0))
    grad_basis = float(train_m.get("grad_post_clip/basis", 0.0))
    grad_total = float(train_m.get("grad_post_clip/total", 0.0))
    coeff_over_basis = grad_coeff / max(1e-12, grad_basis)
    coeff_over_total = grad_coeff / max(1e-12, grad_total)

    metrics: Dict[str, float] = {
        "train_total": float(train_m.get("total", 0.0)),
        "train_recon": float(train_m.get("recon", 0.0)),
        "val_mae": float(val_m.get("mae", 0.0)) if val_m else 0.0,
        "val_img_psnr": float(val_m.get("img_psnr", 0.0)) if val_m else 0.0,
        "grad_post_total": grad_total,
        "grad_post_routing": float(train_m.get("grad_post_clip/routing", 0.0)),
        "grad_post_basis": grad_basis,
        "grad_post_coeff": grad_coeff,
        "grad_post_encoder": float(train_m.get("grad_post_clip/encoder", 0.0)),
        "grad_post_film": float(train_m.get("grad_post_clip/film", 0.0)),
        "grad_ratio_coeff_over_basis": float(coeff_over_basis),
        "grad_ratio_coeff_over_total": float(coeff_over_total),
    }

    if isinstance(val_profiles_m, dict):
        for profile_name, profile_metrics in val_profiles_m.items():
            if not isinstance(profile_metrics, dict):
                continue
            safe_name = _sanitize_stage_name(profile_name)
            if "mae" in profile_metrics:
                metrics[f"val_profile_{safe_name}_mae"] = float(profile_metrics.get("mae", 0.0))
            if "rmse" in profile_metrics:
                metrics[f"val_profile_{safe_name}_rmse"] = float(profile_metrics.get("rmse", 0.0))
            if "img_psnr" in profile_metrics:
                metrics[f"val_profile_{safe_name}_img_psnr"] = float(profile_metrics.get("img_psnr", 0.0))
    return metrics


def _coeff_suite_metrics_from_summary(summary: Dict[str, Any]) -> Dict[str, float]:
    phase1 = summary.get("phase1", {}) if isinstance(summary, dict) else {}
    coeff_alignment = phase1.get("coeff_alignment", {}) if isinstance(phase1, dict) else {}
    pred_vs_oracle = coeff_alignment.get("pred_vs_oracle", {}) if isinstance(coeff_alignment, dict) else {}

    sh = phase1.get("sh_reconstruction", {}) if isinstance(phase1, dict) else {}
    pred = sh.get("pred", {}) if isinstance(sh, dict) else {}
    structured = sh.get("structured_probe", {}) if isinstance(sh, dict) else {}
    oracle = sh.get("oracle", {}) if isinstance(sh, dict) else {}
    gains = sh.get("gains_over_pred_db", {}) if isinstance(sh, dict) else {}

    pred_all = pred.get("all27", {}) if isinstance(pred, dict) else {}
    structured_all = structured.get("all27", {}) if isinstance(structured, dict) else {}
    oracle_all = oracle.get("all27", {}) if isinstance(oracle, dict) else {}

    return {
        "coeff_pred_oracle_mse": float(pred_vs_oracle.get("mse", 0.0)),
        "coeff_pred_oracle_cos": float(pred_vs_oracle.get("cosine_mean", 0.0)),
        "coeff_pred_oracle_r2": float(pred_vs_oracle.get("r2", 0.0)),
        "coeff_pred_sh_psnr": float(pred_all.get("sh_psnr", 0.0)),
        "coeff_structured_sh_psnr": float(structured_all.get("sh_psnr", 0.0)),
        "coeff_oracle_sh_psnr": float(oracle_all.get("sh_psnr", 0.0)),
        "coeff_gap_structured_db": float(gains.get("structured_all27", 0.0)),
        "coeff_gap_all27_db": float(gains.get("oracle_all27", 0.0)),
    }


def _expert_util_metrics_from_summary(summary: Dict[str, Any]) -> Dict[str, float]:
    profiles = summary.get("profiles", {}) if isinstance(summary, dict) else {}
    if not isinstance(profiles, dict) or not profiles:
        return {}
    out: Dict[str, float] = {}
    for profile_name, payload in profiles.items():
        if not isinstance(payload, dict):
            continue
        p_summary = payload.get("summary", {})
        if not isinstance(p_summary, dict):
            continue
        safe = _sanitize_stage_name(profile_name)
        dist_mass = p_summary.get("distribution", {}).get("mass_sum", {})
        ratio = p_summary.get("nonzero_ratio", {})
        out[f"expert_{safe}_mass_entropy_norm"] = float(dist_mass.get("entropy_norm", 0.0))
        out[f"expert_{safe}_mass_effective_k"] = float(dist_mass.get("effective_k", 0.0))
        out[f"expert_{safe}_mass_hhi"] = float(dist_mass.get("hhi", 0.0))
        out[f"expert_{safe}_nonzero_sample_ratio"] = float(ratio.get("sample_count", 0.0))
        out[f"expert_{safe}_nonzero_probe_ratio"] = float(ratio.get("probe_coverage", 0.0))
    return out


def _semantic_drift_metrics_from_summary(summary: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(summary, dict):
        return out
    param = summary.get("parameter_drift", {})
    if isinstance(param, dict):
        out["drift_global_rel_l2_to_a"] = float(param.get("global_rel_l2_to_a", 0.0))
        groups = param.get("per_group", {})
        if isinstance(groups, dict):
            for gname, gpayload in groups.items():
                if not isinstance(gpayload, dict):
                    continue
                safe = _sanitize_stage_name(gname)
                out[f"drift_group_{safe}_rel_l2_to_a"] = float(gpayload.get("rel_l2_to_a", 0.0))
    profiles = summary.get("profiles", {})
    if isinstance(profiles, dict):
        for pname, ppayload in profiles.items():
            if not isinstance(ppayload, dict):
                continue
            metrics = ppayload.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            safe = _sanitize_stage_name(pname)
            out[f"drift_{safe}_b_minus_a_psnr_vs_gt"] = float(metrics.get("b_minus_a_psnr_vs_gt", 0.0))
            out[f"drift_{safe}_compensation_index"] = float(metrics.get("compensation_index", 0.0))
    return out


def _get_train_arg_defaults() -> argparse.Namespace:
    global _TRAIN_ARG_DEFAULTS
    if _TRAIN_ARG_DEFAULTS is None:
        _TRAIN_ARG_DEFAULTS = build_arg_parser().parse_args([])
    return _TRAIN_ARG_DEFAULTS


def normalize_train_args(args: argparse.Namespace) -> argparse.Namespace:
    defaults = _get_train_arg_defaults()
    return normalize_namespace(args, defaults)


def run_training(args: argparse.Namespace) -> None:
    args = normalize_train_args(args)
    _lazy_imports()
    device = _device_from_arg(args.device)
    warmup_epochs = int(getattr(args, "warmup_epochs", 0))
    resume_requested = bool(getattr(args, "resume", None))
    if resume_requested and bool(getattr(args, "load_model", None)):
        raise ValueError("--resume and --load-model are mutually exclusive; use only --resume for strict continuation")
    output_dir = Path(args.output_dir) if args.output_dir else None

    if output_dir is None:
        if resume_requested:
            raw_resume = Path(str(args.resume))
            if raw_resume.parent.name.startswith("stage") and raw_resume.parent.parent.exists():
                output_dir = raw_resume.parent.parent
            else:
                output_dir = raw_resume.parent
        else:
            run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
            output_dir = _ROOT / "3_experiments" / "results" / args.variant / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_checkpoint_path: Path | None = None
    resume_checkpoint: Dict[str, Any] | None = None
    resume_stage_index: int | None = None
    resume_stage_epoch: int = 0
    if resume_requested:
        resume_checkpoint_path = _resolve_resume_checkpoint_path(str(args.resume), output_dir)
        resume_checkpoint = torch.load(str(resume_checkpoint_path), map_location=device, weights_only=False)
        if not isinstance(resume_checkpoint, dict):
            raise ValueError(f"resume checkpoint must be a mapping, got {type(resume_checkpoint)}")
        resume_stage_epoch = int(resume_checkpoint.get("epoch", 0))
        resume_meta = resume_checkpoint.get("meta", {})
        if isinstance(resume_meta, dict):
            training_meta = resume_meta.get("training", {})
            staged_meta = resume_meta.get("staged_training", {})
            if isinstance(training_meta, dict) and training_meta.get("stage_index") is not None:
                resume_stage_index = int(training_meta.get("stage_index"))
            if resume_stage_index is None and isinstance(staged_meta, dict) and staged_meta.get("current_stage") is not None:
                resume_stage_index = int(staged_meta.get("current_stage"))
        if resume_stage_index is None and resume_checkpoint_path is not None:
            resume_stage_index = _infer_stage_index_from_checkpoint_path(resume_checkpoint_path)

    heartbeat_enabled = bool(getattr(args, "heartbeat_enabled", True))
    heartbeat_history_enabled = bool(getattr(args, "heartbeat_history", True))

    heartbeat_path = output_dir / "runtime" / "heartbeat.json"
    heartbeat_history_path = output_dir / "runtime" / "heartbeat.history.jsonl"
    run_context = {
        "pid": int(os.getpid()),
        "variant": str(args.variant),
        "output_dir": str(output_dir),
        "config": str(getattr(args, "config", "")) if getattr(args, "config", None) else None,
        "started_at": _now_iso(),
        "resume": bool(resume_requested),
        "resume_checkpoint": str(resume_checkpoint_path) if resume_checkpoint_path is not None else None,
        "resume_stage_index": int(resume_stage_index) if resume_stage_index is not None else None,
        "resume_stage_epoch": int(resume_stage_epoch),
    }
    _update_heartbeat(
        heartbeat_path=heartbeat_path,
        history_path=heartbeat_history_path,
        enabled=heartbeat_enabled,
        save_history=heartbeat_history_enabled,
        run_context=run_context,
        state="running",
        last_event="init",
        stage_index=0,
        stage_name="init",
        epoch=0,
        num_epochs=int(args.epochs) if args.epochs is not None else None,
    )

    heartbeat_done = {"value": False}

    def _heartbeat_exit_guard() -> None:
        if heartbeat_done["value"]:
            return
        _update_heartbeat(
            heartbeat_path=heartbeat_path,
            history_path=heartbeat_history_path,
            enabled=heartbeat_enabled,
            save_history=heartbeat_history_enabled,
            run_context=run_context,
            state="error",
            last_event="process_exit_before_finish",
            stage_index=None,
            stage_name=None,
            epoch=None,
            num_epochs=None,
        )

    atexit.register(_heartbeat_exit_guard)

    if args.show_progress:
        print("[init] loading dataset and building model...")
    model, adapter, loaders, helpers = build_variant(args.variant, args, device)
    train_loader, val_loader, test_loader = loaders

    if not args.no_init and not resume_requested:
        if args.show_progress:
            print("[init] running K-Means/SVD initialization...")
        helpers["init_fn"](model, train_loader)
        model.to(device)
        if args.show_progress:
            print("[init] initialization done.")
    elif resume_requested and args.show_progress:
        print("[resume] skip random initialization because strict resume is enabled")

    # Optional SH scaling
    scaler_meta = None
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
        scaler_meta = {
            "path": str(scaler_path) if scaler_path is not None else None,
            "l0_mean": [float(x) for x in scaler.l0_mean],
            "l0_std": [float(x) for x in scaler.l0_std],
            "ho_rms": [float(x) for x in scaler.ho_rms],
            "eps": float(getattr(scaler, "eps", 1e-6)),
        }
        l0_mean = [float(x) for x in scaler.l0_mean]
        l0_std = [float(x) for x in scaler.l0_std]
        ho_rms = [float(x) for x in scaler.ho_rms]
        print(f"SH scaler stats: L0 mean={l0_mean}, L0 std={l0_std}, HO rms={ho_rms}")

    if resume_requested:
        assert resume_checkpoint is not None
        raw_state = resume_checkpoint.get("model_state_dict_raw", None)
        if raw_state is not None:
            state = raw_state
        else:
            state = resume_checkpoint.get("model_state_dict", resume_checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if args.show_progress:
            print(
                f"[resume] loaded checkpoint={resume_checkpoint_path} "
                f"epoch={int(resume_checkpoint.get('epoch', 0))} "
                f"stage_index={resume_stage_index}"
            )
        if missing:
            print(f"Warning: Missing keys when resuming model: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys when resuming model: {unexpected}")
    elif args.load_model:
        ckpt = torch.load(args.load_model, map_location=device, weights_only=False)
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

    checkpoint_meta = {
        "variant": args.variant,
        "data_root": str(Path(args.data_root)),
        "manifest": args.manifest,
        "seed": int(args.seed),
        "split": {
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
        },
        "model": {
            "num_gaussians": int(args.num_gaussians),
            "rank": int(args.rank),
            "sh_dim": 27,
            "top_k": int(args.top_k),
            "light_dim": int(args.light_dim),
            "embed_dim": int(args.embed_dim),
            "intensity_dim": int(args.intensity_dim),
            "intensity_offset": int(args.intensity_offset),
            "enable_film": bool(not args.disable_film),
            "light_encoder_mode": str(getattr(model, "light_encoder_mode", getattr(args, "light_encoder_mode", "normal"))),
            "bypass_feature_pairs": [
                [int(p[0]), int(p[1])] for p in list(getattr(model, "bypass_feature_pairs", []))
            ],
            "bypass_feature_norm_mean": [
                float(x) for x in list(getattr(model, "bypass_feature_norm_mean", []))
            ],
            "bypass_feature_norm_std": [
                float(x) for x in list(getattr(model, "bypass_feature_norm_std", []))
            ],
        },
        "training": {
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "dataloader_timeout_seconds": int(getattr(args, "dataloader_timeout_seconds", 0)),
            "dataloader_prefetch_factor": getattr(args, "dataloader_prefetch_factor", None),
            "dataloader_persistent_workers": getattr(args, "dataloader_persistent_workers", None),
            "dataloader_multiprocessing_context": str(getattr(args, "dataloader_multiprocessing_context", "none")),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "lr_scheduler": str(args.lr_scheduler),
            "lr_min": float(args.lr_min),
            "warmup_epochs": warmup_epochs,
            "recon_loss": args.recon_loss,
            "lambda_temporal": float(args.lambda_temporal),
            "lambda_linearity": float(args.lambda_linearity),
            "linearity_aug_pairs": int(args.linearity_aug_pairs),
            "lambda_spatial": float(args.lambda_spatial),
            "spatial_k": int(args.spatial_k),
            "lambda_image": float(args.lambda_image),
            "image_loss_warmup_epochs": int(getattr(args, "image_loss_warmup_epochs", 0)),
            "lambda_routing_balance": float(args.lambda_routing_balance),
            "routing_soft_train": bool(args.routing_soft_train),
            "routing_soft_topk": int(args.routing_soft_topk),
            "routing_temp_start": float(args.routing_temp_start),
            "routing_temp_end": float(args.routing_temp_end),
            "routing_temp_anneal_epochs": int(args.routing_temp_anneal_epochs),
            "image_loss_type": args.image_loss_type,
            "image_samples": int(args.image_samples),
            "image_sample_seed": int(args.image_sample_seed),
            "image_loss_space": args.image_loss_space,
            "enable_weighted_sh_loss": bool(args.enable_weighted_sh_loss),
            "sh_loss_weights": list(args.sh_loss_weights) if args.sh_loss_weights is not None else None,
            "sh_weight_mode": args.sh_weight_mode,
            "amp_mode": str(getattr(args, "amp_mode", "off")),
            "torch_compile": bool(getattr(args, "torch_compile", False)),
            "torch_compile_mode": str(getattr(args, "torch_compile_mode", "reduce-overhead")),
            "torch_compile_dynamic": bool(getattr(args, "torch_compile_dynamic", False)),
            "grad_norm_log_every_steps": int(getattr(args, "grad_norm_log_every_steps", 1)),
            "max_train_batches": int(getattr(args, "max_train_batches", 0)),
            "max_val_batches": int(getattr(args, "max_val_batches", 0)),
            "heartbeat_enabled": bool(heartbeat_enabled),
            "heartbeat_history": bool(heartbeat_history_enabled),
            "resume": bool(resume_requested),
            "resume_checkpoint": str(resume_checkpoint_path) if resume_checkpoint_path is not None else None,
            "resume_save_every": int(getattr(args, "resume_save_every", 1)),
            "resume_checkpoint_name": str(getattr(args, "resume_checkpoint_name", "resume_latest.pt")),
        },
        "sh_scaler": scaler_meta,
    }

    ema_cfg = _resolve_ema_config(args)
    oracle_monitor_cfg = _resolve_oracle_monitor_config(args)
    coeff_suite_cfg = _resolve_coeff_suite_config(args)
    global_best_cfg = _resolve_global_best_config(args)
    val_profiles_cfg = _resolve_val_profiles_config(args)
    proxy_image_loss_cfg = _resolve_proxy_image_loss_config(args)
    routing_balance_anneal_cfg = _resolve_routing_balance_anneal_config(args)
    performance_cfg = _resolve_performance_config(args)
    falcor_periodic_eval_cfg = _resolve_falcor_periodic_eval_config(args)
    expert_util_cfg = _resolve_expert_utilization_audit_config(args)
    semantic_drift_cfg = _resolve_semantic_drift_audit_config(args)
    global_group_lrs = _resolve_global_group_lrs(args)
    staged_specs = _resolve_staged_specs(args)
    total_training_epochs_global = (
        int(sum(int(spec["epochs"]) for spec in staged_specs))
        if staged_specs
        else int(args.epochs)
    )
    routing_balance_decay_end_epoch = int(
        np.ceil(float(total_training_epochs_global) * float(routing_balance_anneal_cfg.get("decay_end_ratio", 0.6)))
    )
    if routing_balance_decay_end_epoch <= 0:
        routing_balance_anneal_epochs_global = 0
    else:
        routing_balance_anneal_epochs_global = max(0, int(routing_balance_decay_end_epoch - 1))

    if bool(proxy_image_loss_cfg.get("enabled", False)):
        args.lambda_image = float(proxy_image_loss_cfg.get("lambda", args.lambda_image))
        args.image_loss_warmup_epochs = int(proxy_image_loss_cfg.get("warmup_epochs", getattr(args, "image_loss_warmup_epochs", 0)))
        if bool(proxy_image_loss_cfg.get("enable_val_image_metrics", True)):
            args.val_image_metrics = True
    args.proxy_image_mix_mode = str(proxy_image_loss_cfg.get("mix_mode", "linear_log_mix"))
    args.proxy_image_log_mix_weight = float(proxy_image_loss_cfg.get("log_mix_weight", 0.3))

    args.amp_mode = str(performance_cfg.get("amp_mode", getattr(args, "amp_mode", "off")))
    args.torch_compile = bool(performance_cfg.get("torch_compile_enabled", getattr(args, "torch_compile", False)))
    args.torch_compile_mode = str(performance_cfg.get("torch_compile_mode", getattr(args, "torch_compile_mode", "reduce-overhead")))
    args.torch_compile_dynamic = bool(
        performance_cfg.get("torch_compile_dynamic", getattr(args, "torch_compile_dynamic", False))
    )
    args.grad_norm_log_every_steps = int(
        performance_cfg.get("grad_norm_log_every_steps", getattr(args, "grad_norm_log_every_steps", 1))
    )
    args.enable_soft_profile_sharing = bool(
        performance_cfg.get("enable_soft_profile_sharing", getattr(args, "enable_soft_profile_sharing", False))
    )
    args.soft_profile_equiv_check_batches = int(
        performance_cfg.get(
            "soft_profile_equiv_check_batches",
            getattr(args, "soft_profile_equiv_check_batches", 2),
        )
    )
    args.soft_profile_mae_tolerance = float(
        performance_cfg.get("soft_profile_mae_tolerance", getattr(args, "soft_profile_mae_tolerance", 1e-6))
    )
    args.soft_profile_img_psnr_tolerance = float(
        performance_cfg.get(
            "soft_profile_img_psnr_tolerance",
            getattr(args, "soft_profile_img_psnr_tolerance", 5e-4),
        )
    )

    checkpoint_meta["training"].update(
        {
            "lambda_image": float(args.lambda_image),
            "image_loss_warmup_epochs": int(getattr(args, "image_loss_warmup_epochs", 0)),
            "proxy_image_mix_mode": str(args.proxy_image_mix_mode),
            "proxy_image_log_mix_weight": float(args.proxy_image_log_mix_weight),
            "amp_mode": str(args.amp_mode),
            "torch_compile": bool(args.torch_compile),
            "torch_compile_mode": str(args.torch_compile_mode),
            "torch_compile_dynamic": bool(args.torch_compile_dynamic),
            "grad_norm_log_every_steps": int(args.grad_norm_log_every_steps),
            "enable_soft_profile_sharing": bool(args.enable_soft_profile_sharing),
            "soft_profile_equiv_check_batches": int(args.soft_profile_equiv_check_batches),
            "soft_profile_mae_tolerance": float(args.soft_profile_mae_tolerance),
            "soft_profile_img_psnr_tolerance": float(args.soft_profile_img_psnr_tolerance),
            "routing_balance_anneal": {
                **routing_balance_anneal_cfg,
                "total_training_epochs": int(total_training_epochs_global),
                "decay_end_epoch": int(routing_balance_decay_end_epoch),
                "anneal_epochs_global": int(routing_balance_anneal_epochs_global),
            },
        }
    )

    checkpoint_meta["training"]["ema"] = ema_cfg
    checkpoint_meta["training"]["oracle_monitor"] = oracle_monitor_cfg
    checkpoint_meta["training"]["coeff_suite"] = coeff_suite_cfg
    checkpoint_meta["training"]["val_profiles"] = val_profiles_cfg
    checkpoint_meta["training"]["proxy_image_loss"] = proxy_image_loss_cfg
    checkpoint_meta["training"]["routing_balance_anneal"] = {
        **routing_balance_anneal_cfg,
        "total_training_epochs": int(total_training_epochs_global),
        "decay_end_epoch": int(routing_balance_decay_end_epoch),
        "anneal_epochs_global": int(routing_balance_anneal_epochs_global),
    }
    checkpoint_meta["training"]["performance"] = performance_cfg
    checkpoint_meta["training"]["falcor_periodic_eval"] = falcor_periodic_eval_cfg
    checkpoint_meta["training"]["expert_utilization_audit"] = expert_util_cfg
    checkpoint_meta["training"]["semantic_drift_audit"] = semantic_drift_cfg
    checkpoint_meta["training"]["group_lrs"] = global_group_lrs
    checkpoint_meta["training"]["global_best"] = global_best_cfg

    if args.show_progress and ema_cfg.get("enabled", False):
        print(
            "[ema] enabled="
            f"{ema_cfg['enabled']} decay={ema_cfg['decay']} "
            f"eval_on_ema={ema_cfg['eval_on_ema']} save_best_with_ema={ema_cfg['save_best_with_ema']} "
            f"raw_eval_every={ema_cfg.get('raw_eval_every_epochs', 1)}"
        )
    if args.show_progress and oracle_monitor_cfg.get("enabled", False):
        print(
            "[oracle-monitor] enabled split="
            f"{oracle_monitor_cfg['split']} period_epochs={oracle_monitor_cfg['period_epochs']} "
            f"stage_end_full={oracle_monitor_cfg['stage_end_full']} "
            f"timeout={oracle_monitor_cfg['timeout_seconds']}s fail_on_timeout={oracle_monitor_cfg['fail_on_timeout']}"
        )
    if args.show_progress and coeff_suite_cfg.get("enabled", False):
        configured_suite_ckpts = list(coeff_suite_cfg.get("checkpoints", []))
        print(
            "[coeff-suite] enabled run_after_training="
            f"{coeff_suite_cfg.get('run_after_training', True)} split={coeff_suite_cfg.get('split', 'test')} "
            f"device={coeff_suite_cfg.get('device', 'cpu')} max_samples={coeff_suite_cfg.get('max_samples', 8192)} "
            f"timeout={coeff_suite_cfg.get('timeout_seconds', 1800)}s "
            f"checkpoints={configured_suite_ckpts if configured_suite_ckpts else 'auto'}"
        )
    if args.show_progress and bool(global_best_cfg.get("enabled", True)):
        print(
            "[global-best] "
            f"track_falcor={bool(global_best_cfg.get('track_falcor', True))} "
            f"track_soft_profile={bool(global_best_cfg.get('track_soft_profile', True))} "
            f"soft_profile={str(global_best_cfg.get('soft_profile_name', 'soft8_t018'))} "
            f"soft_metric={str(global_best_cfg.get('soft_metric', 'img_psnr'))}"
        )
    if args.show_progress and expert_util_cfg.get("enabled", False):
        print(
            "[expert-util-audit] enabled run_after_training="
            f"{expert_util_cfg.get('run_after_training', True)} split={expert_util_cfg.get('split', 'test')} "
            f"device={expert_util_cfg.get('device', 'cpu')} max_samples={expert_util_cfg.get('max_samples', 8192)} "
            f"timeout={expert_util_cfg.get('timeout_seconds', 900)}s"
        )
    if args.show_progress and semantic_drift_cfg.get("enabled", False):
        print(
            "[semantic-drift-audit] enabled run_after_training="
            f"{semantic_drift_cfg.get('run_after_training', True)} split={semantic_drift_cfg.get('split', 'test')} "
            f"device={semantic_drift_cfg.get('device', 'cpu')} max_samples={semantic_drift_cfg.get('max_samples', 8192)} "
            f"timeout={semantic_drift_cfg.get('timeout_seconds', 1200)}s"
        )
    if args.show_progress and val_profiles_cfg.get("enabled", False):
        profile_names = [str(p.get("name", "profile")) for p in val_profiles_cfg.get("profiles", [])]
        print(
            "[val-profiles] enabled best_metric="
            f"{val_profiles_cfg.get('best_metric', 'mae')} "
            f"profiles={profile_names} run_test_compare={val_profiles_cfg.get('run_test_compare', False)}"
        )
    if args.show_progress:
        print(
            "[proxy-image-loss] "
            f"enabled={bool(proxy_image_loss_cfg.get('enabled', False))} "
            f"lambda={float(args.lambda_image):.4g} "
            f"warmup_epochs={int(proxy_image_loss_cfg.get('warmup_epochs', 0))} "
            f"mix_mode={str(args.proxy_image_mix_mode)} "
            f"log_mix_weight={float(args.proxy_image_log_mix_weight):.3f} "
            f"val_image_metrics={bool(args.val_image_metrics)}"
        )
    if args.show_progress and bool(falcor_periodic_eval_cfg.get("enabled", False)):
        profile = falcor_periodic_eval_cfg.get("profile", {})
        print(
            "[falcor-periodic] "
            f"every_n={int(falcor_periodic_eval_cfg.get('every_n_epochs', 0))} "
            f"stage_end_full={bool(falcor_periodic_eval_cfg.get('stage_end_full', True))} "
            f"metric={str(falcor_periodic_eval_cfg.get('best_metric', 'mean_real_render_hdr_psnr'))} "
            f"profile={profile}"
        )
    if args.show_progress and (args.lambda_routing_balance > 0.0 or args.routing_soft_train):
        print(
            "[routing] "
            f"lambda_balance={args.lambda_routing_balance} "
            f"soft_train={args.routing_soft_train} soft_topk={args.routing_soft_topk} "
            f"temp={args.routing_temp_start}->{args.routing_temp_end} "
            f"anneal_epochs={args.routing_temp_anneal_epochs} "
            f"balance_anneal={bool(routing_balance_anneal_cfg.get('enabled', False))} "
            f"balance_start={float(routing_balance_anneal_cfg.get('start', 0.0)):.4g} "
            f"balance_end={float(routing_balance_anneal_cfg.get('end', 0.0)):.4g} "
            f"balance_decay_end_ratio={float(routing_balance_anneal_cfg.get('decay_end_ratio', 0.0)):.3f} "
            f"balance_decay_end_epoch={int(routing_balance_decay_end_epoch)}"
        )
    if args.show_progress and resume_requested:
        print(
            "[resume] strict resume enabled "
            f"checkpoint={resume_checkpoint_path} "
            f"epoch={resume_stage_epoch} stage_index={resume_stage_index} "
            f"save_every={int(getattr(args, 'resume_save_every', 1))} "
            f"snapshot={str(getattr(args, 'resume_checkpoint_name', 'resume_latest.pt'))}"
        )

    if bool(args.torch_compile):
        compile_ok = False
        try:
            model = torch.compile(
                model,
                mode=str(args.torch_compile_mode),
                dynamic=bool(args.torch_compile_dynamic),
            )
            compile_ok = True
        except Exception as compile_error:
            print(f"Warning: torch.compile failed, fallback to eager: {compile_error}")
            model = model.to(device)
        if args.show_progress:
            print(
                "[performance] "
                f"amp_mode={args.amp_mode} "
                f"torch_compile={compile_ok} mode={args.torch_compile_mode} dynamic={bool(args.torch_compile_dynamic)}"
            )
    elif args.show_progress:
        print(
            "[performance] "
            f"amp_mode={args.amp_mode} torch_compile=False grad_norm_every={args.grad_norm_log_every_steps}"
        )

    def _build_trainer(
        *,
        lr: float,
        weight_decay: float,
        lr_scheduler: str,
        lr_min: float,
        warmup_epochs_local: int,
        group_lrs: Dict[str, float],
        routing_balance_epoch_offset: int,
    ):
        return GaussianPhysicsTrainer(
            model=model,
            device=device,
            adapter=adapter,
            lr=lr,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler,
            lr_min=lr_min,
            warmup_epochs=warmup_epochs_local,
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
            image_loss_weight=args.lambda_image,
            image_loss_warmup_epochs=int(proxy_image_loss_cfg.get("warmup_epochs", 0)),
            proxy_image_loss_mix_mode=str(args.proxy_image_mix_mode),
            proxy_image_log_mix_weight=float(args.proxy_image_log_mix_weight),
            lambda_routing_balance=args.lambda_routing_balance,
            routing_balance_anneal_enabled=bool(routing_balance_anneal_cfg.get("enabled", False)),
            routing_balance_anneal_start=float(routing_balance_anneal_cfg.get("start", args.lambda_routing_balance)),
            routing_balance_anneal_end=float(routing_balance_anneal_cfg.get("end", 0.0)),
            routing_balance_anneal_epochs=int(routing_balance_anneal_epochs_global),
            routing_balance_anneal_epoch_offset=int(routing_balance_epoch_offset),
            routing_soft_train=args.routing_soft_train,
            routing_soft_topk=args.routing_soft_topk,
            routing_temp_start=args.routing_temp_start,
            routing_temp_end=args.routing_temp_end,
            routing_temp_anneal_epochs=args.routing_temp_anneal_epochs,
            image_loss_type=args.image_loss_type,
            image_samples=args.image_samples,
            image_sample_seed=args.image_sample_seed,
            image_loss_space=args.image_loss_space,
            enable_weighted_sh_loss=args.enable_weighted_sh_loss,
            sh_loss_weights=args.sh_loss_weights,
            sh_weight_mode=args.sh_weight_mode,
            compute_img_metrics_in_val=args.val_image_metrics,
            compute_superposition_in_val=args.val_superposition,
            param_group_lrs=group_lrs,
            ema_enabled=bool(ema_cfg["enabled"]),
            ema_decay=float(ema_cfg["decay"]),
            ema_eval=bool(ema_cfg["eval_on_ema"]),
            ema_save_best=bool(ema_cfg["save_best_with_ema"]),
            amp_mode=str(args.amp_mode),
            grad_norm_log_every_steps=int(getattr(args, "grad_norm_log_every_steps", 1)),
            raw_eval_every_epochs=int(ema_cfg.get("raw_eval_every_epochs", 1)),
            max_train_batches=int(getattr(args, "max_train_batches", 0)),
            max_val_batches=int(getattr(args, "max_val_batches", 0)),
            enable_soft_profile_sharing=bool(getattr(args, "enable_soft_profile_sharing", False)),
            soft_profile_equiv_check_batches=int(getattr(args, "soft_profile_equiv_check_batches", 2)),
            soft_profile_mae_tolerance=float(getattr(args, "soft_profile_mae_tolerance", 1e-6)),
            soft_profile_img_psnr_tolerance=float(getattr(args, "soft_profile_img_psnr_tolerance", 5e-4)),
            rerun_logger=rerun_logger,
            show_progress=args.show_progress,
        )
    history = _empty_history()
    trainer = None
    test_metrics = None
    phase0_metrics_path = output_dir / "runtime" / "phase0_metrics.jsonl"
    falcor_periodic_history_path = output_dir / "runtime" / "falcor_periodic_eval.jsonl"
    resume_payload_for_fit: Dict[str, Any] | None = resume_checkpoint if resume_requested else None
    resume_consumed = False
    falcor_best_metric_name = str(falcor_periodic_eval_cfg.get("best_metric", "mean_real_render_hdr_psnr"))
    falcor_best_maximize = bool(falcor_periodic_eval_cfg.get("maximize", True))
    falcor_profile_name = str(
        (falcor_periodic_eval_cfg.get("profile", {}) if isinstance(falcor_periodic_eval_cfg, dict) else {}).get("name", "soft8_t018")
    )
    falcor_best_value = float("-inf") if falcor_best_maximize else float("inf")
    falcor_best_ckpt_path = output_dir / f"best_model_val_{_sanitize_stage_name(falcor_profile_name)}_falcor.pt"
    falcor_best_summary_path = output_dir / "falcor_best_summary.json"
    global_best_enabled = bool(global_best_cfg.get("enabled", True))
    global_best_soft_profile_name = str(global_best_cfg.get("soft_profile_name", "soft8_t018"))
    global_best_soft_metric = str(global_best_cfg.get("soft_metric", "img_psnr"))
    global_best_soft_maximize = bool(global_best_cfg.get("soft_metric_maximize", True))
    global_best_track_falcor = bool(global_best_enabled and global_best_cfg.get("track_falcor", True))
    global_best_track_soft = bool(global_best_enabled and global_best_cfg.get("track_soft_profile", True))
    global_best_soft_ckpt_path = output_dir / f"global_best_val_{_sanitize_stage_name(global_best_soft_profile_name)}.pt"
    global_best_soft_summary_path = output_dir / f"global_best_val_{_sanitize_stage_name(global_best_soft_profile_name)}_summary.json"
    global_best_falcor_ckpt_path = output_dir / "global_best_falcor.pt"
    global_best_falcor_summary_path = output_dir / "global_best_falcor_summary.json"
    global_best_soft_value = float("-inf") if global_best_soft_maximize else float("inf")
    global_best_falcor_value = float("-inf") if falcor_best_maximize else float("inf")

    def _metric_is_better(value: float, best: float, maximize: bool) -> bool:
        if not np.isfinite(float(value)):
            return False
        if maximize:
            return float(value) > float(best)
        return float(value) < float(best)

    if global_best_soft_summary_path.exists():
        try:
            global_soft_obj = json.loads(global_best_soft_summary_path.read_text(encoding="utf-8"))
            global_best_soft_value = float(global_soft_obj.get("value", global_best_soft_value))
        except Exception:
            pass
    if global_best_falcor_summary_path.exists():
        try:
            global_falcor_obj = json.loads(global_best_falcor_summary_path.read_text(encoding="utf-8"))
            global_best_falcor_value = float(global_falcor_obj.get("value", global_best_falcor_value))
        except Exception:
            pass

    def _falcor_primary_metric_value(metrics: Dict[str, float]) -> float:
        if falcor_best_metric_name in metrics:
            return float(metrics[falcor_best_metric_name])
        aliases = {
            "hdr_psnr": ["mean_real_render_hdr_psnr", "mean_hdr_psnr", "mean_benchmark_psnr"],
            "real_hdr_psnr": ["mean_real_render_hdr_psnr", "mean_hdr_psnr"],
            "benchmark_psnr": ["mean_benchmark_psnr", "mean_psnr"],
        }
        for candidate in aliases.get(falcor_best_metric_name, []):
            if candidate in metrics:
                return float(metrics[candidate])
        for fallback in ("mean_real_render_hdr_psnr", "mean_hdr_psnr", "mean_benchmark_psnr", "mean_psnr"):
            if fallback in metrics:
                return float(metrics[fallback])
        return float("nan")

    def _falcor_is_better(value: float) -> bool:
        if not np.isfinite(float(value)):
            return False
        if falcor_best_maximize:
            return float(value) > float(falcor_best_value)
        return float(value) < float(falcor_best_value)

    def _update_falcor_best(
        *,
        checkpoint_path: Path,
        summary_path: Path,
        metric_value: float,
        stage_index: int,
        stage_name: str,
        epoch: int,
        kind: str,
    ) -> None:
        nonlocal falcor_best_value, global_best_falcor_value
        falcor_best_value = float(metric_value)
        shutil.copy2(checkpoint_path, falcor_best_ckpt_path)
        payload = {
            "metric_name": falcor_best_metric_name,
            "maximize": bool(falcor_best_maximize),
            "value": float(metric_value),
            "checkpoint": str(falcor_best_ckpt_path),
            "source_checkpoint": str(checkpoint_path),
            "summary": str(summary_path),
            "stage_index": int(stage_index),
            "stage_name": str(stage_name),
            "epoch": int(epoch),
            "kind": str(kind),
            "profile": falcor_periodic_eval_cfg.get("profile", {}),
            "updated_at": _now_iso(),
        }
        falcor_best_summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if global_best_track_falcor and _metric_is_better(float(metric_value), float(global_best_falcor_value), falcor_best_maximize):
            global_best_falcor_value = float(metric_value)
            shutil.copy2(checkpoint_path, global_best_falcor_ckpt_path)
            global_payload = {
                "metric_name": falcor_best_metric_name,
                "maximize": bool(falcor_best_maximize),
                "value": float(metric_value),
                "checkpoint": str(global_best_falcor_ckpt_path),
                "source_checkpoint": str(checkpoint_path),
                "summary": str(summary_path),
                "stage_index": int(stage_index),
                "stage_name": str(stage_name),
                "epoch": int(epoch),
                "kind": str(kind),
                "profile": falcor_periodic_eval_cfg.get("profile", {}),
                "updated_at": _now_iso(),
            }
            global_best_falcor_summary_path.write_text(
                json.dumps(global_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _maybe_update_global_soft_best(
        *,
        source_checkpoint_path: Path,
        metric_value: float,
        stage_index: int,
        stage_name: str,
        epoch: int,
    ) -> None:
        nonlocal global_best_soft_value
        if not global_best_track_soft:
            return
        if not source_checkpoint_path.exists():
            return
        if not _metric_is_better(float(metric_value), float(global_best_soft_value), global_best_soft_maximize):
            return
        global_best_soft_value = float(metric_value)
        shutil.copy2(source_checkpoint_path, global_best_soft_ckpt_path)
        payload = {
            "metric_name": global_best_soft_metric,
            "maximize": bool(global_best_soft_maximize),
            "value": float(metric_value),
            "checkpoint": str(global_best_soft_ckpt_path),
            "source_checkpoint": str(source_checkpoint_path),
            "stage_index": int(stage_index),
            "stage_name": str(stage_name),
            "epoch": int(epoch),
            "profile_name": str(global_best_soft_profile_name),
            "updated_at": _now_iso(),
        }
        global_best_soft_summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    falcor_async_enabled = bool(falcor_periodic_eval_cfg.get("enabled", False)) and bool(
        falcor_periodic_eval_cfg.get("async_mode", True)
    )
    falcor_executor: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=1) if falcor_async_enabled else None
    falcor_pending: List[Dict[str, Any]] = []

    def _falcor_task_runner(*, checkpoint_path: Path, output_dir: Path) -> Dict[str, Any]:
        try:
            summary = _run_falcor_periodic_eval(
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                args=args,
                eval_cfg=falcor_periodic_eval_cfg,
            )
            return {"status": "success", "summary": summary}
        except subprocess.TimeoutExpired as timeout_error:
            return {"status": "timeout", "error": str(timeout_error)}
        except Exception as falcor_error:
            return {"status": "error", "error": str(falcor_error)}

    def _schedule_falcor_task(
        *,
        stage_index: int,
        stage_name: str,
        epoch: int,
        num_epochs: int,
        checkpoint_path: Path,
        run_dir: Path,
    ) -> None:
        if falcor_executor is None:
            return
        future: Future = falcor_executor.submit(
            _falcor_task_runner,
            checkpoint_path=checkpoint_path,
            output_dir=run_dir,
        )
        falcor_pending.append(
            {
                "future": future,
                "stage_index": int(stage_index),
                "stage_name": str(stage_name),
                "epoch": int(epoch),
                "num_epochs": int(num_epochs),
                "checkpoint_path": str(checkpoint_path),
                "run_dir": str(run_dir),
            }
        )

    def _handle_falcor_task_done(task: Dict[str, Any], result: Dict[str, Any]) -> None:
        stage_index = int(task["stage_index"])
        stage_name = str(task["stage_name"])
        epoch = int(task["epoch"])
        num_epochs = int(task["num_epochs"])
        checkpoint_path = Path(str(task["checkpoint_path"]))
        run_dir = Path(str(task["run_dir"]))
        summary_path = run_dir / "benchmark_summary.json"

        status = str(result.get("status", "error"))
        if status == "success":
            summary = result.get("summary", None)
            falcor_metrics = _extract_falcor_metrics_from_summary(summary or {})
            primary = _falcor_primary_metric_value(falcor_metrics)
            payload = {
                "timestamp": _now_iso(),
                "kind": "periodic",
                "stage_index": stage_index,
                "stage_name": stage_name,
                "epoch": epoch,
                "metric_name": falcor_best_metric_name,
                "metric_value": float(primary) if np.isfinite(primary) else None,
                "metrics": falcor_metrics,
                "summary": str(summary_path),
                "checkpoint": str(checkpoint_path),
                "async": True,
            }
            _append_jsonl(falcor_periodic_history_path, payload)
            _append_jsonl(
                phase0_metrics_path,
                {
                    "timestamp": _now_iso(),
                    "kind": "falcor_periodic",
                    "stage_index": stage_index,
                    "stage_name": stage_name,
                    "epoch": epoch,
                    "num_epochs": num_epochs,
                    "metrics": {"falcor_primary": float(primary) if np.isfinite(primary) else 0.0, **falcor_metrics},
                    "summary": str(summary_path),
                    "async": True,
                },
            )
            if _falcor_is_better(primary):
                _update_falcor_best(
                    checkpoint_path=checkpoint_path,
                    summary_path=summary_path,
                    metric_value=float(primary),
                    stage_index=stage_index,
                    stage_name=stage_name,
                    epoch=epoch,
                    kind="periodic_async",
                )
            _update_heartbeat(
                heartbeat_path=heartbeat_path,
                history_path=heartbeat_history_path,
                enabled=heartbeat_enabled,
                save_history=heartbeat_history_enabled,
                run_context=run_context,
                state="running",
                last_event="falcor_periodic_success",
                stage_index=stage_index,
                stage_name=stage_name,
                epoch=epoch,
                num_epochs=num_epochs,
                metrics={"falcor_primary": float(primary) if np.isfinite(primary) else 0.0},
                extra={"summary": str(summary_path), "async": True},
            )
            return

        error_msg = str(result.get("error", "unknown"))
        if status == "timeout":
            timeout_path = run_dir / "timeout.json"
            timeout_path.write_text(
                json.dumps(
                    {
                        "kind": "falcor_periodic_timeout",
                        "stage": stage_name,
                        "epoch": epoch,
                        "timeout_seconds": int(falcor_periodic_eval_cfg.get("timeout_seconds", 1800)),
                        "error": error_msg,
                        "timestamp": _now_iso(),
                        "async": True,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            _update_heartbeat(
                heartbeat_path=heartbeat_path,
                history_path=heartbeat_history_path,
                enabled=heartbeat_enabled,
                save_history=heartbeat_history_enabled,
                run_context=run_context,
                state="running",
                last_event="falcor_periodic_timeout",
                stage_index=stage_index,
                stage_name=stage_name,
                epoch=epoch,
                num_epochs=num_epochs,
                extra={"timeout_file": str(timeout_path), "async": True},
            )
            if bool(falcor_periodic_eval_cfg.get("fail_on_timeout", False)):
                raise RuntimeError(
                    f"Falcor periodic async eval timed out at stage={stage_name} epoch={epoch}"
                )
            return

        _update_heartbeat(
            heartbeat_path=heartbeat_path,
            history_path=heartbeat_history_path,
            enabled=heartbeat_enabled,
            save_history=heartbeat_history_enabled,
            run_context=run_context,
            state="running",
            last_event="falcor_periodic_error",
            stage_index=stage_index,
            stage_name=stage_name,
            epoch=epoch,
            num_epochs=num_epochs,
            extra={"error": error_msg, "async": True},
        )
        if bool(falcor_periodic_eval_cfg.get("fail_on_error", False)):
            raise RuntimeError(
                f"Falcor periodic async eval failed at stage={stage_name} epoch={epoch}: {error_msg}"
            )

    def _drain_falcor_tasks(*, block: bool = False) -> None:
        while True:
            pending_now = list(falcor_pending)
            done_any = False
            for task in pending_now:
                future = task["future"]
                if block:
                    result = future.result()
                    falcor_pending.remove(task)
                    _handle_falcor_task_done(task, result)
                    done_any = True
                    continue
                if future.done():
                    result = future.result()
                    falcor_pending.remove(task)
                    _handle_falcor_task_done(task, result)
                    done_any = True
            if not block or not falcor_pending:
                break
            if not done_any:
                time.sleep(0.2)

    def _run_falcor_periodic_sync(
        *,
        stage_index: int,
        stage_name: str,
        epoch: int,
        num_epochs: int,
        checkpoint_path: Path,
        run_dir: Path,
    ) -> None:
        summary_path = run_dir / "benchmark_summary.json"
        try:
            summary = _run_falcor_periodic_eval(
                checkpoint_path=checkpoint_path,
                output_dir=run_dir,
                args=args,
                eval_cfg=falcor_periodic_eval_cfg,
            )
            falcor_metrics = _extract_falcor_metrics_from_summary(summary or {})
            primary = _falcor_primary_metric_value(falcor_metrics)
            payload = {
                "timestamp": _now_iso(),
                "kind": "periodic",
                "stage_index": int(stage_index),
                "stage_name": str(stage_name),
                "epoch": int(epoch),
                "metric_name": falcor_best_metric_name,
                "metric_value": float(primary) if np.isfinite(primary) else None,
                "metrics": falcor_metrics,
                "summary": str(summary_path),
                "checkpoint": str(checkpoint_path),
                "async": False,
            }
            _append_jsonl(falcor_periodic_history_path, payload)
            _append_jsonl(
                phase0_metrics_path,
                {
                    "timestamp": _now_iso(),
                    "kind": "falcor_periodic",
                    "stage_index": int(stage_index),
                    "stage_name": str(stage_name),
                    "epoch": int(epoch),
                    "num_epochs": int(num_epochs),
                    "metrics": {"falcor_primary": float(primary) if np.isfinite(primary) else 0.0, **falcor_metrics},
                    "summary": str(summary_path),
                    "async": False,
                },
            )
            if _falcor_is_better(primary):
                _update_falcor_best(
                    checkpoint_path=checkpoint_path,
                    summary_path=summary_path,
                    metric_value=float(primary),
                    stage_index=int(stage_index),
                    stage_name=str(stage_name),
                    epoch=int(epoch),
                    kind="periodic_sync",
                )
            _update_heartbeat(
                heartbeat_path=heartbeat_path,
                history_path=heartbeat_history_path,
                enabled=heartbeat_enabled,
                save_history=heartbeat_history_enabled,
                run_context=run_context,
                state="running",
                last_event="falcor_periodic_success",
                stage_index=int(stage_index),
                stage_name=str(stage_name),
                epoch=int(epoch),
                num_epochs=int(num_epochs),
                metrics={"falcor_primary": float(primary) if np.isfinite(primary) else 0.0},
                extra={"summary": str(summary_path), "async": False},
            )
        except subprocess.TimeoutExpired as timeout_error:
            timeout_path = run_dir / "timeout.json"
            timeout_path.write_text(
                json.dumps(
                    {
                        "kind": "falcor_periodic_timeout",
                        "stage": str(stage_name),
                        "epoch": int(epoch),
                        "timeout_seconds": int(falcor_periodic_eval_cfg.get("timeout_seconds", 1800)),
                        "error": str(timeout_error),
                        "timestamp": _now_iso(),
                        "async": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            _update_heartbeat(
                heartbeat_path=heartbeat_path,
                history_path=heartbeat_history_path,
                enabled=heartbeat_enabled,
                save_history=heartbeat_history_enabled,
                run_context=run_context,
                state="running",
                last_event="falcor_periodic_timeout",
                stage_index=int(stage_index),
                stage_name=str(stage_name),
                epoch=int(epoch),
                num_epochs=int(num_epochs),
                extra={"timeout_file": str(timeout_path), "async": False},
            )
            print(
                f"Warning: Falcor periodic eval timeout at stage={stage_name} "
                f"epoch={epoch} timeout={falcor_periodic_eval_cfg.get('timeout_seconds', 1800)}s"
            )
            if bool(falcor_periodic_eval_cfg.get("fail_on_timeout", False)):
                raise RuntimeError(
                    f"Falcor periodic eval timed out at stage={stage_name} epoch={epoch}"
                ) from timeout_error
        except Exception as falcor_error:
            _update_heartbeat(
                heartbeat_path=heartbeat_path,
                history_path=heartbeat_history_path,
                enabled=heartbeat_enabled,
                save_history=heartbeat_history_enabled,
                run_context=run_context,
                state="running",
                last_event="falcor_periodic_error",
                stage_index=int(stage_index),
                stage_name=str(stage_name),
                epoch=int(epoch),
                num_epochs=int(num_epochs),
                extra={"error": str(falcor_error), "async": False},
            )
            print(f"Warning: Falcor periodic eval failed at stage={stage_name} epoch={epoch}: {falcor_error}")
            if bool(falcor_periodic_eval_cfg.get("fail_on_error", False)):
                raise

    falcor_cleanup_done = {"value": False}

    def _finalize_falcor_resources(*, block: bool = True) -> None:
        if falcor_cleanup_done["value"]:
            return
        try:
            _drain_falcor_tasks(block=block)
        except Exception as cleanup_error:
            print(f"Warning: falcor pending task cleanup failed: {cleanup_error}")
        if falcor_executor is not None:
            falcor_executor.shutdown(wait=True)
        falcor_cleanup_done["value"] = True

    def _falcor_exit_guard() -> None:
        _finalize_falcor_resources(block=True)

    atexit.register(_falcor_exit_guard)
    if staged_specs and resume_requested and resume_stage_index is None:
        resume_stage_index = 1
    if staged_specs and resume_requested and resume_stage_index is not None:
        if resume_stage_index < 1 or resume_stage_index > len(staged_specs):
            raise ValueError(
                f"Resume stage index out of range: {resume_stage_index} "
                f"(valid: 1..{len(staged_specs)})"
            )

    if staged_specs:
        staged_summary: Dict[str, Any] = {
            "enabled": True,
            "num_stages": int(len(staged_specs)),
            "stages": [],
        }
        stage_epoch_offset = 0

        for stage_idx, stage in enumerate(staged_specs, start=1):
            stage_name = str(stage["name"])
            stage_slug = _sanitize_stage_name(stage_name)
            stage_dir = output_dir / f"stage{stage_idx:02d}_{stage_slug}"
            stage_dir.mkdir(parents=True, exist_ok=True)

            stage_resume_state: Dict[str, Any] | None = None
            stage_resume_epoch = 0
            if staged_specs and resume_requested and not resume_consumed:
                target_stage = int(resume_stage_index or 1)
                if stage_idx < target_stage:
                    if args.show_progress:
                        print(
                            f"[resume] skip completed stage {stage_idx}/{len(staged_specs)} "
                            f"name={stage_name} (resume target stage={target_stage})"
                        )
                    stage_epoch_offset += int(stage["epochs"])
                    continue
                if stage_idx == target_stage:
                    stage_resume_state = resume_payload_for_fit
                    stage_resume_epoch = int(resume_stage_epoch)
                    if isinstance(stage_resume_state, dict):
                        resume_meta = stage_resume_state.get("meta", {})
                        training_meta = resume_meta.get("training", {}) if isinstance(resume_meta, dict) else {}
                        resume_stage_name = str(training_meta.get("stage_name", "")).strip()
                        if resume_stage_name and resume_stage_name != stage_name:
                            print(
                                f"Warning: resume checkpoint stage_name={resume_stage_name} "
                                f"!= current stage_name={stage_name}"
                            )
                    if stage_resume_epoch >= int(stage["epochs"]):
                        if args.show_progress:
                            print(
                                f"[resume] stage {stage_name} checkpoint epoch={stage_resume_epoch} "
                                f">= stage_epochs={int(stage['epochs'])}, will finalize stage without extra updates"
                            )

            trainability = _set_trainable_groups(model, stage["trainable_groups"])
            if args.show_progress:
                print(
                    f"[staged] stage {stage_idx}/{len(staged_specs)} "
                    f"name={stage_name} epochs={stage['epochs']} "
                    f"groups={trainability['trainable_groups']} "
                    f"trainable={trainability['trainable_param_count']}/{trainability['total_param_count']}"
                )

            trainer = _build_trainer(
                lr=float(stage["lr"]),
                weight_decay=float(stage["weight_decay"]),
                lr_scheduler=str(stage["lr_scheduler"]),
                lr_min=float(stage["lr_min"]),
                warmup_epochs_local=int(stage["warmup_epochs"]),
                group_lrs={**global_group_lrs, **dict(stage.get("group_lrs", {}))},
                routing_balance_epoch_offset=int(stage_epoch_offset),
            )

            stage_meta = deepcopy(checkpoint_meta)
            stage_meta.setdefault("training", {})
            stage_meta["training"].update(
                {
                    "epochs": int(stage["epochs"]),
                    "lr": float(stage["lr"]),
                    "weight_decay": float(stage["weight_decay"]),
                    "lr_scheduler": str(stage["lr_scheduler"]),
                    "lr_min": float(stage["lr_min"]),
                    "warmup_epochs": int(stage["warmup_epochs"]),
                    "staged_enabled": True,
                    "stage_index": int(stage_idx),
                    "stage_name": stage_name,
                    "stage_trainable_groups": list(trainability["trainable_groups"]),
                    "stage_resume_epoch": int(stage_resume_epoch),
                    "resume_checkpoint": str(resume_checkpoint_path) if stage_resume_state is not None and resume_checkpoint_path is not None else None,
                }
            )
            stage_meta["staged_training"] = {
                "enabled": True,
                "num_stages": int(len(staged_specs)),
                "current_stage": int(stage_idx),
                "current_stage_name": stage_name,
                "current_trainable_groups": list(trainability["trainable_groups"]),
                "spec": staged_specs,
            }

            stage_oracle_runs: List[Dict[str, Any]] = []

            _update_heartbeat(
                heartbeat_path=heartbeat_path,
                history_path=heartbeat_history_path,
                enabled=heartbeat_enabled,
                save_history=heartbeat_history_enabled,
                run_context=run_context,
                state="running",
                last_event="stage_start",
                stage_index=int(stage_idx),
                stage_name=stage_name,
                epoch=int(stage_resume_epoch),
                num_epochs=int(stage["epochs"]),
            )

            def _stage_epoch_end_callback(payload: Dict[str, Any]) -> None:
                current_epoch = int(payload.get("epoch", 0))
                phase0_metrics = _phase0_metrics_from_payload(payload)
                _append_jsonl(
                    phase0_metrics_path,
                    {
                        "timestamp": _now_iso(),
                        "kind": "epoch_end",
                        "stage_index": int(stage_idx),
                        "stage_name": stage_name,
                        "epoch": int(current_epoch),
                        "num_epochs": int(stage["epochs"]),
                        "metrics": phase0_metrics,
                    },
                )
                _update_heartbeat(
                    heartbeat_path=heartbeat_path,
                    history_path=heartbeat_history_path,
                    enabled=heartbeat_enabled,
                    save_history=heartbeat_history_enabled,
                    run_context=run_context,
                    state="running",
                    last_event="epoch_end",
                    stage_index=int(stage_idx),
                    stage_name=stage_name,
                    epoch=current_epoch,
                    num_epochs=int(stage["epochs"]),
                    metrics=phase0_metrics,
                )
                if bool(val_profiles_cfg.get("enabled", False)):
                    best_value_by_profile = payload.get("best_value_by_profile", {})
                    if isinstance(best_value_by_profile, dict) and global_best_soft_profile_name in best_value_by_profile:
                        profile_ckpt = stage_dir / _profile_checkpoint_filename(global_best_soft_profile_name)
                        try:
                            _maybe_update_global_soft_best(
                                source_checkpoint_path=profile_ckpt,
                                metric_value=float(best_value_by_profile[global_best_soft_profile_name]),
                                stage_index=int(stage_idx),
                                stage_name=stage_name,
                                epoch=int(current_epoch),
                            )
                        except Exception:
                            pass

                _drain_falcor_tasks(block=False)
                if bool(falcor_periodic_eval_cfg.get("enabled", False)):
                    every_n = int(falcor_periodic_eval_cfg.get("every_n_epochs", 0))
                    if every_n > 0 and current_epoch > 0 and current_epoch % every_n == 0:
                        falcor_run_dir = stage_dir / "falcor_periodic_eval" / f"epoch_{current_epoch:04d}"
                        falcor_ckpt = falcor_run_dir / "checkpoint.pt"
                        falcor_run_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {
                                "epoch": int(current_epoch),
                                "model_state_dict": model.state_dict(),
                                "meta": stage_meta,
                            },
                            falcor_ckpt,
                        )
                        if falcor_async_enabled and falcor_executor is not None:
                            _update_heartbeat(
                                heartbeat_path=heartbeat_path,
                                history_path=heartbeat_history_path,
                                enabled=heartbeat_enabled,
                                save_history=heartbeat_history_enabled,
                                run_context=run_context,
                                state="falcor_eval_running",
                                last_event="falcor_periodic_start",
                                stage_index=int(stage_idx),
                                stage_name=stage_name,
                                epoch=current_epoch,
                                num_epochs=int(stage["epochs"]),
                                extra={"checkpoint": str(falcor_ckpt), "output_dir": str(falcor_run_dir), "async": True},
                            )
                            _schedule_falcor_task(
                                stage_index=int(stage_idx),
                                stage_name=stage_name,
                                epoch=current_epoch,
                                num_epochs=int(stage["epochs"]),
                                checkpoint_path=falcor_ckpt,
                                run_dir=falcor_run_dir,
                            )
                        else:
                            _run_falcor_periodic_sync(
                                stage_index=int(stage_idx),
                                stage_name=stage_name,
                                epoch=int(current_epoch),
                                num_epochs=int(stage["epochs"]),
                                checkpoint_path=falcor_ckpt,
                                run_dir=falcor_run_dir,
                            )

                if not bool(oracle_monitor_cfg.get("enabled", False)):
                    return
                period_epochs = int(oracle_monitor_cfg.get("period_epochs", 0))
                if period_epochs <= 0:
                    return
                if current_epoch <= 0 or current_epoch % period_epochs != 0:
                    return

                monitor_dir = stage_dir / "oracle_monitor"
                checkpoint_path = monitor_dir / f"epoch_{current_epoch:04d}.pt"
                summary_path = monitor_dir / f"epoch_{current_epoch:04d}_light.json"
                monitor_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": current_epoch,
                        "model_state_dict": model.state_dict(),
                        "meta": stage_meta,
                    },
                    checkpoint_path,
                )
                try:
                    _update_heartbeat(
                        heartbeat_path=heartbeat_path,
                        history_path=heartbeat_history_path,
                        enabled=heartbeat_enabled,
                        save_history=heartbeat_history_enabled,
                        run_context=run_context,
                        state="oracle_running",
                        last_event="oracle_light_start",
                        stage_index=int(stage_idx),
                        stage_name=stage_name,
                        epoch=current_epoch,
                        num_epochs=int(stage["epochs"]),
                        extra={"checkpoint": str(checkpoint_path), "summary": str(summary_path)},
                    )
                    summary = _run_oracle_monitor(
                        checkpoint_path=checkpoint_path,
                        output_json_path=summary_path,
                        args=args,
                        oracle_cfg=oracle_monitor_cfg,
                        max_samples=int(oracle_monitor_cfg.get("lightweight_max_samples", 4096)),
                        sample_seed=int(oracle_monitor_cfg.get("lightweight_sample_seed", args.seed)) + current_epoch,
                        timeout_seconds=int(oracle_monitor_cfg.get("timeout_seconds", 180)),
                    )
                    if summary is not None:
                        metrics = _oracle_metrics_from_summary(summary)
                        stage_oracle_runs.append(
                            {
                                "kind": "lightweight",
                                "epoch": current_epoch,
                                "summary": str(summary_path),
                                "metrics": metrics,
                            }
                        )
                        if rerun_logger is not None:
                            rerun_logger.log_scalars(current_epoch, metrics, f"oracle/{stage_slug}/light")
                        _append_jsonl(
                            phase0_metrics_path,
                            {
                                "timestamp": _now_iso(),
                                "kind": "oracle_light",
                                "stage_index": int(stage_idx),
                                "stage_name": stage_name,
                                "epoch": int(current_epoch),
                                "num_epochs": int(stage["epochs"]),
                                "metrics": metrics,
                                "summary": str(summary_path),
                            },
                        )
                    _update_heartbeat(
                        heartbeat_path=heartbeat_path,
                        history_path=heartbeat_history_path,
                        enabled=heartbeat_enabled,
                        save_history=heartbeat_history_enabled,
                        run_context=run_context,
                        state="running",
                        last_event="oracle_light_success",
                        stage_index=int(stage_idx),
                        stage_name=stage_name,
                        epoch=current_epoch,
                        num_epochs=int(stage["epochs"]),
                        metrics=metrics,
                    )
                except subprocess.TimeoutExpired as timeout_error:
                    timeout_path = monitor_dir / f"epoch_{current_epoch:04d}_light_timeout.json"
                    timeout_payload = {
                        "kind": "lightweight_timeout",
                        "stage": stage_name,
                        "epoch": int(current_epoch),
                        "timeout_seconds": int(oracle_monitor_cfg.get("timeout_seconds", 180)),
                        "error": str(timeout_error),
                        "timestamp": _now_iso(),
                    }
                    timeout_path.write_text(json.dumps(timeout_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    stage_oracle_runs.append(
                        {
                            "kind": "lightweight_timeout",
                            "epoch": int(current_epoch),
                            "summary": str(timeout_path),
                            "metrics": {},
                        }
                    )
                    _update_heartbeat(
                        heartbeat_path=heartbeat_path,
                        history_path=heartbeat_history_path,
                        enabled=heartbeat_enabled,
                        save_history=heartbeat_history_enabled,
                        run_context=run_context,
                        state="running",
                        last_event="oracle_light_timeout",
                        stage_index=int(stage_idx),
                        stage_name=stage_name,
                        epoch=current_epoch,
                        num_epochs=int(stage["epochs"]),
                        extra={"timeout_file": str(timeout_path)},
                    )
                    print(
                        f"Warning: Oracle monitor timeout at stage={stage_name} "
                        f"epoch={current_epoch} timeout={oracle_monitor_cfg.get('timeout_seconds', 180)}s"
                    )
                    if bool(oracle_monitor_cfg.get("fail_on_timeout", False)):
                        raise RuntimeError(
                            f"Oracle monitor timed out at stage={stage_name} epoch={current_epoch}"
                        ) from timeout_error
                except Exception as monitor_error:
                    _update_heartbeat(
                        heartbeat_path=heartbeat_path,
                        history_path=heartbeat_history_path,
                        enabled=heartbeat_enabled,
                        save_history=heartbeat_history_enabled,
                        run_context=run_context,
                        state="running",
                        last_event="oracle_light_error",
                        stage_index=int(stage_idx),
                        stage_name=stage_name,
                        epoch=current_epoch,
                        num_epochs=int(stage["epochs"]),
                        extra={"error": str(monitor_error)},
                    )
                    print(f"Warning: Oracle monitor failed at stage={stage_name} epoch={current_epoch}: {monitor_error}")

                # Falcor periodic evaluation is executed before Oracle early-return checks.

            stage_history = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=int(stage["epochs"]),
                output_dir=stage_dir,
                checkpoint_meta=stage_meta,
                save_best=True,
                best_metric=str(stage["best_metric"]),
                val_profiles=list(val_profiles_cfg.get("profiles", [])) if bool(val_profiles_cfg.get("enabled", False)) else None,
                save_best_profiles=bool(val_profiles_cfg.get("enabled", False)),
                best_metric_per_profile=str(val_profiles_cfg.get("best_metric", "mae")),
                epoch_end_callback=_stage_epoch_end_callback,
                resume_state=stage_resume_state,
                save_resume_every=int(getattr(args, "resume_save_every", 1)),
                resume_checkpoint_name=str(getattr(args, "resume_checkpoint_name", "resume_latest.pt")),
            )
            if stage_resume_state is not None:
                resume_consumed = True
            _merge_history(history, stage_history)

            if bool(oracle_monitor_cfg.get("enabled", False)) and bool(oracle_monitor_cfg.get("stage_end_full", True)):
                stage_monitor_dir = stage_dir / "oracle_monitor"
                stage_monitor_dir.mkdir(parents=True, exist_ok=True)
                stage_best = stage_dir / "best_model.pt"
                stage_last = stage_dir / "last_model.pt"
                stage_ckpt = stage_best if stage_best.exists() else stage_last
                if stage_ckpt.exists():
                    full_summary_path = stage_monitor_dir / "stage_end_full.json"
                    try:
                        _update_heartbeat(
                            heartbeat_path=heartbeat_path,
                            history_path=heartbeat_history_path,
                            enabled=heartbeat_enabled,
                            save_history=heartbeat_history_enabled,
                            run_context=run_context,
                            state="oracle_running",
                            last_event="oracle_stage_end_start",
                            stage_index=int(stage_idx),
                            stage_name=stage_name,
                            epoch=int(stage["epochs"]),
                            num_epochs=int(stage["epochs"]),
                            extra={"checkpoint": str(stage_ckpt), "summary": str(full_summary_path)},
                        )
                        full_summary = _run_oracle_monitor(
                            checkpoint_path=stage_ckpt,
                            output_json_path=full_summary_path,
                            args=args,
                            oracle_cfg=oracle_monitor_cfg,
                            max_samples=int(oracle_monitor_cfg.get("full_max_samples", 0)),
                            sample_seed=int(oracle_monitor_cfg.get("lightweight_sample_seed", args.seed)),
                            timeout_seconds=int(oracle_monitor_cfg.get("timeout_seconds", 180)),
                        )
                        if full_summary is not None:
                            full_metrics = _oracle_metrics_from_summary(full_summary)
                            stage_oracle_runs.append(
                                {
                                    "kind": "stage_end_full",
                                    "epoch": int(stage["epochs"]),
                                    "summary": str(full_summary_path),
                                    "metrics": full_metrics,
                                }
                            )
                            if rerun_logger is not None:
                                rerun_logger.log_scalars(int(stage["epochs"]), full_metrics, f"oracle/{stage_slug}/full")
                            _update_heartbeat(
                                heartbeat_path=heartbeat_path,
                                history_path=heartbeat_history_path,
                                enabled=heartbeat_enabled,
                                save_history=heartbeat_history_enabled,
                                run_context=run_context,
                                state="running",
                                last_event="oracle_stage_end_success",
                                stage_index=int(stage_idx),
                                stage_name=stage_name,
                                epoch=int(stage["epochs"]),
                                num_epochs=int(stage["epochs"]),
                                metrics=full_metrics,
                            )
                    except subprocess.TimeoutExpired as timeout_error:
                        timeout_path = stage_monitor_dir / "stage_end_full_timeout.json"
                        timeout_payload = {
                            "kind": "stage_end_full_timeout",
                            "stage": stage_name,
                            "epoch": int(stage["epochs"]),
                            "timeout_seconds": int(oracle_monitor_cfg.get("timeout_seconds", 180)),
                            "error": str(timeout_error),
                            "timestamp": _now_iso(),
                        }
                        timeout_path.write_text(json.dumps(timeout_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                        stage_oracle_runs.append(
                            {
                                "kind": "stage_end_full_timeout",
                                "epoch": int(stage["epochs"]),
                                "summary": str(timeout_path),
                                "metrics": {},
                            }
                        )
                        _update_heartbeat(
                            heartbeat_path=heartbeat_path,
                            history_path=heartbeat_history_path,
                            enabled=heartbeat_enabled,
                            save_history=heartbeat_history_enabled,
                            run_context=run_context,
                            state="running",
                            last_event="oracle_stage_end_timeout",
                            stage_index=int(stage_idx),
                            stage_name=stage_name,
                            epoch=int(stage["epochs"]),
                            num_epochs=int(stage["epochs"]),
                            extra={"timeout_file": str(timeout_path)},
                        )
                        print(
                            f"Warning: Oracle stage-end monitor timeout at stage={stage_name} "
                            f"timeout={oracle_monitor_cfg.get('timeout_seconds', 180)}s"
                        )
                        if bool(oracle_monitor_cfg.get("fail_on_timeout", False)):
                            raise RuntimeError(f"Oracle stage-end monitor timed out at stage={stage_name}") from timeout_error
                    except Exception as monitor_error:
                        _update_heartbeat(
                            heartbeat_path=heartbeat_path,
                            history_path=heartbeat_history_path,
                            enabled=heartbeat_enabled,
                            save_history=heartbeat_history_enabled,
                            run_context=run_context,
                            state="running",
                            last_event="oracle_stage_end_error",
                            stage_index=int(stage_idx),
                            stage_name=stage_name,
                            epoch=int(stage["epochs"]),
                            num_epochs=int(stage["epochs"]),
                            extra={"error": str(monitor_error)},
                        )
                        print(f"Warning: Oracle stage-end monitor failed at stage={stage_name}: {monitor_error}")

            _drain_falcor_tasks(block=True)
            if bool(falcor_periodic_eval_cfg.get("enabled", False)) and bool(falcor_periodic_eval_cfg.get("stage_end_full", True)):
                stage_falcor_dir = stage_dir / "falcor_periodic_eval" / "stage_end_full"
                stage_falcor_dir.mkdir(parents=True, exist_ok=True)
                stage_best = stage_dir / "best_model.pt"
                stage_last = stage_dir / "last_model.pt"
                stage_ckpt = stage_best if stage_best.exists() else stage_last
                if stage_ckpt.exists():
                    stage_end_summary_path = stage_falcor_dir / "benchmark_summary.json"
                    try:
                        _update_heartbeat(
                            heartbeat_path=heartbeat_path,
                            history_path=heartbeat_history_path,
                            enabled=heartbeat_enabled,
                            save_history=heartbeat_history_enabled,
                            run_context=run_context,
                            state="falcor_eval_running",
                            last_event="falcor_stage_end_start",
                            stage_index=int(stage_idx),
                            stage_name=stage_name,
                            epoch=int(stage["epochs"]),
                            num_epochs=int(stage["epochs"]),
                            extra={"checkpoint": str(stage_ckpt), "output_dir": str(stage_falcor_dir)},
                        )
                        stage_summary = _run_falcor_periodic_eval(
                            checkpoint_path=stage_ckpt,
                            output_dir=stage_falcor_dir,
                            args=args,
                            eval_cfg=falcor_periodic_eval_cfg,
                        )
                        stage_metrics = _extract_falcor_metrics_from_summary(stage_summary or {})
                        primary = _falcor_primary_metric_value(stage_metrics)
                        _append_jsonl(
                            falcor_periodic_history_path,
                            {
                                "timestamp": _now_iso(),
                                "kind": "stage_end_full",
                                "stage_index": int(stage_idx),
                                "stage_name": stage_name,
                                "epoch": int(stage["epochs"]),
                                "metric_name": falcor_best_metric_name,
                                "metric_value": float(primary) if np.isfinite(primary) else None,
                                "metrics": stage_metrics,
                                "summary": str(stage_end_summary_path),
                                "checkpoint": str(stage_ckpt),
                            },
                        )
                        if _falcor_is_better(primary):
                            _update_falcor_best(
                                checkpoint_path=stage_ckpt,
                                summary_path=stage_end_summary_path,
                                metric_value=float(primary),
                                stage_index=int(stage_idx),
                                stage_name=stage_name,
                                epoch=int(stage["epochs"]),
                                kind="stage_end_full",
                            )
                        _update_heartbeat(
                            heartbeat_path=heartbeat_path,
                            history_path=heartbeat_history_path,
                            enabled=heartbeat_enabled,
                            save_history=heartbeat_history_enabled,
                            run_context=run_context,
                            state="running",
                            last_event="falcor_stage_end_success",
                            stage_index=int(stage_idx),
                            stage_name=stage_name,
                            epoch=int(stage["epochs"]),
                            num_epochs=int(stage["epochs"]),
                            metrics={"falcor_primary": float(primary) if np.isfinite(primary) else 0.0},
                            extra={"summary": str(stage_end_summary_path)},
                        )
                    except subprocess.TimeoutExpired as timeout_error:
                        timeout_path = stage_falcor_dir / "timeout.json"
                        timeout_path.write_text(
                            json.dumps(
                                {
                                    "kind": "falcor_stage_end_timeout",
                                    "stage": stage_name,
                                    "epoch": int(stage["epochs"]),
                                    "timeout_seconds": int(falcor_periodic_eval_cfg.get("timeout_seconds", 1800)),
                                    "error": str(timeout_error),
                                    "timestamp": _now_iso(),
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        _update_heartbeat(
                            heartbeat_path=heartbeat_path,
                            history_path=heartbeat_history_path,
                            enabled=heartbeat_enabled,
                            save_history=heartbeat_history_enabled,
                            run_context=run_context,
                            state="running",
                            last_event="falcor_stage_end_timeout",
                            stage_index=int(stage_idx),
                            stage_name=stage_name,
                            epoch=int(stage["epochs"]),
                            num_epochs=int(stage["epochs"]),
                            extra={"timeout_file": str(timeout_path)},
                        )
                        print(
                            f"Warning: Falcor stage-end eval timeout at stage={stage_name} "
                            f"timeout={falcor_periodic_eval_cfg.get('timeout_seconds', 1800)}s"
                        )
                        if bool(falcor_periodic_eval_cfg.get("fail_on_timeout", False)):
                            raise RuntimeError(f"Falcor stage-end eval timed out at stage={stage_name}") from timeout_error
                    except Exception as falcor_error:
                        _update_heartbeat(
                            heartbeat_path=heartbeat_path,
                            history_path=heartbeat_history_path,
                            enabled=heartbeat_enabled,
                            save_history=heartbeat_history_enabled,
                            run_context=run_context,
                            state="running",
                            last_event="falcor_stage_end_error",
                            stage_index=int(stage_idx),
                            stage_name=stage_name,
                            epoch=int(stage["epochs"]),
                            num_epochs=int(stage["epochs"]),
                            extra={"error": str(falcor_error)},
                        )
                        print(f"Warning: Falcor stage-end eval failed at stage={stage_name}: {falcor_error}")
                        if bool(falcor_periodic_eval_cfg.get("fail_on_error", False)):
                            raise

            _update_heartbeat(
                heartbeat_path=heartbeat_path,
                history_path=heartbeat_history_path,
                enabled=heartbeat_enabled,
                save_history=heartbeat_history_enabled,
                run_context=run_context,
                state="running",
                last_event="stage_finished",
                stage_index=int(stage_idx),
                stage_name=stage_name,
                epoch=int(stage["epochs"]),
                num_epochs=int(stage["epochs"]),
            )

            train_last = {k: (v[-1] if v else None) for k, v in stage_history["train"].items()}
            val_last = {k: (v[-1] if v else None) for k, v in stage_history["val"].items()}
            staged_summary["stages"].append(
                {
                    "index": int(stage_idx),
                    "name": stage_name,
                    "epochs": int(stage["epochs"]),
                    "best_metric": str(stage["best_metric"]),
                    "lr": float(stage["lr"]),
                    "weight_decay": float(stage["weight_decay"]),
                    "lr_scheduler": str(stage["lr_scheduler"]),
                    "lr_min": float(stage["lr_min"]),
                    "warmup_epochs": int(stage["warmup_epochs"]),
                    "group_lrs": {**global_group_lrs, **dict(stage.get("group_lrs", {}))},
                    "ema": ema_cfg,
                    "oracle_runs": stage_oracle_runs,
                    "output_dir": str(stage_dir),
                    "best_model": str(stage_dir / "best_model.pt"),
                    "val_profile_best_models": {
                        str(profile.get("name", "profile")): str(stage_dir / _profile_checkpoint_filename(str(profile.get("name", "profile"))))
                        for profile in (val_profiles_cfg.get("profiles", []) if bool(val_profiles_cfg.get("enabled", False)) else [])
                    },
                    "last_model": str(stage_dir / "last_model.pt"),
                    "trainability": trainability,
                    "train_last": {k: v for k, v in train_last.items() if v is not None},
                    "val_last": {k: v for k, v in val_last.items() if v is not None},
                }
            )
            stage_epoch_offset += int(stage["epochs"])

        final_stage = staged_summary["stages"][-1]
        final_stage_dir = Path(final_stage["output_dir"])
        profile_ckpt_names = []
        if bool(val_profiles_cfg.get("enabled", False)):
            profile_ckpt_names = [
                _profile_checkpoint_filename(str(profile.get("name", "profile")))
                for profile in val_profiles_cfg.get("profiles", [])
            ]
        for ckpt_name in ["best_model.pt", "last_model.pt", *profile_ckpt_names]:
            src = final_stage_dir / ckpt_name
            dst = output_dir / ckpt_name
            if src.exists():
                shutil.copy2(src, dst)

        staged_summary_path = output_dir / "staged_training_summary.json"
        staged_summary_path.write_text(json.dumps(staged_summary, indent=2), encoding="utf-8")
        print(f"Wrote staged summary: {staged_summary_path}")
    else:
        _set_trainable_groups(model, ["all"])
        single_stage_resume_epoch = int(resume_stage_epoch if resume_requested else 0)
        trainer = _build_trainer(
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            lr_scheduler=str(args.lr_scheduler),
            lr_min=float(args.lr_min),
            warmup_epochs_local=int(warmup_epochs),
            group_lrs=global_group_lrs,
            routing_balance_epoch_offset=0,
        )

        _update_heartbeat(
            heartbeat_path=heartbeat_path,
            history_path=heartbeat_history_path,
            enabled=heartbeat_enabled,
            save_history=heartbeat_history_enabled,
            run_context=run_context,
            state="running",
            last_event="stage_start",
            stage_index=1,
            stage_name="single_stage",
            epoch=single_stage_resume_epoch,
            num_epochs=int(args.epochs),
        )

        def _single_stage_epoch_end_callback(payload: Dict[str, Any]) -> None:
            current_epoch = int(payload.get("epoch", 0))
            phase0_metrics = _phase0_metrics_from_payload(payload)
            _append_jsonl(
                phase0_metrics_path,
                {
                    "timestamp": _now_iso(),
                    "kind": "epoch_end",
                    "stage_index": 1,
                    "stage_name": "single_stage",
                    "epoch": int(current_epoch),
                    "num_epochs": int(args.epochs),
                    "metrics": phase0_metrics,
                },
            )
            _update_heartbeat(
                heartbeat_path=heartbeat_path,
                history_path=heartbeat_history_path,
                enabled=heartbeat_enabled,
                save_history=heartbeat_history_enabled,
                run_context=run_context,
                state="running",
                last_event="epoch_end",
                stage_index=1,
                stage_name="single_stage",
                epoch=current_epoch,
                num_epochs=int(args.epochs),
                metrics=phase0_metrics,
            )
            if bool(val_profiles_cfg.get("enabled", False)):
                best_value_by_profile = payload.get("best_value_by_profile", {})
                if isinstance(best_value_by_profile, dict) and global_best_soft_profile_name in best_value_by_profile:
                    profile_ckpt = output_dir / _profile_checkpoint_filename(global_best_soft_profile_name)
                    try:
                        _maybe_update_global_soft_best(
                            source_checkpoint_path=profile_ckpt,
                            metric_value=float(best_value_by_profile[global_best_soft_profile_name]),
                            stage_index=1,
                            stage_name="single_stage",
                            epoch=int(current_epoch),
                        )
                    except Exception:
                        pass

            _drain_falcor_tasks(block=False)
            if bool(falcor_periodic_eval_cfg.get("enabled", False)):
                every_n = int(falcor_periodic_eval_cfg.get("every_n_epochs", 0))
                if every_n > 0 and current_epoch > 0 and current_epoch % every_n == 0:
                    falcor_run_dir = output_dir / "falcor_periodic_eval" / f"epoch_{current_epoch:04d}"
                    falcor_ckpt = falcor_run_dir / "checkpoint.pt"
                    falcor_run_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "epoch": int(current_epoch),
                            "model_state_dict": model.state_dict(),
                            "meta": checkpoint_meta,
                        },
                        falcor_ckpt,
                    )
                    if falcor_async_enabled and falcor_executor is not None:
                        _update_heartbeat(
                            heartbeat_path=heartbeat_path,
                            history_path=heartbeat_history_path,
                            enabled=heartbeat_enabled,
                            save_history=heartbeat_history_enabled,
                            run_context=run_context,
                            state="falcor_eval_running",
                            last_event="falcor_periodic_start",
                            stage_index=1,
                            stage_name="single_stage",
                            epoch=current_epoch,
                            num_epochs=int(args.epochs),
                            extra={"checkpoint": str(falcor_ckpt), "output_dir": str(falcor_run_dir), "async": True},
                        )
                        _schedule_falcor_task(
                            stage_index=1,
                            stage_name="single_stage",
                            epoch=current_epoch,
                            num_epochs=int(args.epochs),
                            checkpoint_path=falcor_ckpt,
                            run_dir=falcor_run_dir,
                        )
                    else:
                        _run_falcor_periodic_sync(
                            stage_index=1,
                            stage_name="single_stage",
                            epoch=int(current_epoch),
                            num_epochs=int(args.epochs),
                            checkpoint_path=falcor_ckpt,
                            run_dir=falcor_run_dir,
                        )

            if not bool(oracle_monitor_cfg.get("enabled", False)):
                return
            period_epochs = int(oracle_monitor_cfg.get("period_epochs", 0))
            if period_epochs <= 0:
                return
            if current_epoch <= 0 or current_epoch % period_epochs != 0:
                return

            monitor_dir = output_dir / "oracle_monitor"
            checkpoint_path = monitor_dir / f"epoch_{current_epoch:04d}.pt"
            summary_path = monitor_dir / f"epoch_{current_epoch:04d}_light.json"
            monitor_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": current_epoch,
                    "model_state_dict": model.state_dict(),
                    "meta": checkpoint_meta,
                },
                checkpoint_path,
            )
            try:
                _update_heartbeat(
                    heartbeat_path=heartbeat_path,
                    history_path=heartbeat_history_path,
                    enabled=heartbeat_enabled,
                    save_history=heartbeat_history_enabled,
                    run_context=run_context,
                    state="oracle_running",
                    last_event="oracle_light_start",
                    stage_index=1,
                    stage_name="single_stage",
                    epoch=current_epoch,
                    num_epochs=int(args.epochs),
                    extra={"checkpoint": str(checkpoint_path), "summary": str(summary_path)},
                )
                summary = _run_oracle_monitor(
                    checkpoint_path=checkpoint_path,
                    output_json_path=summary_path,
                    args=args,
                    oracle_cfg=oracle_monitor_cfg,
                    max_samples=int(oracle_monitor_cfg.get("lightweight_max_samples", 4096)),
                    sample_seed=int(oracle_monitor_cfg.get("lightweight_sample_seed", args.seed)) + current_epoch,
                    timeout_seconds=int(oracle_monitor_cfg.get("timeout_seconds", 180)),
                )
                if summary is not None:
                    metrics = _oracle_metrics_from_summary(summary)
                    if rerun_logger is not None:
                        rerun_logger.log_scalars(current_epoch, metrics, "oracle/main/light")
                    _append_jsonl(
                        phase0_metrics_path,
                        {
                            "timestamp": _now_iso(),
                            "kind": "oracle_light",
                            "stage_index": 1,
                            "stage_name": "single_stage",
                            "epoch": int(current_epoch),
                            "num_epochs": int(args.epochs),
                            "metrics": metrics,
                            "summary": str(summary_path),
                        },
                    )
                    _update_heartbeat(
                        heartbeat_path=heartbeat_path,
                        history_path=heartbeat_history_path,
                        enabled=heartbeat_enabled,
                        save_history=heartbeat_history_enabled,
                        run_context=run_context,
                        state="running",
                        last_event="oracle_light_success",
                        stage_index=1,
                        stage_name="single_stage",
                        epoch=current_epoch,
                        num_epochs=int(args.epochs),
                        metrics=metrics,
                    )
            except subprocess.TimeoutExpired as timeout_error:
                timeout_path = monitor_dir / f"epoch_{current_epoch:04d}_light_timeout.json"
                timeout_payload = {
                    "kind": "lightweight_timeout",
                    "stage": "single_stage",
                    "epoch": int(current_epoch),
                    "timeout_seconds": int(oracle_monitor_cfg.get("timeout_seconds", 180)),
                    "error": str(timeout_error),
                    "timestamp": _now_iso(),
                }
                timeout_path.write_text(json.dumps(timeout_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                _update_heartbeat(
                    heartbeat_path=heartbeat_path,
                    history_path=heartbeat_history_path,
                    enabled=heartbeat_enabled,
                    save_history=heartbeat_history_enabled,
                    run_context=run_context,
                    state="running",
                    last_event="oracle_light_timeout",
                    stage_index=1,
                    stage_name="single_stage",
                    epoch=current_epoch,
                    num_epochs=int(args.epochs),
                    extra={"timeout_file": str(timeout_path)},
                )
                print(
                    f"Warning: Oracle monitor timeout at epoch={current_epoch} "
                    f"timeout={oracle_monitor_cfg.get('timeout_seconds', 180)}s"
                )
                if bool(oracle_monitor_cfg.get("fail_on_timeout", False)):
                    raise RuntimeError(f"Oracle monitor timed out at epoch={current_epoch}") from timeout_error
            except Exception as monitor_error:
                _update_heartbeat(
                    heartbeat_path=heartbeat_path,
                    history_path=heartbeat_history_path,
                    enabled=heartbeat_enabled,
                    save_history=heartbeat_history_enabled,
                    run_context=run_context,
                    state="running",
                    last_event="oracle_light_error",
                    stage_index=1,
                    stage_name="single_stage",
                    epoch=current_epoch,
                    num_epochs=int(args.epochs),
                    extra={"error": str(monitor_error)},
                )
                print(f"Warning: Oracle monitor failed at epoch={current_epoch}: {monitor_error}")

        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            output_dir=output_dir,
            checkpoint_meta=checkpoint_meta,
            save_best=output_dir is not None,
            best_metric="mae",
            val_profiles=list(val_profiles_cfg.get("profiles", [])) if bool(val_profiles_cfg.get("enabled", False)) else None,
            save_best_profiles=bool(val_profiles_cfg.get("enabled", False)),
            best_metric_per_profile=str(val_profiles_cfg.get("best_metric", "mae")),
            epoch_end_callback=_single_stage_epoch_end_callback,
            resume_state=resume_payload_for_fit if resume_requested else None,
            save_resume_every=int(getattr(args, "resume_save_every", 1)),
            resume_checkpoint_name=str(getattr(args, "resume_checkpoint_name", "resume_latest.pt")),
        )

        if bool(oracle_monitor_cfg.get("enabled", False)) and bool(oracle_monitor_cfg.get("stage_end_full", True)):
            monitor_dir = output_dir / "oracle_monitor"
            monitor_dir.mkdir(parents=True, exist_ok=True)
            best_ckpt = output_dir / "best_model.pt"
            last_ckpt = output_dir / "last_model.pt"
            final_ckpt = best_ckpt if best_ckpt.exists() else last_ckpt
            if final_ckpt.exists():
                full_summary_path = monitor_dir / "stage_end_full.json"
                try:
                    _update_heartbeat(
                        heartbeat_path=heartbeat_path,
                        history_path=heartbeat_history_path,
                        enabled=heartbeat_enabled,
                        save_history=heartbeat_history_enabled,
                        run_context=run_context,
                        state="oracle_running",
                        last_event="oracle_stage_end_start",
                        stage_index=1,
                        stage_name="single_stage",
                        epoch=int(args.epochs),
                        num_epochs=int(args.epochs),
                        extra={"checkpoint": str(final_ckpt), "summary": str(full_summary_path)},
                    )
                    full_summary = _run_oracle_monitor(
                        checkpoint_path=final_ckpt,
                        output_json_path=full_summary_path,
                        args=args,
                        oracle_cfg=oracle_monitor_cfg,
                        max_samples=int(oracle_monitor_cfg.get("full_max_samples", 0)),
                        sample_seed=int(oracle_monitor_cfg.get("lightweight_sample_seed", args.seed)),
                        timeout_seconds=int(oracle_monitor_cfg.get("timeout_seconds", 180)),
                    )
                    if full_summary is not None:
                        full_metrics = _oracle_metrics_from_summary(full_summary)
                        if rerun_logger is not None:
                            rerun_logger.log_scalars(int(args.epochs), full_metrics, "oracle/main/full")
                        _update_heartbeat(
                            heartbeat_path=heartbeat_path,
                            history_path=heartbeat_history_path,
                            enabled=heartbeat_enabled,
                            save_history=heartbeat_history_enabled,
                            run_context=run_context,
                            state="running",
                            last_event="oracle_stage_end_success",
                            stage_index=1,
                            stage_name="single_stage",
                            epoch=int(args.epochs),
                            num_epochs=int(args.epochs),
                            metrics=full_metrics,
                        )
                except subprocess.TimeoutExpired as timeout_error:
                    timeout_path = monitor_dir / "stage_end_full_timeout.json"
                    timeout_payload = {
                        "kind": "stage_end_full_timeout",
                        "stage": "single_stage",
                        "epoch": int(args.epochs),
                        "timeout_seconds": int(oracle_monitor_cfg.get("timeout_seconds", 180)),
                        "error": str(timeout_error),
                        "timestamp": _now_iso(),
                    }
                    timeout_path.write_text(json.dumps(timeout_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    _update_heartbeat(
                        heartbeat_path=heartbeat_path,
                        history_path=heartbeat_history_path,
                        enabled=heartbeat_enabled,
                        save_history=heartbeat_history_enabled,
                        run_context=run_context,
                        state="running",
                        last_event="oracle_stage_end_timeout",
                        stage_index=1,
                        stage_name="single_stage",
                        epoch=int(args.epochs),
                        num_epochs=int(args.epochs),
                        extra={"timeout_file": str(timeout_path)},
                    )
                    print(
                        f"Warning: Oracle stage-end monitor timeout timeout={oracle_monitor_cfg.get('timeout_seconds', 180)}s"
                    )
                    if bool(oracle_monitor_cfg.get("fail_on_timeout", False)):
                        raise RuntimeError("Oracle stage-end monitor timed out") from timeout_error
                except Exception as monitor_error:
                    _update_heartbeat(
                        heartbeat_path=heartbeat_path,
                        history_path=heartbeat_history_path,
                        enabled=heartbeat_enabled,
                        save_history=heartbeat_history_enabled,
                        run_context=run_context,
                        state="running",
                        last_event="oracle_stage_end_error",
                        stage_index=1,
                        stage_name="single_stage",
                        epoch=int(args.epochs),
                        num_epochs=int(args.epochs),
                        extra={"error": str(monitor_error)},
                    )
                    print(f"Warning: Oracle stage-end monitor failed: {monitor_error}")

        _drain_falcor_tasks(block=True)
        if bool(falcor_periodic_eval_cfg.get("enabled", False)) and bool(falcor_periodic_eval_cfg.get("stage_end_full", True)):
            stage_falcor_dir = output_dir / "falcor_periodic_eval" / "stage_end_full"
            stage_falcor_dir.mkdir(parents=True, exist_ok=True)
            best_ckpt = output_dir / "best_model.pt"
            last_ckpt = output_dir / "last_model.pt"
            final_ckpt = best_ckpt if best_ckpt.exists() else last_ckpt
            if final_ckpt.exists():
                stage_end_summary_path = stage_falcor_dir / "benchmark_summary.json"
                try:
                    _update_heartbeat(
                        heartbeat_path=heartbeat_path,
                        history_path=heartbeat_history_path,
                        enabled=heartbeat_enabled,
                        save_history=heartbeat_history_enabled,
                        run_context=run_context,
                        state="falcor_eval_running",
                        last_event="falcor_stage_end_start",
                        stage_index=1,
                        stage_name="single_stage",
                        epoch=int(args.epochs),
                        num_epochs=int(args.epochs),
                        extra={"checkpoint": str(final_ckpt), "output_dir": str(stage_falcor_dir)},
                    )
                    stage_summary = _run_falcor_periodic_eval(
                        checkpoint_path=final_ckpt,
                        output_dir=stage_falcor_dir,
                        args=args,
                        eval_cfg=falcor_periodic_eval_cfg,
                    )
                    stage_metrics = _extract_falcor_metrics_from_summary(stage_summary or {})
                    primary = _falcor_primary_metric_value(stage_metrics)
                    _append_jsonl(
                        falcor_periodic_history_path,
                        {
                            "timestamp": _now_iso(),
                            "kind": "stage_end_full",
                            "stage_index": 1,
                            "stage_name": "single_stage",
                            "epoch": int(args.epochs),
                            "metric_name": falcor_best_metric_name,
                            "metric_value": float(primary) if np.isfinite(primary) else None,
                            "metrics": stage_metrics,
                            "summary": str(stage_end_summary_path),
                            "checkpoint": str(final_ckpt),
                        },
                    )
                    if _falcor_is_better(primary):
                        _update_falcor_best(
                            checkpoint_path=final_ckpt,
                            summary_path=stage_end_summary_path,
                            metric_value=float(primary),
                            stage_index=1,
                            stage_name="single_stage",
                            epoch=int(args.epochs),
                            kind="stage_end_full",
                        )
                    _update_heartbeat(
                        heartbeat_path=heartbeat_path,
                        history_path=heartbeat_history_path,
                        enabled=heartbeat_enabled,
                        save_history=heartbeat_history_enabled,
                        run_context=run_context,
                        state="running",
                        last_event="falcor_stage_end_success",
                        stage_index=1,
                        stage_name="single_stage",
                        epoch=int(args.epochs),
                        num_epochs=int(args.epochs),
                        metrics={"falcor_primary": float(primary) if np.isfinite(primary) else 0.0},
                        extra={"summary": str(stage_end_summary_path)},
                    )
                except subprocess.TimeoutExpired as timeout_error:
                    timeout_path = stage_falcor_dir / "timeout.json"
                    timeout_path.write_text(
                        json.dumps(
                            {
                                "kind": "falcor_stage_end_timeout",
                                "stage": "single_stage",
                                "epoch": int(args.epochs),
                                "timeout_seconds": int(falcor_periodic_eval_cfg.get("timeout_seconds", 1800)),
                                "error": str(timeout_error),
                                "timestamp": _now_iso(),
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    _update_heartbeat(
                        heartbeat_path=heartbeat_path,
                        history_path=heartbeat_history_path,
                        enabled=heartbeat_enabled,
                        save_history=heartbeat_history_enabled,
                        run_context=run_context,
                        state="running",
                        last_event="falcor_stage_end_timeout",
                        stage_index=1,
                        stage_name="single_stage",
                        epoch=int(args.epochs),
                        num_epochs=int(args.epochs),
                        extra={"timeout_file": str(timeout_path)},
                    )
                    print(
                        f"Warning: Falcor stage-end eval timeout timeout={falcor_periodic_eval_cfg.get('timeout_seconds', 1800)}s"
                    )
                    if bool(falcor_periodic_eval_cfg.get("fail_on_timeout", False)):
                        raise RuntimeError("Falcor stage-end eval timed out") from timeout_error
                except Exception as falcor_error:
                    _update_heartbeat(
                        heartbeat_path=heartbeat_path,
                        history_path=heartbeat_history_path,
                        enabled=heartbeat_enabled,
                        save_history=heartbeat_history_enabled,
                        run_context=run_context,
                        state="running",
                        last_event="falcor_stage_end_error",
                        stage_index=1,
                        stage_name="single_stage",
                        epoch=int(args.epochs),
                        num_epochs=int(args.epochs),
                        extra={"error": str(falcor_error)},
                    )
                    print(f"Warning: Falcor stage-end eval failed: {falcor_error}")
                    if bool(falcor_periodic_eval_cfg.get("fail_on_error", False)):
                        raise

        _update_heartbeat(
            heartbeat_path=heartbeat_path,
            history_path=heartbeat_history_path,
            enabled=heartbeat_enabled,
            save_history=heartbeat_history_enabled,
            run_context=run_context,
            state="running",
            last_event="stage_finished",
            stage_index=1,
            stage_name="single_stage",
            epoch=int(args.epochs),
            num_epochs=int(args.epochs),
        )

    if test_loader is not None and trainer is not None:
        test_metrics = trainer.validate_epoch(test_loader)
        print("Test metrics:", test_metrics)

    val_profile_test_compare: Dict[str, Any] | None = None
    val_profile_best_summary: Dict[str, Any] | None = None
    if bool(val_profiles_cfg.get("enabled", False)):
        profile_summary: Dict[str, Any] = {
            "best_metric": str(val_profiles_cfg.get("best_metric", "mae")),
            "profiles": {},
        }
        for profile in val_profiles_cfg.get("profiles", []):
            profile_name = str(profile.get("name", "profile"))
            ckpt_path = output_dir / _profile_checkpoint_filename(profile_name)
            entry: Dict[str, Any] = {
                "checkpoint": str(ckpt_path),
                "exists": bool(ckpt_path.exists()),
                "profile": profile,
            }
            if ckpt_path.exists():
                try:
                    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                    entry.update(
                        {
                            "epoch": int(ckpt.get("epoch", 0)),
                            "best_metric": ckpt.get("best_metric", None),
                            "best_metric_name": ckpt.get("best_metric_name", None),
                        }
                    )
                except Exception as summary_error:
                    entry["load_error"] = str(summary_error)
            profile_summary["profiles"][profile_name] = entry

        val_profile_best_summary = profile_summary
        (output_dir / "val_profile_best_summary.json").write_text(
            json.dumps(profile_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if (
        trainer is not None
        and bool(val_profiles_cfg.get("enabled", False))
        and bool(val_profiles_cfg.get("run_test_compare", False))
    ):
        try:
            _update_heartbeat(
                heartbeat_path=heartbeat_path,
                history_path=heartbeat_history_path,
                enabled=heartbeat_enabled,
                save_history=heartbeat_history_enabled,
                run_context=run_context,
                state="profile_test_compare_running",
                last_event="profile_test_compare_start",
                stage_index=int(len(staged_specs)) if staged_specs else 1,
                stage_name="staged" if staged_specs else "single_stage",
                epoch=int(args.epochs),
                num_epochs=int(args.epochs),
            )
            val_profile_test_compare = _run_profile_test_compare(
                trainer=trainer,
                model=model,
                test_loader=test_loader,
                output_dir=output_dir,
                profile_cfg=val_profiles_cfg,
            )
            if val_profile_test_compare is not None:
                _update_heartbeat(
                    heartbeat_path=heartbeat_path,
                    history_path=heartbeat_history_path,
                    enabled=heartbeat_enabled,
                    save_history=heartbeat_history_enabled,
                    run_context=run_context,
                    state="running",
                    last_event="profile_test_compare_success",
                    stage_index=int(len(staged_specs)) if staged_specs else 1,
                    stage_name="staged" if staged_specs else "single_stage",
                    epoch=int(args.epochs),
                    num_epochs=int(args.epochs),
                    extra={"summary": str(output_dir / "val_profile_test_compare.json")},
                )
        except Exception as compare_error:
            _update_heartbeat(
                heartbeat_path=heartbeat_path,
                history_path=heartbeat_history_path,
                enabled=heartbeat_enabled,
                save_history=heartbeat_history_enabled,
                run_context=run_context,
                state="running",
                last_event="profile_test_compare_error",
                stage_index=int(len(staged_specs)) if staged_specs else 1,
                stage_name="staged" if staged_specs else "single_stage",
                epoch=int(args.epochs),
                num_epochs=int(args.epochs),
                extra={"error": str(compare_error)},
            )
            print(f"Warning: val profile test compare failed: {compare_error}")

    coeff_suite_summary: Dict[str, Any] | None = None
    coeff_suite_multi_summary: Dict[str, Any] | None = None
    expert_util_summary: Dict[str, Any] | None = None
    semantic_drift_summary: Dict[str, Any] | None = None
    if bool(coeff_suite_cfg.get("enabled", False)) and bool(coeff_suite_cfg.get("run_after_training", True)):
        timeout_seconds = int(coeff_suite_cfg.get("timeout_seconds", 1800))
        configured_ckpts = [str(x) for x in coeff_suite_cfg.get("checkpoints", [])]
        auto_ckpts: List[str] = []
        if global_best_track_falcor:
            auto_ckpts.append(str(global_best_falcor_ckpt_path.name))
        if global_best_track_soft:
            auto_ckpts.append(str(global_best_soft_ckpt_path.name))
        auto_ckpts.extend(["best_model.pt", "last_model.pt"])
        raw_candidates = configured_ckpts if configured_ckpts else auto_ckpts

        def _resolve_suite_checkpoint(raw_name: str) -> Path:
            token = str(raw_name).strip()
            alias = token.lower()
            alias_map = {
                "global_best_falcor": global_best_falcor_ckpt_path,
                "global_best_falcor.pt": global_best_falcor_ckpt_path,
                "global_best_soft": global_best_soft_ckpt_path,
                "global_best_soft_profile": global_best_soft_ckpt_path,
                "global_best_soft.pt": global_best_soft_ckpt_path,
                "best_model": output_dir / "best_model.pt",
                "best_model.pt": output_dir / "best_model.pt",
                "last_model": output_dir / "last_model.pt",
                "last_model.pt": output_dir / "last_model.pt",
            }
            if alias in alias_map:
                return alias_map[alias]
            ckpt = Path(token)
            if not ckpt.is_absolute():
                ckpt = output_dir / ckpt
            return ckpt

        suite_runs: List[Dict[str, Any]] = []
        seen_suite_paths: set[str] = set()
        successful_runs = 0
        for raw_name in raw_candidates:
            suite_ckpt = _resolve_suite_checkpoint(raw_name)
            if suite_ckpt.exists():
                key = str(suite_ckpt.resolve())
            else:
                key = str(suite_ckpt)
            if key in seen_suite_paths:
                continue
            seen_suite_paths.add(key)

            label_base = str(Path(str(raw_name)).stem or str(raw_name))
            label = _sanitize_stage_name(label_base)
            if not suite_ckpt.exists():
                suite_runs.append(
                    {
                        "name": label,
                        "checkpoint": str(suite_ckpt),
                        "exists": False,
                        "status": "missing",
                    }
                )
                continue

            if successful_runs == 0:
                suite_out_dir = output_dir / "diagnostics" / "coeff_suite"
            else:
                suite_out_dir = output_dir / "diagnostics" / "coeff_suite_multi" / f"{successful_runs:02d}_{label}"
            try:
                _update_heartbeat(
                    heartbeat_path=heartbeat_path,
                    history_path=heartbeat_history_path,
                    enabled=heartbeat_enabled,
                    save_history=heartbeat_history_enabled,
                    run_context=run_context,
                    state="coeff_suite_running",
                    last_event="coeff_suite_start",
                    stage_index=int(len(staged_specs)) if staged_specs else 1,
                    stage_name="staged" if staged_specs else "single_stage",
                    epoch=int(args.epochs),
                    num_epochs=int(args.epochs),
                    extra={"checkpoint": str(suite_ckpt), "output_dir": str(suite_out_dir), "name": label},
                )
                summary_obj = _run_coeff_suite(
                    checkpoint_path=suite_ckpt,
                    output_dir=suite_out_dir,
                    args=args,
                    suite_cfg=coeff_suite_cfg,
                    timeout_seconds=timeout_seconds,
                )
                metrics_obj = _coeff_suite_metrics_from_summary(summary_obj or {})
                if rerun_logger is not None and metrics_obj:
                    rerun_logger.log_scalars(int(args.epochs), metrics_obj, f"coeff_suite/{label}")
                _append_jsonl(
                    phase0_metrics_path,
                    {
                        "timestamp": _now_iso(),
                        "kind": "coeff_suite",
                        "stage_index": int(len(staged_specs)) if staged_specs else 1,
                        "stage_name": "staged" if staged_specs else "single_stage",
                        "epoch": int(args.epochs),
                        "num_epochs": int(args.epochs),
                        "metrics": metrics_obj,
                        "summary": str(suite_out_dir / "phase1_summary.json"),
                        "checkpoint": str(suite_ckpt),
                        "name": label,
                    },
                )
                _update_heartbeat(
                    heartbeat_path=heartbeat_path,
                    history_path=heartbeat_history_path,
                    enabled=heartbeat_enabled,
                    save_history=heartbeat_history_enabled,
                    run_context=run_context,
                    state="running",
                    last_event="coeff_suite_success",
                    stage_index=int(len(staged_specs)) if staged_specs else 1,
                    stage_name="staged" if staged_specs else "single_stage",
                    epoch=int(args.epochs),
                    num_epochs=int(args.epochs),
                    metrics=metrics_obj,
                    extra={"summary": str(suite_out_dir / "phase1_summary.json"), "checkpoint": str(suite_ckpt), "name": label},
                )
                suite_runs.append(
                    {
                        "name": label,
                        "checkpoint": str(suite_ckpt),
                        "exists": True,
                        "status": "success",
                        "summary": str(suite_out_dir / "phase1_summary.json"),
                        "metrics": metrics_obj,
                    }
                )
                if coeff_suite_summary is None:
                    coeff_suite_summary = summary_obj
                successful_runs += 1
            except subprocess.TimeoutExpired as timeout_error:
                timeout_path = output_dir / "diagnostics" / "coeff_suite_multi" / f"{label}_timeout.json"
                timeout_payload = {
                    "kind": "coeff_suite_timeout",
                    "timeout_seconds": timeout_seconds,
                    "error": str(timeout_error),
                    "checkpoint": str(suite_ckpt),
                    "name": label,
                    "timestamp": _now_iso(),
                }
                timeout_path.parent.mkdir(parents=True, exist_ok=True)
                timeout_path.write_text(json.dumps(timeout_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                _update_heartbeat(
                    heartbeat_path=heartbeat_path,
                    history_path=heartbeat_history_path,
                    enabled=heartbeat_enabled,
                    save_history=heartbeat_history_enabled,
                    run_context=run_context,
                    state="running",
                    last_event="coeff_suite_timeout",
                    stage_index=int(len(staged_specs)) if staged_specs else 1,
                    stage_name="staged" if staged_specs else "single_stage",
                    epoch=int(args.epochs),
                    num_epochs=int(args.epochs),
                    extra={"timeout_file": str(timeout_path), "checkpoint": str(suite_ckpt), "name": label},
                )
                suite_runs.append(
                    {
                        "name": label,
                        "checkpoint": str(suite_ckpt),
                        "exists": True,
                        "status": "timeout",
                        "timeout_file": str(timeout_path),
                    }
                )
                print(f"Warning: coeff suite timed out timeout={timeout_seconds}s checkpoint={suite_ckpt}")
                if bool(coeff_suite_cfg.get("fail_on_timeout", False)):
                    raise RuntimeError("Coeff suite timed out") from timeout_error
            except Exception as coeff_error:
                _update_heartbeat(
                    heartbeat_path=heartbeat_path,
                    history_path=heartbeat_history_path,
                    enabled=heartbeat_enabled,
                    save_history=heartbeat_history_enabled,
                    run_context=run_context,
                    state="running",
                    last_event="coeff_suite_error",
                    stage_index=int(len(staged_specs)) if staged_specs else 1,
                    stage_name="staged" if staged_specs else "single_stage",
                    epoch=int(args.epochs),
                    num_epochs=int(args.epochs),
                    extra={"error": str(coeff_error), "checkpoint": str(suite_ckpt), "name": label},
                )
                suite_runs.append(
                    {
                        "name": label,
                        "checkpoint": str(suite_ckpt),
                        "exists": True,
                        "status": "error",
                        "error": str(coeff_error),
                    }
                )
                print(f"Warning: coeff suite failed checkpoint={suite_ckpt}: {coeff_error}")

        if suite_runs:
            coeff_suite_multi_summary = {
                "timestamp": _now_iso(),
                "configured_checkpoints": configured_ckpts,
                "resolved_default_checkpoints": auto_ckpts,
                "runs": suite_runs,
            }
            multi_summary_path = output_dir / "diagnostics" / "coeff_suite_multi" / "coeff_suite_multi_summary.json"
            multi_summary_path.parent.mkdir(parents=True, exist_ok=True)
            multi_summary_path.write_text(
                json.dumps(coeff_suite_multi_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _resolve_audit_profiles(explicit_profiles: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
        if explicit_profiles:
            return explicit_profiles
        if bool(val_profiles_cfg.get("enabled", False)):
            return list(val_profiles_cfg.get("profiles", []))
        return [
            {
                "name": f"hard{int(args.top_k)}",
                "top_k": int(args.top_k),
                "training_soft_routing": False,
                "routing_soft_topk": None,
                "routing_temperature": None,
            }
        ]

    best_ckpt = output_dir / "best_model.pt"
    last_ckpt = output_dir / "last_model.pt"

    if bool(expert_util_cfg.get("enabled", False)) and bool(expert_util_cfg.get("run_after_training", True)):
        cfg_ckpt = expert_util_cfg.get("checkpoint", None)
        if cfg_ckpt:
            ckpt_path = Path(str(cfg_ckpt))
            if not ckpt_path.is_absolute():
                ckpt_path = output_dir / ckpt_path
        else:
            ckpt_path = best_ckpt if best_ckpt.exists() else last_ckpt
        if ckpt_path.exists():
            audit_profiles = _resolve_audit_profiles(list(expert_util_cfg.get("profiles", [])))
            out_path = output_dir / "diagnostics" / "expert_utilization" / "expert_utilization_summary.json"
            timeout_seconds = int(expert_util_cfg.get("timeout_seconds", 900))
            try:
                _update_heartbeat(
                    heartbeat_path=heartbeat_path,
                    history_path=heartbeat_history_path,
                    enabled=heartbeat_enabled,
                    save_history=heartbeat_history_enabled,
                    run_context=run_context,
                    state="expert_utilization_audit_running",
                    last_event="expert_utilization_audit_start",
                    stage_index=int(len(staged_specs)) if staged_specs else 1,
                    stage_name="staged" if staged_specs else "single_stage",
                    epoch=int(args.epochs),
                    num_epochs=int(args.epochs),
                    extra={"checkpoint": str(ckpt_path), "summary": str(out_path)},
                )
                expert_util_summary = _run_expert_utilization_audit(
                    checkpoint_path=ckpt_path,
                    output_json_path=out_path,
                    args=args,
                    audit_cfg=expert_util_cfg,
                    profiles=audit_profiles,
                    timeout_seconds=timeout_seconds,
                )
                expert_metrics = _expert_util_metrics_from_summary(expert_util_summary or {})
                if rerun_logger is not None and expert_metrics:
                    rerun_logger.log_scalars(int(args.epochs), expert_metrics, "expert_utilization")
                _append_jsonl(
                    phase0_metrics_path,
                    {
                        "timestamp": _now_iso(),
                        "kind": "expert_utilization_audit",
                        "stage_index": int(len(staged_specs)) if staged_specs else 1,
                        "stage_name": "staged" if staged_specs else "single_stage",
                        "epoch": int(args.epochs),
                        "num_epochs": int(args.epochs),
                        "metrics": expert_metrics,
                        "summary": str(out_path),
                    },
                )
                _update_heartbeat(
                    heartbeat_path=heartbeat_path,
                    history_path=heartbeat_history_path,
                    enabled=heartbeat_enabled,
                    save_history=heartbeat_history_enabled,
                    run_context=run_context,
                    state="running",
                    last_event="expert_utilization_audit_success",
                    stage_index=int(len(staged_specs)) if staged_specs else 1,
                    stage_name="staged" if staged_specs else "single_stage",
                    epoch=int(args.epochs),
                    num_epochs=int(args.epochs),
                    metrics=expert_metrics,
                    extra={"summary": str(out_path)},
                )
            except subprocess.TimeoutExpired as timeout_error:
                timeout_path = output_dir / "diagnostics" / "expert_utilization_timeout.json"
                timeout_path.parent.mkdir(parents=True, exist_ok=True)
                timeout_path.write_text(
                    json.dumps(
                        {
                            "kind": "expert_utilization_timeout",
                            "timeout_seconds": timeout_seconds,
                            "error": str(timeout_error),
                            "timestamp": _now_iso(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"Warning: expert utilization audit timed out timeout={timeout_seconds}s")
                if bool(expert_util_cfg.get("fail_on_timeout", False)):
                    raise RuntimeError("expert utilization audit timed out") from timeout_error
            except Exception as audit_error:
                print(f"Warning: expert utilization audit failed: {audit_error}")

    if bool(semantic_drift_cfg.get("enabled", False)) and bool(semantic_drift_cfg.get("run_after_training", True)):
        pairs = list(semantic_drift_cfg.get("pairs", []))
        if not pairs and best_ckpt.exists() and last_ckpt.exists():
            pairs = [
                {
                    "name": "best_vs_last",
                    "checkpoint_a": str(best_ckpt),
                    "checkpoint_b": str(last_ckpt),
                }
            ]
        drift_profiles = _resolve_audit_profiles(list(semantic_drift_cfg.get("profiles", [])))
        pair_summaries: Dict[str, Any] = {}
        timeout_seconds = int(semantic_drift_cfg.get("timeout_seconds", 1200))
        for pair in pairs:
            pair_name = str(pair.get("name", "pair"))
            ckpt_a = Path(str(pair.get("checkpoint_a", "")))
            ckpt_b = Path(str(pair.get("checkpoint_b", "")))
            if not ckpt_a.is_absolute():
                ckpt_a = output_dir / ckpt_a
            if not ckpt_b.is_absolute():
                ckpt_b = output_dir / ckpt_b
            if not ckpt_a.exists() or not ckpt_b.exists():
                print(f"Warning: semantic drift pair skipped (missing ckpt): {pair_name}")
                continue
            out_path = output_dir / "diagnostics" / "semantic_drift" / f"{_sanitize_stage_name(pair_name)}.json"
            try:
                pair_summary = _run_semantic_drift_audit(
                    checkpoint_a=ckpt_a,
                    checkpoint_b=ckpt_b,
                    output_json_path=out_path,
                    pair_name=pair_name,
                    args=args,
                    audit_cfg=semantic_drift_cfg,
                    profiles=drift_profiles,
                    timeout_seconds=timeout_seconds,
                )
                if pair_summary is not None:
                    pair_summaries[pair_name] = pair_summary
            except subprocess.TimeoutExpired as timeout_error:
                print(f"Warning: semantic drift audit timed out for {pair_name}: {timeout_error}")
                if bool(semantic_drift_cfg.get("fail_on_timeout", False)):
                    raise RuntimeError(f"semantic drift audit timed out for {pair_name}") from timeout_error
            except Exception as drift_error:
                print(f"Warning: semantic drift audit failed for {pair_name}: {drift_error}")
        if pair_summaries:
            semantic_drift_summary = {
                "pairs": pair_summaries,
                "meta": {
                    "num_pairs": int(len(pair_summaries)),
                },
            }
            summary_path = output_dir / "diagnostics" / "semantic_drift" / "semantic_drift_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(semantic_drift_summary, ensure_ascii=False, indent=2), encoding="utf-8")

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
        if isinstance(val_profile_test_compare, dict):
            for profile_name, payload in (val_profile_test_compare.get("results", {}) or {}).items():
                if not isinstance(payload, dict):
                    continue
                metrics = payload.get("metrics", {})
                if not isinstance(metrics, dict):
                    continue
                safe_name = _sanitize_stage_name(profile_name)
                for key, value in metrics.items():
                    result_metrics[f"test_profile_{safe_name}_{key}"] = value
        if coeff_suite_summary is not None:
            coeff_metrics = _coeff_suite_metrics_from_summary(coeff_suite_summary)
            result_metrics.update({f"coeff_suite_{k}": v for k, v in coeff_metrics.items()})
        if isinstance(coeff_suite_multi_summary, dict):
            runs = coeff_suite_multi_summary.get("runs", [])
            if isinstance(runs, list):
                result_metrics["coeff_suite_multi_runs"] = len(runs)
                result_metrics["coeff_suite_multi_success"] = sum(
                    1 for r in runs if isinstance(r, dict) and r.get("status") == "success"
                )
        if expert_util_summary is not None:
            expert_metrics = _expert_util_metrics_from_summary(expert_util_summary)
            result_metrics.update({f"expert_util_{k}": v for k, v in expert_metrics.items()})
        if isinstance(semantic_drift_summary, dict):
            for pair_name, pair_summary in (semantic_drift_summary.get("pairs", {}) or {}).items():
                if not isinstance(pair_summary, dict):
                    continue
                safe_pair = _sanitize_stage_name(pair_name)
                drift_metrics = _semantic_drift_metrics_from_summary(pair_summary)
                for key, value in drift_metrics.items():
                    result_metrics[f"semantic_drift_{safe_pair}_{key}"] = value
        if falcor_best_summary_path.exists():
            try:
                falcor_best_obj = json.loads(falcor_best_summary_path.read_text(encoding="utf-8"))
                result_metrics["falcor_best_metric_name"] = falcor_best_obj.get("metric_name")
                result_metrics["falcor_best_metric_value"] = falcor_best_obj.get("value")
                result_metrics["falcor_best_stage_index"] = falcor_best_obj.get("stage_index")
                result_metrics["falcor_best_epoch"] = falcor_best_obj.get("epoch")
            except Exception:
                pass
        if global_best_falcor_summary_path.exists():
            try:
                global_falcor_obj = json.loads(global_best_falcor_summary_path.read_text(encoding="utf-8"))
                result_metrics["global_falcor_best_metric_name"] = global_falcor_obj.get("metric_name")
                result_metrics["global_falcor_best_metric_value"] = global_falcor_obj.get("value")
                result_metrics["global_falcor_best_stage_index"] = global_falcor_obj.get("stage_index")
                result_metrics["global_falcor_best_epoch"] = global_falcor_obj.get("epoch")
            except Exception:
                pass
        if global_best_soft_summary_path.exists():
            try:
                global_soft_obj = json.loads(global_best_soft_summary_path.read_text(encoding="utf-8"))
                result_metrics["global_soft_best_metric_name"] = global_soft_obj.get("metric_name")
                result_metrics["global_soft_best_metric_value"] = global_soft_obj.get("value")
                result_metrics["global_soft_best_stage_index"] = global_soft_obj.get("stage_index")
                result_metrics["global_soft_best_epoch"] = global_soft_obj.get("epoch")
                result_metrics["global_soft_best_profile_name"] = global_soft_obj.get("profile_name")
            except Exception:
                pass
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
            "num_workers": args.num_workers,
            "dataloader_timeout_seconds": int(getattr(args, "dataloader_timeout_seconds", 0)),
            "dataloader_prefetch_factor": getattr(args, "dataloader_prefetch_factor", None),
            "dataloader_persistent_workers": getattr(args, "dataloader_persistent_workers", None),
            "dataloader_multiprocessing_context": str(getattr(args, "dataloader_multiprocessing_context", "none")),
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lr_scheduler": args.lr_scheduler,
            "lr_min": args.lr_min,
            "warmup_epochs": warmup_epochs,
            "recon_loss": args.recon_loss,
            "charbonnier_eps": args.charbonnier_eps,
            "lambda_temporal": args.lambda_temporal,
            "lambda_linearity": args.lambda_linearity,
            "linearity_aug_pairs": args.linearity_aug_pairs,
            "lambda_spatial": args.lambda_spatial,
            "spatial_k": args.spatial_k,
            "lambda_image": args.lambda_image,
            "image_loss_warmup_epochs": int(proxy_image_loss_cfg.get("warmup_epochs", 0)),
            "lambda_routing_balance": args.lambda_routing_balance,
            "routing_soft_train": args.routing_soft_train,
            "routing_soft_topk": args.routing_soft_topk,
            "routing_temp_start": args.routing_temp_start,
            "routing_temp_end": args.routing_temp_end,
            "routing_temp_anneal_epochs": args.routing_temp_anneal_epochs,
            "image_loss_type": args.image_loss_type,
            "image_samples": args.image_samples,
            "image_sample_seed": args.image_sample_seed,
            "image_loss_space": args.image_loss_space,
            "enable_weighted_sh_loss": args.enable_weighted_sh_loss,
            "sh_loss_weights": args.sh_loss_weights,
            "sh_weight_mode": args.sh_weight_mode,
            "resume": bool(resume_requested),
            "resume_checkpoint": str(resume_checkpoint_path) if resume_checkpoint_path is not None else None,
            "resume_stage_index": int(resume_stage_index) if resume_stage_index is not None else None,
            "resume_stage_epoch": int(resume_stage_epoch),
            "resume_save_every": int(getattr(args, "resume_save_every", 1)),
            "resume_checkpoint_name": str(getattr(args, "resume_checkpoint_name", "resume_latest.pt")),
            "light_dim": args.light_dim,
            "embed_dim": args.embed_dim,
            "intensity_dim": args.intensity_dim,
            "intensity_offset": args.intensity_offset,
            "light_encoder_mode": str(getattr(model, "light_encoder_mode", getattr(args, "light_encoder_mode", "normal"))),
            "bypass_feature_pairs": [
                [int(p[0]), int(p[1])] for p in list(getattr(model, "bypass_feature_pairs", []))
            ],
            "bypass_feature_norm_mean": [
                float(x) for x in list(getattr(model, "bypass_feature_norm_mean", []))
            ],
            "bypass_feature_norm_std": [
                float(x) for x in list(getattr(model, "bypass_feature_norm_std", []))
            ],
            "ema_enabled": bool(ema_cfg.get("enabled", False)),
            "ema_decay": float(ema_cfg.get("decay", 0.999)),
            "ema_eval_on_ema": bool(ema_cfg.get("eval_on_ema", False)),
            "ema_save_best_with_ema": bool(ema_cfg.get("save_best_with_ema", False)),
            "oracle_monitor_enabled": bool(oracle_monitor_cfg.get("enabled", False)),
            "oracle_monitor_period_epochs": int(oracle_monitor_cfg.get("period_epochs", 0)),
            "oracle_monitor_lightweight_max_samples": int(oracle_monitor_cfg.get("lightweight_max_samples", 0)),
            "oracle_monitor_stage_end_full": bool(oracle_monitor_cfg.get("stage_end_full", False)),
            "oracle_monitor_timeout_seconds": int(oracle_monitor_cfg.get("timeout_seconds", 0)),
            "oracle_monitor_fail_on_timeout": bool(oracle_monitor_cfg.get("fail_on_timeout", False)),
            "coeff_suite_enabled": bool(coeff_suite_cfg.get("enabled", False)),
            "coeff_suite_run_after_training": bool(coeff_suite_cfg.get("run_after_training", False)),
            "coeff_suite_split": str(coeff_suite_cfg.get("split", "test")),
            "coeff_suite_device": str(coeff_suite_cfg.get("device", "cpu")),
            "coeff_suite_max_samples": int(coeff_suite_cfg.get("max_samples", 0)),
            "coeff_suite_timeout_seconds": int(coeff_suite_cfg.get("timeout_seconds", 0)),
            "coeff_suite_fail_on_timeout": bool(coeff_suite_cfg.get("fail_on_timeout", False)),
            "coeff_suite_checkpoints": list(coeff_suite_cfg.get("checkpoints", [])),
            "global_best_enabled": bool(global_best_cfg.get("enabled", True)),
            "global_best_track_falcor": bool(global_best_cfg.get("track_falcor", True)),
            "global_best_track_soft_profile": bool(global_best_cfg.get("track_soft_profile", True)),
            "global_best_soft_profile_name": str(global_best_cfg.get("soft_profile_name", "soft8_t018")),
            "global_best_soft_metric": str(global_best_cfg.get("soft_metric", "img_psnr")),
            "global_best_soft_metric_maximize": bool(global_best_cfg.get("soft_metric_maximize", True)),
            "val_profiles_enabled": bool(val_profiles_cfg.get("enabled", False)),
            "val_profiles_best_metric": str(val_profiles_cfg.get("best_metric", "mae")),
            "val_profiles_names": [str(p.get("name", "profile")) for p in val_profiles_cfg.get("profiles", [])],
            "val_profiles_run_test_compare": bool(val_profiles_cfg.get("run_test_compare", False)),
            "proxy_image_loss_enabled": bool(proxy_image_loss_cfg.get("enabled", False)),
            "proxy_image_loss_lambda": float(proxy_image_loss_cfg.get("lambda", 0.0)),
            "proxy_image_loss_warmup_epochs": int(proxy_image_loss_cfg.get("warmup_epochs", 0)),
            "proxy_image_loss_mix_mode": str(proxy_image_loss_cfg.get("mix_mode", "linear_log_mix")),
            "proxy_image_loss_log_mix_weight": float(proxy_image_loss_cfg.get("log_mix_weight", 0.3)),
            "routing_balance_anneal_enabled": bool(routing_balance_anneal_cfg.get("enabled", False)),
            "routing_balance_anneal_start": float(routing_balance_anneal_cfg.get("start", args.lambda_routing_balance)),
            "routing_balance_anneal_end": float(routing_balance_anneal_cfg.get("end", 0.0)),
            "routing_balance_anneal_decay_end_ratio": float(routing_balance_anneal_cfg.get("decay_end_ratio", 0.6)),
            "routing_balance_anneal_decay_end_epoch": int(routing_balance_decay_end_epoch),
            "routing_balance_anneal_epochs_global": int(routing_balance_anneal_epochs_global),
            "routing_balance_anneal_total_training_epochs": int(total_training_epochs_global),
            "falcor_periodic_eval_enabled": bool(falcor_periodic_eval_cfg.get("enabled", False)),
            "falcor_periodic_eval_every_n_epochs": int(falcor_periodic_eval_cfg.get("every_n_epochs", 0)),
            "falcor_periodic_eval_stage_end_full": bool(falcor_periodic_eval_cfg.get("stage_end_full", False)),
            "falcor_periodic_eval_metric": str(falcor_periodic_eval_cfg.get("best_metric", "")),
            "falcor_periodic_eval_profile": falcor_periodic_eval_cfg.get("profile", {}),
            "expert_utilization_audit_enabled": bool(expert_util_cfg.get("enabled", False)),
            "expert_utilization_audit_run_after_training": bool(expert_util_cfg.get("run_after_training", False)),
            "expert_utilization_audit_split": str(expert_util_cfg.get("split", "test")),
            "expert_utilization_audit_device": str(expert_util_cfg.get("device", "cpu")),
            "semantic_drift_audit_enabled": bool(semantic_drift_cfg.get("enabled", False)),
            "semantic_drift_audit_run_after_training": bool(semantic_drift_cfg.get("run_after_training", False)),
            "semantic_drift_audit_split": str(semantic_drift_cfg.get("split", "test")),
            "semantic_drift_audit_device": str(semantic_drift_cfg.get("device", "cpu")),
            "heartbeat_enabled": bool(heartbeat_enabled),
            "heartbeat_history": bool(heartbeat_history_enabled),
            "group_lrs": global_group_lrs,
            "staged_enabled": bool(len(staged_specs) > 0),
            "staged_num_stages": int(len(staged_specs)),
            "staged_trainable_groups": [spec["trainable_groups"] for spec in staged_specs],
            "staged_group_lrs": [spec.get("group_lrs", {}) for spec in staged_specs],
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

    _finalize_falcor_resources(block=True)
    heartbeat_done["value"] = True
    _update_heartbeat(
        heartbeat_path=heartbeat_path,
        history_path=heartbeat_history_path,
        enabled=heartbeat_enabled,
        save_history=heartbeat_history_enabled,
        run_context=run_context,
        state="finished",
        last_event="train_end",
        stage_index=int(len(staged_specs)) if staged_specs else 1,
        stage_name="staged" if staged_specs else "single_stage",
        epoch=int(args.epochs),
        num_epochs=int(args.epochs),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Gaussian-Physics trainer")

    parser.add_argument("--variant", choices=["unified_set"], required=False)
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
    parser.add_argument("--dataloader-timeout-seconds", type=int, default=0,
                        help="DataLoader timeout (seconds). 0 disables timeout.")
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=None,
                        help="Optional DataLoader prefetch_factor when num_workers > 0.")
    parser.add_argument("--dataloader-persistent-workers", action="store_true", dest="dataloader_persistent_workers",
                        help="Enable DataLoader persistent_workers when num_workers > 0.")
    parser.add_argument("--no-dataloader-persistent-workers", action="store_false", dest="dataloader_persistent_workers",
                        help="Disable DataLoader persistent_workers.")
    parser.set_defaults(dataloader_persistent_workers=None)
    parser.add_argument("--dataloader-multiprocessing-context", choices=["none", "fork", "spawn", "forkserver"], default="none",
                        help="Multiprocessing context for DataLoader workers when num_workers > 0.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine", "plateau"], default="none")
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Linear warmup epochs before scheduler stepping.")
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
    parser.add_argument("--lambda-image", type=float, default=0.0,
                        help="Weight for differentiable SH image-space loss.")
    parser.add_argument("--image-loss-warmup-epochs", type=int, default=0,
                        help="Linear warmup epochs for image loss weight (0 disables warmup).")
    parser.add_argument("--lambda-routing-balance", type=float, default=0.0,
                        help="Weight for routing load-balance regularization (CV^2 usage loss).")
    parser.add_argument("--routing-soft-train", action="store_true",
                        help="Enable training-time soft routing (inference remains hard top-k).")
    parser.add_argument("--no-routing-soft-train", action="store_false", dest="routing_soft_train",
                        help="Disable training-time soft routing.")
    parser.set_defaults(routing_soft_train=False)
    parser.add_argument("--routing-soft-topk", type=int, default=0,
                        help="Candidate gaussian count for soft routing (0 means all gaussians).")
    parser.add_argument("--routing-temp-start", type=float, default=1.0,
                        help="Starting soft-routing temperature for annealing.")
    parser.add_argument("--routing-temp-end", type=float, default=1.0,
                        help="Ending soft-routing temperature.")
    parser.add_argument("--routing-temp-anneal-epochs", type=int, default=0,
                        help="Temperature anneal epochs for soft routing (0 disables anneal).")
    parser.add_argument("--amp-mode", choices=["off", "bf16", "fp16"], default="off",
                        help="Mixed precision mode for train/val forward.")
    parser.add_argument("--torch-compile", action="store_true", dest="torch_compile",
                        help="Enable torch.compile for model forward.")
    parser.add_argument("--no-torch-compile", action="store_false", dest="torch_compile",
                        help="Disable torch.compile.")
    parser.set_defaults(torch_compile=False)
    parser.add_argument("--torch-compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="reduce-overhead",
                        help="torch.compile mode.")
    parser.add_argument("--torch-compile-dynamic", action="store_true", dest="torch_compile_dynamic",
                        help="Enable dynamic shape support in torch.compile.")
    parser.add_argument("--no-torch-compile-dynamic", action="store_false", dest="torch_compile_dynamic",
                        help="Disable dynamic shape support in torch.compile.")
    parser.set_defaults(torch_compile_dynamic=False)
    parser.add_argument("--grad-norm-log-every-steps", type=int, default=1,
                        help="Log gradient norms every N steps (0 disables grad norm logging).")
    parser.add_argument("--max-train-batches", type=int, default=0,
                        help="Optional cap on train batches per epoch (0 disables).")
    parser.add_argument("--max-val-batches", type=int, default=0,
                        help="Optional cap on val batches per epoch (0 disables).")
    parser.add_argument("--ema-raw-eval-every-epochs", type=int, default=1,
                        help="When EMA eval is enabled, run raw eval every N epochs.")
    parser.add_argument("--enable-soft-profile-sharing", action="store_true", dest="enable_soft_profile_sharing",
                        help="Enable strict-equivalence soft profile shared forward in validation.")
    parser.add_argument("--no-soft-profile-sharing", action="store_false", dest="enable_soft_profile_sharing",
                        help="Disable soft profile shared forward.")
    parser.set_defaults(enable_soft_profile_sharing=False)
    parser.add_argument("--soft-profile-equiv-check-batches", type=int, default=2,
                        help="Equivalence guard batches for soft profile shared forward.")
    parser.add_argument("--soft-profile-mae-tolerance", type=float, default=1e-6,
                        help="Max allowed MAE delta between shared and legacy soft profile paths.")
    parser.add_argument("--soft-profile-img-psnr-tolerance", type=float, default=5e-4,
                        help="Max allowed image-PSNR delta between shared and legacy soft profile paths.")
    parser.add_argument("--image-loss-type", choices=["mse", "charbonnier"], default="mse",
                        help="Loss type for image-space supervision.")
    parser.add_argument("--image-samples", type=int, default=64,
                        help="Number of spherical directions sampled per batch for image loss.")
    parser.add_argument("--image-sample-seed", type=int, default=42,
                        help="Random seed for image-loss direction sampling.")
    parser.add_argument("--image-loss-space", choices=["linear", "srgb"], default="linear",
                        help="Image space used for image loss/metrics.")
    parser.add_argument("--enable-weighted-sh-loss", action="store_true",
                        help="Enable basis-weighted SH reconstruction loss.")
    parser.add_argument("--no-weighted-sh-loss", action="store_false", dest="enable_weighted_sh_loss")
    parser.set_defaults(enable_weighted_sh_loss=False)
    parser.add_argument("--sh-loss-weights", type=float, nargs=9, default=None,
                        help="Nine basis weights for weighted SH loss.")
    parser.add_argument("--sh-weight-mode", choices=["basis"], default="basis",
                        help="Weight broadcast mode for SH coefficients.")
    parser.add_argument("--val-image-metrics", action="store_true",
                        help="Compute image-space metrics in validation (slower).")
    parser.add_argument("--no-val-image-metrics", action="store_false", dest="val_image_metrics")
    parser.set_defaults(val_image_metrics=False)
    parser.add_argument("--val-superposition", action="store_true",
                        help="Compute superposition metric in validation (extra forwards).")
    parser.add_argument("--no-val-superposition", action="store_false", dest="val_superposition")
    parser.set_defaults(val_superposition=False)

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
    parser.add_argument("--light-encoder-mode", choices=["normal", "bypass_fixed", "bypass_linear", "bypass_mlp_16_32"], default="normal",
                        help="Light encoder mode: normal learned encoder or bypass modes for diagnostics.")
    parser.add_argument("--bypass-feature-pairs", type=int, nargs="*", default=None,
                        help="Bypass feature pairs as flattened indices: l0 d0 l1 d1 ...")
    parser.add_argument("--bypass-feature-norm-mean", type=float, nargs="*", default=None,
                        help="Optional z-score mean per bypass feature.")
    parser.add_argument("--bypass-feature-norm-std", type=float, nargs="*", default=None,
                        help="Optional z-score std per bypass feature.")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to checkpoint (.pt) to load model weights from.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Strict resume checkpoint (.pt): restore model/optimizer/scheduler/EMA/stage progress.")
    parser.add_argument("--resume-save-every", type=int, default=1,
                        help="Save strict-resume checkpoint every N epochs (0 disables periodic resume snapshots).")
    parser.add_argument("--resume-checkpoint-name", type=str, default="resume_latest.pt",
                        help="Filename for periodic strict-resume checkpoint snapshots.")

    parser.add_argument("--enable-sh-scaler", action="store_true",
                        help="Enable adaptive SH scaling (L0 mean/std + higher-order RMS).")
    parser.add_argument("--sh-scaler-path", type=str, default=None,
                        help="Path to load/save SH scaler (npz).")
    parser.add_argument("--sh-scaler-max-samples", type=int, default=200000,
                        help="Max samples to estimate SH scaler statistics.")

    parser.add_argument("--no-init", action="store_true", help="Skip K-Means initialization")
    parser.add_argument("--print-history", action="store_true")

    parser.add_argument("--heartbeat-enabled", action="store_true", dest="heartbeat_enabled",
                        help="Write heartbeat status file during training.")
    parser.add_argument("--no-heartbeat", action="store_false", dest="heartbeat_enabled",
                        help="Disable heartbeat status file.")
    parser.set_defaults(heartbeat_enabled=True)
    parser.add_argument("--heartbeat-history", action="store_true", dest="heartbeat_history",
                        help="Append heartbeat history events to jsonl.")
    parser.add_argument("--no-heartbeat-history", action="store_false", dest="heartbeat_history",
                        help="Disable heartbeat history jsonl.")
    parser.set_defaults(heartbeat_history=True)

    # Rerun visualization arguments
    parser.add_argument("--enable-rerun", action="store_true", dest="enable_rerun",
                       help="Enable Rerun visualization (default off)")
    parser.add_argument("--no-rerun", action="store_false", dest="enable_rerun",
                       help="Disable Rerun visualization")
    parser.set_defaults(enable_rerun=False)
    parser.add_argument("--rerun-log-freq", type=int, default=10,
                       help="Log visualizations every N epochs")
    parser.add_argument("--rerun-save-path", type=str, default=None,
                       help="Save .rrd file to path (default: output_dir/train.rrd)")

    # Progress display
    parser.add_argument("--show-progress", action="store_true", dest="show_progress",
                       help="Enable progress bars and init status logs")
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
    apply_variant_defaults_from_profile(args)


if __name__ == "__main__":  # pragma: no cover
    parser = build_arg_parser()
    defaults = parser.parse_args([])
    args = parser.parse_args()

    if args.config:
        cfg = _load_yaml(args.config)
        args._config_obj = cfg
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
