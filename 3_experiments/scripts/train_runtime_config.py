"""Runtime config resolvers for the unified training entrypoint.

This module centralizes YAML->runtime config normalization so train.py can stay
focused on orchestration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


_VAL_PROFILE_ALLOWED_METRICS = {
    "mae",
    "rmse",
    "charbonnier",
    "img_mae",
    "img_rmse",
    "img_psnr",
}


def _as_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _get_training_cfg(args: argparse.Namespace) -> Dict[str, Any] | None:
    cfg = getattr(args, "_config_obj", None)
    if not isinstance(cfg, dict):
        return None
    training_cfg = cfg.get("training")
    if not isinstance(training_cfg, dict):
        return None
    return training_cfg


def resolve_staged_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    training_cfg = _get_training_cfg(args)
    if training_cfg is None:
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

        specs.append(
            {
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
        )

    return specs


def resolve_ema_config(args: argparse.Namespace) -> Dict[str, Any]:
    raw_eval_default = max(1, int(getattr(args, "ema_raw_eval_every_epochs", 1)))
    defaults = {
        "enabled": False,
        "decay": 0.999,
        "eval_on_ema": False,
        "save_best_with_ema": False,
        "raw_eval_every_epochs": raw_eval_default,
    }

    training_cfg = _get_training_cfg(args)
    if training_cfg is None:
        return defaults
    ema_cfg = training_cfg.get("ema")
    if not isinstance(ema_cfg, dict):
        return defaults

    enabled = bool(ema_cfg.get("enabled", False))
    return {
        "enabled": enabled,
        "decay": float(ema_cfg.get("decay", 0.999)),
        "eval_on_ema": bool(ema_cfg.get("eval_on_ema", True if enabled else False)),
        "save_best_with_ema": bool(ema_cfg.get("save_best_with_ema", True if enabled else False)),
        "raw_eval_every_epochs": max(1, int(ema_cfg.get("raw_eval_every_epochs", raw_eval_default))),
    }


def resolve_oracle_monitor_config(args: argparse.Namespace) -> Dict[str, Any]:
    training_cfg = _get_training_cfg(args)
    if training_cfg is None:
        return {"enabled": False}

    oracle_cfg = training_cfg.get("oracle_monitor")
    if not isinstance(oracle_cfg, dict):
        return {"enabled": False}
    if not bool(oracle_cfg.get("enabled", False)):
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


def resolve_coeff_suite_config(args: argparse.Namespace) -> Dict[str, Any]:
    training_cfg = _get_training_cfg(args)
    if training_cfg is None:
        return {"enabled": False}

    suite_cfg = training_cfg.get("coeff_suite")
    if not isinstance(suite_cfg, dict):
        return {"enabled": False}
    if not bool(suite_cfg.get("enabled", False)):
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


def resolve_global_group_lrs(args: argparse.Namespace) -> Dict[str, float]:
    training_cfg = _get_training_cfg(args)
    if training_cfg is None:
        return {}
    raw = training_cfg.get("group_lrs")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("training.group_lrs must be a mapping")
    return {str(k): float(v) for k, v in raw.items()}


def resolve_global_best_config(args: argparse.Namespace) -> Dict[str, Any]:
    defaults = {
        "enabled": True,
        "track_falcor": True,
        "track_soft_profile": True,
        "soft_profile_name": "soft8_t018",
        "soft_metric": "img_psnr",
        "soft_metric_maximize": True,
    }
    training_cfg = _get_training_cfg(args)
    if training_cfg is None:
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


def resolve_proxy_image_loss_config(args: argparse.Namespace) -> Dict[str, Any]:
    lambda_default = float(getattr(args, "lambda_image", 0.0))
    training_cfg = _get_training_cfg(args)
    loss_weights_cfg = _as_mapping(training_cfg.get("loss_weights")) if training_cfg is not None else {}
    image_weight_override = loss_weights_cfg.get("image")
    if image_weight_override is not None:
        lambda_default = float(image_weight_override)
    warmup_default = int(getattr(args, "image_loss_warmup_epochs", 0))
    mix_mode_default = str(getattr(args, "proxy_image_mix_mode", "linear_log_mix")).strip().lower()
    log_mix_weight_default = float(getattr(args, "proxy_image_log_mix_weight", 0.3))
    loss_type_default = str(getattr(args, "image_loss_type", "mse")).strip().lower()
    huber_delta_default = max(1e-8, float(getattr(args, "image_loss_huber_delta", 0.1)))
    domain_default = str(getattr(args, "image_loss_domain", "radiance")).strip().lower()
    sampling_mode_default = str(getattr(args, "image_sampling_mode", "fixed")).strip().lower()
    soft_sat_enabled_default = bool(getattr(args, "image_soft_saturation_enabled", False))
    soft_sat_mode_default = str(getattr(args, "image_soft_saturation_mode", "exp")).strip().lower()
    soft_sat_k_default = max(1e-8, float(getattr(args, "image_soft_saturation_k", 1.0)))

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

    def _normalize_loss_type(raw: Any) -> str:
        value = str(raw if raw is not None else loss_type_default).strip().lower()
        if value not in {"mse", "charbonnier", "huber"}:
            raise ValueError("training.proxy_image_loss.loss_type must be one of: mse, charbonnier, huber")
        return value

    def _normalize_sampling_mode(raw: Any) -> str:
        value = str(raw if raw is not None else sampling_mode_default).strip().lower()
        if value not in {"fixed", "per_epoch", "per_step"}:
            raise ValueError("training.proxy_image_loss.sampling_mode must be one of: fixed, per_epoch, per_step")
        return value

    def _normalize_domain(raw: Any) -> str:
        value = str(raw if raw is not None else domain_default).strip().lower()
        if value not in {"radiance", "irradiance"}:
            raise ValueError("training.proxy_image_loss.domain must be one of: radiance, irradiance")
        return value

    def _normalize_soft_sat_mode(raw: Any) -> str:
        value = str(raw if raw is not None else soft_sat_mode_default).strip().lower()
        if value not in {"exp"}:
            raise ValueError("training.proxy_image_loss.soft_saturation.mode must be one of: exp")
        return value

    def _normalize_positive(raw: Any, default: float) -> float:
        value = float(raw if raw is not None else default)
        return max(1e-8, value)

    defaults = {
        "enabled": bool(lambda_default > 0.0),
        "lambda": lambda_default,
        "warmup_epochs": warmup_default,
        "enable_val_image_metrics": bool(lambda_default > 0.0),
        "mix_mode": _normalize_mix_mode(mix_mode_default),
        "log_mix_weight": _normalize_weight(log_mix_weight_default),
        "loss_type": _normalize_loss_type(loss_type_default),
        "huber_delta": _normalize_positive(huber_delta_default, huber_delta_default),
        "domain": _normalize_domain(domain_default),
        "sampling_mode": _normalize_sampling_mode(sampling_mode_default),
        "soft_saturation_enabled": bool(soft_sat_enabled_default),
        "soft_saturation_mode": _normalize_soft_sat_mode(soft_sat_mode_default),
        "soft_saturation_k": _normalize_positive(soft_sat_k_default, soft_sat_k_default),
    }

    if training_cfg is None:
        return defaults
    raw = training_cfg.get("proxy_image_loss")
    if not isinstance(raw, dict):
        return defaults

    lambda_value = float(raw.get("lambda", lambda_default))
    if image_weight_override is not None:
        lambda_value = float(image_weight_override)
    enabled = bool(raw.get("enabled", lambda_value > 0.0))
    soft_sat_raw = raw.get("soft_saturation", {})
    if soft_sat_raw is None:
        soft_sat_raw = {}
    if not isinstance(soft_sat_raw, dict):
        raise ValueError("training.proxy_image_loss.soft_saturation must be a mapping when provided")

    return {
        "enabled": enabled,
        "lambda": float(lambda_value),
        "warmup_epochs": int(raw.get("warmup_epochs", warmup_default)),
        "enable_val_image_metrics": bool(raw.get("enable_val_image_metrics", True if enabled else False)),
        "mix_mode": _normalize_mix_mode(raw.get("mix_mode", raw.get("mode", mix_mode_default))),
        "log_mix_weight": _normalize_weight(raw.get("log_mix_weight", raw.get("mix_weight", log_mix_weight_default))),
        "loss_type": _normalize_loss_type(raw.get("loss_type", raw.get("image_loss_type", loss_type_default))),
        "huber_delta": _normalize_positive(raw.get("huber_delta", raw.get("image_loss_huber_delta", huber_delta_default)), huber_delta_default),
        "domain": _normalize_domain(raw.get("domain", raw.get("image_loss_domain", domain_default))),
        "sampling_mode": _normalize_sampling_mode(raw.get("sampling_mode", raw.get("image_sampling_mode", sampling_mode_default))),
        "soft_saturation_enabled": bool(
            soft_sat_raw.get(
                "enabled",
                raw.get("soft_saturation_enabled", soft_sat_enabled_default),
            )
        ),
        "soft_saturation_mode": _normalize_soft_sat_mode(
            soft_sat_raw.get(
                "mode",
                raw.get("soft_saturation_mode", soft_sat_mode_default),
            )
        ),
        "soft_saturation_k": _normalize_positive(
            soft_sat_raw.get(
                "k",
                raw.get("soft_saturation_k", soft_sat_k_default),
            ),
            soft_sat_k_default,
        ),
    }


def resolve_gbuffer_image_loss_config(args: argparse.Namespace) -> Dict[str, Any]:
    lambda_default = float(getattr(args, "gbuffer_image_loss_lambda", 0.0))
    training_cfg = _get_training_cfg(args)
    loss_weights_cfg = _as_mapping(training_cfg.get("loss_weights")) if training_cfg is not None else {}
    gbuffer_weight_override = loss_weights_cfg.get("gbuffer")
    if gbuffer_weight_override is not None:
        lambda_default = float(gbuffer_weight_override)
    warmup_default = int(getattr(args, "gbuffer_image_loss_warmup_epochs", 0))
    every_steps_default = int(getattr(args, "gbuffer_image_loss_every_steps", 16))
    loss_type_default = str(getattr(args, "gbuffer_image_loss_type", "charbonnier")).strip().lower()
    pixel_sample_count_default = int(getattr(args, "gbuffer_image_loss_pixel_sample_count", 8192))
    dataset_root_default = str(getattr(args, "gbuffer_image_loss_dataset_root", "") or "").strip()
    enabled_default = bool(getattr(args, "gbuffer_image_loss_enabled", False))
    domain_default = str(getattr(args, "gbuffer_image_loss_domain", "linear")).strip().lower()
    gt_color_space_default = str(getattr(args, "gbuffer_image_loss_gt_color_space", "linear")).strip().lower()
    strict_keys_default = bool(getattr(args, "gbuffer_image_loss_strict_keys", True))
    pos_key_default = str(getattr(args, "gbuffer_image_loss_pos_key", "posW") or "").strip() or "posW"
    normal_key_default = str(getattr(args, "gbuffer_image_loss_normal_key", "normW") or "").strip() or "normW"
    albedo_key_default = str(getattr(args, "gbuffer_image_loss_albedo_key", "albedo") or "").strip() or "albedo"
    gt_linear_key_default = str(getattr(args, "gbuffer_image_loss_gt_linear_key", "gt_linear") or "").strip() or "gt_linear"
    light_params_key_default = str(getattr(args, "gbuffer_image_loss_light_params_key", "light_params") or "").strip() or "light_params"
    light_mask_key_default = str(getattr(args, "gbuffer_image_loss_light_mask_key", "light_mask") or "").strip() or "light_mask"
    valid_mask_key_default = str(getattr(args, "gbuffer_image_loss_valid_mask_key", "valid_mask") or "").strip() or "valid_mask"
    frame_idx_key_default = str(getattr(args, "gbuffer_image_loss_frame_idx_key", "frame_idx") or "").strip() or "frame_idx"
    config_idx_key_default = str(getattr(args, "gbuffer_image_loss_config_idx_key", "config_idx") or "").strip() or "config_idx"
    target_source_default = (
        str(getattr(args, "gbuffer_image_loss_target_source", "dataset_sh") or "").strip().lower() or "dataset_sh"
    )
    gt_knn_default = int(getattr(args, "gbuffer_image_loss_gt_knn", 8))
    gt_weight_eps_default = float(getattr(args, "gbuffer_image_loss_gt_weight_eps", 0.1))
    gt_chunk_size_default = int(getattr(args, "gbuffer_image_loss_gt_chunk_size", 32768))

    def _normalize_loss_type(raw: Any) -> str:
        value = str(raw if raw is not None else loss_type_default).strip().lower()
        if value not in {"charbonnier"}:
            raise ValueError("training.gbuffer_image_loss.loss_type must be: charbonnier")
        return value

    def _normalize_domain(raw: Any) -> str:
        value = str(raw if raw is not None else domain_default).strip().lower()
        if value not in {"linear"}:
            raise ValueError("training.gbuffer_image_loss.domain must be: linear")
        return value

    def _normalize_color_space(raw: Any) -> str:
        value = str(raw if raw is not None else gt_color_space_default).strip().lower()
        if value not in {"linear", "srgb"}:
            raise ValueError("training.gbuffer_image_loss.gt_color_space must be one of: linear, srgb")
        return value

    def _normalize_key(raw: Any, default_value: str, field_name: str) -> str:
        value = str(default_value if raw is None else raw).strip()
        if not value:
            raise ValueError(f"training.gbuffer_image_loss.{field_name} must be a non-empty string")
        return value

    def _normalize_target_source(raw: Any) -> str:
        value = str(target_source_default if raw is None else raw).strip().lower()
        if value not in {"dataset_sh", "gbuffer_linear"}:
            raise ValueError(
                "training.gbuffer_image_loss.target_source must be one of: dataset_sh, gbuffer_linear"
            )
        return value

    defaults = {
        "enabled": bool(enabled_default or lambda_default > 0.0),
        "lambda": float(max(0.0, lambda_default)),
        "warmup_epochs": int(max(0, warmup_default)),
        "every_steps": int(max(1, every_steps_default)),
        "loss_type": _normalize_loss_type(loss_type_default),
        "dataset_root": dataset_root_default,
        "pixel_sample_count": int(max(1, pixel_sample_count_default)),
        "domain": _normalize_domain(domain_default),
        "gt_color_space": _normalize_color_space(gt_color_space_default),
        "strict_keys": bool(strict_keys_default),
        "pos_key": _normalize_key(pos_key_default, "posW", "pos_key"),
        "normal_key": _normalize_key(normal_key_default, "normW", "normal_key"),
        "albedo_key": _normalize_key(albedo_key_default, "albedo", "albedo_key"),
        "gt_linear_key": _normalize_key(gt_linear_key_default, "gt_linear", "gt_linear_key"),
        "light_params_key": _normalize_key(light_params_key_default, "light_params", "light_params_key"),
        "light_mask_key": _normalize_key(light_mask_key_default, "light_mask", "light_mask_key"),
        "valid_mask_key": _normalize_key(valid_mask_key_default, "valid_mask", "valid_mask_key"),
        "frame_idx_key": _normalize_key(frame_idx_key_default, "frame_idx", "frame_idx_key"),
        "config_idx_key": _normalize_key(config_idx_key_default, "config_idx", "config_idx_key"),
        "target_source": _normalize_target_source(target_source_default),
        "gt_knn": int(max(1, gt_knn_default)),
        "gt_weight_eps": float(max(1e-8, gt_weight_eps_default)),
        "gt_chunk_size": int(max(1, gt_chunk_size_default)),
    }

    if training_cfg is None:
        return defaults
    raw = training_cfg.get("gbuffer_image_loss")
    if not isinstance(raw, dict):
        return defaults

    lambda_value = float(max(0.0, raw.get("lambda", defaults["lambda"])))
    if gbuffer_weight_override is not None:
        lambda_value = float(max(0.0, gbuffer_weight_override))
    enabled = bool(raw.get("enabled", defaults["enabled"]))
    out = {
        "enabled": enabled,
        "lambda": float(lambda_value),
        "warmup_epochs": int(max(0, raw.get("warmup_epochs", defaults["warmup_epochs"]))),
        "every_steps": int(max(1, raw.get("every_steps", defaults["every_steps"]))),
        "loss_type": _normalize_loss_type(raw.get("loss_type", defaults["loss_type"])),
        "dataset_root": str(raw.get("dataset_root", defaults["dataset_root"]) or "").strip(),
        "pixel_sample_count": int(max(1, raw.get("pixel_sample_count", defaults["pixel_sample_count"]))),
        "domain": _normalize_domain(raw.get("domain", defaults["domain"])),
        "gt_color_space": _normalize_color_space(raw.get("gt_color_space", defaults["gt_color_space"])),
        "strict_keys": bool(raw.get("strict_keys", defaults["strict_keys"])),
        "pos_key": _normalize_key(raw.get("pos_key", defaults["pos_key"]), defaults["pos_key"], "pos_key"),
        "normal_key": _normalize_key(raw.get("normal_key", defaults["normal_key"]), defaults["normal_key"], "normal_key"),
        "albedo_key": _normalize_key(raw.get("albedo_key", defaults["albedo_key"]), defaults["albedo_key"], "albedo_key"),
        "gt_linear_key": _normalize_key(raw.get("gt_linear_key", defaults["gt_linear_key"]), defaults["gt_linear_key"], "gt_linear_key"),
        "light_params_key": _normalize_key(
            raw.get("light_params_key", defaults["light_params_key"]),
            defaults["light_params_key"],
            "light_params_key",
        ),
        "light_mask_key": _normalize_key(
            raw.get("light_mask_key", defaults["light_mask_key"]),
            defaults["light_mask_key"],
            "light_mask_key",
        ),
        "valid_mask_key": _normalize_key(
            raw.get("valid_mask_key", defaults["valid_mask_key"]),
            defaults["valid_mask_key"],
            "valid_mask_key",
        ),
        "frame_idx_key": _normalize_key(
            raw.get("frame_idx_key", defaults["frame_idx_key"]),
            defaults["frame_idx_key"],
            "frame_idx_key",
        ),
        "config_idx_key": _normalize_key(
            raw.get("config_idx_key", defaults["config_idx_key"]),
            defaults["config_idx_key"],
            "config_idx_key",
        ),
        "target_source": _normalize_target_source(
            raw.get("target_source", defaults["target_source"])
        ),
        "gt_knn": int(max(1, raw.get("gt_knn", defaults["gt_knn"]))),
        "gt_weight_eps": float(max(1e-8, raw.get("gt_weight_eps", defaults["gt_weight_eps"]))),
        "gt_chunk_size": int(max(1, raw.get("gt_chunk_size", defaults["gt_chunk_size"]))),
    }
    if out["enabled"] and not out["dataset_root"]:
        raise ValueError("training.gbuffer_image_loss.dataset_root is required when enabled=true")
    return out


def resolve_proxy_gbuffer_handover_config(args: argparse.Namespace) -> Dict[str, Any]:
    defaults = {
        "enabled": False,
        "start_epoch": 160,
        "end_epoch": 220,
        "proxy_start_scale": 1.0,
        "proxy_end_scale": 0.0,
        "gbuffer_start_scale": 0.0,
        "gbuffer_end_scale": 1.0,
    }

    def _normalize_nonneg_float(raw: Any, *, field_name: str) -> float:
        value = float(raw)
        if value < 0.0:
            raise ValueError(f"training.proxy_gbuffer_handover.{field_name} must be >= 0")
        return value

    def _normalize_epoch(raw: Any, *, field_name: str) -> int:
        value = int(raw)
        if value < 1:
            raise ValueError(f"training.proxy_gbuffer_handover.{field_name} must be >= 1")
        return value

    training_cfg = _get_training_cfg(args)
    if training_cfg is None:
        return dict(defaults)
    raw = training_cfg.get("proxy_gbuffer_handover")
    if raw is None:
        return dict(defaults)
    if not isinstance(raw, dict):
        raise ValueError("training.proxy_gbuffer_handover must be a mapping when provided")

    out = dict(defaults)
    out["enabled"] = bool(raw.get("enabled", out["enabled"]))
    out["start_epoch"] = _normalize_epoch(raw.get("start_epoch", out["start_epoch"]), field_name="start_epoch")
    out["end_epoch"] = _normalize_epoch(raw.get("end_epoch", out["end_epoch"]), field_name="end_epoch")
    if int(out["end_epoch"]) < int(out["start_epoch"]):
        raise ValueError("training.proxy_gbuffer_handover.end_epoch must be >= start_epoch")
    out["proxy_start_scale"] = _normalize_nonneg_float(
        raw.get("proxy_start_scale", out["proxy_start_scale"]),
        field_name="proxy_start_scale",
    )
    out["proxy_end_scale"] = _normalize_nonneg_float(
        raw.get("proxy_end_scale", out["proxy_end_scale"]),
        field_name="proxy_end_scale",
    )
    out["gbuffer_start_scale"] = _normalize_nonneg_float(
        raw.get("gbuffer_start_scale", out["gbuffer_start_scale"]),
        field_name="gbuffer_start_scale",
    )
    out["gbuffer_end_scale"] = _normalize_nonneg_float(
        raw.get("gbuffer_end_scale", out["gbuffer_end_scale"]),
        field_name="gbuffer_end_scale",
    )
    return out


def resolve_loss_effective_contribution_config(args: argparse.Namespace) -> Dict[str, Any]:
    defaults = {
        "enabled": False,
        "ema_decay": 0.98,
        "warmup_steps": 0,
        "eps": 1e-8,
        "clamp_min_scale": 0.25,
        "clamp_max_scale": 4.0,
        "frequency_aware": True,
        "min_target_contribution": 0.0,
        "max_target_contribution": 0.0,
        "target_ratio_default": 0.25,
        "target_ratios": {
            "image": 0.25,
            "gbuffer": 0.25,
            "temporal": 0.10,
            "linearity": 0.10,
            "spatial": 0.10,
            "routing_balance": 0.05,
        },
        "normalize_terms": {},
    }

    training_cfg = _get_training_cfg(args)
    if training_cfg is None:
        return defaults

    raw = training_cfg.get("loss_effective_contribution")
    if not isinstance(raw, dict):
        return defaults

    out = dict(defaults)
    out["enabled"] = bool(raw.get("enabled", out["enabled"]))
    out["ema_decay"] = float(np.clip(float(raw.get("ema_decay", out["ema_decay"])), 0.0, 1.0))
    out["warmup_steps"] = max(0, int(raw.get("warmup_steps", out["warmup_steps"])))
    out["eps"] = max(1e-12, float(raw.get("eps", out["eps"])))
    out["clamp_min_scale"] = max(0.0, float(raw.get("clamp_min_scale", out["clamp_min_scale"])))
    out["clamp_max_scale"] = max(out["clamp_min_scale"], float(raw.get("clamp_max_scale", out["clamp_max_scale"])))
    out["frequency_aware"] = bool(raw.get("frequency_aware", out["frequency_aware"]))
    out["min_target_contribution"] = max(
        0.0, float(raw.get("min_target_contribution", out["min_target_contribution"]))
    )
    out["max_target_contribution"] = max(
        0.0, float(raw.get("max_target_contribution", out["max_target_contribution"]))
    )
    out["target_ratio_default"] = max(0.0, float(raw.get("target_ratio_default", out["target_ratio_default"])))
    target_ratios = raw.get("target_ratios", out["target_ratios"])
    if not isinstance(target_ratios, dict):
        raise ValueError("training.loss_effective_contribution.target_ratios must be a mapping when provided")
    out["target_ratios"] = {str(k): max(0.0, float(v)) for k, v in target_ratios.items()}
    normalize_terms = raw.get("normalize_terms", out["normalize_terms"])
    if not isinstance(normalize_terms, dict):
        raise ValueError("training.loss_effective_contribution.normalize_terms must be a mapping when provided")
    out["normalize_terms"] = {str(k): bool(v) for k, v in normalize_terms.items()}
    return out


def resolve_optimization_monitoring_config(args: argparse.Namespace) -> Dict[str, Any]:
    defaults = {
        "enabled": True,
        "shared_groups": ["routing", "basis", "coeff", "encoder", "film"],
        "grad_diagnostics_every_steps": 50,
        "update_ratio_every_steps": 20,
        "spectrum_every_epochs": 1,
        "pulse_recovery_tolerance": 0.02,
        "pulse_recovery_max_steps": 64,
        "gns_enabled": True,
    }

    training_cfg = _get_training_cfg(args)
    if training_cfg is None:
        return defaults
    raw = training_cfg.get("optimization_monitoring")
    if not isinstance(raw, dict):
        return defaults

    out = dict(defaults)
    out["enabled"] = bool(raw.get("enabled", out["enabled"]))
    groups_raw = raw.get("shared_groups", out["shared_groups"])
    if isinstance(groups_raw, str):
        groups = [x.strip() for x in groups_raw.split(",") if x.strip()]
    elif isinstance(groups_raw, list):
        groups = [str(x).strip() for x in groups_raw if str(x).strip()]
    else:
        groups = list(out["shared_groups"])
    out["shared_groups"] = groups if groups else list(defaults["shared_groups"])
    out["grad_diagnostics_every_steps"] = max(
        0, int(raw.get("grad_diagnostics_every_steps", out["grad_diagnostics_every_steps"]))
    )
    out["update_ratio_every_steps"] = max(
        0, int(raw.get("update_ratio_every_steps", out["update_ratio_every_steps"]))
    )
    out["spectrum_every_epochs"] = max(
        0, int(raw.get("spectrum_every_epochs", out["spectrum_every_epochs"]))
    )
    out["pulse_recovery_tolerance"] = max(
        0.0, float(raw.get("pulse_recovery_tolerance", out["pulse_recovery_tolerance"]))
    )
    out["pulse_recovery_max_steps"] = max(
        1, int(raw.get("pulse_recovery_max_steps", out["pulse_recovery_max_steps"]))
    )
    out["gns_enabled"] = bool(raw.get("gns_enabled", out["gns_enabled"]))
    return out


def resolve_routing_balance_anneal_config(args: argparse.Namespace) -> Dict[str, Any]:
    default_lambda = max(0.0, float(getattr(args, "lambda_routing_balance", 0.0)))
    default_ratio = 0.6
    defaults = {
        "enabled": False,
        "start": float(default_lambda),
        "end": 0.0,
        "decay_end_ratio": float(default_ratio),
    }

    training_cfg = _get_training_cfg(args)
    if training_cfg is None:
        return defaults

    raw = training_cfg.get("routing_balance_anneal")
    if not isinstance(raw, dict):
        return defaults

    ratio = float(np.clip(float(raw.get("decay_end_ratio", default_ratio)), 0.0, 1.0))
    return {
        "enabled": bool(raw.get("enabled", False)),
        "start": float(max(0.0, float(raw.get("start", default_lambda)))),
        "end": float(max(0.0, float(raw.get("end", 0.0)))),
        "decay_end_ratio": ratio,
    }


def resolve_performance_config(args: argparse.Namespace) -> Dict[str, Any]:
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
        "grad_norm_log_every_steps": int(getattr(args, "grad_norm_log_every_steps", 100)),
        "train_routing_param_mode": str(getattr(args, "train_routing_param_mode", "gather")),
        "contraction_mode": str(getattr(args, "contraction_mode", "fused")),
        "cuda_graph_train": bool(getattr(args, "cuda_graph_train", False)),
        "cuda_graph_mode": str(getattr(args, "cuda_graph_mode", "dual")),
        "cuda_graph_warmup_steps": int(getattr(args, "cuda_graph_warmup_steps", 10)),
        "cuda_graph_fallback_eager": bool(getattr(args, "cuda_graph_fallback_eager", True)),
        "enable_soft_profile_sharing": bool(getattr(args, "enable_soft_profile_sharing", False)),
        "soft_profile_equiv_check_batches": int(getattr(args, "soft_profile_equiv_check_batches", 2)),
        "soft_profile_mae_tolerance": float(getattr(args, "soft_profile_mae_tolerance", 1e-6)),
        "soft_profile_img_psnr_tolerance": float(getattr(args, "soft_profile_img_psnr_tolerance", 5e-4)),
        "profile_train_enabled": bool(getattr(args, "profile_train", False)),
        "profile_dir": getattr(args, "profile_dir", None),
        "profile_wait_steps": max(0, int(getattr(args, "profile_wait", 1))),
        "profile_warmup_steps": max(0, int(getattr(args, "profile_warmup", 1))),
        "profile_active_steps": max(1, int(getattr(args, "profile_active", 3))),
        "profile_repeat": max(1, int(getattr(args, "profile_repeat", 1))),
        "profile_record_shapes": bool(getattr(args, "profile_record_shapes", False)),
        "profile_with_stack": bool(getattr(args, "profile_with_stack", False)),
        "profile_profile_memory": bool(getattr(args, "profile_memory", False)),
    }

    training_cfg = _get_training_cfg(args)
    if training_cfg is None:
        return defaults

    raw = training_cfg.get("performance", {})
    if not isinstance(raw, dict):
        return defaults

    compile_raw = raw.get("torch_compile", {})
    if compile_raw is None or not isinstance(compile_raw, dict):
        compile_raw = {}

    soft_profile_raw = raw.get("soft_profile_sharing", {})
    if soft_profile_raw is None or not isinstance(soft_profile_raw, dict):
        soft_profile_raw = {}
    cuda_graph_raw = raw.get("cuda_graph", {})
    if cuda_graph_raw is None or not isinstance(cuda_graph_raw, dict):
        cuda_graph_raw = {}
    profiler_raw = raw.get("profiler", {})
    if profiler_raw is None or not isinstance(profiler_raw, dict):
        profiler_raw = {}

    out = dict(defaults)
    out["amp_mode"] = _normalize_amp_mode(raw.get("amp_mode", out["amp_mode"]))
    out["torch_compile_enabled"] = bool(compile_raw.get("enabled", out["torch_compile_enabled"]))
    out["torch_compile_mode"] = str(compile_raw.get("mode", out["torch_compile_mode"]))
    out["torch_compile_dynamic"] = bool(compile_raw.get("dynamic", out["torch_compile_dynamic"]))
    out["grad_norm_log_every_steps"] = max(0, int(raw.get("grad_norm_log_every_steps", out["grad_norm_log_every_steps"])))
    out["train_routing_param_mode"] = str(
        raw.get("train_routing_param_mode", out["train_routing_param_mode"])
    ).strip().lower()
    if out["train_routing_param_mode"] not in {"gather", "dense_masked"}:
        out["train_routing_param_mode"] = "gather"
    out["contraction_mode"] = str(
        raw.get("contraction_mode", out["contraction_mode"])
    ).strip().lower()
    if out["contraction_mode"] not in {"legacy", "fused"}:
        out["contraction_mode"] = "fused"
    out["cuda_graph_train"] = bool(cuda_graph_raw.get("enabled", out["cuda_graph_train"]))
    out["cuda_graph_mode"] = str(cuda_graph_raw.get("mode", out["cuda_graph_mode"])).strip().lower()
    if out["cuda_graph_mode"] not in {"single", "dual"}:
        out["cuda_graph_mode"] = "dual"
    out["cuda_graph_warmup_steps"] = max(
        0,
        int(cuda_graph_raw.get("warmup_steps", out["cuda_graph_warmup_steps"])),
    )
    out["cuda_graph_fallback_eager"] = bool(
        cuda_graph_raw.get("fallback_eager", out["cuda_graph_fallback_eager"])
    )
    out["enable_soft_profile_sharing"] = bool(soft_profile_raw.get("enabled", out["enable_soft_profile_sharing"]))
    out["soft_profile_equiv_check_batches"] = max(
        1,
        int(soft_profile_raw.get("equiv_check_batches", out["soft_profile_equiv_check_batches"])),
    )
    out["soft_profile_mae_tolerance"] = float(
        soft_profile_raw.get("mae_tolerance", out["soft_profile_mae_tolerance"])
    )
    out["soft_profile_img_psnr_tolerance"] = float(
        soft_profile_raw.get("img_psnr_tolerance", out["soft_profile_img_psnr_tolerance"])
    )
    out["profile_train_enabled"] = bool(profiler_raw.get("enabled", out["profile_train_enabled"]))
    out["profile_dir"] = profiler_raw.get("dir", out["profile_dir"])
    out["profile_wait_steps"] = max(0, int(profiler_raw.get("wait", out["profile_wait_steps"])))
    out["profile_warmup_steps"] = max(0, int(profiler_raw.get("warmup", out["profile_warmup_steps"])))
    out["profile_active_steps"] = max(1, int(profiler_raw.get("active", out["profile_active_steps"])))
    out["profile_repeat"] = max(1, int(profiler_raw.get("repeat", out["profile_repeat"])))
    out["profile_record_shapes"] = bool(profiler_raw.get("record_shapes", out["profile_record_shapes"]))
    out["profile_with_stack"] = bool(profiler_raw.get("with_stack", out["profile_with_stack"]))
    out["profile_profile_memory"] = bool(profiler_raw.get("profile_memory", out["profile_profile_memory"]))
    return out


def resolve_falcor_periodic_eval_config(args: argparse.Namespace) -> Dict[str, Any]:
    training_cfg = _get_training_cfg(args)
    if training_cfg is None:
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


def resolve_val_profiles_config(args: argparse.Namespace) -> Dict[str, Any]:
    defaults = {
        "enabled": False,
        "best_metric": "mae",
        "profiles": [],
        "run_test_compare": False,
    }
    training_cfg = _get_training_cfg(args)
    if training_cfg is None:
        return defaults

    raw = training_cfg.get("val_profiles")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return defaults

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
    if best_metric not in _VAL_PROFILE_ALLOWED_METRICS:
        raise ValueError(
            "training.val_profiles.best_metric must be one of: "
            + ", ".join(sorted(_VAL_PROFILE_ALLOWED_METRICS))
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


def normalize_route_profiles(raw_profiles: Any, default_top_k: int) -> List[Dict[str, Any]]:
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


def resolve_expert_utilization_audit_config(args: argparse.Namespace) -> Dict[str, Any]:
    training_cfg = _get_training_cfg(args)
    if training_cfg is None:
        return {"enabled": False}

    raw = training_cfg.get("expert_utilization_audit")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return {"enabled": False}

    profiles = normalize_route_profiles(raw.get("profiles", []), int(args.top_k))
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


def resolve_semantic_drift_audit_config(args: argparse.Namespace) -> Dict[str, Any]:
    training_cfg = _get_training_cfg(args)
    if training_cfg is None:
        return {"enabled": False}

    raw = training_cfg.get("semantic_drift_audit")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return {"enabled": False}

    profiles = normalize_route_profiles(raw.get("profiles", []), int(args.top_k))
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


@dataclass(frozen=True)
class RuntimeConfigBundle:
    ema_cfg: Dict[str, Any]
    oracle_monitor_cfg: Dict[str, Any]
    coeff_suite_cfg: Dict[str, Any]
    global_best_cfg: Dict[str, Any]
    val_profiles_cfg: Dict[str, Any]
    proxy_image_loss_cfg: Dict[str, Any]
    gbuffer_image_loss_cfg: Dict[str, Any]
    proxy_gbuffer_handover_cfg: Dict[str, Any]
    loss_effective_contribution_cfg: Dict[str, Any]
    optimization_monitoring_cfg: Dict[str, Any]
    routing_balance_anneal_cfg: Dict[str, Any]
    performance_cfg: Dict[str, Any]
    falcor_periodic_eval_cfg: Dict[str, Any]
    expert_util_cfg: Dict[str, Any]
    semantic_drift_cfg: Dict[str, Any]
    global_group_lrs: Dict[str, float]
    staged_specs: List[Dict[str, Any]]
    total_training_epochs_global: int
    routing_balance_decay_end_epoch: int
    routing_balance_anneal_epochs_global: int


def apply_runtime_overrides(args: argparse.Namespace, runtime_cfg: RuntimeConfigBundle) -> None:
    proxy_image_loss_cfg = runtime_cfg.proxy_image_loss_cfg
    if bool(proxy_image_loss_cfg.get("enabled", False)):
        args.lambda_image = float(proxy_image_loss_cfg.get("lambda", args.lambda_image))
        args.image_loss_warmup_epochs = int(
            proxy_image_loss_cfg.get(
                "warmup_epochs",
                getattr(args, "image_loss_warmup_epochs", 0),
            )
        )
        if bool(proxy_image_loss_cfg.get("enable_val_image_metrics", True)):
            args.val_image_metrics = True
    args.proxy_image_mix_mode = str(proxy_image_loss_cfg.get("mix_mode", "linear_log_mix"))
    args.proxy_image_log_mix_weight = float(proxy_image_loss_cfg.get("log_mix_weight", 0.3))
    args.image_loss_type = str(proxy_image_loss_cfg.get("loss_type", getattr(args, "image_loss_type", "mse")))
    args.image_loss_huber_delta = float(
        proxy_image_loss_cfg.get("huber_delta", getattr(args, "image_loss_huber_delta", 0.1))
    )
    args.image_loss_domain = str(
        proxy_image_loss_cfg.get("domain", getattr(args, "image_loss_domain", "radiance"))
    )
    args.image_sampling_mode = str(
        proxy_image_loss_cfg.get("sampling_mode", getattr(args, "image_sampling_mode", "fixed"))
    )
    args.image_soft_saturation_enabled = bool(
        proxy_image_loss_cfg.get(
            "soft_saturation_enabled",
            getattr(args, "image_soft_saturation_enabled", False),
        )
    )
    args.image_soft_saturation_mode = str(
        proxy_image_loss_cfg.get(
            "soft_saturation_mode",
            getattr(args, "image_soft_saturation_mode", "exp"),
        )
    )
    args.image_soft_saturation_k = float(
        proxy_image_loss_cfg.get(
            "soft_saturation_k",
            getattr(args, "image_soft_saturation_k", 1.0),
        )
    )
    gbuffer_image_loss_cfg = runtime_cfg.gbuffer_image_loss_cfg
    args.gbuffer_image_loss_enabled = bool(gbuffer_image_loss_cfg.get("enabled", False))
    args.gbuffer_image_loss_lambda = float(
        gbuffer_image_loss_cfg.get("lambda", getattr(args, "gbuffer_image_loss_lambda", 0.0))
    )
    args.gbuffer_image_loss_warmup_epochs = int(
        gbuffer_image_loss_cfg.get(
            "warmup_epochs",
            getattr(args, "gbuffer_image_loss_warmup_epochs", 0),
        )
    )
    args.gbuffer_image_loss_every_steps = int(
        gbuffer_image_loss_cfg.get(
            "every_steps",
            getattr(args, "gbuffer_image_loss_every_steps", 16),
        )
    )
    args.gbuffer_image_loss_type = str(
        gbuffer_image_loss_cfg.get(
            "loss_type",
            getattr(args, "gbuffer_image_loss_type", "charbonnier"),
        )
    )
    args.gbuffer_image_loss_dataset_root = str(
        gbuffer_image_loss_cfg.get(
            "dataset_root",
            getattr(args, "gbuffer_image_loss_dataset_root", ""),
        )
    )
    args.gbuffer_image_loss_pixel_sample_count = int(
        gbuffer_image_loss_cfg.get(
            "pixel_sample_count",
            getattr(args, "gbuffer_image_loss_pixel_sample_count", 8192),
        )
    )
    args.gbuffer_image_loss_domain = str(
        gbuffer_image_loss_cfg.get(
            "domain",
            getattr(args, "gbuffer_image_loss_domain", "linear"),
        )
    )
    args.gbuffer_image_loss_gt_color_space = str(
        gbuffer_image_loss_cfg.get(
            "gt_color_space",
            getattr(args, "gbuffer_image_loss_gt_color_space", "linear"),
        )
    )
    args.gbuffer_image_loss_strict_keys = bool(
        gbuffer_image_loss_cfg.get(
            "strict_keys",
            getattr(args, "gbuffer_image_loss_strict_keys", True),
        )
    )
    args.gbuffer_image_loss_pos_key = str(
        gbuffer_image_loss_cfg.get(
            "pos_key",
            getattr(args, "gbuffer_image_loss_pos_key", "posW"),
        )
    )
    args.gbuffer_image_loss_normal_key = str(
        gbuffer_image_loss_cfg.get(
            "normal_key",
            getattr(args, "gbuffer_image_loss_normal_key", "normW"),
        )
    )
    args.gbuffer_image_loss_albedo_key = str(
        gbuffer_image_loss_cfg.get(
            "albedo_key",
            getattr(args, "gbuffer_image_loss_albedo_key", "albedo"),
        )
    )
    args.gbuffer_image_loss_gt_linear_key = str(
        gbuffer_image_loss_cfg.get(
            "gt_linear_key",
            getattr(args, "gbuffer_image_loss_gt_linear_key", "gt_linear"),
        )
    )
    args.gbuffer_image_loss_light_params_key = str(
        gbuffer_image_loss_cfg.get(
            "light_params_key",
            getattr(args, "gbuffer_image_loss_light_params_key", "light_params"),
        )
    )
    args.gbuffer_image_loss_light_mask_key = str(
        gbuffer_image_loss_cfg.get(
            "light_mask_key",
            getattr(args, "gbuffer_image_loss_light_mask_key", "light_mask"),
        )
    )
    args.gbuffer_image_loss_valid_mask_key = str(
        gbuffer_image_loss_cfg.get(
            "valid_mask_key",
            getattr(args, "gbuffer_image_loss_valid_mask_key", "valid_mask"),
        )
    )
    args.gbuffer_image_loss_frame_idx_key = str(
        gbuffer_image_loss_cfg.get(
            "frame_idx_key",
            getattr(args, "gbuffer_image_loss_frame_idx_key", "frame_idx"),
        )
    )
    args.gbuffer_image_loss_config_idx_key = str(
        gbuffer_image_loss_cfg.get(
            "config_idx_key",
            getattr(args, "gbuffer_image_loss_config_idx_key", "config_idx"),
        )
    )
    args.gbuffer_image_loss_target_source = str(
        gbuffer_image_loss_cfg.get(
            "target_source",
            getattr(args, "gbuffer_image_loss_target_source", "dataset_sh"),
        )
    )
    args.gbuffer_image_loss_gt_knn = int(
        gbuffer_image_loss_cfg.get(
            "gt_knn",
            getattr(args, "gbuffer_image_loss_gt_knn", 8),
        )
    )
    args.gbuffer_image_loss_gt_weight_eps = float(
        gbuffer_image_loss_cfg.get(
            "gt_weight_eps",
            getattr(args, "gbuffer_image_loss_gt_weight_eps", 0.1),
        )
    )
    args.gbuffer_image_loss_gt_chunk_size = int(
        gbuffer_image_loss_cfg.get(
            "gt_chunk_size",
            getattr(args, "gbuffer_image_loss_gt_chunk_size", 32768),
        )
    )
    proxy_gbuffer_handover_cfg = runtime_cfg.proxy_gbuffer_handover_cfg
    args.proxy_gbuffer_handover_enabled = bool(
        proxy_gbuffer_handover_cfg.get("enabled", getattr(args, "proxy_gbuffer_handover_enabled", False))
    )
    args.proxy_gbuffer_handover_start_epoch = int(
        proxy_gbuffer_handover_cfg.get(
            "start_epoch",
            getattr(args, "proxy_gbuffer_handover_start_epoch", 160),
        )
    )
    args.proxy_gbuffer_handover_end_epoch = int(
        proxy_gbuffer_handover_cfg.get(
            "end_epoch",
            getattr(args, "proxy_gbuffer_handover_end_epoch", 220),
        )
    )
    args.proxy_gbuffer_handover_proxy_start_scale = float(
        proxy_gbuffer_handover_cfg.get(
            "proxy_start_scale",
            getattr(args, "proxy_gbuffer_handover_proxy_start_scale", 1.0),
        )
    )
    args.proxy_gbuffer_handover_proxy_end_scale = float(
        proxy_gbuffer_handover_cfg.get(
            "proxy_end_scale",
            getattr(args, "proxy_gbuffer_handover_proxy_end_scale", 0.0),
        )
    )
    args.proxy_gbuffer_handover_gbuffer_start_scale = float(
        proxy_gbuffer_handover_cfg.get(
            "gbuffer_start_scale",
            getattr(args, "proxy_gbuffer_handover_gbuffer_start_scale", 0.0),
        )
    )
    args.proxy_gbuffer_handover_gbuffer_end_scale = float(
        proxy_gbuffer_handover_cfg.get(
            "gbuffer_end_scale",
            getattr(args, "proxy_gbuffer_handover_gbuffer_end_scale", 1.0),
        )
    )
    args.loss_effective_contribution_cfg = dict(
        runtime_cfg.loss_effective_contribution_cfg
    )
    args.optimization_monitoring_cfg = dict(
        runtime_cfg.optimization_monitoring_cfg
    )

    performance_cfg = runtime_cfg.performance_cfg
    args.amp_mode = str(performance_cfg.get("amp_mode", getattr(args, "amp_mode", "off")))
    args.torch_compile = bool(
        performance_cfg.get("torch_compile_enabled", getattr(args, "torch_compile", False))
    )
    args.torch_compile_mode = str(
        performance_cfg.get("torch_compile_mode", getattr(args, "torch_compile_mode", "reduce-overhead"))
    )
    args.torch_compile_dynamic = bool(
        performance_cfg.get("torch_compile_dynamic", getattr(args, "torch_compile_dynamic", False))
    )
    args.grad_norm_log_every_steps = int(
        performance_cfg.get("grad_norm_log_every_steps", getattr(args, "grad_norm_log_every_steps", 100))
    )
    args.train_routing_param_mode = str(
        performance_cfg.get("train_routing_param_mode", getattr(args, "train_routing_param_mode", "gather"))
    ).strip().lower()
    if args.train_routing_param_mode not in {"gather", "dense_masked"}:
        args.train_routing_param_mode = "gather"
    args.contraction_mode = str(
        performance_cfg.get("contraction_mode", getattr(args, "contraction_mode", "fused"))
    ).strip().lower()
    if args.contraction_mode not in {"legacy", "fused"}:
        args.contraction_mode = "fused"
    args.cuda_graph_train = bool(
        performance_cfg.get("cuda_graph_train", getattr(args, "cuda_graph_train", False))
    )
    args.cuda_graph_mode = str(
        performance_cfg.get("cuda_graph_mode", getattr(args, "cuda_graph_mode", "dual"))
    ).strip().lower()
    if args.cuda_graph_mode not in {"single", "dual"}:
        args.cuda_graph_mode = "dual"
    args.cuda_graph_warmup_steps = max(
        0,
        int(
            performance_cfg.get(
                "cuda_graph_warmup_steps",
                getattr(args, "cuda_graph_warmup_steps", 10),
            )
        ),
    )
    args.cuda_graph_fallback_eager = bool(
        performance_cfg.get(
            "cuda_graph_fallback_eager",
            getattr(args, "cuda_graph_fallback_eager", True),
        )
    )
    args.enable_soft_profile_sharing = bool(
        performance_cfg.get(
            "enable_soft_profile_sharing",
            getattr(args, "enable_soft_profile_sharing", False),
        )
    )
    args.soft_profile_equiv_check_batches = int(
        performance_cfg.get(
            "soft_profile_equiv_check_batches",
            getattr(args, "soft_profile_equiv_check_batches", 2),
        )
    )
    args.soft_profile_mae_tolerance = float(
        performance_cfg.get(
            "soft_profile_mae_tolerance",
            getattr(args, "soft_profile_mae_tolerance", 1e-6),
        )
    )
    args.soft_profile_img_psnr_tolerance = float(
        performance_cfg.get(
            "soft_profile_img_psnr_tolerance",
            getattr(args, "soft_profile_img_psnr_tolerance", 5e-4),
        )
    )
    args.profile_train = bool(
        performance_cfg.get("profile_train_enabled", getattr(args, "profile_train", False))
    )
    args.profile_dir = performance_cfg.get("profile_dir", getattr(args, "profile_dir", None))
    args.profile_wait = int(
        performance_cfg.get("profile_wait_steps", getattr(args, "profile_wait", 1))
    )
    args.profile_warmup = int(
        performance_cfg.get("profile_warmup_steps", getattr(args, "profile_warmup", 1))
    )
    args.profile_active = int(
        performance_cfg.get("profile_active_steps", getattr(args, "profile_active", 3))
    )
    args.profile_repeat = int(
        performance_cfg.get("profile_repeat", getattr(args, "profile_repeat", 1))
    )
    args.profile_record_shapes = bool(
        performance_cfg.get("profile_record_shapes", getattr(args, "profile_record_shapes", False))
    )
    args.profile_with_stack = bool(
        performance_cfg.get("profile_with_stack", getattr(args, "profile_with_stack", False))
    )
    args.profile_memory = bool(
        performance_cfg.get("profile_profile_memory", getattr(args, "profile_memory", False))
    )


def resolve_runtime_config_bundle(args: argparse.Namespace) -> RuntimeConfigBundle:
    proxy_image_loss_cfg = resolve_proxy_image_loss_config(args)
    gbuffer_image_loss_cfg = resolve_gbuffer_image_loss_config(args)
    proxy_gbuffer_handover_cfg = resolve_proxy_gbuffer_handover_config(args)
    loss_effective_contribution_cfg = resolve_loss_effective_contribution_config(args)
    optimization_monitoring_cfg = resolve_optimization_monitoring_config(args)
    routing_balance_anneal_cfg = resolve_routing_balance_anneal_config(args)
    staged_specs = resolve_staged_specs(args)

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

    return RuntimeConfigBundle(
        ema_cfg=resolve_ema_config(args),
        oracle_monitor_cfg=resolve_oracle_monitor_config(args),
        coeff_suite_cfg=resolve_coeff_suite_config(args),
        global_best_cfg=resolve_global_best_config(args),
        val_profiles_cfg=resolve_val_profiles_config(args),
        proxy_image_loss_cfg=proxy_image_loss_cfg,
        gbuffer_image_loss_cfg=gbuffer_image_loss_cfg,
        proxy_gbuffer_handover_cfg=proxy_gbuffer_handover_cfg,
        loss_effective_contribution_cfg=loss_effective_contribution_cfg,
        optimization_monitoring_cfg=optimization_monitoring_cfg,
        routing_balance_anneal_cfg=routing_balance_anneal_cfg,
        performance_cfg=resolve_performance_config(args),
        falcor_periodic_eval_cfg=resolve_falcor_periodic_eval_config(args),
        expert_util_cfg=resolve_expert_utilization_audit_config(args),
        semantic_drift_cfg=resolve_semantic_drift_audit_config(args),
        global_group_lrs=resolve_global_group_lrs(args),
        staged_specs=staged_specs,
        total_training_epochs_global=total_training_epochs_global,
        routing_balance_decay_end_epoch=routing_balance_decay_end_epoch,
        routing_balance_anneal_epochs_global=routing_balance_anneal_epochs_global,
    )
