from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runtime_cfg = _load_module(
    "train_runtime_config_test",
    ROOT / "3_experiments" / "scripts" / "train_runtime_config.py",
)


def test_resolve_runtime_config_bundle_uses_staged_total_epochs() -> None:
    args = Namespace(
        epochs=999,
        lr=1e-3,
        weight_decay=0.0,
        lr_scheduler="none",
        lr_min=1e-4,
        warmup_epochs=0,
        lambda_image=0.0,
        image_loss_warmup_epochs=0,
        proxy_image_mix_mode="linear_log_mix",
        proxy_image_log_mix_weight=0.3,
        lambda_routing_balance=0.01,
        top_k=3,
        batch_size=256,
        seed=7,
        device="cpu",
        amp_mode="off",
        torch_compile=False,
        torch_compile_mode="reduce-overhead",
        torch_compile_dynamic=False,
        grad_norm_log_every_steps=100,
        train_routing_param_mode="gather",
        contraction_mode="fused",
        cuda_graph_train=False,
        cuda_graph_mode="dual",
        cuda_graph_warmup_steps=10,
        cuda_graph_fallback_eager=True,
        enable_soft_profile_sharing=False,
        soft_profile_equiv_check_batches=2,
        soft_profile_mae_tolerance=1e-6,
        soft_profile_img_psnr_tolerance=5e-4,
        profile_train=False,
        profile_dir=None,
        profile_wait=1,
        profile_warmup=1,
        profile_active=3,
        profile_repeat=1,
        profile_record_shapes=False,
        profile_with_stack=False,
        profile_memory=False,
        _config_obj={
            "training": {
                "staged": {
                    "enabled": True,
                    "stages": [
                        {"name": "s1", "epochs": 10, "trainable_groups": ["coeff"]},
                        {"name": "s2", "epochs": 30, "trainable_groups": ["all"]},
                    ],
                },
                "routing_balance_anneal": {
                    "enabled": True,
                    "start": 0.01,
                    "end": 0.0,
                    "decay_end_ratio": 0.5,
                },
            }
        },
    )
    bundle = runtime_cfg.resolve_runtime_config_bundle(args)
    assert len(bundle.staged_specs) == 2
    assert bundle.total_training_epochs_global == 40
    assert bundle.routing_balance_decay_end_epoch == 20
    assert bundle.routing_balance_anneal_epochs_global == 19


def test_resolve_proxy_image_loss_config_invalid_mode() -> None:
    args = Namespace(
        lambda_image=0.1,
        image_loss_warmup_epochs=0,
        proxy_image_mix_mode="linear_log_mix",
        proxy_image_log_mix_weight=0.3,
        _config_obj={"training": {"proxy_image_loss": {"enabled": True, "mix_mode": "bad_mode"}}},
    )
    with pytest.raises(ValueError):
        runtime_cfg.resolve_proxy_image_loss_config(args)


def test_apply_runtime_overrides_updates_args() -> None:
    args = Namespace(
        epochs=5,
        lr=1e-3,
        weight_decay=0.0,
        lr_scheduler="none",
        lr_min=1e-4,
        warmup_epochs=0,
        lambda_image=0.0,
        image_loss_warmup_epochs=0,
        proxy_image_mix_mode="linear_log_mix",
        proxy_image_log_mix_weight=0.3,
        val_image_metrics=False,
        lambda_routing_balance=0.0,
        top_k=3,
        batch_size=8,
        seed=7,
        device="cpu",
        amp_mode="off",
        torch_compile=False,
        torch_compile_mode="reduce-overhead",
        torch_compile_dynamic=False,
        grad_norm_log_every_steps=100,
        train_routing_param_mode="gather",
        contraction_mode="fused",
        cuda_graph_train=False,
        cuda_graph_mode="dual",
        cuda_graph_warmup_steps=10,
        cuda_graph_fallback_eager=True,
        enable_soft_profile_sharing=False,
        soft_profile_equiv_check_batches=2,
        soft_profile_mae_tolerance=1e-6,
        soft_profile_img_psnr_tolerance=5e-4,
        profile_train=False,
        profile_dir=None,
        profile_wait=1,
        profile_warmup=1,
        profile_active=3,
        profile_repeat=1,
        profile_record_shapes=False,
        profile_with_stack=False,
        profile_memory=False,
        _config_obj={
            "training": {
                "proxy_image_loss": {
                    "enabled": True,
                    "lambda": 0.2,
                    "warmup_epochs": 10,
                    "enable_val_image_metrics": True,
                    "mix_mode": "log_only",
                    "log_mix_weight": 0.8,
                },
                "gbuffer_image_loss": {
                    "enabled": True,
                    "lambda": 0.05,
                    "warmup_epochs": 3,
                    "every_steps": 8,
                    "loss_type": "charbonnier",
                    "dataset_root": "tmp/gbuffer",
                    "pixel_sample_count": 2048,
                    "domain": "linear",
                    "gt_color_space": "srgb",
                    "strict_keys": True,
                    "pos_key": "posW",
                    "normal_key": "normW",
                    "albedo_key": "albedo",
                    "gt_linear_key": "gt_linear",
                    "light_params_key": "light_params",
                    "light_mask_key": "light_mask",
                    "valid_mask_key": "valid_mask",
                },
                "proxy_gbuffer_handover": {
                    "enabled": True,
                    "start_epoch": 150,
                    "end_epoch": 220,
                    "proxy_start_scale": 1.0,
                    "proxy_end_scale": 0.0,
                    "gbuffer_start_scale": 0.1,
                    "gbuffer_end_scale": 1.2,
                },
                "performance": {
                    "amp_mode": "bf16",
                    "torch_compile": {"enabled": True, "mode": "default", "dynamic": True},
                    "grad_norm_log_every_steps": 4,
                    "train_routing_param_mode": "dense_masked",
                    "contraction_mode": "legacy",
                    "cuda_graph": {
                        "enabled": True,
                        "mode": "single",
                        "warmup_steps": 5,
                        "fallback_eager": False,
                    },
                    "profiler": {
                        "enabled": True,
                        "dir": "runs/profile_case",
                        "wait": 2,
                        "warmup": 3,
                        "active": 4,
                        "repeat": 2,
                        "record_shapes": True,
                        "with_stack": True,
                        "profile_memory": True,
                    },
                    "soft_profile_sharing": {
                        "enabled": True,
                        "equiv_check_batches": 3,
                        "mae_tolerance": 1e-5,
                        "img_psnr_tolerance": 1e-3,
                    },
                },
            }
        },
    )
    bundle = runtime_cfg.resolve_runtime_config_bundle(args)
    runtime_cfg.apply_runtime_overrides(args, bundle)

    assert args.lambda_image == 0.2
    assert args.image_loss_warmup_epochs == 10
    assert args.val_image_metrics is True
    assert args.proxy_image_mix_mode == "log_only"
    assert abs(args.proxy_image_log_mix_weight - 0.8) < 1e-12
    assert args.gbuffer_image_loss_enabled is True
    assert abs(args.gbuffer_image_loss_lambda - 0.05) < 1e-12
    assert args.gbuffer_image_loss_warmup_epochs == 3
    assert args.gbuffer_image_loss_every_steps == 8
    assert args.gbuffer_image_loss_type == "charbonnier"
    assert args.gbuffer_image_loss_dataset_root == "tmp/gbuffer"
    assert args.gbuffer_image_loss_pixel_sample_count == 2048
    assert args.gbuffer_image_loss_domain == "linear"
    assert args.gbuffer_image_loss_gt_color_space == "srgb"
    assert args.gbuffer_image_loss_strict_keys is True
    assert args.gbuffer_image_loss_pos_key == "posW"
    assert args.gbuffer_image_loss_normal_key == "normW"
    assert args.gbuffer_image_loss_albedo_key == "albedo"
    assert args.gbuffer_image_loss_gt_linear_key == "gt_linear"
    assert args.gbuffer_image_loss_light_params_key == "light_params"
    assert args.gbuffer_image_loss_light_mask_key == "light_mask"
    assert args.gbuffer_image_loss_valid_mask_key == "valid_mask"
    assert args.proxy_gbuffer_handover_enabled is True
    assert args.proxy_gbuffer_handover_start_epoch == 150
    assert args.proxy_gbuffer_handover_end_epoch == 220
    assert abs(args.proxy_gbuffer_handover_proxy_start_scale - 1.0) < 1e-12
    assert abs(args.proxy_gbuffer_handover_proxy_end_scale - 0.0) < 1e-12
    assert abs(args.proxy_gbuffer_handover_gbuffer_start_scale - 0.1) < 1e-12
    assert abs(args.proxy_gbuffer_handover_gbuffer_end_scale - 1.2) < 1e-12
    assert args.amp_mode == "bf16"
    assert args.torch_compile is True
    assert args.torch_compile_mode == "default"
    assert args.torch_compile_dynamic is True
    assert args.grad_norm_log_every_steps == 4
    assert args.train_routing_param_mode == "dense_masked"
    assert args.contraction_mode == "legacy"
    assert args.cuda_graph_train is True
    assert args.cuda_graph_mode == "single"
    assert args.cuda_graph_warmup_steps == 5
    assert args.cuda_graph_fallback_eager is False
    assert args.enable_soft_profile_sharing is True
    assert args.soft_profile_equiv_check_batches == 3
    assert abs(args.soft_profile_mae_tolerance - 1e-5) < 1e-12
    assert abs(args.soft_profile_img_psnr_tolerance - 1e-3) < 1e-12
    assert args.profile_train is True
    assert args.profile_dir == "runs/profile_case"
    assert args.profile_wait == 2
    assert args.profile_warmup == 3
    assert args.profile_active == 4
    assert args.profile_repeat == 2
    assert args.profile_record_shapes is True
    assert args.profile_with_stack is True
    assert args.profile_memory is True


def test_resolve_performance_config_defaults_grad_norm_to_100() -> None:
    args = Namespace(
        amp_mode="off",
        torch_compile=False,
        torch_compile_mode="reduce-overhead",
        torch_compile_dynamic=False,
        train_routing_param_mode="gather",
        contraction_mode="fused",
        cuda_graph_train=False,
        cuda_graph_mode="dual",
        cuda_graph_warmup_steps=10,
        cuda_graph_fallback_eager=True,
        enable_soft_profile_sharing=False,
        soft_profile_equiv_check_batches=2,
        soft_profile_mae_tolerance=1e-6,
        soft_profile_img_psnr_tolerance=5e-4,
        profile_train=False,
        profile_dir=None,
        profile_wait=1,
        profile_warmup=1,
        profile_active=3,
        profile_repeat=1,
        profile_record_shapes=False,
        profile_with_stack=False,
        profile_memory=False,
        _config_obj={},
    )
    cfg = runtime_cfg.resolve_performance_config(args)
    assert cfg["grad_norm_log_every_steps"] == 100
    assert cfg["train_routing_param_mode"] == "gather"
    assert cfg["contraction_mode"] == "fused"
    assert cfg["cuda_graph_train"] is False


def test_resolve_gbuffer_image_loss_config_requires_dataset_when_enabled() -> None:
    args = Namespace(
        gbuffer_image_loss_enabled=False,
        gbuffer_image_loss_lambda=0.0,
        gbuffer_image_loss_warmup_epochs=0,
        gbuffer_image_loss_every_steps=16,
        gbuffer_image_loss_type="charbonnier",
        gbuffer_image_loss_dataset_root="",
        gbuffer_image_loss_pixel_sample_count=8192,
        _config_obj={
            "training": {
                "gbuffer_image_loss": {
                    "enabled": True,
                    "lambda": 0.05,
                    "dataset_root": "",
                }
            }
        },
    )
    with pytest.raises(ValueError):
        runtime_cfg.resolve_gbuffer_image_loss_config(args)


def test_resolve_gbuffer_image_loss_config_success() -> None:
    args = Namespace(
        gbuffer_image_loss_enabled=False,
        gbuffer_image_loss_lambda=0.0,
        gbuffer_image_loss_warmup_epochs=0,
        gbuffer_image_loss_every_steps=16,
        gbuffer_image_loss_type="charbonnier",
        gbuffer_image_loss_dataset_root="",
        gbuffer_image_loss_pixel_sample_count=8192,
        _config_obj={
            "training": {
                "gbuffer_image_loss": {
                    "enabled": True,
                    "lambda": 0.05,
                    "warmup_epochs": 3,
                    "every_steps": 8,
                    "loss_type": "charbonnier",
                    "dataset_root": "tmp/gbuffer",
                    "pixel_sample_count": 2048,
                    "domain": "linear",
                    "gt_color_space": "srgb",
                    "strict_keys": False,
                    "pos_key": "world_pos",
                    "normal_key": "world_normal",
                    "albedo_key": "base_color",
                    "gt_linear_key": "gt_image",
                    "light_params_key": "lights",
                    "light_mask_key": "lights_mask",
                    "valid_mask_key": "valid",
                }
            }
        },
    )
    cfg = runtime_cfg.resolve_gbuffer_image_loss_config(args)
    assert cfg["enabled"] is True
    assert cfg["lambda"] == 0.05
    assert cfg["warmup_epochs"] == 3
    assert cfg["every_steps"] == 8
    assert cfg["loss_type"] == "charbonnier"
    assert cfg["dataset_root"] == "tmp/gbuffer"
    assert cfg["pixel_sample_count"] == 2048
    assert cfg["domain"] == "linear"
    assert cfg["gt_color_space"] == "srgb"
    assert cfg["strict_keys"] is False
    assert cfg["pos_key"] == "world_pos"
    assert cfg["normal_key"] == "world_normal"
    assert cfg["albedo_key"] == "base_color"
    assert cfg["gt_linear_key"] == "gt_image"
    assert cfg["light_params_key"] == "lights"
    assert cfg["light_mask_key"] == "lights_mask"
    assert cfg["valid_mask_key"] == "valid"


def test_resolve_gbuffer_image_loss_config_rejects_invalid_domain() -> None:
    args = Namespace(
        gbuffer_image_loss_enabled=False,
        gbuffer_image_loss_lambda=0.0,
        gbuffer_image_loss_warmup_epochs=0,
        gbuffer_image_loss_every_steps=16,
        gbuffer_image_loss_type="charbonnier",
        gbuffer_image_loss_dataset_root="",
        gbuffer_image_loss_pixel_sample_count=8192,
        _config_obj={
            "training": {
                "gbuffer_image_loss": {
                    "enabled": True,
                    "lambda": 0.05,
                    "dataset_root": "tmp/gbuffer",
                    "domain": "srgb",
                }
            }
        },
    )
    with pytest.raises(ValueError):
        runtime_cfg.resolve_gbuffer_image_loss_config(args)


def test_resolve_proxy_gbuffer_handover_config_rejects_invalid_epoch_order() -> None:
    args = Namespace(
        _config_obj={
            "training": {
                "proxy_gbuffer_handover": {
                    "enabled": True,
                    "start_epoch": 220,
                    "end_epoch": 180,
                }
            }
        }
    )
    with pytest.raises(ValueError):
        runtime_cfg.resolve_proxy_gbuffer_handover_config(args)
