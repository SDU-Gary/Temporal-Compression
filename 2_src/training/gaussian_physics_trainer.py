"""Shared trainer for Gaussian-Physics model variants."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import json
import os
import random
import re
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from .loss_effective_contribution import LossEffectiveContributionController


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    """Charbonnier loss (smooth L1) for robust reconstruction."""
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + epsilon * epsilon))


@dataclass
class BatchAdapter:
    """Map dataset batch keys into (positions, params, targets)."""

    positions_key: str = "probe_position"
    params_key: str = "light_params"
    target_key: str = "sh_coeffs"
    mask_key: Optional[str] = None
    params_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    target_inverse: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    non_blocking_transfer: bool = True

    def unpack(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        non_blocking = bool(self.non_blocking_transfer)
        positions = batch[self.positions_key].to(device, non_blocking=non_blocking)
        params = batch[self.params_key].to(device, non_blocking=non_blocking)
        targets = batch[self.target_key].to(device, non_blocking=non_blocking)

        if self.params_transform is not None:
            params = self.params_transform(params)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        if self.mask_key is not None and self.mask_key in batch:
            mask = batch[self.mask_key].to(device, non_blocking=non_blocking)
            return positions, params, targets, mask

        return positions, params, targets

    def inverse_targets(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.target_inverse is None:
            return preds, targets
        return self.target_inverse(preds), self.target_inverse(targets)


try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None


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
    if name.startswith("bypass_proj."):
        return "encoder"
    if name.startswith("bypass_mlp."):
        return "encoder"
    if name.startswith("gamma.") or name.startswith("beta."):
        return "film"
    return "other"


class GaussianPhysicsTrainer:
    """Standardized trainer for Gaussian-Physics variants."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        adapter: BatchAdapter,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        lr_scheduler: str = "none",
        lr_min: float = 1e-4,
        warmup_epochs: int = 0,
        recon_loss: str = "mse",
        recon_weight: float = 1.0,
        charbonnier_eps: float = 1e-3,
        temporal_weight: float = 0.0,
        temporal_loss_fn: Optional[Callable[[nn.Module], torch.Tensor]] = None,
        top_k: int = 3,
        grad_clip: Optional[float] = 1.0,
        linearity_weight: float = 0.0,
        linearity_aug_pairs: int = 0,
        linearity_every_steps: int = 4,
        spatial_weight: float = 0.0,
        spatial_k: int = 1,
        image_loss_weight: float = 0.0,
        image_loss_warmup_epochs: int = 0,
        image_loss_type: str = "mse",
        image_loss_huber_delta: float = 0.1,
        image_loss_domain: str = "radiance",
        image_samples: int = 64,
        image_sample_seed: int = 42,
        image_sampling_mode: str = "fixed",
        image_soft_saturation_enabled: bool = False,
        image_soft_saturation_mode: str = "exp",
        image_soft_saturation_k: float = 1.0,
        image_loss_space: str = "linear",
        proxy_image_loss_mix_mode: str = "linear_log_mix",
        proxy_image_log_mix_weight: float = 0.3,
        enable_weighted_sh_loss: bool = False,
        sh_loss_weights: Optional[list[float]] = None,
        sh_weight_mode: str = "basis",
        compute_img_metrics_in_val: bool = False,
        compute_superposition_in_val: bool = False,
        param_group_lrs: Optional[Dict[str, float]] = None,
        lambda_routing_balance: float = 0.0,
        routing_balance_anneal_enabled: bool = False,
        routing_balance_anneal_start: Optional[float] = None,
        routing_balance_anneal_end: float = 0.0,
        routing_balance_anneal_epochs: int = 0,
        routing_balance_anneal_epoch_offset: int = 0,
        routing_soft_train: bool = False,
        routing_soft_topk: int = 0,
        routing_temp_start: float = 1.0,
        routing_temp_end: float = 1.0,
        routing_temp_anneal_epochs: int = 0,
        train_routing_param_mode: str = "gather",
        ema_enabled: bool = False,
        ema_decay: float = 0.999,
        ema_eval: bool = False,
        ema_save_best: bool = False,
        rerun_logger=None,  # Optional RerunLogger instance
        show_progress: bool = True,
        amp_mode: str = "off",
        grad_norm_log_every_steps: int = 100,
        raw_eval_every_epochs: int = 1,
        max_train_batches: int = 0,
        max_val_batches: int = 0,
        enable_soft_profile_sharing: bool = False,
        soft_profile_equiv_check_batches: int = 2,
        soft_profile_mae_tolerance: float = 1e-6,
        soft_profile_img_psnr_tolerance: float = 5e-4,
        gbuffer_image_loss_cfg: Optional[Dict[str, Any]] = None,
        proxy_gbuffer_handover_cfg: Optional[Dict[str, Any]] = None,
        proxy_gbuffer_handover_epoch_offset: int = 0,
        loss_effective_contribution_cfg: Optional[Dict[str, Any]] = None,
        optimization_monitoring_cfg: Optional[Dict[str, Any]] = None,
        gbuffer_loader: Optional[torch.utils.data.DataLoader] = None,
        gbuffer_probe_min: Optional[Any] = None,
        gbuffer_probe_max: Optional[Any] = None,
        gbuffer_gt_sh_tensor: Optional[Any] = None,
        gbuffer_gt_probe_positions: Optional[Any] = None,
        gbuffer_gt_light_configs: Optional[Any] = None,
        gbuffer_gt_light_mask: Optional[Any] = None,
        gbuffer_gt_frame_indices: Optional[Any] = None,
        gbuffer_gt_knn: int = 8,
        gbuffer_gt_weight_eps: float = 0.1,
        gbuffer_gt_chunk_size: int = 32768,
        profile_train: bool = False,
        profile_dir: Optional[str] = None,
        profile_wait: int = 1,
        profile_warmup: int = 1,
        profile_active: int = 3,
        profile_repeat: int = 1,
        profile_record_shapes: bool = False,
        profile_with_stack: bool = False,
        profile_memory: bool = False,
        cuda_graph_train: bool = False,
        cuda_graph_mode: str = "dual",
        cuda_graph_warmup_steps: int = 10,
        cuda_graph_fallback_eager: bool = True,
    ):
        self.model = model
        self.device = device
        self.adapter = adapter
        self.recon_loss = recon_loss
        self.recon_weight = max(0.0, float(recon_weight))
        self.charbonnier_eps = charbonnier_eps
        self.lr_scheduler = lr_scheduler
        self.lr_min = lr_min
        self.warmup_epochs = warmup_epochs
        self.base_lr = lr
        self.temporal_weight = temporal_weight
        self.temporal_loss_fn = temporal_loss_fn
        self.top_k = top_k
        self.grad_clip = grad_clip
        self.linearity_weight = linearity_weight
        self.linearity_aug_pairs = linearity_aug_pairs
        self.linearity_every_steps = max(1, int(linearity_every_steps))
        self.spatial_weight = spatial_weight
        self.spatial_k = spatial_k
        self.image_loss_weight_target = max(0.0, float(image_loss_weight))
        self.image_loss_warmup_epochs = max(0, int(image_loss_warmup_epochs))
        self._current_image_loss_weight = float(self.image_loss_weight_target)
        # Backward-compatible alias used by existing logs/checkpoints.
        self.image_loss_weight = float(self.image_loss_weight_target)
        self.image_loss_type = str(image_loss_type)
        self.image_loss_huber_delta = max(1e-8, float(image_loss_huber_delta))
        self.image_loss_domain = str(image_loss_domain).strip().lower()
        self.image_samples = max(1, int(image_samples))
        self.image_sample_seed = int(image_sample_seed)
        self.image_sampling_mode = str(image_sampling_mode).strip().lower()
        self.image_soft_saturation_enabled = bool(image_soft_saturation_enabled)
        self.image_soft_saturation_mode = str(image_soft_saturation_mode).strip().lower()
        self.image_soft_saturation_k = max(1e-8, float(image_soft_saturation_k))
        self._image_sampling_seed_current = int(self.image_sample_seed)
        self._image_sampling_epoch = 0
        self.image_loss_space = str(image_loss_space)
        self.proxy_image_loss_mix_mode = str(proxy_image_loss_mix_mode).strip().lower()
        self.proxy_image_log_mix_weight = float(np.clip(float(proxy_image_log_mix_weight), 0.0, 1.0))
        gbuffer_cfg = dict(gbuffer_image_loss_cfg or {})
        self.gbuffer_image_loss_enabled = bool(gbuffer_cfg.get("enabled", False))
        self.gbuffer_image_loss_weight_target = max(
            0.0,
            float(gbuffer_cfg.get("lambda", 0.0)),
        )
        self.gbuffer_image_loss_warmup_epochs = max(0, int(gbuffer_cfg.get("warmup_epochs", 0)))
        self.gbuffer_image_loss_every_steps = max(1, int(gbuffer_cfg.get("every_steps", 16)))
        # Enforce smooth loss for differentiable image-loss branch.
        self.gbuffer_image_loss_type = "charbonnier"
        self.gbuffer_image_loss_pixel_sample_count = max(1, int(gbuffer_cfg.get("pixel_sample_count", 8192)))
        self.gbuffer_image_loss_dataset_root = str(gbuffer_cfg.get("dataset_root", "") or "").strip()
        self.gbuffer_image_loss_domain = str(gbuffer_cfg.get("domain", "linear")).strip().lower()
        self.gbuffer_image_loss_gt_color_space = str(gbuffer_cfg.get("gt_color_space", "linear")).strip().lower()
        self.gbuffer_image_loss_strict_keys = bool(gbuffer_cfg.get("strict_keys", True))
        self.gbuffer_image_loss_pos_key = str(gbuffer_cfg.get("pos_key", "posW") or "").strip() or "posW"
        self.gbuffer_image_loss_normal_key = str(gbuffer_cfg.get("normal_key", "normW") or "").strip() or "normW"
        self.gbuffer_image_loss_albedo_key = str(gbuffer_cfg.get("albedo_key", "albedo") or "").strip() or "albedo"
        self.gbuffer_image_loss_gt_linear_key = (
            str(gbuffer_cfg.get("gt_linear_key", "gt_linear") or "").strip() or "gt_linear"
        )
        self.gbuffer_image_loss_light_params_key = (
            str(gbuffer_cfg.get("light_params_key", "light_params") or "").strip() or "light_params"
        )
        self.gbuffer_image_loss_light_mask_key = (
            str(gbuffer_cfg.get("light_mask_key", "light_mask") or "").strip() or "light_mask"
        )
        self.gbuffer_image_loss_valid_mask_key = (
            str(gbuffer_cfg.get("valid_mask_key", "valid_mask") or "").strip() or "valid_mask"
        )
        self.gbuffer_image_loss_frame_idx_key = (
            str(gbuffer_cfg.get("frame_idx_key", "frame_idx") or "").strip() or "frame_idx"
        )
        target_source_raw = str(gbuffer_cfg.get("target_source", "dataset_sh")).strip().lower()
        if (
            "target_source" not in gbuffer_cfg
            and (gbuffer_gt_probe_positions is None or gbuffer_gt_sh_tensor is None)
        ):
            target_source_raw = "gbuffer_linear"
        self.gbuffer_image_loss_target_source = target_source_raw
        self.gbuffer_image_loss_gt_knn = max(
            1,
            int(gbuffer_cfg.get("gt_knn", gbuffer_gt_knn)),
        )
        self.gbuffer_image_loss_gt_weight_eps = max(
            1e-8,
            float(gbuffer_cfg.get("gt_weight_eps", gbuffer_gt_weight_eps)),
        )
        self.gbuffer_image_loss_gt_chunk_size = max(
            1,
            int(gbuffer_cfg.get("gt_chunk_size", gbuffer_gt_chunk_size)),
        )
        self._current_gbuffer_image_loss_weight = float(self.gbuffer_image_loss_weight_target)
        self._gbuffer_loader = gbuffer_loader
        self._gbuffer_iter = None
        self._gbuffer_probe_min = None
        self._gbuffer_probe_max = None
        self._gbuffer_gt_probe_positions_world: Optional[torch.Tensor] = None
        self._gbuffer_gt_sh_tensor_cpu: Optional[torch.Tensor] = None
        self._gbuffer_gt_light_configs_cpu: Optional[torch.Tensor] = None
        self._gbuffer_gt_light_mask_cpu: Optional[torch.Tensor] = None
        self._gbuffer_gt_frame_to_cfg_idx: Dict[int, int] = {}
        if self.gbuffer_image_loss_enabled:
            if self._gbuffer_loader is None:
                raise ValueError("gbuffer_image_loss enabled but gbuffer_loader is None")
            if gbuffer_probe_min is None or gbuffer_probe_max is None:
                raise ValueError("gbuffer_image_loss enabled but probe_min/probe_max are missing")
            min_np = np.asarray(gbuffer_probe_min, dtype=np.float32).reshape(1, 3)
            max_np = np.asarray(gbuffer_probe_max, dtype=np.float32).reshape(1, 3)
            self._gbuffer_probe_min = torch.from_numpy(min_np).to(self.device)
            self._gbuffer_probe_max = torch.from_numpy(max_np).to(self.device)
            if self.gbuffer_image_loss_target_source not in {"dataset_sh", "gbuffer_linear"}:
                raise ValueError(
                    f"Unsupported gbuffer_image_loss target_source: {self.gbuffer_image_loss_target_source}. "
                    "Expected one of {'dataset_sh', 'gbuffer_linear'}."
                )
            if self.gbuffer_image_loss_target_source == "dataset_sh":
                if gbuffer_gt_probe_positions is None or gbuffer_gt_sh_tensor is None:
                    raise ValueError(
                        "gbuffer_image_loss target_source=dataset_sh requires "
                        "gbuffer_gt_probe_positions and gbuffer_gt_sh_tensor."
                    )
                if gbuffer_gt_light_configs is None:
                    raise ValueError(
                        "gbuffer_image_loss target_source=dataset_sh requires gbuffer_gt_light_configs."
                    )
                probe_np = np.asarray(gbuffer_gt_probe_positions, dtype=np.float32).reshape(-1, 3)
                gt_sh_np = np.asarray(gbuffer_gt_sh_tensor, dtype=np.float32)
                if gt_sh_np.ndim != 3 or gt_sh_np.shape[-1] != 27:
                    raise ValueError(
                        f"gbuffer_gt_sh_tensor must be [P,M,27], got {tuple(gt_sh_np.shape)}"
                    )
                if gt_sh_np.shape[0] != probe_np.shape[0]:
                    raise ValueError(
                        "gbuffer_gt_sh_tensor/probe_positions size mismatch: "
                        f"{gt_sh_np.shape[0]} vs {probe_np.shape[0]}"
                    )
                light_cfg_np = np.asarray(gbuffer_gt_light_configs, dtype=np.float32)
                if light_cfg_np.ndim == 2:
                    light_cfg_np = light_cfg_np[:, None, :]
                if light_cfg_np.ndim != 3:
                    raise ValueError(
                        f"gbuffer_gt_light_configs must be [M,S,F], got {tuple(light_cfg_np.shape)}"
                    )
                if light_cfg_np.shape[0] != gt_sh_np.shape[1]:
                    raise ValueError(
                        "gbuffer_gt_light_configs config count mismatch with gbuffer_gt_sh_tensor: "
                        f"{light_cfg_np.shape[0]} vs {gt_sh_np.shape[1]}"
                    )
                if gbuffer_gt_light_mask is None:
                    light_mask_np = np.ones(light_cfg_np.shape[:2], dtype=np.float32)
                else:
                    light_mask_np = np.asarray(gbuffer_gt_light_mask, dtype=np.float32)
                    if light_mask_np.ndim == 1:
                        light_mask_np = light_mask_np[:, None]
                    if light_mask_np.shape[0] != light_cfg_np.shape[0]:
                        raise ValueError(
                            "gbuffer_gt_light_mask config count mismatch with gbuffer_gt_light_configs: "
                            f"{light_mask_np.shape[0]} vs {light_cfg_np.shape[0]}"
                        )
                    if light_mask_np.shape[1] != light_cfg_np.shape[1]:
                        raise ValueError(
                            "gbuffer_gt_light_mask slot count mismatch with gbuffer_gt_light_configs: "
                            f"{light_mask_np.shape[1]} vs {light_cfg_np.shape[1]}"
                        )
                if gbuffer_gt_frame_indices is None:
                    frame_idx_np = np.arange(gt_sh_np.shape[1], dtype=np.int64)
                else:
                    frame_idx_np = np.asarray(gbuffer_gt_frame_indices, dtype=np.int64).reshape(-1)
                    if frame_idx_np.shape[0] != gt_sh_np.shape[1]:
                        raise ValueError(
                            "gbuffer_gt_frame_indices length mismatch with config count: "
                            f"{frame_idx_np.shape[0]} vs {gt_sh_np.shape[1]}"
                        )
                self._gbuffer_gt_probe_positions_world = torch.from_numpy(
                    np.ascontiguousarray(probe_np, dtype=np.float32)
                ).to(self.device)
                self._gbuffer_gt_sh_tensor_cpu = torch.from_numpy(
                    np.ascontiguousarray(gt_sh_np, dtype=np.float32)
                )
                self._gbuffer_gt_light_configs_cpu = torch.from_numpy(
                    np.ascontiguousarray(light_cfg_np, dtype=np.float32)
                )
                self._gbuffer_gt_light_mask_cpu = torch.from_numpy(
                    np.ascontiguousarray(light_mask_np, dtype=np.float32)
                )
                self._gbuffer_gt_frame_to_cfg_idx = {
                    int(frame_idx_np[i]): int(i) for i in range(int(frame_idx_np.shape[0]))
                }
        handover_cfg = dict(proxy_gbuffer_handover_cfg or {})
        self.proxy_gbuffer_handover_enabled = bool(handover_cfg.get("enabled", False))
        self.proxy_gbuffer_handover_epoch_offset = max(0, int(proxy_gbuffer_handover_epoch_offset))
        self.proxy_gbuffer_handover_start_epoch = max(1, int(handover_cfg.get("start_epoch", 160)))
        self.proxy_gbuffer_handover_end_epoch = max(
            self.proxy_gbuffer_handover_start_epoch,
            int(handover_cfg.get("end_epoch", 220)),
        )
        self.proxy_gbuffer_handover_proxy_start_scale = max(
            0.0,
            float(handover_cfg.get("proxy_start_scale", 1.0)),
        )
        self.proxy_gbuffer_handover_proxy_end_scale = max(
            0.0,
            float(handover_cfg.get("proxy_end_scale", 0.0)),
        )
        self.proxy_gbuffer_handover_gbuffer_start_scale = max(
            0.0,
            float(handover_cfg.get("gbuffer_start_scale", 0.0)),
        )
        self.proxy_gbuffer_handover_gbuffer_end_scale = max(
            0.0,
            float(handover_cfg.get("gbuffer_end_scale", 1.0)),
        )
        if self.proxy_gbuffer_handover_enabled and (
            (not self.gbuffer_image_loss_enabled)
            or float(self.gbuffer_image_loss_weight_target) <= 0.0
        ):
            raise ValueError(
                "proxy_gbuffer_handover enabled but gbuffer_image_loss is disabled "
                "or has zero lambda target."
            )
        self._current_proxy_gbuffer_handover_progress = 0.0
        self._current_proxy_gbuffer_proxy_scale = 1.0
        self._current_proxy_gbuffer_gbuffer_scale = 1.0
        self.loss_effective_contribution_cfg = dict(loss_effective_contribution_cfg or {})
        self._loss_effective = LossEffectiveContributionController.from_config(
            self.loss_effective_contribution_cfg
        )
        mon_cfg = dict(optimization_monitoring_cfg or {})
        self.optimization_monitoring_cfg = mon_cfg
        self.monitor_enabled = bool(mon_cfg.get("enabled", True))
        shared_groups_raw = mon_cfg.get(
            "shared_groups",
            ["routing", "basis", "coeff", "encoder", "film"],
        )
        if isinstance(shared_groups_raw, str):
            shared_groups = [x.strip() for x in shared_groups_raw.split(",") if x.strip()]
        elif isinstance(shared_groups_raw, (list, tuple)):
            shared_groups = [str(x).strip() for x in shared_groups_raw if str(x).strip()]
        else:
            shared_groups = ["routing", "basis", "coeff", "encoder", "film"]
        self.monitor_shared_groups = tuple(shared_groups)
        self.monitor_shared_group_set = set(self.monitor_shared_groups)
        self.monitor_grad_diagnostics_every_steps = max(
            0, int(mon_cfg.get("grad_diagnostics_every_steps", 50))
        )
        self.monitor_update_ratio_every_steps = max(
            0, int(mon_cfg.get("update_ratio_every_steps", 20))
        )
        self.monitor_spectrum_every_epochs = max(
            0, int(mon_cfg.get("spectrum_every_epochs", 1))
        )
        self.monitor_pulse_recovery_tolerance = max(
            0.0, float(mon_cfg.get("pulse_recovery_tolerance", 0.02))
        )
        self.monitor_pulse_max_steps = max(
            1,
            int(
                mon_cfg.get(
                    "pulse_recovery_max_steps",
                    max(1, int(self.gbuffer_image_loss_every_steps) * 4),
                )
            ),
        )
        self.monitor_gns_enabled = bool(mon_cfg.get("gns_enabled", True))
        self._monitor_prev_total_grad_vec: Optional[torch.Tensor] = None
        self._current_epoch_for_monitor = 0
        self.enable_weighted_sh_loss = bool(enable_weighted_sh_loss)
        self.sh_weight_mode = str(sh_weight_mode)
        self.compute_img_metrics_in_val = bool(compute_img_metrics_in_val)
        self.compute_superposition_in_val = bool(compute_superposition_in_val)
        self.param_group_lrs = dict(param_group_lrs or {})
        self.lambda_routing_balance = max(0.0, float(lambda_routing_balance))
        self.routing_balance_anneal_enabled = bool(routing_balance_anneal_enabled)
        self.routing_balance_anneal_start = (
            self.lambda_routing_balance
            if routing_balance_anneal_start is None
            else max(0.0, float(routing_balance_anneal_start))
        )
        self.routing_balance_anneal_end = max(0.0, float(routing_balance_anneal_end))
        self.routing_balance_anneal_epochs = max(0, int(routing_balance_anneal_epochs))
        self.routing_balance_anneal_epoch_offset = max(0, int(routing_balance_anneal_epoch_offset))
        self._current_lambda_routing_balance = float(self.lambda_routing_balance)
        self.routing_soft_train = bool(routing_soft_train)
        self.routing_soft_topk = int(routing_soft_topk)
        self.routing_temp_start = float(routing_temp_start)
        self.routing_temp_end = float(routing_temp_end)
        self.routing_temp_anneal_epochs = max(0, int(routing_temp_anneal_epochs))
        self._current_routing_temp = float(self.routing_temp_start)
        self._train_routing_mode = False
        self.train_routing_param_mode = str(train_routing_param_mode).strip().lower()
        if self.train_routing_param_mode not in {"gather", "dense_masked"}:
            raise ValueError(
                f"Unsupported train_routing_param_mode: {train_routing_param_mode}. "
                "Expected one of {'gather', 'dense_masked'}."
            )
        self.ema_enabled = bool(ema_enabled)
        self.ema_decay = float(ema_decay)
        self.ema_eval = bool(ema_eval and self.ema_enabled)
        self.ema_save_best = bool(ema_save_best and self.ema_enabled)
        self.rerun_logger = rerun_logger
        self.show_progress = show_progress
        self._progress_prefix = ""
        self.grad_norm_log_every_steps = max(0, int(grad_norm_log_every_steps))
        self.raw_eval_every_epochs = max(1, int(raw_eval_every_epochs))
        self.max_train_batches = max(0, int(max_train_batches))
        self.max_val_batches = max(0, int(max_val_batches))
        self.enable_soft_profile_sharing = bool(enable_soft_profile_sharing)
        self.soft_profile_equiv_check_batches = max(1, int(soft_profile_equiv_check_batches))
        self.soft_profile_mae_tolerance = float(max(0.0, soft_profile_mae_tolerance))
        self.soft_profile_img_psnr_tolerance = float(max(0.0, soft_profile_img_psnr_tolerance))
        self.profile_train = bool(profile_train)
        self.profile_dir = profile_dir
        self.profile_wait = max(0, int(profile_wait))
        self.profile_warmup = max(0, int(profile_warmup))
        self.profile_active = max(1, int(profile_active))
        self.profile_repeat = max(1, int(profile_repeat))
        self.profile_record_shapes = bool(profile_record_shapes)
        self.profile_with_stack = bool(profile_with_stack)
        self.profile_memory = bool(profile_memory)
        self.cuda_graph_train = bool(cuda_graph_train)
        self.cuda_graph_mode = str(cuda_graph_mode).strip().lower()
        if self.cuda_graph_mode not in {"single", "dual"}:
            raise ValueError(
                f"Unsupported cuda_graph_mode: {cuda_graph_mode}. "
                "Expected one of {'single', 'dual'}."
            )
        self.cuda_graph_warmup_steps = max(0, int(cuda_graph_warmup_steps))
        self.cuda_graph_fallback_eager = bool(cuda_graph_fallback_eager)
        self._cuda_graph_requested = bool(self.cuda_graph_train)
        self._cuda_graph_enabled = bool(self.cuda_graph_train and self.device.type == "cuda")
        self._cuda_graph_optimizer_capturable = bool(self._cuda_graph_enabled)
        self._profiler: Any = None
        self._profiler_output_dir: Optional[Path] = None
        self._profiler_trace_count = 0
        self._profiler_step_count = 0
        self._profiler_error_reported = False
        self._disabled_soft_profile_share_keys: set[tuple[int, Optional[int]]] = set()
        self._verified_soft_profile_share_keys: set[tuple[int, Optional[int]]] = set()
        self._cuda_graph_states: dict[str, Dict[str, Any]] = {}
        self._cuda_graph_stats: Dict[str, int] = {}
        self._probe_positions = None
        self._probe_neighbors = None
        self._optimizer_group_names: list[str] = []
        self._optimizer_group_base_lrs: list[float] = []
        self._ema_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.amp_mode = str(amp_mode).strip().lower()
        if self.amp_mode not in {"off", "bf16", "fp16"}:
            raise ValueError(f"Unsupported amp_mode: {amp_mode}")
        if self.device.type != "cuda":
            self.amp_mode = "off"
        self._use_amp = bool(self.amp_mode in {"bf16", "fp16"} and self.device.type == "cuda")
        self._amp_dtype = (
            torch.bfloat16 if self.amp_mode == "bf16" else (torch.float16 if self.amp_mode == "fp16" else None)
        )
        scaler_device = "cuda" if self.device.type == "cuda" else "cpu"
        self._grad_scaler = torch.amp.GradScaler(
            scaler_device,
            enabled=(self.amp_mode == "fp16" and self.device.type == "cuda"),
        )

        if self.image_loss_type not in {"mse", "charbonnier", "huber"}:
            raise ValueError(f"Unsupported image_loss_type: {self.image_loss_type}")
        if self.image_loss_domain not in {"radiance", "irradiance"}:
            raise ValueError(f"Unsupported image_loss_domain: {self.image_loss_domain}")
        if self.image_loss_space not in {"linear", "srgb"}:
            raise ValueError(f"Unsupported image_loss_space: {self.image_loss_space}")
        if self.image_sampling_mode not in {"fixed", "per_epoch", "per_step"}:
            raise ValueError(f"Unsupported image_sampling_mode: {self.image_sampling_mode}")
        if self.image_soft_saturation_mode not in {"exp"}:
            raise ValueError(f"Unsupported image_soft_saturation_mode: {self.image_soft_saturation_mode}")
        if self.proxy_image_loss_mix_mode not in {"linear_log_mix", "linear_only", "log_only"}:
            raise ValueError(f"Unsupported proxy_image_loss_mix_mode: {self.proxy_image_loss_mix_mode}")
        if self.gbuffer_image_loss_type not in {"charbonnier"}:
            raise ValueError(
                f"Unsupported gbuffer_image_loss_type: {self.gbuffer_image_loss_type}"
            )
        if self.gbuffer_image_loss_target_source not in {"dataset_sh", "gbuffer_linear"}:
            raise ValueError(
                f"Unsupported gbuffer_image_loss_target_source: {self.gbuffer_image_loss_target_source}"
            )
        if self.gbuffer_image_loss_domain not in {"linear"}:
            raise ValueError(
                f"Unsupported gbuffer_image_loss_domain: {self.gbuffer_image_loss_domain}. "
                "Expected 'linear'."
            )
        if self.gbuffer_image_loss_gt_color_space not in {"linear", "srgb"}:
            raise ValueError(
                f"Unsupported gbuffer_image_loss_gt_color_space: {self.gbuffer_image_loss_gt_color_space}"
            )
        if self.sh_weight_mode != "basis":
            raise ValueError(f"Unsupported sh_weight_mode: {self.sh_weight_mode}")

        if sh_loss_weights is None:
            sh_loss_weights = [3.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5]
        if len(sh_loss_weights) != 9:
            raise ValueError("sh_loss_weights must contain exactly 9 values")
        sh_w9 = torch.tensor(sh_loss_weights, dtype=torch.float32)
        self._sh_loss_weights_27 = torch.cat([sh_w9, sh_w9, sh_w9], dim=0)
        self._image_basis_cache: Dict[tuple[str, str, int, str], torch.Tensor] = {}
        self._irradiance_band_weights_9 = torch.tensor(
            [
                float(np.pi),
                float(2.0 * np.pi / 3.0),
                float(2.0 * np.pi / 3.0),
                float(2.0 * np.pi / 3.0),
                float(np.pi / 4.0),
                float(np.pi / 4.0),
                float(np.pi / 4.0),
                float(np.pi / 4.0),
                float(np.pi / 4.0),
            ],
            dtype=torch.float32,
        )
        self._irradiance_positive_beta = 8.0

        self.optimizer = self._build_optimizer(lr=lr, weight_decay=weight_decay)
        self._init_cuda_graph_state()
        if self.ema_enabled:
            self._init_ema_state()

        if recon_loss == "mse":
            self.criterion = nn.MSELoss()
        elif recon_loss == "l1":
            self.criterion = nn.L1Loss()
        elif recon_loss == "charbonnier":
            self.criterion = None
        else:
            raise ValueError(f"Unknown recon_loss: {recon_loss}")

    def _build_optimizer(self, lr: float, weight_decay: float) -> torch.optim.Optimizer:
        named_trainable_params = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]
        if not named_trainable_params:
            raise ValueError("No trainable parameters found when building optimizer")

        group_overrides: Dict[str, float] = {}
        if self.param_group_lrs:
            for key, value in self.param_group_lrs.items():
                group_overrides[str(key)] = float(value)
        allowed_groups = {"all", "routing", "basis", "coeff", "encoder", "film", "other"}
        unknown_groups = sorted(k for k in group_overrides.keys() if k not in allowed_groups)
        if unknown_groups:
            raise ValueError(f"Unknown param_group_lrs keys: {unknown_groups}; allowed={sorted(allowed_groups)}")

        default_lr = float(group_overrides.get("all", lr))
        grouped_params: Dict[str, list[torch.nn.Parameter]] = {}
        for name, param in named_trainable_params:
            group = _param_group_from_name(name)
            grouped_params.setdefault(group, []).append(param)

        optimizer_groups = []
        self._optimizer_group_names = []
        self._optimizer_group_base_lrs = []
        for group_name, params in grouped_params.items():
            group_lr = float(group_overrides.get(group_name, default_lr))
            optimizer_groups.append(
                {
                    "params": params,
                    "lr": group_lr,
                    "group": group_name,
                }
            )
            self._optimizer_group_names.append(group_name)
            self._optimizer_group_base_lrs.append(group_lr)

        adam_kwargs: Dict[str, Any] = {}
        if self._cuda_graph_optimizer_capturable and self.device.type == "cuda":
            adam_kwargs["capturable"] = True
        try:
            return torch.optim.Adam(
                optimizer_groups,
                lr=default_lr,
                weight_decay=weight_decay,
                **adam_kwargs,
            )
        except TypeError:
            self._cuda_graph_optimizer_capturable = False
            return torch.optim.Adam(
                optimizer_groups,
                lr=default_lr,
                weight_decay=weight_decay,
            )

    def _autocast_context(self):
        if not self._use_amp or self._amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=True)

    def _resolve_profile_output_dir(self, output_dir: Optional[Path]) -> Optional[Path]:
        if self.profile_dir is not None and str(self.profile_dir).strip():
            return Path(str(self.profile_dir))
        if output_dir is None:
            return None
        return Path(output_dir) / "profiling"

    def _profile_range(self, name: str):
        if self._profiler is None:
            return nullcontext()
        try:
            return torch.profiler.record_function(name)
        except Exception:
            return nullcontext()

    def _init_profiler(self, output_dir: Optional[Path]) -> None:
        self._profiler = None
        self._profiler_output_dir = None
        self._profiler_trace_count = 0
        self._profiler_step_count = 0
        self._profiler_error_reported = False
        if not self.profile_train:
            return
        if not hasattr(torch, "profiler") or not hasattr(torch.profiler, "profile"):
            print("Warning: torch.profiler is unavailable; disable profiling.")
            return

        profiler_output_dir = self._resolve_profile_output_dir(output_dir)
        if profiler_output_dir is None:
            print("Warning: profiling is enabled but no output directory is available; disable profiling.")
            return
        profiler_output_dir.mkdir(parents=True, exist_ok=True)

        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device.type == "cuda" and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        schedule = torch.profiler.schedule(
            wait=int(self.profile_wait),
            warmup=int(self.profile_warmup),
            active=int(self.profile_active),
            repeat=int(self.profile_repeat),
        )
        sort_key = (
            "self_cuda_time_total"
            if torch.profiler.ProfilerActivity.CUDA in activities
            else "self_cpu_time_total"
        )

        def _trace_handler(trace_profiler: torch.profiler.profile) -> None:
            trace_idx = int(self._profiler_trace_count)
            self._profiler_trace_count += 1
            trace_path = profiler_output_dir / f"trace_{trace_idx:03d}.json"
            summary_path = profiler_output_dir / f"summary_{trace_idx:03d}.txt"
            try:
                trace_profiler.export_chrome_trace(str(trace_path))
            except Exception as exc:
                print(f"Warning: failed to export profiler trace {trace_path}: {exc}")
            try:
                summary_table = trace_profiler.key_averages(
                    group_by_input_shape=bool(self.profile_record_shapes)
                ).table(sort_by=str(sort_key), row_limit=200)
                summary_path.write_text(summary_table, encoding="utf-8")
            except Exception as exc:
                print(f"Warning: failed to export profiler summary {summary_path}: {exc}")

        try:
            profiler = torch.profiler.profile(
                activities=activities,
                schedule=schedule,
                on_trace_ready=_trace_handler,
                record_shapes=bool(self.profile_record_shapes),
                profile_memory=bool(self.profile_memory),
                with_stack=bool(self.profile_with_stack),
            )
            profiler.__enter__()
        except Exception as exc:
            print(f"Warning: failed to initialize torch.profiler: {exc}")
            return

        self._profiler = profiler
        self._profiler_output_dir = profiler_output_dir
        config_payload = {
            "enabled": True,
            "activities": [
                "cuda" if activity == torch.profiler.ProfilerActivity.CUDA else "cpu"
                for activity in activities
            ],
            "wait": int(self.profile_wait),
            "warmup": int(self.profile_warmup),
            "active": int(self.profile_active),
            "repeat": int(self.profile_repeat),
            "record_shapes": bool(self.profile_record_shapes),
            "with_stack": bool(self.profile_with_stack),
            "profile_memory": bool(self.profile_memory),
        }
        try:
            (profiler_output_dir / "config.json").write_text(
                json.dumps(config_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        if self.show_progress:
            print(f"[profiler] enabled -> {profiler_output_dir}")

    def _profile_step(self) -> None:
        if self._profiler is None:
            return
        try:
            self._profiler.step()
            self._profiler_step_count += 1
        except Exception as exc:
            if not self._profiler_error_reported:
                print(f"Warning: profiler.step failed, disable profiling: {exc}")
                self._profiler_error_reported = True
            self._close_profiler()

    def _close_profiler(self) -> None:
        profiler = self._profiler
        profiler_output_dir = self._profiler_output_dir
        self._profiler = None
        self._profiler_output_dir = None
        if profiler is None:
            return
        try:
            profiler.__exit__(None, None, None)
        except Exception as exc:
            print(f"Warning: failed to close profiler cleanly: {exc}")
        if profiler_output_dir is not None:
            try:
                summary_payload = {
                    "steps": int(self._profiler_step_count),
                    "trace_count": int(self._profiler_trace_count),
                }
                (profiler_output_dir / "run_summary.json").write_text(
                    json.dumps(summary_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
            if self.show_progress:
                print(
                    f"[profiler] completed steps={self._profiler_step_count} "
                    f"traces={self._profiler_trace_count} dir={profiler_output_dir}"
                )

    def _init_cuda_graph_state(self) -> None:
        self._cuda_graph_states = {
            "regular": {
                "captured": False,
                "disabled": False,
                "disable_reason": "",
                "warmup_remaining": int(self.cuda_graph_warmup_steps),
                "graph": None,
                "inputs": None,
                "outputs": None,
            },
            "linearity": {
                "captured": False,
                "disabled": False,
                "disable_reason": "",
                "warmup_remaining": int(self.cuda_graph_warmup_steps),
                "graph": None,
                "inputs": None,
                "outputs": None,
            },
        }
        self._cuda_graph_stats = {
            "requested": int(self._cuda_graph_requested),
            "enabled": int(self._cuda_graph_enabled),
            "capture_regular": 0,
            "capture_linearity": 0,
            "replay_regular": 0,
            "replay_linearity": 0,
            "fallback_eager": 0,
            "fallback_ineligible": 0,
            "fallback_warmup": 0,
            "fallback_capture_error": 0,
            "fallback_shape_mismatch": 0,
            "fallback_grad_norm_step": 0,
        }

    def _cuda_graph_branch_key(self, should_compute_linearity: bool) -> str:
        return "linearity" if bool(should_compute_linearity) else "regular"

    def _cuda_graph_can_use(self, *, should_compute_linearity: bool, should_log_grad: bool) -> tuple[bool, str]:
        if not self._cuda_graph_enabled:
            return False, "disabled"
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return False, "cuda_unavailable"
        if self.profile_train:
            return False, "torch_profiler_enabled"
        if self._grad_scaler.is_enabled():
            return False, "grad_scaler_enabled"
        if not self._cuda_graph_optimizer_capturable:
            return False, "optimizer_not_capturable"
        if self._current_lambda_routing_balance > 0.0:
            return False, "routing_balance_enabled"
        if self.spatial_weight > 0.0:
            return False, "spatial_loss_enabled"
        if float(self._current_image_loss_weight) > 0.0:
            return False, "image_loss_enabled"
        if self.gbuffer_image_loss_enabled and float(self._current_gbuffer_image_loss_weight) > 0.0:
            return False, "gbuffer_image_loss_enabled"
        if bool(should_log_grad):
            return False, "grad_norm_logging_step"
        if should_compute_linearity and self.cuda_graph_mode != "dual":
            return False, "linearity_graph_disabled"
        return True, ""

    @staticmethod
    def _cuda_graph_tensor_shape_dtype_device(t: Optional[torch.Tensor]) -> tuple[Any, Any, Any]:
        if t is None:
            return None, None, None
        return tuple(t.shape), str(t.dtype), str(t.device)

    def _cuda_graph_inputs_match(
        self,
        ref_inputs: Dict[str, Optional[torch.Tensor]],
        positions: torch.Tensor,
        params: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor],
        probe_idx: Optional[torch.Tensor],
    ) -> bool:
        checks = (
            ("positions", positions),
            ("params", params),
            ("targets", targets),
            ("mask", mask),
            ("probe_idx", probe_idx),
        )
        for key, value in checks:
            ref = ref_inputs.get(key)
            if (ref is None) != (value is None):
                return False
            if ref is None:
                continue
            if self._cuda_graph_tensor_shape_dtype_device(ref) != self._cuda_graph_tensor_shape_dtype_device(value):
                return False
        return True

    @staticmethod
    def _cuda_graph_copy_inputs(
        dst_inputs: Dict[str, Optional[torch.Tensor]],
        positions: torch.Tensor,
        params: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor],
        probe_idx: Optional[torch.Tensor],
    ) -> None:
        dst_inputs["positions"].copy_(positions, non_blocking=False)
        dst_inputs["params"].copy_(params, non_blocking=False)
        dst_inputs["targets"].copy_(targets, non_blocking=False)
        if dst_inputs.get("mask") is not None and mask is not None:
            dst_inputs["mask"].copy_(mask, non_blocking=False)
        if dst_inputs.get("probe_idx") is not None and probe_idx is not None:
            dst_inputs["probe_idx"].copy_(probe_idx, non_blocking=False)

    def _model_state_dict_clone(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        for key, value in self.model.state_dict().items():
            if torch.is_tensor(value):
                state[key] = value.detach().clone()
            else:
                state[key] = value
        return state

    def _model_state_dict_with_ema(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        ema_state = self._ema_state_dict or {}
        for key, value in self.model.state_dict().items():
            if not torch.is_tensor(value):
                state[key] = value
                continue
            if key in ema_state:
                state[key] = ema_state[key].to(device=value.device, dtype=value.dtype).detach().clone()
            else:
                state[key] = value.detach().clone()
        return state

    def _capture_rng_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            try:
                state["torch_cuda"] = torch.cuda.get_rng_state_all()
            except Exception:
                pass
        return state

    def _restore_rng_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        if "python" in state:
            try:
                random.setstate(state["python"])
            except Exception:
                pass
        if "numpy" in state:
            try:
                np.random.set_state(state["numpy"])
            except Exception:
                pass
        if "torch_cpu" in state:
            try:
                torch.random.set_rng_state(state["torch_cpu"])
            except Exception:
                pass
        if "torch_cuda" in state and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(state["torch_cuda"])
            except Exception:
                pass

    def _init_ema_state(self) -> None:
        self._ema_state_dict = {}
        for key, value in self.model.state_dict().items():
            if torch.is_tensor(value) and value.dtype.is_floating_point:
                self._ema_state_dict[key] = value.detach().clone()

    @torch.no_grad()
    def _update_ema(self) -> None:
        if not self.ema_enabled or self._ema_state_dict is None:
            return
        decay = float(self.ema_decay)
        for key, value in self.model.state_dict().items():
            if key not in self._ema_state_dict:
                continue
            src = value.detach()
            dst = self._ema_state_dict[key]
            if dst.device != src.device or dst.dtype != src.dtype:
                dst = dst.to(device=src.device, dtype=src.dtype)
                self._ema_state_dict[key] = dst
            dst.mul_(decay).add_(src, alpha=1.0 - decay)

    @contextmanager
    def _use_ema_weights(self):
        if not self.ema_enabled or self._ema_state_dict is None:
            yield
            return
        backup: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            model_state = self.model.state_dict()
            for key, ema_val in self._ema_state_dict.items():
                if key not in model_state:
                    continue
                current = model_state[key]
                if not torch.is_tensor(current):
                    continue
                backup[key] = current.detach().clone()
                current.copy_(ema_val.to(device=current.device, dtype=current.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                model_state = self.model.state_dict()
                for key, backup_val in backup.items():
                    if key in model_state and torch.is_tensor(model_state[key]):
                        model_state[key].copy_(backup_val)

    def _apply_warmup(self, epoch: int) -> bool:
        if self.warmup_epochs <= 0:
            return False
        if epoch > self.warmup_epochs:
            return False
        scale = float(epoch) / float(max(1, self.warmup_epochs))
        for param_group, base_lr in zip(self.optimizer.param_groups, self._optimizer_group_base_lrs):
            param_group["lr"] = float(base_lr * scale)
        return True

    def _current_lr_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for param_group in self.optimizer.param_groups:
            group_name = str(param_group.get("group", "group"))
            metrics[f"{group_name}"] = float(param_group.get("lr", 0.0))
        return metrics

    def _compute_temporal_loss(self) -> torch.Tensor:
        if self.temporal_weight <= 0:
            return torch.tensor(0.0, device=self.device)

        if self.temporal_loss_fn is not None:
            return self.temporal_loss_fn(self.model)

        if hasattr(self.model, "compute_temporal_smoothness_loss"):
            return self.model.compute_temporal_smoothness_loss()

        raise AttributeError(
            "temporal_weight > 0 but model has no compute_temporal_smoothness_loss and no temporal_loss_fn provided."
        )

    def _init_spatial_neighbors(self, train_loader: torch.utils.data.DataLoader) -> None:
        if self.spatial_weight <= 0 or self.spatial_k <= 0:
            return
        dataset = getattr(train_loader, "dataset", None)
        if dataset is None or not hasattr(dataset, "get_full_probe_positions"):
            return
        positions = dataset.get_full_probe_positions(normalized=True)
        if positions is None or len(positions) == 0:
            return
        pos = np.asarray(positions, dtype=np.float32)
        dists = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        np.fill_diagonal(dists, np.inf)
        k = min(self.spatial_k, dists.shape[1] - 1)
        if k <= 0:
            return
        neighbors = np.argsort(dists, axis=1)[:, :k]
        self._probe_positions = torch.from_numpy(pos).to(self.device)
        self._probe_neighbors = torch.from_numpy(neighbors.astype(np.int64)).to(self.device)

    def _compute_spatial_loss(
        self,
        preds: torch.Tensor,
        params: torch.Tensor,
        mask: Optional[torch.Tensor],
        probe_idx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.spatial_weight <= 0:
            return torch.tensor(0.0, device=self.device)
        if probe_idx is None or self._probe_neighbors is None or self._probe_positions is None:
            return torch.tensor(0.0, device=self.device)
        probe_idx = probe_idx.to(self.device)
        neighbors = self._probe_neighbors[probe_idx]  # [B, K]
        neighbor_pos = self._probe_positions[neighbors]  # [B, K, 3]
        B, K, _ = neighbor_pos.shape
        if K == 0:
            return torch.tensor(0.0, device=self.device)

        neighbor_pos_flat = neighbor_pos.reshape(B * K, -1)
        # TODO(perf): 这里会将 params/mask 扩展为 [B*K, ...]，邻居数大时显存与带宽开销明显。
        # 可考虑低频计算、采样邻居或重用同一 probe 的局部结果，减少重复前向。
        params_flat = params[:, None, ...].expand(B, K, *params.shape[1:]).reshape(B * K, *params.shape[1:])
        if mask is not None:
            mask_flat = mask[:, None, ...].expand(B, K, *mask.shape[1:]).reshape(B * K, *mask.shape[1:])
        else:
            mask_flat = None

        preds_neighbor = self._forward(neighbor_pos_flat, params_flat, mask_flat)
        preds_neighbor = preds_neighbor.reshape(B, K, -1)
        preds_self = preds[:, None, :].expand(B, K, -1)
        return torch.mean(torch.abs(preds_self - preds_neighbor))

    def _compute_routing(
        self,
        positions: torch.Tensor,
        for_training: bool = False,
        *,
        top_k: Optional[int] = None,
        training_soft_routing: Optional[bool] = None,
        routing_temperature: Optional[float] = None,
        routing_soft_topk: Optional[int] = None,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        effective_top_k = int(self.top_k if top_k is None else top_k)
        if training_soft_routing is None:
            effective_soft_routing = bool(for_training and self.routing_soft_train)
        else:
            effective_soft_routing = bool(training_soft_routing)
        effective_temperature = float(self._current_routing_temp if routing_temperature is None else routing_temperature)
        if routing_soft_topk is None:
            effective_soft_topk = self.routing_soft_topk if self.routing_soft_topk > 0 else None
        else:
            effective_soft_topk = int(routing_soft_topk)
            if effective_soft_topk <= 0:
                effective_soft_topk = None

        if hasattr(self.model, "compute_gaussian_routing"):
            try:
                return self.model.compute_gaussian_routing(
                    positions,
                    top_k=effective_top_k,
                    training_soft_routing=effective_soft_routing,
                    routing_temperature=effective_temperature,
                    routing_soft_topk=effective_soft_topk,
                )
            except TypeError:
                return self.model.compute_gaussian_routing(positions, top_k=effective_top_k)
        return None

    def _select_routing(
        self,
        routing: Optional[tuple[torch.Tensor, torch.Tensor]],
        index: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if routing is None:
            return None
        weights, topk_indices = routing
        return weights[index], topk_indices[index]

    def _forward(
        self,
        positions: torch.Tensor,
        params: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        routing: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        *,
        top_k: Optional[int] = None,
        training_soft_routing: Optional[bool] = None,
        routing_temperature: Optional[float] = None,
        routing_soft_topk: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward wrapper that handles optional light masks."""
        if routing is not None and hasattr(self.model, "forward_with_routing"):
            effective_param_mode = self.train_routing_param_mode if self._train_routing_mode else "gather"
            try:
                return self.model.forward_with_routing(
                    routing,
                    params,
                    light_mask=mask,
                    param_select_mode=effective_param_mode,
                )
            except TypeError:
                return self.model.forward_with_routing(routing, params)

        effective_top_k = int(self.top_k if top_k is None else top_k)
        use_soft_routing = bool(self._train_routing_mode and self.routing_soft_train)
        if training_soft_routing is not None:
            use_soft_routing = bool(training_soft_routing)
        effective_routing_temperature = float(
            self._current_routing_temp if routing_temperature is None else routing_temperature
        )
        if routing_soft_topk is None:
            effective_routing_soft_topk = self.routing_soft_topk if self.routing_soft_topk > 0 else None
        else:
            effective_routing_soft_topk = int(routing_soft_topk)
            if effective_routing_soft_topk <= 0:
                effective_routing_soft_topk = None

        if use_soft_routing:
            try:
                return self.model(
                    positions,
                    params,
                    top_k=effective_top_k,
                    light_mask=mask,
                    training_soft_routing=True,
                    routing_temperature=effective_routing_temperature,
                    routing_soft_topk=effective_routing_soft_topk,
                )
            except TypeError:
                pass

        if mask is None:
            return self.model(positions, params, top_k=effective_top_k)
        try:
            return self.model(positions, params, top_k=effective_top_k, light_mask=mask)
        except TypeError:
            # Fallback for models without mask support
            return self.model(positions, params, top_k=effective_top_k)

    def _compute_routing_balance_loss(
        self,
        routing: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        if routing is None:
            return torch.tensor(0.0, device=self.device), {
                "routing_balance_loss": 0.0,
                "routing_entropy": 0.0,
                "routing_nonzero_ratio": 0.0,
            }

        weights, indices = routing  # [B, K'], [B, K']
        if weights.numel() == 0 or indices.numel() == 0:
            return torch.tensor(0.0, device=self.device), {
                "routing_balance_loss": 0.0,
                "routing_entropy": 0.0,
                "routing_nonzero_ratio": 0.0,
            }

        flat_weights = weights.reshape(-1)
        flat_indices = indices.reshape(-1).to(dtype=torch.long)

        num_experts = int(getattr(self.model, "K", 0))
        if num_experts <= 0 and hasattr(self.model, "mu") and torch.is_tensor(getattr(self.model, "mu")):
            num_experts = int(getattr(self.model, "mu").shape[0])
        if flat_indices.numel() > 0:
            num_experts = max(num_experts, int(flat_indices.max().item()) + 1)
        if num_experts <= 0:
            return torch.tensor(0.0, device=self.device), {
                "routing_balance_loss": 0.0,
                "routing_entropy": 0.0,
                "routing_nonzero_ratio": 0.0,
            }

        # Aggregate usage by global expert id, not local top-k slot id.
        usage_sum = torch.zeros((num_experts,), device=flat_weights.device, dtype=flat_weights.dtype)
        usage_sum.scatter_add_(0, flat_indices, flat_weights)
        usage = usage_sum / (torch.sum(usage_sum) + 1e-8)
        mean_usage = torch.mean(usage)
        var_usage = torch.mean((usage - mean_usage) ** 2)
        balance_loss = var_usage / (mean_usage * mean_usage + 1e-8)

        entropy = -torch.sum(usage * torch.log(usage + 1e-8))
        entropy = entropy / torch.log(torch.tensor(float(max(2, usage.shape[0])), device=usage.device))

        nonzero_threshold = 0.5 / float(max(1, usage.shape[0]))
        nonzero_ratio = torch.mean((usage > nonzero_threshold).to(dtype=usage.dtype))

        stats = {
            "routing_balance_loss": float(balance_loss.detach().item()),
            "routing_entropy": float(entropy.detach().item()),
            "routing_nonzero_ratio": float(nonzero_ratio.detach().item()),
        }
        return balance_loss, stats

    def _routing_temperature_for_epoch(self, epoch: int) -> float:
        if not self.routing_soft_train:
            return float(self.routing_temp_start)
        if self.routing_temp_anneal_epochs <= 0:
            return float(self.routing_temp_end)
        progress = min(1.0, max(0.0, float(epoch - 1) / float(self.routing_temp_anneal_epochs)))
        return float(self.routing_temp_start + (self.routing_temp_end - self.routing_temp_start) * progress)

    def _image_loss_weight_for_epoch(self, epoch: int) -> float:
        base = float(self.image_loss_weight_target)
        if base <= 0.0:
            return 0.0
        if self.image_loss_warmup_epochs <= 0:
            return base
        progress = min(1.0, max(0.0, float(epoch) / float(self.image_loss_warmup_epochs)))
        return float(base * progress)

    def _gbuffer_image_loss_weight_for_epoch(self, epoch: int) -> float:
        base = float(self.gbuffer_image_loss_weight_target)
        if (not self.gbuffer_image_loss_enabled) or base <= 0.0:
            return 0.0
        if self.gbuffer_image_loss_warmup_epochs <= 0:
            return base
        progress = min(1.0, max(0.0, float(epoch) / float(self.gbuffer_image_loss_warmup_epochs)))
        return float(base * progress)

    def _routing_balance_weight_for_epoch(self, epoch: int) -> float:
        if not self.routing_balance_anneal_enabled:
            return float(self.lambda_routing_balance)
        start = float(self.routing_balance_anneal_start)
        end = float(self.routing_balance_anneal_end)
        if self.routing_balance_anneal_epochs <= 0:
            return end
        global_epoch = max(1, int(self.routing_balance_anneal_epoch_offset) + int(epoch))
        progress = min(1.0, max(0.0, float(global_epoch - 1) / float(self.routing_balance_anneal_epochs)))
        return float(start + (end - start) * progress)

    def _proxy_gbuffer_handover_scales_for_epoch(self, epoch: int) -> tuple[float, float, float]:
        if not self.proxy_gbuffer_handover_enabled:
            return 0.0, 1.0, 1.0
        global_epoch = int(self.proxy_gbuffer_handover_epoch_offset) + int(epoch)
        if global_epoch <= self.proxy_gbuffer_handover_start_epoch:
            progress = 0.0
        elif global_epoch >= self.proxy_gbuffer_handover_end_epoch:
            progress = 1.0
        else:
            denom = max(1, self.proxy_gbuffer_handover_end_epoch - self.proxy_gbuffer_handover_start_epoch)
            progress = float(global_epoch - self.proxy_gbuffer_handover_start_epoch) / float(denom)
            progress = min(1.0, max(0.0, progress))
        proxy_scale = float(
            self.proxy_gbuffer_handover_proxy_start_scale
            + (self.proxy_gbuffer_handover_proxy_end_scale - self.proxy_gbuffer_handover_proxy_start_scale) * progress
        )
        gbuffer_scale = float(
            self.proxy_gbuffer_handover_gbuffer_start_scale
            + (self.proxy_gbuffer_handover_gbuffer_end_scale - self.proxy_gbuffer_handover_gbuffer_start_scale) * progress
        )
        return float(progress), max(0.0, proxy_scale), max(0.0, gbuffer_scale)

    def _aux_image_weights_for_epoch(self, epoch: int) -> tuple[float, float]:
        proxy_base = float(self._image_loss_weight_for_epoch(epoch))
        gbuffer_base = float(self._gbuffer_image_loss_weight_for_epoch(epoch))
        progress, proxy_scale, gbuffer_scale = self._proxy_gbuffer_handover_scales_for_epoch(epoch)
        self._current_proxy_gbuffer_handover_progress = float(progress)
        self._current_proxy_gbuffer_proxy_scale = float(proxy_scale)
        self._current_proxy_gbuffer_gbuffer_scale = float(gbuffer_scale)
        return float(proxy_base * proxy_scale), float(gbuffer_base * gbuffer_scale)

    @staticmethod
    def _write_effective_term_metrics(
        metrics: Dict[str, float],
        *,
        term: str,
        stats: Dict[str, float],
    ) -> None:
        t = str(term)
        metrics[f"loss_eff_scale_{t}"] = float(stats.get("scale", 1.0))
        metrics[f"loss_eff_weight_base_{t}"] = float(stats.get("weight_base", 0.0))
        metrics[f"loss_eff_weight_{t}"] = float(stats.get("weight_effective", 0.0))
        metrics[f"loss_eff_contrib_base_{t}"] = float(stats.get("contrib_base", 0.0))
        metrics[f"loss_eff_contrib_{t}"] = float(stats.get("contrib_effective", 0.0))
        metrics[f"loss_eff_target_{t}"] = float(stats.get("target", 0.0))
        metrics[f"loss_eff_expected_{t}"] = float(stats.get("expected", 0.0))
        metrics[f"loss_eff_ema_{t}"] = float(stats.get("ema_loss", 0.0))
        metrics[f"loss_eff_frequency_{t}"] = float(stats.get("frequency", 1.0))

    def _get_monitor_shared_named_params(self) -> list[tuple[str, nn.Parameter]]:
        if not self.monitor_enabled:
            return []
        out: list[tuple[str, nn.Parameter]] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            group = _param_group_from_name(name)
            if self.monitor_shared_group_set and group not in self.monitor_shared_group_set:
                continue
            out.append((name, param))
        return out

    @staticmethod
    def _flatten_grad_tensors(grads: list[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
        parts = []
        for grad in grads:
            if grad is None:
                continue
            g = grad.detach()
            if g.numel() <= 0:
                continue
            parts.append(g.reshape(-1))
        if not parts:
            return None
        return torch.cat(parts, dim=0)

    def _compute_grad_diagnostics(
        self,
        *,
        loss_main: torch.Tensor,
        loss_aux: torch.Tensor,
    ) -> Dict[str, float]:
        if not self.monitor_enabled:
            return {}
        shared_named = self._get_monitor_shared_named_params()
        if not shared_named:
            return {}
        params = [p for _, p in shared_named]
        if not loss_main.requires_grad:
            return {}
        if not loss_aux.requires_grad:
            return {}

        try:
            grad_main = torch.autograd.grad(
                loss_main,
                params,
                retain_graph=True,
                allow_unused=True,
                create_graph=False,
            )
            grad_aux = torch.autograd.grad(
                loss_aux,
                params,
                retain_graph=True,
                allow_unused=True,
                create_graph=False,
            )
        except Exception:
            return {}

        vec_main = self._flatten_grad_tensors(list(grad_main))
        vec_aux = self._flatten_grad_tensors(list(grad_aux))
        if vec_main is None or vec_aux is None:
            return {}

        eps = 1e-12
        norm_main = float(torch.linalg.vector_norm(vec_main).detach().item())
        norm_aux = float(torch.linalg.vector_norm(vec_aux).detach().item())
        ratio = float(norm_aux / max(eps, norm_main))
        denom = max(eps, norm_main * norm_aux)
        cos = float(torch.dot(vec_main, vec_aux).detach().item() / denom)
        cos = max(-1.0, min(1.0, cos))
        conflict = 1.0 if cos < 0.0 else 0.0

        gns_value = 0.0
        gns_valid = 0.0
        if self.monitor_gns_enabled:
            total_vec_cpu = (vec_main + vec_aux).detach().to(dtype=torch.float32, device="cpu")
            prev = self._monitor_prev_total_grad_vec
            if prev is not None and prev.shape == total_vec_cpu.shape:
                diff = total_vec_cpu - prev
                var_term = 0.5 * float(torch.dot(diff, diff).item())
                dot_pp = float(torch.dot(total_vec_cpu, total_vec_cpu).item())
                dot_pq = float(torch.dot(prev, prev).item())
                dot_cross = float(torch.dot(total_vec_cpu, prev).item())
                mean_term = 0.25 * (dot_pp + dot_pq + 2.0 * dot_cross)
                gns_value = float(var_term / max(eps, mean_term))
                gns_valid = 1.0
            self._monitor_prev_total_grad_vec = total_vec_cpu

        return {
            "grad_diag_sample": 1.0,
            "grad_ratio_aux_main_shared": ratio,
            "grad_cos_aux_main_shared": cos,
            "grad_conflict_flag": conflict,
            "grad_norm_main_shared": norm_main,
            "grad_norm_aux_shared": norm_aux,
            "gns_proxy_value": float(gns_value),
            "gns_proxy_valid": float(gns_valid),
        }

    def _compute_routing_logit_stats(
        self,
        *,
        positions: torch.Tensor,
        routing: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[float, float]:
        if routing is None:
            return 0.0, 0.0
        if not hasattr(self.model, "mu") or not hasattr(self.model, "log_scale"):
            return 0.0, 0.0
        weights, topk_indices = routing
        if weights.numel() <= 0 or topk_indices.numel() <= 0:
            return 0.0, 0.0
        try:
            mu = getattr(self.model, "mu")
            log_scale = getattr(self.model, "log_scale")
            selected_mu = mu[topk_indices]
            selected_scale = torch.exp(log_scale[topk_indices])
            pos_expanded = positions.unsqueeze(1)
            diff = pos_expanded - selected_mu
            weighted_diff = diff / (selected_scale + 1e-12)
            logits_raw = -0.5 * torch.sum(weighted_diff * weighted_diff, dim=-1)
            logits_scaled = logits_raw
            if bool(self.routing_soft_train):
                temperature = max(1e-6, float(self._current_routing_temp))
                logits_scaled = logits_scaled / temperature
            scaled_rms = torch.sqrt(torch.mean(logits_scaled * logits_scaled) + 1e-12)
            raw_rms = torch.sqrt(torch.mean(logits_raw * logits_raw) + 1e-12)
            return float(scaled_rms.detach().item()), float(raw_rms.detach().item())
        except Exception:
            return 0.0, 0.0

    def _compute_routing_logit_rms(
        self,
        *,
        positions: torch.Tensor,
        routing: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> float:
        scaled_rms, _ = self._compute_routing_logit_stats(
            positions=positions,
            routing=routing,
        )
        return float(scaled_rms)

    def _compute_routing_usage_vector(
        self,
        routing: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        if routing is None:
            return None
        weights, indices = routing
        if weights.numel() <= 0 or indices.numel() <= 0:
            return None
        try:
            num_experts = int(getattr(self.model, "K", 0))
            if num_experts <= 0 and hasattr(self.model, "mu") and torch.is_tensor(getattr(self.model, "mu")):
                num_experts = int(getattr(self.model, "mu").shape[0])
            if indices.numel() > 0:
                num_experts = max(num_experts, int(indices.max().item()) + 1)
            if num_experts <= 0:
                return None
            usage = torch.zeros((num_experts,), device=weights.device, dtype=torch.float32)
            usage.scatter_add_(
                0,
                indices.reshape(-1).to(dtype=torch.long),
                weights.reshape(-1).to(dtype=torch.float32),
            )
            return usage
        except Exception:
            return None

    def _compute_active_coeff_spectrum_metrics(
        self,
        *,
        routing_usage_sum: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        if not self.monitor_enabled or routing_usage_sum is None:
            return {}
        coeffs = getattr(self.model, "coeffs", None)
        if not torch.is_tensor(coeffs) or coeffs.numel() <= 0:
            return {}
        try:
            usage_cpu = routing_usage_sum.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
            if usage_cpu.numel() <= 0:
                return {}
            num_experts = int(coeffs.shape[0])
            if usage_cpu.shape[0] < num_experts:
                pad = torch.zeros((num_experts - usage_cpu.shape[0],), dtype=usage_cpu.dtype)
                usage_cpu = torch.cat([usage_cpu, pad], dim=0)
            elif usage_cpu.shape[0] > num_experts:
                usage_cpu = usage_cpu[:num_experts]

            total_usage = float(torch.sum(usage_cpu).item())
            if total_usage <= 1e-12:
                return {}

            topk = min(num_experts, max(4, int(np.ceil(float(num_experts) / 3.0))))
            if topk <= 0:
                return {}
            top_values, top_indices = torch.topk(usage_cpu, k=topk, largest=True, sorted=True)
            usage_mass_topk = float(torch.sum(top_values).item() / max(1e-12, total_usage))

            coeffs_cpu = coeffs.detach().to(dtype=torch.float32, device="cpu")
            active_mat = coeffs_cpu[top_indices].reshape(-1, coeffs_cpu.shape[-1])
            if active_mat.numel() <= 0:
                return {}
            sv = torch.linalg.svdvals(active_mat)
            if sv.numel() <= 0:
                return {}
            sv_sum = float(torch.sum(sv).item())
            sv_max = float(torch.max(sv).item())
            fro_sq = float(torch.sum(sv * sv).item())
            if sv_sum <= 1e-12 or sv_max <= 1e-12:
                eff_rank = 0.0
                stable_rank = 0.0
            else:
                p = sv / (sv_sum + 1e-12)
                entropy = float((-torch.sum(p * torch.log(p + 1e-12))).item())
                eff_rank = float(np.exp(entropy))
                stable_rank = float(fro_sq / max(1e-12, sv_max * sv_max))
            return {
                "active_coeff_effective_rank": float(eff_rank),
                "active_coeff_stable_rank": float(stable_rank),
                "active_coeff_topk": float(topk),
                "active_usage_mass_topk": float(usage_mass_topk),
            }
        except Exception:
            return {}

    def _compute_weight_spectrum_metrics(self) -> Dict[str, float]:
        if not self.monitor_enabled:
            return {}
        metrics: Dict[str, float] = {}
        with torch.no_grad():
            for name, tensor_name in (("basis", "U"), ("coeff", "coeffs")):
                tensor = getattr(self.model, tensor_name, None)
                if not torch.is_tensor(tensor):
                    continue
                mat = tensor.detach().to(dtype=torch.float32)
                if mat.numel() <= 0:
                    continue
                mat2d = mat.reshape(-1, mat.shape[-1]).to(device="cpu")
                try:
                    sv = torch.linalg.svdvals(mat2d)
                except Exception:
                    continue
                if sv.numel() <= 0:
                    continue
                sv_sum = float(torch.sum(sv).item())
                sv_max = float(torch.max(sv).item())
                fro_sq = float(torch.sum(sv * sv).item())
                if sv_sum <= 1e-12 or sv_max <= 1e-12:
                    eff_rank = 0.0
                    stable_rank = 0.0
                else:
                    p = sv / (sv_sum + 1e-12)
                    entropy = float((-torch.sum(p * torch.log(p + 1e-12))).item())
                    eff_rank = float(np.exp(entropy))
                    stable_rank = float(fro_sq / max(1e-12, sv_max * sv_max))
                metrics[f"spectrum_{name}_effective_rank"] = float(eff_rank)
                metrics[f"spectrum_{name}_stable_rank"] = float(stable_rank)
                metrics[f"spectrum_{name}_sv_max"] = float(sv_max)
                metrics[f"spectrum_{name}_sv_mean"] = float(torch.mean(sv).item())
                metrics[f"spectrum_{name}_fro_norm"] = float(torch.linalg.vector_norm(mat2d).item())
        return metrics

    @staticmethod
    def _metric_higher_is_better(metric_name: str) -> bool:
        return str(metric_name).strip().lower() in {"img_psnr"}

    def _metric_init_best_value(self, metric_name: str) -> float:
        if self._metric_higher_is_better(metric_name):
            return float("-inf")
        return float("inf")

    def _metric_is_better(self, metric_name: str, current: Optional[float], best: float) -> bool:
        if current is None:
            return False
        try:
            current_f = float(current)
        except Exception:
            return False
        if not np.isfinite(current_f):
            return False
        if self._metric_higher_is_better(metric_name):
            return current_f > float(best)
        return current_f < float(best)

    def _metric_for_plateau_scheduler(self, metric_name: str, metrics: Dict[str, float]) -> float:
        fallback = float(metrics.get("mae", 0.0))
        value = metrics.get(metric_name, fallback)
        try:
            value_f = float(value)
        except Exception:
            value_f = fallback
        if self._metric_higher_is_better(metric_name):
            # ReduceLROnPlateau is configured in "min" mode.
            return -value_f
        return value_f

    def _get_sh_loss_weights(self, ref: torch.Tensor) -> torch.Tensor:
        w = self._sh_loss_weights_27.to(device=ref.device, dtype=ref.dtype)
        view_shape = [1] * (ref.dim() - 1) + [27]
        return w.view(*view_shape)

    def _tone_map_reinhard(self, image: torch.Tensor) -> torch.Tensor:
        image = torch.clamp(image, min=0.0)
        return image / (1.0 + image)

    def _srgb_encode(self, image: torch.Tensor) -> torch.Tensor:
        image = torch.clamp(image, min=0.0)
        threshold = 0.0031308
        low = 12.92 * image
        high = 1.055 * torch.pow(image, 1.0 / 2.4) - 0.055
        return torch.where(image <= threshold, low, high)

    def _set_image_sampling_context_for_epoch(self, epoch: int) -> None:
        epoch_i = max(1, int(epoch))
        self._image_sampling_epoch = epoch_i
        if self.image_sampling_mode == "fixed":
            seed = int(self.image_sample_seed)
        elif self.image_sampling_mode == "per_epoch":
            seed = int(self.image_sample_seed + (epoch_i - 1))
        else:
            # per_step mode reseeds each step; epoch context keeps validation deterministic.
            seed = int(self.image_sample_seed + (epoch_i - 1) * 100000)
        self._image_sampling_seed_current = seed

    def _set_image_sampling_context_for_step(self, step_idx: int) -> None:
        if self.image_sampling_mode != "per_step":
            return
        epoch_i = max(1, int(self._image_sampling_epoch))
        step_i = max(1, int(step_idx))
        self._image_sampling_seed_current = int(
            self.image_sample_seed + (epoch_i - 1) * 100000 + (step_i - 1)
        )

    def _build_seeded_rotation_matrix(self, ref: torch.Tensor) -> torch.Tensor:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(self._image_sampling_seed_current))
        u1, u2, u3 = torch.rand((3,), generator=gen, dtype=torch.float32)
        two_pi = float(2.0 * np.pi)
        qx = torch.sqrt(1.0 - u1) * torch.sin(two_pi * u2)
        qy = torch.sqrt(1.0 - u1) * torch.cos(two_pi * u2)
        qz = torch.sqrt(u1) * torch.sin(two_pi * u3)
        qw = torch.sqrt(u1) * torch.cos(two_pi * u3)
        x, y, z, w = qx, qy, qz, qw
        row0 = torch.stack(
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            dim=0,
        )
        row1 = torch.stack(
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            dim=0,
        )
        row2 = torch.stack(
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            dim=0,
        )
        rot = torch.stack([row0, row1, row2], dim=0).to(dtype=torch.float32)
        return rot.to(device=ref.device, dtype=ref.dtype)

    def _build_random_dirs(self, ref: torch.Tensor) -> torch.Tensor:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(self._image_sampling_seed_current))
        u = torch.rand((self.image_samples,), generator=gen, dtype=torch.float32)
        v = torch.rand((self.image_samples,), generator=gen, dtype=torch.float32)
        z = 2.0 * u - 1.0
        phi = 2.0 * np.pi * v
        r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        return torch.stack([x, y, z], dim=-1).to(device=ref.device, dtype=ref.dtype)

    def _build_fibonacci_dirs(self, ref: torch.Tensor) -> torch.Tensor:
        n = int(self.image_samples)
        i = torch.arange(n, dtype=torch.float32) + 0.5
        y = 1.0 - 2.0 * i / float(n)
        r = torch.sqrt(torch.clamp(1.0 - y * y, min=0.0))
        golden = float(np.pi * (3.0 - np.sqrt(5.0)))
        theta = golden * i
        x = r * torch.cos(theta)
        z = r * torch.sin(theta)
        dirs = torch.stack([x, y, z], dim=-1).to(device=ref.device, dtype=ref.dtype)
        rot = self._build_seeded_rotation_matrix(ref)
        return torch.matmul(dirs, rot.transpose(0, 1))

    def _build_sh_basis_from_dirs(self, dirs: torch.Tensor) -> torch.Tensor:
        xb, yb, zb = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        return torch.stack(
            [
                torch.full_like(xb, 0.282095),
                0.488603 * yb,
                0.488603 * zb,
                0.488603 * xb,
                1.092548 * xb * yb,
                1.092548 * yb * zb,
                0.315392 * (3.0 * zb * zb - 1.0),
                1.092548 * xb * zb,
                0.546274 * (xb * xb - yb * yb),
            ],
            dim=-1,
        )

    def _build_image_basis(self, ref: torch.Tensor, basis_kind: str = "radiance") -> torch.Tensor:
        key = (str(ref.device), str(ref.dtype), int(self._image_sampling_seed_current), str(basis_kind))
        cached = self._image_basis_cache.get(key)
        if cached is not None:
            return cached

        if str(basis_kind) == "irradiance":
            dirs = self._build_fibonacci_dirs(ref)
        else:
            dirs = self._build_random_dirs(ref)
        basis = self._build_sh_basis_from_dirs(dirs)
        self._image_basis_cache[key] = basis
        return basis

    def _render_sh_to_samples(self, sh: torch.Tensor, basis_kind: str = "radiance") -> torch.Tensor:
        basis = self._build_image_basis(sh, basis_kind=basis_kind)
        sh_rgb = torch.stack([sh[..., 0:9], sh[..., 9:18], sh[..., 18:27]], dim=-2)
        image = torch.einsum("...cn,sn->...sc", sh_rgb, basis)
        if self.image_loss_space == "srgb":
            image = self._srgb_encode(self._tone_map_reinhard(image))
        else:
            image = torch.clamp(image, min=0.0)
        return image

    def _render_irradiance_to_samples(self, sh: torch.Tensor) -> torch.Tensor:
        basis = self._build_image_basis(sh, basis_kind="irradiance")
        sh_rgb = torch.stack([sh[..., 0:9], sh[..., 9:18], sh[..., 18:27]], dim=-2)
        w = self._irradiance_band_weights_9.to(device=sh.device, dtype=sh.dtype)
        view_shape = [1] * (sh_rgb.dim() - 1) + [9]
        sh_rgb_weighted = sh_rgb * w.view(*view_shape)
        image = torch.einsum("...cn,sn->...sc", sh_rgb_weighted, basis)
        if self.image_loss_space == "srgb":
            image = self._srgb_encode(self._tone_map_reinhard(image))
        else:
            image = torch.clamp(image, min=0.0)
        return image

    def _apply_image_soft_saturation(self, image: torch.Tensor) -> torch.Tensor:
        if not self.image_soft_saturation_enabled:
            return image
        x = torch.clamp(image, min=0.0)
        if self.image_soft_saturation_mode == "exp":
            k = float(max(1e-8, self.image_soft_saturation_k))
            return 1.0 - torch.exp(-x / k)
        return x

    def _image_loss_from_samples(self, pred_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        if self.image_loss_type == "charbonnier":
            return charbonnier_loss(pred_img, target_img, epsilon=self.charbonnier_eps)
        if self.image_loss_type == "huber":
            return torch.nn.functional.huber_loss(
                pred_img,
                target_img,
                reduction="mean",
                delta=float(self.image_loss_huber_delta),
            )
        return torch.mean((pred_img - target_img) ** 2)

    def _compute_image_loss_components(
        self, pred_sh: torch.Tensor, target_sh: torch.Tensor
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = torch.tensor(0.0, device=pred_sh.device, dtype=pred_sh.dtype)
        components: Dict[str, torch.Tensor] = {
            "linear": zero,
            "log": zero,
            "mix": zero,
        }
        if self._current_image_loss_weight <= 0:
            return zero, components

        if self.image_loss_domain == "irradiance":
            pred_img = self._render_irradiance_to_samples(pred_sh)
            target_img = self._render_irradiance_to_samples(target_sh)
        else:
            pred_img = self._render_sh_to_samples(pred_sh, basis_kind="radiance")
            target_img = self._render_sh_to_samples(target_sh, basis_kind="radiance")
        pred_img_loss = self._apply_image_soft_saturation(pred_img)
        target_img_loss = self._apply_image_soft_saturation(target_img)

        mode = self.proxy_image_loss_mix_mode
        linear_loss = zero
        log_loss = zero

        if mode in {"linear_log_mix", "linear_only"}:
            linear_loss = self._image_loss_from_samples(pred_img_loss, target_img_loss)
        if mode in {"linear_log_mix", "log_only"}:
            pred_log = torch.log1p(torch.clamp(pred_img_loss, min=0.0))
            target_log = torch.log1p(torch.clamp(target_img_loss, min=0.0))
            log_loss = self._image_loss_from_samples(pred_log, target_log)

        if mode == "linear_only":
            mix_loss = linear_loss
        elif mode == "log_only":
            mix_loss = log_loss
        else:
            w = float(self.proxy_image_log_mix_weight)
            mix_loss = (1.0 - w) * linear_loss + w * log_loss

        components["linear"] = linear_loss
        components["log"] = log_loss
        components["mix"] = mix_loss
        return mix_loss, components

    def _compute_image_loss(self, pred_sh: torch.Tensor, target_sh: torch.Tensor) -> torch.Tensor:
        loss, _ = self._compute_image_loss_components(pred_sh, target_sh)
        return loss

    @staticmethod
    def _image_metric_tensors(pred_img: torch.Tensor, target_img: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred32 = pred_img.to(dtype=torch.float32)
        target32 = target_img.to(dtype=torch.float32)
        diff = pred32 - target32
        mse = torch.mean(diff * diff)
        mae = torch.mean(torch.abs(diff))
        rmse = torch.sqrt(torch.clamp(mse, min=1e-12))
        psnr = 10.0 * torch.log10(torch.tensor(1.0, device=pred32.device, dtype=pred32.dtype) / torch.clamp(mse, min=1e-12))
        return {
            "img_mae": mae,
            "img_rmse": rmse,
            "img_psnr": psnr,
        }

    def _compute_image_metrics(self, pred_sh: torch.Tensor, target_sh: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred_img = self._render_sh_to_samples(pred_sh)
        target_img = self._render_sh_to_samples(target_sh)
        return self._image_metric_tensors(pred_img=pred_img, target_img=target_img)

    def _next_gbuffer_batch(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self.gbuffer_image_loss_enabled or self._gbuffer_loader is None:
            return None
        if self._gbuffer_iter is None:
            self._gbuffer_iter = iter(self._gbuffer_loader)
        try:
            raw_batch = next(self._gbuffer_iter)
        except StopIteration:
            self._gbuffer_iter = iter(self._gbuffer_loader)
            raw_batch = next(self._gbuffer_iter)
        if not isinstance(raw_batch, dict):
            return None

        batch: Dict[str, torch.Tensor] = {}
        for key in ("posW", "normW", "albedo", "gt_linear", "light_params", "light_mask", "frame_idx", "config_idx"):
            value = raw_batch.get(key)
            if value is None:
                continue
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, dtype=torch.float32)
            value = value.to(self.device, non_blocking=self.adapter.non_blocking_transfer)
            if value.ndim >= 1 and value.shape[0] == 1:
                value = value.squeeze(0)
            batch[key] = value
        return batch

    def _normalize_world_pos_to_probe_range(self, pos_world: torch.Tensor) -> torch.Tensor:
        if self._gbuffer_probe_min is None or self._gbuffer_probe_max is None:
            raise RuntimeError("GBuffer probe bounds are not initialized")
        min_v = self._gbuffer_probe_min.to(device=pos_world.device, dtype=pos_world.dtype)
        max_v = self._gbuffer_probe_max.to(device=pos_world.device, dtype=pos_world.dtype)
        return 2.0 * (pos_world - min_v) / (max_v - min_v + 1e-8) - 1.0

    @staticmethod
    def _tensor_first_int(x: Optional[torch.Tensor], default: int = -1) -> int:
        if x is None:
            return int(default)
        if not torch.is_tensor(x):
            try:
                return int(x)
            except Exception:
                return int(default)
        if x.numel() <= 0:
            return int(default)
        return int(x.reshape(-1)[0].item())

    def _resolve_gbuffer_cfg_idx(self, frame_idx: int, config_idx: int) -> int:
        if config_idx >= 0:
            return int(config_idx)
        if frame_idx in self._gbuffer_gt_frame_to_cfg_idx:
            return int(self._gbuffer_gt_frame_to_cfg_idx[frame_idx])
        if frame_idx < 0:
            raise KeyError("Missing frame_idx/config_idx for gbuffer sample")
        # Fallback to identity mapping when frame list is unavailable.
        return int(frame_idx)

    def _interpolate_gbuffer_gt_sh(
        self,
        pos_world: torch.Tensor,
        *,
        cfg_idx: int,
    ) -> torch.Tensor:
        probe_pos = self._gbuffer_gt_probe_positions_world
        gt_tensor_cpu = self._gbuffer_gt_sh_tensor_cpu
        if probe_pos is None or gt_tensor_cpu is None:
            raise RuntimeError("GBuffer dataset_sh target requested but GT SH tensors are not initialized")
        num_probe = int(probe_pos.shape[0])
        cfg_i = int(cfg_idx)
        if cfg_i < 0 or cfg_i >= int(gt_tensor_cpu.shape[1]):
            raise IndexError(
                f"cfg_idx out of range for gbuffer GT SH tensor: cfg_idx={cfg_i}, num_cfg={int(gt_tensor_cpu.shape[1])}"
            )
        gt_frame = gt_tensor_cpu[:, cfg_i, :].to(
            device=pos_world.device,
            dtype=pos_world.dtype,
            non_blocking=self.adapter.non_blocking_transfer,
        )
        k_eff = min(max(1, int(self.gbuffer_image_loss_gt_knn)), num_probe)
        eps2 = float(self.gbuffer_image_loss_gt_weight_eps) ** 2
        chunk = max(1, int(self.gbuffer_image_loss_gt_chunk_size))
        out = torch.empty((int(pos_world.shape[0]), 27), device=pos_world.device, dtype=pos_world.dtype)
        for start in range(0, int(pos_world.shape[0]), chunk):
            end = min(int(pos_world.shape[0]), start + chunk)
            p = pos_world[start:end]
            d2 = torch.sum((p[:, None, :] - probe_pos[None, :, :]) ** 2, dim=-1)
            d2_sel, idx = torch.topk(d2, k=k_eff, dim=1, largest=False, sorted=False)
            w = 1.0 / (d2_sel + eps2)
            w = w / (torch.sum(w, dim=1, keepdim=True) + 1e-8)
            sh_sel = gt_frame[idx]  # [C,K,27]
            out[start:end] = torch.sum(w.unsqueeze(-1) * sh_sel, dim=1)
        return out

    def _gbuffer_image_loss_from_linear(
        self,
        pred_rgb: torch.Tensor,
        gt_rgb: torch.Tensor,
    ) -> torch.Tensor:
        return charbonnier_loss(pred_rgb, gt_rgb, epsilon=self.charbonnier_eps)

    def _render_irradiance_to_normals(
        self,
        pred_sh: torch.Tensor,
        normal: torch.Tensor,
        albedo: torch.Tensor,
    ) -> torch.Tensor:
        n = torch.nn.functional.normalize(normal, dim=-1, eps=1e-8)
        basis = self._build_sh_basis_from_dirs(n)
        sh_rgb = torch.stack([pred_sh[..., 0:9], pred_sh[..., 9:18], pred_sh[..., 18:27]], dim=-2)
        w = self._irradiance_band_weights_9.to(device=pred_sh.device, dtype=pred_sh.dtype)
        sh_rgb_weighted = sh_rgb * w.view(1, 1, 9)
        irradiance = torch.einsum("...cn,...n->...c", sh_rgb_weighted, basis)
        irradiance = torch.nn.functional.softplus(
            irradiance,
            beta=float(self._irradiance_positive_beta),
        )
        return albedo * irradiance / torch.pi

    def _compute_gbuffer_image_loss(self, *, step_idx: int) -> tuple[torch.Tensor, float]:
        zero = torch.tensor(0.0, device=self.device)
        if (
            (not self.gbuffer_image_loss_enabled)
            or self._current_gbuffer_image_loss_weight <= 0.0
            or self._gbuffer_loader is None
        ):
            return zero, 0.0
        if int(step_idx) % int(self.gbuffer_image_loss_every_steps) != 0:
            return zero, 0.0

        batch = self._next_gbuffer_batch()
        if batch is None:
            return zero, 0.0

        pos_world = batch.get("posW")
        normal = batch.get("normW")
        albedo = batch.get("albedo")
        frame_idx = self._tensor_first_int(batch.get("frame_idx"), default=-1)
        config_idx = self._tensor_first_int(batch.get("config_idx"), default=-1)
        if pos_world is None or normal is None or albedo is None:
            return zero, 0.0

        pos_world = pos_world.reshape(-1, 3)
        normal = normal.reshape(-1, 3)
        albedo = albedo.reshape(-1, 3)
        n_pixels = int(pos_world.shape[0])
        if n_pixels <= 0:
            return zero, 0.0

        pos_norm = self._normalize_world_pos_to_probe_range(pos_world)
        gt_linear = None
        light_params = None
        light_mask = None

        if self.gbuffer_image_loss_target_source == "dataset_sh":
            cfg_idx = self._resolve_gbuffer_cfg_idx(frame_idx=frame_idx, config_idx=config_idx)
            light_cfg_cpu = self._gbuffer_gt_light_configs_cpu
            if light_cfg_cpu is None:
                raise RuntimeError("gbuffer GT light configs are not initialized")
            if cfg_idx < 0 or cfg_idx >= int(light_cfg_cpu.shape[0]):
                raise IndexError(
                    f"cfg_idx out of range for gbuffer GT light configs: cfg_idx={cfg_idx}, "
                    f"num_cfg={int(light_cfg_cpu.shape[0])}, frame_idx={frame_idx}"
                )
            light_params = light_cfg_cpu[cfg_idx].to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=self.adapter.non_blocking_transfer,
            )
            if light_params.ndim == 1:
                light_params = light_params.unsqueeze(0)
            light_params = light_params.unsqueeze(0).expand(n_pixels, -1, -1)
            light_mask_cpu = self._gbuffer_gt_light_mask_cpu
            if light_mask_cpu is None:
                light_mask = None
            else:
                light_mask = light_mask_cpu[cfg_idx].to(
                    device=self.device,
                    dtype=torch.float32,
                    non_blocking=self.adapter.non_blocking_transfer,
                )
                if light_mask.ndim == 0:
                    light_mask = light_mask.reshape(1)
                light_mask = light_mask.unsqueeze(0).expand(n_pixels, -1)
            with torch.no_grad():
                gt_sh = self._interpolate_gbuffer_gt_sh(pos_world=pos_world, cfg_idx=cfg_idx)
                gt_linear = self._render_irradiance_to_normals(pred_sh=gt_sh, normal=normal, albedo=albedo)
        else:
            gt_linear = batch.get("gt_linear")
            light_params = batch.get("light_params")
            light_mask = batch.get("light_mask")
            if gt_linear is None or light_params is None:
                return zero, 0.0
            gt_linear = gt_linear.reshape(-1, 3)
            if light_params.ndim == 2:
                light_params = light_params.unsqueeze(0).expand(n_pixels, -1, -1)
            elif light_params.ndim == 3 and light_params.shape[0] == 1:
                light_params = light_params.expand(n_pixels, -1, -1)
            elif light_params.ndim == 3 and light_params.shape[0] == n_pixels:
                pass
            else:
                raise ValueError(f"Unsupported light_params shape for gbuffer loss: {tuple(light_params.shape)}")

            if light_mask is not None:
                if light_mask.ndim == 1:
                    light_mask = light_mask.unsqueeze(0).expand(n_pixels, -1)
                elif light_mask.ndim == 2 and light_mask.shape[0] == 1:
                    light_mask = light_mask.expand(n_pixels, -1)
                elif light_mask.ndim == 2 and light_mask.shape[0] == n_pixels:
                    pass
                else:
                    raise ValueError(f"Unsupported light_mask shape for gbuffer loss: {tuple(light_mask.shape)}")

        # Keep routing policy identical to the main training branch.
        routing = self._compute_routing(pos_norm, for_training=True)
        pred_sh = self._forward(
            pos_norm,
            light_params,
            light_mask,
            routing=routing,
        )
        pred_rgb = self._render_irradiance_to_normals(pred_sh=pred_sh, normal=normal, albedo=albedo)
        if gt_linear is None:
            return zero, 0.0
        loss = self._gbuffer_image_loss_from_linear(pred_rgb, gt_linear)
        return loss, float(n_pixels)

    def _recon_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_eval, target_eval = pred, target
        if self.enable_weighted_sh_loss and self.adapter.target_inverse is not None:
            pred_eval, target_eval = self.adapter.inverse_targets(pred, target)

        if self.enable_weighted_sh_loss:
            diff = (pred_eval - target_eval) * self._get_sh_loss_weights(pred_eval)
            if self.recon_loss == "charbonnier":
                return torch.mean(torch.sqrt(diff * diff + self.charbonnier_eps * self.charbonnier_eps))
            if self.recon_loss == "l1":
                return torch.mean(torch.abs(diff))
            return torch.mean(diff * diff)

        if self.recon_loss == "charbonnier":
            return charbonnier_loss(pred, target, epsilon=self.charbonnier_eps)
        return self.criterion(pred, target)

    def _compute_linearity_loss(
        self,
        positions: torch.Tensor,
        params: torch.Tensor,
        mask: Optional[torch.Tensor],
        preds_full: torch.Tensor,
        routing_full: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute superposition linearity loss by splitting light sets.

        Uses random partition of lights into A/B and enforces:
        f(A+B) ≈ f(A) + f(B)
        """
        if self.linearity_weight <= 0:
            return torch.tensor(0.0, device=self.device)

        if mask is None or params.dim() != 3:
            return torch.tensor(0.0, device=self.device)

        num_lights = params.shape[1]
        if num_lights < 2:
            return torch.tensor(0.0, device=self.device)

        # Random split of lights per sample
        rand = torch.rand_like(mask.float())
        mask_a = (rand > 0.5) & (mask > 0)
        mask_b = (mask > 0) & (~mask_a)

        valid = (mask_a.sum(dim=-1) > 0) & (mask_b.sum(dim=-1) > 0)
        if not torch.any(valid):
            return torch.tensor(0.0, device=self.device)

        positions_v = positions[valid]
        params_v = params[valid]
        mask_a_v = mask_a[valid]
        mask_b_v = mask_b[valid]
        preds_full_v = preds_full[valid]
        routing_v = self._select_routing(routing_full, valid) if routing_full is not None else None

        # 将 A/B 两路前向合并为一次批量前向，减少重复调度开销。
        positions_cat = torch.cat([positions_v, positions_v], dim=0)
        params_cat = torch.cat([params_v, params_v], dim=0)
        mask_cat = torch.cat([mask_a_v, mask_b_v], dim=0)
        routing_cat = None
        if routing_v is not None:
            rw, ri = routing_v
            routing_cat = (torch.cat([rw, rw], dim=0), torch.cat([ri, ri], dim=0))

        preds_cat = self._forward(positions_cat, params_cat, mask_cat, routing=routing_cat)
        preds_a, preds_b = torch.chunk(preds_cat, 2, dim=0)

        if self.adapter.target_inverse is not None:
            preds_full_v = self.adapter.target_inverse(preds_full_v)
            preds_a = self.adapter.target_inverse(preds_a)
            preds_b = self.adapter.target_inverse(preds_b)

        return torch.mean(torch.abs(preds_full_v - (preds_a + preds_b)))

    def _compute_linearity_aug_loss(
        self,
        positions: torch.Tensor,
        params: torch.Tensor,
        mask: Optional[torch.Tensor],
        routing_full: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """On-the-fly linearity augmentation using pairs of single-light samples."""
        if self.linearity_weight <= 0 or self.linearity_aug_pairs <= 0:
            return torch.tensor(0.0, device=self.device)

        if mask is None or params.dim() != 3:
            return torch.tensor(0.0, device=self.device)

        # Identify single-light samples
        light_counts = mask.sum(dim=-1)
        single_idx = torch.nonzero(light_counts == 1, as_tuple=False).squeeze(-1)
        if single_idx.numel() < 2:
            return torch.tensor(0.0, device=self.device)

        # Limit number of pairs
        num_pairs = min(self.linearity_aug_pairs, single_idx.numel() // 2)
        if num_pairs <= 0:
            return torch.tensor(0.0, device=self.device)

        # Shuffle indices
        perm = single_idx[torch.randperm(single_idx.numel(), device=single_idx.device)]
        pairs = perm[: 2 * num_pairs].view(num_pairs, 2)

        total = 0.0
        # 按 pair 批量化构造并一次性前向，避免 Python 逐 pair 循环。
        idx_i = pairs[:, 0]
        idx_j = pairs[:, 1]

        pos = positions[idx_i]
        params_a = params[idx_i]
        params_b = params[idx_j]
        mask_a = mask[idx_i]
        mask_b = mask[idx_j]

        # Extract single light descriptors
        pair_ids = torch.arange(num_pairs, device=positions.device)
        idx_a = torch.argmax(mask_a, dim=-1)
        idx_b = torch.argmax(mask_b, dim=-1)
        light_a = params_a[pair_ids, idx_a]
        light_b = params_b[pair_ids, idx_b]

        # Build combined params (slots 0 and 1)
        _, N, D = params_a.shape
        combined = torch.zeros((num_pairs, N, D), device=params.device, dtype=params.dtype)
        combined_mask = torch.zeros((num_pairs, N), device=params.device, dtype=mask.dtype)
        combined[:, 0] = light_a
        combined[:, 1] = light_b
        combined_mask[:, 0] = 1.0
        combined_mask[:, 1] = 1.0

        positions_cat = torch.cat([pos, pos, pos], dim=0)
        params_cat = torch.cat([params_a, params_b, combined], dim=0)
        mask_cat = torch.cat([mask_a, mask_b, combined_mask], dim=0)

        routing_cat = None
        if routing_full is not None:
            rw, ri = routing_full
            rw_p = rw[idx_i]
            ri_p = ri[idx_i]
            routing_cat = (
                torch.cat([rw_p, rw_p, rw_p], dim=0),
                torch.cat([ri_p, ri_p, ri_p], dim=0),
            )

        pred_cat = self._forward(positions_cat, params_cat, mask_cat, routing=routing_cat)
        pred_a, pred_b, pred_ab = torch.chunk(pred_cat, 3, dim=0)

        if self.adapter.target_inverse is not None:
            pred_a = self.adapter.target_inverse(pred_a)
            pred_b = self.adapter.target_inverse(pred_b)
            pred_ab = self.adapter.target_inverse(pred_ab)

        total = torch.mean(torch.abs(pred_ab - (pred_a + pred_b)))
        return total

    def _compute_train_step_outputs(
        self,
        *,
        positions: torch.Tensor,
        params: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor],
        probe_idx: Optional[torch.Tensor],
        step_idx: int,
        should_compute_linearity: bool,
    ) -> Dict[str, Any]:
        routing = self._compute_routing(positions, for_training=True)
        preds = self._forward(positions, params, mask, routing=routing)

        loss_recon = self._recon_loss(preds, targets)
        preds_eval, targets_eval = preds, targets
        if self.adapter.target_inverse is not None:
            preds_eval, targets_eval = self.adapter.inverse_targets(preds, targets)

        loss_image, image_loss_components = self._compute_image_loss_components(preds_eval, targets_eval)
        loss_temporal = self._compute_temporal_loss()
        if should_compute_linearity:
            loss_linearity = self._compute_linearity_loss(
                positions,
                params,
                mask,
                preds,
                routing_full=routing,
            )
            loss_linearity_aug = self._compute_linearity_aug_loss(
                positions,
                params,
                mask,
                routing_full=routing,
            )
        else:
            loss_linearity = torch.tensor(0.0, device=self.device)
            loss_linearity_aug = torch.tensor(0.0, device=self.device)

        loss_spatial = self._compute_spatial_loss(preds, params, mask, probe_idx)
        loss_routing_balance = torch.tensor(0.0, device=self.device)
        routing_stats = {
            "routing_balance_loss": 0.0,
            "routing_entropy": 0.0,
            "routing_nonzero_ratio": 0.0,
        }
        if self._current_lambda_routing_balance > 0.0:
            loss_routing_balance, routing_stats = self._compute_routing_balance_loss(routing)
        routing_logit_rms, routing_raw_logit_rms = self._compute_routing_logit_stats(
            positions=positions,
            routing=routing,
        )
        routing_usage = self._compute_routing_usage_vector(routing)

        loss_recon_main = float(self.recon_weight) * loss_recon
        self._loss_effective.begin_step(loss_recon_main)
        loss_linearity_total = loss_linearity + loss_linearity_aug

        eff_metrics: Dict[str, float] = {
            "loss_eff_enabled": 1.0 if self._loss_effective.enabled else 0.0,
            "loss_eff_recon_ema": float(self._loss_effective.recon_ema or 0.0),
        }
        eff_w_image, image_stats = self._loss_effective.compute_weight(
            term="image",
            base_weight=float(self._current_image_loss_weight),
            raw_loss=loss_image,
            frequency=1.0,
        )
        self._write_effective_term_metrics(eff_metrics, term="image", stats=image_stats)

        eff_w_temporal, temporal_stats = self._loss_effective.compute_weight(
            term="temporal",
            base_weight=float(self.temporal_weight),
            raw_loss=loss_temporal,
            frequency=1.0,
        )
        self._write_effective_term_metrics(eff_metrics, term="temporal", stats=temporal_stats)

        eff_w_linearity, linearity_stats = self._loss_effective.compute_weight(
            term="linearity",
            base_weight=float(self.linearity_weight),
            raw_loss=loss_linearity_total,
            frequency=1.0,
        )
        self._write_effective_term_metrics(eff_metrics, term="linearity", stats=linearity_stats)

        eff_w_spatial, spatial_stats = self._loss_effective.compute_weight(
            term="spatial",
            base_weight=float(self.spatial_weight),
            raw_loss=loss_spatial,
            frequency=1.0,
        )
        self._write_effective_term_metrics(eff_metrics, term="spatial", stats=spatial_stats)

        eff_w_routing, routing_balance_stats = self._loss_effective.compute_weight(
            term="routing_balance",
            base_weight=float(self._current_lambda_routing_balance),
            raw_loss=loss_routing_balance,
            frequency=1.0,
        )
        self._write_effective_term_metrics(eff_metrics, term="routing_balance", stats=routing_balance_stats)

        loss_total = (
            loss_recon_main
            + eff_w_image * loss_image
            + eff_w_temporal * loss_temporal
            + eff_w_linearity * loss_linearity_total
            + eff_w_spatial * loss_spatial
            + eff_w_routing * loss_routing_balance
        )
        return {
            "loss_total": loss_total,
            "loss_recon": loss_recon,
            "loss_recon_main": loss_recon_main,
            "loss_image": loss_image,
            "loss_image_linear": image_loss_components["linear"],
            "loss_image_log": image_loss_components["log"],
            "loss_temporal": loss_temporal,
            "loss_linearity": loss_linearity,
            "loss_linearity_aug": loss_linearity_aug,
            "loss_spatial": loss_spatial,
            "loss_eff_metrics": eff_metrics,
            "routing_stats": routing_stats,
            "routing_logit_rms": float(routing_logit_rms),
            "routing_raw_logit_rms": float(routing_raw_logit_rms),
            "routing_usage": routing_usage,
            "step_idx": int(step_idx),
        }

    def _run_train_step_eager(
        self,
        *,
        positions: torch.Tensor,
        params: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor],
        probe_idx: Optional[torch.Tensor],
        step_idx: int,
        should_compute_linearity: bool,
        should_log_grad: bool,
        pre_clip_norms_list: list[Dict[str, float]],
        post_clip_norms_list: list[Dict[str, float]],
    ) -> Dict[str, Any]:
        self.optimizer.zero_grad(set_to_none=True)
        monitor_stats: Dict[str, float] = {}
        should_run_grad_diag = bool(
            self.monitor_enabled
            and self.monitor_grad_diagnostics_every_steps > 0
            and (int(step_idx) % int(self.monitor_grad_diagnostics_every_steps) == 0)
        )
        should_run_update_ratio = bool(
            self.monitor_enabled
            and self.monitor_update_ratio_every_steps > 0
            and (int(step_idx) % int(self.monitor_update_ratio_every_steps) == 0)
        )
        update_named_params: list[tuple[str, nn.Parameter]] = []
        update_snapshots: list[torch.Tensor] = []
        with self._profile_range("train/forward_loss"):
            with self._autocast_context():
                step_outputs = self._compute_train_step_outputs(
                    positions=positions,
                    params=params,
                    targets=targets,
                    mask=mask,
                    probe_idx=probe_idx,
                    step_idx=step_idx,
                    should_compute_linearity=should_compute_linearity,
                )
                loss_gbuffer, gbuffer_samples = self._compute_gbuffer_image_loss(step_idx=step_idx)
                step_outputs["loss_gbuffer"] = loss_gbuffer
                step_outputs["gbuffer_samples"] = float(gbuffer_samples)
                eff_metrics = step_outputs.get("loss_eff_metrics", {})
                if float(gbuffer_samples) > 0.0 and float(self._current_gbuffer_image_loss_weight) > 0.0:
                    eff_w_gbuffer, gbuffer_stats = self._loss_effective.compute_weight(
                        term="gbuffer",
                        base_weight=float(self._current_gbuffer_image_loss_weight),
                        raw_loss=loss_gbuffer,
                        frequency=(1.0 / float(max(1, self.gbuffer_image_loss_every_steps))),
                    )
                    self._write_effective_term_metrics(eff_metrics, term="gbuffer", stats=gbuffer_stats)
                    step_outputs["loss_total"] = (
                        step_outputs["loss_total"]
                        + float(eff_w_gbuffer) * loss_gbuffer
                    )
                else:
                    gbuffer_stats = {
                        "scale": 1.0,
                        "weight_base": float(self._current_gbuffer_image_loss_weight),
                        "weight_effective": float(self._current_gbuffer_image_loss_weight),
                        "contrib_base": 0.0,
                        "contrib_effective": 0.0,
                        "target": 0.0,
                        "expected": 0.0,
                        "ema_loss": 0.0,
                        "frequency": (1.0 / float(max(1, self.gbuffer_image_loss_every_steps))),
                        "enabled": 0.0,
                    }
                    self._write_effective_term_metrics(eff_metrics, term="gbuffer", stats=gbuffer_stats)
                step_outputs["loss_eff_metrics"] = eff_metrics
                if should_run_grad_diag:
                    loss_main = step_outputs.get(
                        "loss_recon_main",
                        float(self.recon_weight) * step_outputs["loss_recon"],
                    )
                    aux_loss = step_outputs["loss_total"] - loss_main
                    diag_stats = self._compute_grad_diagnostics(
                        loss_main=loss_main,
                        loss_aux=aux_loss,
                    )
                    monitor_stats.update(diag_stats)
                if should_run_update_ratio:
                    update_named_params = self._get_monitor_shared_named_params()
                    if update_named_params:
                        update_snapshots = [
                            p.detach().clone()
                            for _, p in update_named_params
                        ]
                monitor_stats["routing_logit_rms"] = float(step_outputs.get("routing_logit_rms", 0.0))
                monitor_stats["routing_raw_logit_rms"] = float(step_outputs.get("routing_raw_logit_rms", 0.0))
                monitor_stats["routing_logit_sample"] = 1.0

        with self._profile_range("train/backward"):
            if self._grad_scaler.is_enabled():
                self._grad_scaler.scale(step_outputs["loss_total"]).backward()
                self._grad_scaler.unscale_(self.optimizer)
            else:
                step_outputs["loss_total"].backward()

        if should_log_grad:
            pre_clip_norms = self._compute_grad_norms(prefix="grad_pre_clip")
            pre_clip_norms_list.append(pre_clip_norms)

        clip_norm_before = 0.0
        clip_triggered = 0.0
        if self.grad_clip is not None:
            clip_total = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            clip_norm_before = float(clip_total.detach().item() if torch.is_tensor(clip_total) else clip_total)
            clip_triggered = 1.0 if clip_norm_before > float(self.grad_clip) else 0.0
        monitor_stats["grad_clip_norm_before"] = float(clip_norm_before)
        monitor_stats["grad_clip_triggered"] = float(clip_triggered)
        monitor_stats["grad_clip_step"] = 1.0 if self.grad_clip is not None else 0.0

        if should_log_grad:
            post_clip_norms = self._compute_grad_norms(prefix="grad_post_clip")
            post_clip_norms_list.append(post_clip_norms)

        with self._profile_range("train/optimizer_step"):
            if self._grad_scaler.is_enabled():
                self._grad_scaler.step(self.optimizer)
                self._grad_scaler.update()
            else:
                self.optimizer.step()
        if update_snapshots and update_named_params:
            with torch.no_grad():
                delta_sq = 0.0
                weight_sq = 0.0
                for (_, param), pre in zip(update_named_params, update_snapshots):
                    cur = param.detach()
                    diff = cur - pre
                    delta_sq += float(torch.sum(diff * diff).item())
                    weight_sq += float(torch.sum(pre * pre).item())
                ratio = float(np.sqrt(delta_sq / max(1e-12, weight_sq)))
                monitor_stats["param_update_ratio_shared"] = ratio
                monitor_stats["param_update_sample"] = 1.0
        else:
            monitor_stats["param_update_ratio_shared"] = 0.0
            monitor_stats["param_update_sample"] = 0.0
        if "grad_diag_sample" not in monitor_stats:
            monitor_stats["grad_diag_sample"] = 0.0
            monitor_stats["grad_ratio_aux_main_shared"] = 0.0
            monitor_stats["grad_cos_aux_main_shared"] = 0.0
            monitor_stats["grad_conflict_flag"] = 0.0
            monitor_stats["grad_norm_main_shared"] = 0.0
            monitor_stats["grad_norm_aux_shared"] = 0.0
            monitor_stats["gns_proxy_value"] = 0.0
            monitor_stats["gns_proxy_valid"] = 0.0
        step_outputs["monitor_stats"] = monitor_stats
        self._update_ema()
        return step_outputs

    def _capture_cuda_graph_branch(
        self,
        *,
        branch_key: str,
        positions: torch.Tensor,
        params: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor],
        probe_idx: Optional[torch.Tensor],
        should_compute_linearity: bool,
    ) -> None:
        if self.device.type != "cuda":
            raise RuntimeError("CUDA graph capture requires CUDA device")

        static_inputs: Dict[str, Optional[torch.Tensor]] = {
            "positions": positions.detach().clone(),
            "params": params.detach().clone(),
            "targets": targets.detach().clone(),
            "mask": mask.detach().clone() if mask is not None else None,
            "probe_idx": probe_idx.detach().clone() if probe_idx is not None else None,
        }
        output_buffers: Dict[str, torch.Tensor] = {
            "loss_total": torch.zeros((), device=self.device),
            "loss_recon": torch.zeros((), device=self.device),
            "loss_recon_main": torch.zeros((), device=self.device),
            "loss_image": torch.zeros((), device=self.device),
            "loss_image_linear": torch.zeros((), device=self.device),
            "loss_image_log": torch.zeros((), device=self.device),
            "loss_temporal": torch.zeros((), device=self.device),
            "loss_linearity": torch.zeros((), device=self.device),
            "loss_linearity_aug": torch.zeros((), device=self.device),
            "loss_spatial": torch.zeros((), device=self.device),
        }

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(device=self.device)
        with torch.cuda.graph(graph):
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast_context():
                outputs = self._compute_train_step_outputs(
                    positions=static_inputs["positions"],
                    params=static_inputs["params"],
                    targets=static_inputs["targets"],
                    mask=static_inputs["mask"],
                    probe_idx=static_inputs["probe_idx"],
                    step_idx=0,
                    should_compute_linearity=should_compute_linearity,
                )
            outputs["loss_total"].backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            output_buffers["loss_total"].copy_(outputs["loss_total"].detach())
            output_buffers["loss_recon"].copy_(outputs["loss_recon"].detach())
            output_buffers["loss_recon_main"].copy_(outputs["loss_recon_main"].detach())
            output_buffers["loss_image"].copy_(outputs["loss_image"].detach())
            output_buffers["loss_image_linear"].copy_(outputs["loss_image_linear"].detach())
            output_buffers["loss_image_log"].copy_(outputs["loss_image_log"].detach())
            output_buffers["loss_temporal"].copy_(outputs["loss_temporal"].detach())
            output_buffers["loss_linearity"].copy_(outputs["loss_linearity"].detach())
            output_buffers["loss_linearity_aug"].copy_(outputs["loss_linearity_aug"].detach())
            output_buffers["loss_spatial"].copy_(outputs["loss_spatial"].detach())

        state = self._cuda_graph_states[branch_key]
        state["graph"] = graph
        state["inputs"] = static_inputs
        state["outputs"] = output_buffers
        state["captured"] = True
        self._cuda_graph_stats[f"capture_{branch_key}"] += 1

    def _run_train_step_graph(
        self,
        *,
        positions: torch.Tensor,
        params: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor],
        probe_idx: Optional[torch.Tensor],
        should_compute_linearity: bool,
        should_log_grad: bool,
    ) -> Optional[Dict[str, Any]]:
        can_use, reason = self._cuda_graph_can_use(
            should_compute_linearity=should_compute_linearity,
            should_log_grad=should_log_grad,
        )
        if not can_use:
            if reason == "grad_norm_logging_step":
                self._cuda_graph_stats["fallback_grad_norm_step"] += 1
            else:
                self._cuda_graph_stats["fallback_ineligible"] += 1
            if not self.cuda_graph_fallback_eager:
                raise RuntimeError(f"CUDA graph disabled for current step: {reason}")
            return None

        branch_key = self._cuda_graph_branch_key(should_compute_linearity)
        state = self._cuda_graph_states[branch_key]
        if state.get("disabled", False):
            self._cuda_graph_stats["fallback_ineligible"] += 1
            if not self.cuda_graph_fallback_eager:
                raise RuntimeError(
                    f"CUDA graph branch '{branch_key}' disabled: {state.get('disable_reason', 'unknown')}"
                )
            return None

        captured_now = False
        if not state.get("captured", False):
            warmup_remaining = int(state.get("warmup_remaining", 0))
            if warmup_remaining > 0:
                state["warmup_remaining"] = warmup_remaining - 1
                self._cuda_graph_stats["fallback_warmup"] += 1
                return None
            try:
                self._capture_cuda_graph_branch(
                    branch_key=branch_key,
                    positions=positions,
                    params=params,
                    targets=targets,
                    mask=mask,
                    probe_idx=probe_idx,
                    should_compute_linearity=should_compute_linearity,
                )
            except Exception as exc:
                state["disabled"] = True
                state["disable_reason"] = str(exc)
                self._cuda_graph_stats["fallback_capture_error"] += 1
                if self.show_progress:
                    print(f"[cuda-graph] disable branch={branch_key} reason={exc}")
                if not self.cuda_graph_fallback_eager:
                    raise RuntimeError(f"CUDA graph capture failed for branch '{branch_key}': {exc}") from exc
                return None
            captured_now = True

        if captured_now:
            self._update_ema()
            self._cuda_graph_stats[f"replay_{branch_key}"] += 1
            outputs = state.get("outputs", {})
            if not isinstance(outputs, dict):
                return None
            return {
                "loss_total": outputs["loss_total"],
                "loss_recon": outputs["loss_recon"],
                "loss_recon_main": outputs["loss_recon_main"],
                "loss_image": outputs["loss_image"],
                "loss_image_linear": outputs["loss_image_linear"],
                "loss_image_log": outputs["loss_image_log"],
                "loss_temporal": outputs["loss_temporal"],
                "loss_linearity": outputs["loss_linearity"],
                "loss_linearity_aug": outputs["loss_linearity_aug"],
                "loss_spatial": outputs["loss_spatial"],
                "routing_stats": {
                    "routing_balance_loss": 0.0,
                    "routing_entropy": 0.0,
                    "routing_nonzero_ratio": 0.0,
                },
                "routing_logit_rms": 0.0,
                "monitor_stats": {},
                "step_idx": 0,
            }

        static_inputs = state.get("inputs")
        if not isinstance(static_inputs, dict) or not self._cuda_graph_inputs_match(
            static_inputs,
            positions,
            params,
            targets,
            mask,
            probe_idx,
        ):
            self._cuda_graph_stats["fallback_shape_mismatch"] += 1
            if not self.cuda_graph_fallback_eager:
                raise RuntimeError(f"CUDA graph input mismatch for branch '{branch_key}'")
            return None

        self._cuda_graph_copy_inputs(
            static_inputs,
            positions,
            params,
            targets,
            mask,
            probe_idx,
        )
        with self._profile_range(f"train/graph_replay/{branch_key}"):
            state["graph"].replay()
        self._update_ema()
        self._cuda_graph_stats[f"replay_{branch_key}"] += 1

        outputs = state.get("outputs", {})
        if not isinstance(outputs, dict):
            return None
        return {
            "loss_total": outputs["loss_total"],
            "loss_recon": outputs["loss_recon"],
            "loss_recon_main": outputs["loss_recon_main"],
            "loss_image": outputs["loss_image"],
            "loss_image_linear": outputs["loss_image_linear"],
            "loss_image_log": outputs["loss_image_log"],
            "loss_temporal": outputs["loss_temporal"],
            "loss_linearity": outputs["loss_linearity"],
            "loss_linearity_aug": outputs["loss_linearity_aug"],
            "loss_spatial": outputs["loss_spatial"],
            "routing_stats": {
                "routing_balance_loss": 0.0,
                "routing_entropy": 0.0,
                "routing_nonzero_ratio": 0.0,
            },
            "routing_logit_rms": 0.0,
            "monitor_stats": {},
            "step_idx": 0,
        }

    def _compute_grad_norms(self, prefix: str = "") -> Dict[str, float]:
        """Compute gradient norms for all parameter groups.

        Args:
            prefix: Prefix for metric keys (e.g., "grad_pre_clip", "grad_post_clip")

        Returns:
            Dictionary with gradient norms for total and each parameter group
        """
        grad_norms = {}

        # TODO(perf): 该函数每 batch 调用两次（pre/post clip），且包含多次 .item()。
        # .item() 会触发 CPU-GPU 同步；若追求吞吐，可降低记录频率或按 epoch 聚合。
        # Compute total gradient norm
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms[f'{prefix}/total'] = total_norm

        # Compute per-optimizer-group norms (routing/basis/coeff/encoder/film/...)
        for opt_group in self.optimizer.param_groups:
            group_name = str(opt_group.get("group", "other"))
            group_total = 0.0
            for param in opt_group.get("params", []):
                if param.grad is None:
                    continue
                param_norm = param.grad.detach().data.norm(2)
                group_total += param_norm.item() ** 2
            grad_norms[f"{prefix}/{group_name}"] = group_total ** 0.5

        return grad_norms

    def _accumulate_grad_norms(self, grad_norms_list: list[Dict[str, float]]) -> Dict[str, float]:
        """Average gradient norms across batches.

        Args:
            grad_norms_list: List of gradient norm dictionaries from each batch

        Returns:
            Averaged gradient norms
        """
        if not grad_norms_list:
            return {}

        # Get all keys from first entry
        keys = grad_norms_list[0].keys()
        averaged = {}

        for key in keys:
            values = [gn[key] for gn in grad_norms_list if key in gn]
            averaged[key] = sum(values) / len(values) if values else 0.0

        return averaged

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.train()
        self._train_routing_mode = True

        totals = {
            "total": torch.tensor(0.0, device=self.device),
            "recon": torch.tensor(0.0, device=self.device),
            "image": torch.tensor(0.0, device=self.device),
            "gbuffer": torch.tensor(0.0, device=self.device),
            "image_linear": torch.tensor(0.0, device=self.device),
            "image_log": torch.tensor(0.0, device=self.device),
            "coeff_l1": torch.tensor(0.0, device=self.device),
            "linearity": torch.tensor(0.0, device=self.device),
            "spatial": torch.tensor(0.0, device=self.device),
            "routing_balance": 0.0,
            "routing_entropy": 0.0,
            "routing_nonzero_ratio": 0.0,
            "gbuffer_samples": 0.0,
        }
        loss_eff_totals: Dict[str, float] = {}
        monitor_totals: Dict[str, float] = {
            "grad_diag_samples": 0.0,
            "grad_ratio_aux_main_sum": 0.0,
            "grad_cos_sum": 0.0,
            "grad_conflict_sum": 0.0,
            "gns_sum": 0.0,
            "gns_samples": 0.0,
            "param_update_ratio_sum": 0.0,
            "param_update_samples": 0.0,
            "grad_clip_trigger_sum": 0.0,
            "grad_clip_steps": 0.0,
            "grad_clip_norm_sum": 0.0,
            "routing_logit_rms_sum": 0.0,
            "routing_logit_rms_max": 0.0,
            "routing_raw_logit_rms_sum": 0.0,
            "routing_raw_logit_rms_max": 0.0,
            "routing_logit_samples": 0.0,
            "pulse_count": 0.0,
            "pulse_recovery_sum": 0.0,
            "pulse_recovery_count": 0.0,
            "pulse_unresolved_count": 0.0,
        }
        routing_usage_sum: Optional[torch.Tensor] = None
        pulse_pending_baseline: Optional[float] = None
        pulse_pending_steps = 0
        pre_clip_norms_list = []
        post_clip_norms_list = []
        num_batches = 0
        linearity_eval_steps = 0
        linearity_skip_steps = 0
        for key in (
            "capture_regular",
            "capture_linearity",
            "replay_regular",
            "replay_linearity",
            "fallback_eager",
            "fallback_ineligible",
            "fallback_warmup",
            "fallback_capture_error",
            "fallback_shape_mismatch",
            "fallback_grad_norm_step",
        ):
            self._cuda_graph_stats[key] = 0

        try:
            loader = train_loader
            if self.show_progress and _tqdm is not None:
                desc = self._progress_prefix + " [train]" if self._progress_prefix else "train"
                loader = _tqdm(train_loader, desc=desc, leave=False, unit="batch")

            for step_idx, batch in enumerate(loader, start=1):
                if self.max_train_batches > 0 and step_idx > self.max_train_batches:
                    break
                with self._profile_range("train/unpack"):
                    unpacked = self.adapter.unpack(batch, self.device)
                if len(unpacked) == 4:
                    positions, params, targets, mask = unpacked
                else:
                    positions, params, targets = unpacked
                    mask = None

                self._set_image_sampling_context_for_step(step_idx)
                should_compute_linearity = (
                    self.linearity_weight > 0.0
                    and (step_idx % self.linearity_every_steps == 0)
                )
                if should_compute_linearity:
                    linearity_eval_steps += 1
                elif self.linearity_weight > 0.0:
                    linearity_skip_steps += 1
                probe_idx = batch.get("probe_idx") if isinstance(batch, dict) else None
                should_log_grad = self.grad_norm_log_every_steps > 0 and (step_idx % self.grad_norm_log_every_steps == 0)
                graph_outputs = None
                if self._cuda_graph_enabled:
                    graph_outputs = self._run_train_step_graph(
                        positions=positions,
                        params=params,
                        targets=targets,
                        mask=mask,
                        probe_idx=probe_idx,
                        should_compute_linearity=should_compute_linearity,
                        should_log_grad=should_log_grad,
                    )
                if graph_outputs is None:
                    if self._cuda_graph_enabled:
                        self._cuda_graph_stats["fallback_eager"] += 1
                    step_outputs = self._run_train_step_eager(
                        positions=positions,
                        params=params,
                        targets=targets,
                        mask=mask,
                        probe_idx=probe_idx,
                        step_idx=step_idx,
                        should_compute_linearity=should_compute_linearity,
                        should_log_grad=should_log_grad,
                        pre_clip_norms_list=pre_clip_norms_list,
                        post_clip_norms_list=post_clip_norms_list,
                    )
                else:
                    step_outputs = graph_outputs

                totals["total"] += step_outputs["loss_total"].detach()
                totals["recon"] += step_outputs["loss_recon"].detach()
                totals["image"] += step_outputs["loss_image"].detach()
                totals["gbuffer"] += step_outputs.get("loss_gbuffer", torch.tensor(0.0, device=self.device)).detach()
                totals["image_linear"] += step_outputs["loss_image_linear"].detach()
                totals["image_log"] += step_outputs["loss_image_log"].detach()
                totals["coeff_l1"] += step_outputs["loss_temporal"].detach()
                totals["linearity"] += (
                    step_outputs["loss_linearity"].detach()
                    + step_outputs["loss_linearity_aug"].detach()
                )
                totals["spatial"] += step_outputs["loss_spatial"].detach()
                routing_stats = step_outputs["routing_stats"]
                totals["routing_balance"] += float(routing_stats["routing_balance_loss"])
                totals["routing_entropy"] += float(routing_stats["routing_entropy"])
                totals["routing_nonzero_ratio"] += float(routing_stats["routing_nonzero_ratio"])
                routing_usage_step = step_outputs.get("routing_usage", None)
                if torch.is_tensor(routing_usage_step):
                    usage_detached = routing_usage_step.detach().to(dtype=torch.float32)
                    if routing_usage_sum is None:
                        routing_usage_sum = torch.zeros_like(usage_detached)
                    if routing_usage_sum.shape != usage_detached.shape:
                        min_len = min(int(routing_usage_sum.numel()), int(usage_detached.numel()))
                        merged = torch.zeros(
                            (max(int(routing_usage_sum.numel()), int(usage_detached.numel())),),
                            device=usage_detached.device,
                            dtype=torch.float32,
                        )
                        merged[: int(routing_usage_sum.numel())] += routing_usage_sum.reshape(-1)
                        merged[: int(usage_detached.numel())] += usage_detached.reshape(-1)
                        routing_usage_sum = merged
                    else:
                        routing_usage_sum = routing_usage_sum + usage_detached
                totals["gbuffer_samples"] += float(step_outputs.get("gbuffer_samples", 0.0))
                loss_eff_step = step_outputs.get("loss_eff_metrics", {})
                if isinstance(loss_eff_step, dict):
                    for key, value in loss_eff_step.items():
                        try:
                            loss_eff_totals[str(key)] = loss_eff_totals.get(str(key), 0.0) + float(value)
                        except Exception:
                            continue
                monitor_step = step_outputs.get("monitor_stats", {})
                if isinstance(monitor_step, dict):
                    diag_sample = float(monitor_step.get("grad_diag_sample", 0.0))
                    monitor_totals["grad_diag_samples"] += diag_sample
                    monitor_totals["grad_ratio_aux_main_sum"] += float(
                        monitor_step.get("grad_ratio_aux_main_shared", 0.0)
                    ) * diag_sample
                    monitor_totals["grad_cos_sum"] += float(
                        monitor_step.get("grad_cos_aux_main_shared", 0.0)
                    ) * diag_sample
                    monitor_totals["grad_conflict_sum"] += float(
                        monitor_step.get("grad_conflict_flag", 0.0)
                    ) * diag_sample

                    gns_valid = float(monitor_step.get("gns_proxy_valid", 0.0))
                    monitor_totals["gns_samples"] += gns_valid
                    monitor_totals["gns_sum"] += float(
                        monitor_step.get("gns_proxy_value", 0.0)
                    ) * gns_valid

                    update_sample = float(monitor_step.get("param_update_sample", 0.0))
                    monitor_totals["param_update_samples"] += update_sample
                    monitor_totals["param_update_ratio_sum"] += float(
                        monitor_step.get("param_update_ratio_shared", 0.0)
                    ) * update_sample

                    clip_step = float(monitor_step.get("grad_clip_step", 0.0))
                    monitor_totals["grad_clip_steps"] += clip_step
                    monitor_totals["grad_clip_trigger_sum"] += float(
                        monitor_step.get("grad_clip_triggered", 0.0)
                    ) * clip_step
                    monitor_totals["grad_clip_norm_sum"] += float(
                        monitor_step.get("grad_clip_norm_before", 0.0)
                    ) * clip_step

                    logit_sample = float(monitor_step.get("routing_logit_sample", 0.0))
                    logit_rms = float(monitor_step.get("routing_logit_rms", 0.0))
                    raw_logit_rms = float(monitor_step.get("routing_raw_logit_rms", 0.0))
                    monitor_totals["routing_logit_samples"] += logit_sample
                    monitor_totals["routing_logit_rms_sum"] += logit_rms * logit_sample
                    monitor_totals["routing_raw_logit_rms_sum"] += raw_logit_rms * logit_sample
                    if logit_rms > monitor_totals["routing_logit_rms_max"]:
                        monitor_totals["routing_logit_rms_max"] = logit_rms
                    if raw_logit_rms > monitor_totals["routing_raw_logit_rms_max"]:
                        monitor_totals["routing_raw_logit_rms_max"] = raw_logit_rms

                recon_step = float(step_outputs["loss_recon"].detach().item())
                gbuffer_trigger = bool(
                    float(step_outputs.get("gbuffer_samples", 0.0)) > 0.0
                    and float(self._current_gbuffer_image_loss_weight) > 0.0
                )
                if gbuffer_trigger:
                    monitor_totals["pulse_count"] += 1.0
                    if pulse_pending_baseline is not None:
                        monitor_totals["pulse_unresolved_count"] += 1.0
                    baseline = (
                        float(self._loss_effective.recon_ema)
                        if self._loss_effective.recon_ema is not None
                        else recon_step
                    )
                    pulse_pending_baseline = baseline
                    pulse_pending_steps = 0
                elif pulse_pending_baseline is not None:
                    pulse_pending_steps += 1
                    recovery_threshold = pulse_pending_baseline * (
                        1.0 + float(self.monitor_pulse_recovery_tolerance)
                    )
                    if recon_step <= recovery_threshold:
                        monitor_totals["pulse_recovery_sum"] += float(pulse_pending_steps)
                        monitor_totals["pulse_recovery_count"] += 1.0
                        pulse_pending_baseline = None
                        pulse_pending_steps = 0
                    elif pulse_pending_steps >= int(self.monitor_pulse_max_steps):
                        monitor_totals["pulse_unresolved_count"] += 1.0
                        pulse_pending_baseline = None
                        pulse_pending_steps = 0
                num_batches += 1
                self._profile_step()
        finally:
            self._train_routing_mode = False
        if pulse_pending_baseline is not None:
            monitor_totals["pulse_unresolved_count"] += 1.0

        # Average loss metrics
        metrics = {}
        denom = max(1, num_batches)
        for k, v in totals.items():
            if torch.is_tensor(v):
                metrics[k] = float((v / denom).detach().item())
            else:
                metrics[k] = float(v / denom)
        for key, value in loss_eff_totals.items():
            metrics[key] = float(value / denom)
        metrics["routing_temp"] = float(self._current_routing_temp)
        metrics["image_loss_weight"] = float(self._current_image_loss_weight)
        metrics["gbuffer_image_loss_weight"] = float(self._current_gbuffer_image_loss_weight)
        metrics["proxy_gbuffer_handover_progress"] = float(self._current_proxy_gbuffer_handover_progress)
        metrics["proxy_gbuffer_proxy_scale"] = float(self._current_proxy_gbuffer_proxy_scale)
        metrics["proxy_gbuffer_gbuffer_scale"] = float(self._current_proxy_gbuffer_gbuffer_scale)
        metrics["image_sampling_seed"] = float(self._image_sampling_seed_current)
        metrics["routing_balance_lambda"] = float(self._current_lambda_routing_balance)
        metrics["recon_weight"] = float(self.recon_weight)
        metrics["linearity_eval_steps"] = float(linearity_eval_steps)
        metrics["linearity_skip_steps"] = float(linearity_skip_steps)
        for key, value in self._cuda_graph_stats.items():
            metrics[f"cuda_graph_{key}"] = float(value)

        grad_diag_denom = max(1.0, monitor_totals["grad_diag_samples"])
        metrics["grad_ratio_aux_main_shared"] = float(
            monitor_totals["grad_ratio_aux_main_sum"] / grad_diag_denom
        )
        metrics["grad_cos_aux_main_shared"] = float(
            monitor_totals["grad_cos_sum"] / grad_diag_denom
        )
        metrics["grad_conflict_rate"] = float(
            monitor_totals["grad_conflict_sum"] / grad_diag_denom
        )
        metrics["grad_diag_samples"] = float(monitor_totals["grad_diag_samples"])

        gns_denom = max(1.0, monitor_totals["gns_samples"])
        metrics["gns_proxy"] = float(monitor_totals["gns_sum"] / gns_denom)
        metrics["gns_proxy_samples"] = float(monitor_totals["gns_samples"])

        upd_denom = max(1.0, monitor_totals["param_update_samples"])
        metrics["param_update_ratio_shared"] = float(
            monitor_totals["param_update_ratio_sum"] / upd_denom
        )
        metrics["param_update_samples"] = float(monitor_totals["param_update_samples"])

        clip_denom = max(1.0, monitor_totals["grad_clip_steps"])
        metrics["grad_clip_trigger_rate"] = float(
            monitor_totals["grad_clip_trigger_sum"] / clip_denom
        )
        metrics["grad_clip_norm_before"] = float(
            monitor_totals["grad_clip_norm_sum"] / clip_denom
        )
        metrics["grad_clip_steps"] = float(monitor_totals["grad_clip_steps"])

        logit_denom = max(1.0, monitor_totals["routing_logit_samples"])
        metrics["routing_logit_rms"] = float(
            monitor_totals["routing_logit_rms_sum"] / logit_denom
        )
        metrics["routing_logit_rms_max"] = float(monitor_totals["routing_logit_rms_max"])
        metrics["routing_raw_logit_rms"] = float(
            monitor_totals["routing_raw_logit_rms_sum"] / logit_denom
        )
        metrics["routing_raw_logit_rms_max"] = float(monitor_totals["routing_raw_logit_rms_max"])
        metrics["routing_logit_samples"] = float(monitor_totals["routing_logit_samples"])

        pulse_count = float(monitor_totals["pulse_count"])
        pulse_recovery_count = float(monitor_totals["pulse_recovery_count"])
        pulse_recovery_denom = max(1.0, pulse_recovery_count)
        pulse_unresolved = float(monitor_totals["pulse_unresolved_count"])
        metrics["pulse_count"] = pulse_count
        metrics["pulse_recovery_steps_mean"] = float(
            monitor_totals["pulse_recovery_sum"] / pulse_recovery_denom
        )
        metrics["pulse_recovery_count"] = pulse_recovery_count
        metrics["pulse_unresolved_count"] = pulse_unresolved
        metrics["pulse_unresolved_rate"] = float(pulse_unresolved / max(1.0, pulse_count))

        if (
            self.monitor_enabled
            and self.monitor_spectrum_every_epochs > 0
            and (int(self._current_epoch_for_monitor) % int(self.monitor_spectrum_every_epochs) == 0)
        ):
            metrics.update(self._compute_weight_spectrum_metrics())
            metrics.update(
                self._compute_active_coeff_spectrum_metrics(
                    routing_usage_sum=routing_usage_sum,
                )
            )

        # Average and add gradient norms
        avg_pre_clip = self._accumulate_grad_norms(pre_clip_norms_list)
        avg_post_clip = self._accumulate_grad_norms(post_clip_norms_list)
        metrics.update(avg_pre_clip)
        metrics.update(avg_post_clip)

        return metrics

    @torch.no_grad()
    def validate_epoch(
        self,
        val_loader: torch.utils.data.DataLoader,
        *,
        top_k: Optional[int] = None,
        training_soft_routing: Optional[bool] = None,
        routing_soft_topk: Optional[int] = None,
        routing_temperature: Optional[float] = None,
    ) -> Dict[str, float]:
        self.model.eval()

        sums = {
            "mae": torch.tensor(0.0, device=self.device),
            "rmse": torch.tensor(0.0, device=self.device),
            "charbonnier": torch.tensor(0.0, device=self.device),
            "superposition": torch.tensor(0.0, device=self.device),
            "img_mae": torch.tensor(0.0, device=self.device),
            "img_rmse": torch.tensor(0.0, device=self.device),
            "img_psnr": torch.tensor(0.0, device=self.device),
        }
        num_batches = 0

        loader = val_loader
        if self.show_progress and _tqdm is not None:
            desc = self._progress_prefix + " [val]" if self._progress_prefix else "val"
            loader = _tqdm(val_loader, desc=desc, leave=False, unit="batch")

        for step_idx, batch in enumerate(loader, start=1):
            if self.max_val_batches > 0 and step_idx > self.max_val_batches:
                break
            with self._profile_range("val/unpack"):
                unpacked = self.adapter.unpack(batch, self.device)
            if len(unpacked) == 4:
                positions, params, targets, mask = unpacked
            else:
                positions, params, targets = unpacked
                mask = None

            with self._profile_range("val/forward_metrics"):
                with self._autocast_context():
                    preds = self._forward(
                        positions,
                        params,
                        mask,
                        top_k=top_k,
                        training_soft_routing=training_soft_routing,
                        routing_soft_topk=routing_soft_topk,
                        routing_temperature=routing_temperature,
                    )
                    preds_eval, targets_eval = self.adapter.inverse_targets(preds, targets)

                pred32 = preds_eval.to(dtype=torch.float32)
                target32 = targets_eval.to(dtype=torch.float32)
                mae = torch.mean(torch.abs(pred32 - target32))
                rmse = torch.sqrt(torch.mean((pred32 - target32) ** 2))

                if self.recon_loss == "charbonnier":
                    charbonnier = charbonnier_loss(preds_eval, targets_eval, epsilon=self.charbonnier_eps).to(dtype=torch.float32)
                else:
                    charbonnier = torch.tensor(0.0, device=self.device, dtype=torch.float32)

                sums["mae"] += mae
                sums["rmse"] += rmse
                sums["charbonnier"] += charbonnier

                if self.compute_img_metrics_in_val:
                    img_metrics = self._compute_image_metrics(preds_eval, targets_eval)
                    sums["img_mae"] += img_metrics["img_mae"]
                    sums["img_rmse"] += img_metrics["img_rmse"]
                    sums["img_psnr"] += img_metrics["img_psnr"]

                if self.compute_superposition_in_val and self.linearity_weight > 0:
                    sup_err = self._compute_linearity_loss(positions, params, mask, preds).to(dtype=torch.float32)
                    sums["superposition"] += sup_err
            num_batches += 1
            self._profile_step()
            self._profile_step()

        denom = max(1, num_batches)
        return {k: float((v / denom).detach().item()) for k, v in sums.items()}

    @staticmethod
    def _soft_profile_share_key(profile_kwargs: Dict[str, Any]) -> tuple[int, Optional[int]]:
        top_k = int(profile_kwargs.get("top_k", 0))
        soft_topk_raw = profile_kwargs.get("routing_soft_topk", None)
        soft_topk = None if soft_topk_raw is None else int(soft_topk_raw)
        return (top_k, soft_topk)

    def _compute_soft_routing_exponent(
        self,
        positions: torch.Tensor,
        *,
        top_k: int,
        routing_soft_topk: Optional[int],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not (hasattr(self.model, "mu") and hasattr(self.model, "log_scale")):
            return None
        if not (torch.is_tensor(getattr(self.model, "mu")) and torch.is_tensor(getattr(self.model, "log_scale"))):
            return None

        mu = getattr(self.model, "mu")
        log_scale = getattr(self.model, "log_scale")
        num_experts = int(getattr(self.model, "K", int(mu.shape[0])))
        if routing_soft_topk is None or int(routing_soft_topk) <= 0:
            soft_k = num_experts
        else:
            soft_k = min(num_experts, max(int(routing_soft_topk), int(top_k)))

        pos_expanded = positions.unsqueeze(1)
        diff_all = mu.unsqueeze(0) - pos_expanded
        distances = torch.sum(diff_all * diff_all, dim=-1)
        _, topk_indices = torch.topk(distances, k=soft_k, largest=False, dim=-1)

        selected_mu = mu[topk_indices]
        selected_scale = torch.exp(log_scale[topk_indices])
        diff = pos_expanded - selected_mu
        exponent = -0.5 * torch.sum((diff / selected_scale) ** 2, dim=-1)
        return exponent, topk_indices

    @torch.no_grad()
    def validate_epoch_soft_profile_group(
        self,
        val_loader: torch.utils.data.DataLoader,
        *,
        profiles: list[Dict[str, Any]],
        check_equivalence: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        if not profiles:
            return {}

        first = profiles[0]
        top_k = int(first.get("top_k", self.top_k))
        routing_soft_topk = first.get("routing_soft_topk", None)

        sums: Dict[str, Dict[str, torch.Tensor]] = {}
        for profile in profiles:
            name = str(profile["name"])
            sums[name] = {
                "mae": torch.tensor(0.0, device=self.device),
                "rmse": torch.tensor(0.0, device=self.device),
                "charbonnier": torch.tensor(0.0, device=self.device),
                "superposition": torch.tensor(0.0, device=self.device),
                "img_mae": torch.tensor(0.0, device=self.device),
                "img_rmse": torch.tensor(0.0, device=self.device),
                "img_psnr": torch.tensor(0.0, device=self.device),
            }

        loader = val_loader
        if self.show_progress and _tqdm is not None:
            desc = self._progress_prefix + " [val-shared]" if self._progress_prefix else "val-shared"
            loader = _tqdm(val_loader, desc=desc, leave=False, unit="batch")

        num_batches = 0
        for step_idx, batch in enumerate(loader, start=1):
            if self.max_val_batches > 0 and step_idx > self.max_val_batches:
                break

            unpacked = self.adapter.unpack(batch, self.device)
            if len(unpacked) == 4:
                positions, params, targets, mask = unpacked
            else:
                positions, params, targets = unpacked
                mask = None

            with self._autocast_context():
                routing_base = self._compute_soft_routing_exponent(
                    positions,
                    top_k=top_k,
                    routing_soft_topk=routing_soft_topk,
                )
                if routing_base is None:
                    raise RuntimeError("Soft profile sharing is unavailable for current model.")
                exponent, topk_indices = routing_base

            for profile in profiles:
                profile_name = str(profile["name"])
                temp = float(profile.get("routing_temperature", self._current_routing_temp))
                temp = max(1e-6, temp)
                with self._autocast_context():
                    weights = torch.softmax(exponent / temp, dim=-1)
                    preds = self._forward(positions, params, mask, routing=(weights, topk_indices), top_k=top_k)
                    preds_eval, targets_eval = self.adapter.inverse_targets(preds, targets)

                pred32 = preds_eval.to(dtype=torch.float32)
                target32 = targets_eval.to(dtype=torch.float32)
                mae = torch.mean(torch.abs(pred32 - target32))
                rmse = torch.sqrt(torch.mean((pred32 - target32) ** 2))
                if self.recon_loss == "charbonnier":
                    charbonnier = charbonnier_loss(preds_eval, targets_eval, epsilon=self.charbonnier_eps).to(dtype=torch.float32)
                else:
                    charbonnier = torch.tensor(0.0, device=self.device, dtype=torch.float32)

                sums[profile_name]["mae"] += mae
                sums[profile_name]["rmse"] += rmse
                sums[profile_name]["charbonnier"] += charbonnier

                if self.compute_img_metrics_in_val:
                    img_m = self._compute_image_metrics(preds_eval, targets_eval)
                    sums[profile_name]["img_mae"] += img_m["img_mae"]
                    sums[profile_name]["img_rmse"] += img_m["img_rmse"]
                    sums[profile_name]["img_psnr"] += img_m["img_psnr"]

                if self.compute_superposition_in_val and self.linearity_weight > 0:
                    sup_err = self._compute_linearity_loss(positions, params, mask, preds).to(dtype=torch.float32)
                    sums[profile_name]["superposition"] += sup_err

                if check_equivalence and step_idx <= self.soft_profile_equiv_check_batches:
                    with self._autocast_context():
                        legacy_preds = self._forward(
                            positions,
                            params,
                            mask,
                            top_k=top_k,
                            training_soft_routing=True,
                            routing_soft_topk=routing_soft_topk,
                            routing_temperature=temp,
                        )
                        legacy_eval, _ = self.adapter.inverse_targets(legacy_preds, targets)
                    legacy32 = legacy_eval.to(dtype=torch.float32)
                    delta_mae = torch.mean(torch.abs(pred32 - legacy32))
                    if float(delta_mae.detach().item()) > self.soft_profile_mae_tolerance:
                        raise RuntimeError(
                            f"soft-profile sharing equivalence failed for {profile_name}: "
                            f"delta_mae={float(delta_mae.detach().item()):.6g}"
                        )
                    if self.compute_img_metrics_in_val:
                        shared_psnr = self._compute_image_metrics(preds_eval, targets_eval)["img_psnr"]
                        legacy_psnr = self._compute_image_metrics(legacy_eval, targets_eval)["img_psnr"]
                        delta_psnr = torch.abs(shared_psnr - legacy_psnr)
                        if float(delta_psnr.detach().item()) > self.soft_profile_img_psnr_tolerance:
                            raise RuntimeError(
                                f"soft-profile sharing equivalence failed for {profile_name}: "
                                f"delta_img_psnr={float(delta_psnr.detach().item()):.6g}"
                            )

            num_batches += 1

        denom = max(1, num_batches)
        out: Dict[str, Dict[str, float]] = {}
        for profile in profiles:
            name = str(profile["name"])
            out[name] = {
                key: float((value / denom).detach().item())
                for key, value in sums[name].items()
            }
        return out

    def _log_visualizations(self, epoch: int, val_loader: torch.utils.data.DataLoader):
        """Log 3D visualizations and SH comparisons to Rerun.

        Args:
            epoch: Current epoch number
            val_loader: Validation data loader
        """
        if self.rerun_logger is None:
            return

        import numpy as np
        from utils.rerun_logger import compute_per_probe_errors

        # 1. Log Gaussian centers
        if hasattr(self.model, 'mu'):
            centers = self.model.mu.detach().cpu().numpy()
            self.rerun_logger.log_gaussian_centers(epoch, centers)

        # 2. Compute and log probe errors
        positions = None
        try:
            positions, errors = compute_per_probe_errors(
                self.model, val_loader.dataset, self.device, self.adapter, max_samples=100
            )
            self.rerun_logger.log_probe_errors(epoch, positions, errors, split="val")
        except Exception as e:
            print(f"Warning: Could not compute probe errors: {e}")

        # 3. Log Gaussian coverage
        if hasattr(self.model, 'mu') and hasattr(self.model, 'log_scale'):
            centers = self.model.mu.detach().cpu().numpy()
            scales = torch.exp(self.model.log_scale).detach().cpu().numpy()
            try:
                if positions is not None:
                    self.rerun_logger.log_gaussian_coverage(epoch, centers, scales, positions)
            except Exception as e:
                print(f"Warning: Could not log Gaussian coverage: {e}")

        # 4. Log best/worst SH comparisons
        try:
            best_sh_gt, best_sh_pred, worst_sh_gt, worst_sh_pred = self._find_best_worst_samples(val_loader)
            self.rerun_logger.log_sh_comparison(epoch, best_sh_gt, best_sh_pred, "best")
            self.rerun_logger.log_sh_comparison(epoch, worst_sh_gt, worst_sh_pred, "worst")

            # 5. Log rendered image comparisons (if enabled)
            render_enabled = os.environ.get("RERUN_DISABLE_RENDER", "0") != "1"
            if render_enabled and hasattr(self.rerun_logger, 'log_rendered_comparison'):
                try:
                    self.rerun_logger.log_rendered_comparison(
                        epoch, best_sh_gt, best_sh_pred, "best"
                    )
                    self.rerun_logger.log_rendered_comparison(
                        epoch, worst_sh_gt, worst_sh_pred, "worst"
                    )
                except Exception as render_error:
                    print(f"Warning: Could not log rendered comparisons: {render_error}")

        except Exception as e:
            print(f"Warning: Could not log SH comparisons: {e}")

    def _find_best_worst_samples(self, val_loader: torch.utils.data.DataLoader):
        """Find best and worst validation samples for visualization.

        Returns:
            best_sh_gt, best_sh_pred, worst_sh_gt, worst_sh_pred: SH coefficients [27]
        """
        import numpy as np

        self.model.eval()
        sample_errors = []
        sample_gts = []
        sample_preds = []

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 20:  # Limit to first 20 batches
                    break

                unpacked = self.adapter.unpack(batch, self.device)
                if len(unpacked) == 4:
                    positions, params, targets, mask = unpacked
                else:
                    positions, params, targets = unpacked
                    mask = None

                preds = self._forward(positions, params, mask)

                # Compute per-sample MAE
                errors = torch.mean(torch.abs(preds - targets), dim=1).cpu().numpy()

                sample_errors.extend(errors)
                sample_gts.extend(targets.cpu().numpy())
                sample_preds.extend(preds.cpu().numpy())

        # Find best and worst
        sample_errors = np.array(sample_errors)
        best_idx = np.argmin(sample_errors)
        worst_idx = np.argmax(sample_errors)

        best_gt = torch.from_numpy(sample_gts[best_idx]).to(self.device)
        best_pred = torch.from_numpy(sample_preds[best_idx]).to(self.device)
        worst_gt = torch.from_numpy(sample_gts[worst_idx]).to(self.device)
        worst_pred = torch.from_numpy(sample_preds[worst_idx]).to(self.device)

        best_pred, best_gt = self.adapter.inverse_targets(best_pred, best_gt)
        worst_pred, worst_gt = self.adapter.inverse_targets(worst_pred, worst_gt)

        return (
            best_gt.detach().cpu().numpy(),
            best_pred.detach().cpu().numpy(),
            worst_gt.detach().cpu().numpy(),
            worst_pred.detach().cpu().numpy(),
        )

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        num_epochs: int,
        output_dir: Optional[Path] = None,
        checkpoint_meta: Optional[Dict[str, Any]] = None,
        save_best: bool = True,
        best_metric: str = "mae",
        val_profiles: Optional[list[Dict[str, Any]]] = None,
        save_best_profiles: bool = False,
        best_metric_per_profile: str = "mae",
        log_every: int = 0,
        epoch_end_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        save_resume_every: int = 1,
        resume_checkpoint_name: str = "resume_latest.pt",
    ) -> Dict[str, Dict[str, list]]:
        history = {
            "train": {
                "total": [],
                "recon": [],
                "image": [],
                "gbuffer": [],
                "coeff_l1": [],
                "linearity": [],
                "spatial": [],
            },
            "val": {"mae": [], "rmse": [], "charbonnier": [], "superposition": [], "img_mae": [], "img_rmse": [], "img_psnr": []},
        }

        best_value = self._metric_init_best_value(best_metric)
        val_profiles_resolved: list[Dict[str, Any]] = []
        if val_profiles:
            for profile in val_profiles:
                if not isinstance(profile, dict):
                    continue
                profile_name = str(profile.get("name", "profile")).strip()
                if not profile_name:
                    continue
                val_profiles_resolved.append(
                    {
                        "name": profile_name,
                        "top_k": int(profile.get("top_k", self.top_k)),
                        "training_soft_routing": bool(profile.get("training_soft_routing", False)),
                        "routing_soft_topk": profile.get("routing_soft_topk", None),
                        "routing_temperature": profile.get("routing_temperature", None),
                    }
                )

        best_value_by_profile: Dict[str, float] = {
            profile["name"]: self._metric_init_best_value(best_metric_per_profile)
            for profile in val_profiles_resolved
        }

        scheduler = None

        def _profile_checkpoint_filename(name: str) -> str:
            safe_name = re.sub(r"[^0-9a-zA-Z_\-]+", "_", str(name).strip())
            safe_name = re.sub(r"_+", "_", safe_name).strip("_")
            if not safe_name:
                safe_name = "profile"
            return f"best_model_val_{safe_name}.pt"

        def _save_checkpoint(
            *,
            path: Path,
            epoch: int,
            best_value_to_save: Optional[float],
            metric_name: str,
            profile_name: Optional[str] = None,
            profile_config: Optional[Dict[str, Any]] = None,
            save_with_ema_weights: bool = False,
        ) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            model_state_to_save = self._model_state_dict_clone()
            raw_state_for_ref = None
            if save_with_ema_weights and self.ema_save_best:
                raw_state_for_ref = model_state_to_save
                model_state_to_save = self._model_state_dict_with_ema()
            torch.save(
                {
                    "epoch": int(epoch),
                    "model_state_dict": model_state_to_save,
                    "model_state_dict_raw": raw_state_for_ref,
                    "ema_state_dict": self._ema_state_dict,
                    "ema_enabled": bool(self.ema_enabled),
                    "ema_eval": bool(self.ema_eval),
                    "ema_save_best": bool(self.ema_save_best),
                    "saved_with_ema_weights": bool(save_with_ema_weights and self.ema_save_best),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "best_metric": best_value_to_save,
                    "best_metric_name": metric_name,
                    "best_metric_by_profile": dict(best_value_by_profile),
                    "val_profile_name": profile_name,
                    "val_profile_config": profile_config,
                    "history": history,
                    "rng_state": self._capture_rng_state(),
                    "loss_effective_contribution_state": self._loss_effective.state_dict(),
                    "meta": checkpoint_meta,
                },
                path,
            )

        if val_profiles_resolved:
            history["val_profiles"] = {
                profile["name"]: {k: [] for k in history["val"]}
                for profile in val_profiles_resolved
            }

        # Log model metadata at start
        if self.rerun_logger:
            num_params = sum(p.numel() for p in self.model.parameters())
            self.rerun_logger.log_model_metadata(
                num_gaussians=getattr(self.model, 'K', 0),
                rank=getattr(self.model, 'rank', 0),
                num_params=num_params
            )

        if self.lr_scheduler == "cosine":
            scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=self.lr_min)
        elif self.lr_scheduler == "plateau":
            scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=20, min_lr=self.lr_min)

        start_epoch = 0
        if isinstance(resume_state, dict):
            start_epoch = int(resume_state.get("epoch", 0))
            start_epoch = max(0, min(int(num_epochs), start_epoch))

            optimizer_state = resume_state.get("optimizer_state_dict")
            if isinstance(optimizer_state, dict):
                self.optimizer.load_state_dict(optimizer_state)

            if scheduler is not None:
                scheduler_state = resume_state.get("scheduler_state_dict")
                if isinstance(scheduler_state, dict):
                    scheduler.load_state_dict(scheduler_state)
                elif start_epoch > 0 and isinstance(scheduler, CosineAnnealingLR):
                    # Backward compatibility: older checkpoints may not store scheduler state.
                    for _ in range(start_epoch):
                        scheduler.step()

            if self.ema_enabled and isinstance(resume_state.get("ema_state_dict"), dict):
                restored_ema: Dict[str, torch.Tensor] = {}
                for key, value in resume_state["ema_state_dict"].items():
                    if torch.is_tensor(value):
                        restored_ema[key] = value.detach().clone()
                self._ema_state_dict = restored_ema

            loss_eff_state = resume_state.get("loss_effective_contribution_state")
            if isinstance(loss_eff_state, dict):
                self._loss_effective.load_state_dict(loss_eff_state)

            history_state = resume_state.get("history")
            if isinstance(history_state, dict):
                if isinstance(history_state.get("train"), dict):
                    for key in history["train"]:
                        values = history_state["train"].get(key)
                        if isinstance(values, list):
                            history["train"][key] = list(values)
                if isinstance(history_state.get("val"), dict):
                    for key in history["val"]:
                        values = history_state["val"].get(key)
                        if isinstance(values, list):
                            history["val"][key] = list(values)
                if isinstance(history_state.get("val_profiles"), dict):
                    history["val_profiles"] = history_state["val_profiles"]

            if str(resume_state.get("best_metric_name", "")) == str(best_metric):
                resumed_best = resume_state.get("best_metric")
                if resumed_best is not None:
                    best_value = float(resumed_best)

            resumed_profile_best = resume_state.get("best_metric_by_profile")
            if isinstance(resumed_profile_best, dict):
                for profile_name in list(best_value_by_profile.keys()):
                    if profile_name in resumed_profile_best:
                        try:
                            best_value_by_profile[profile_name] = float(resumed_profile_best[profile_name])
                        except Exception:
                            pass

            rng_state = resume_state.get("rng_state")
            if isinstance(rng_state, dict):
                self._restore_rng_state(rng_state)

        self._init_spatial_neighbors(train_loader)
        self._init_profiler(output_dir)
        if self.show_progress and self._cuda_graph_requested:
            print(
                "[cuda-graph] "
                f"requested={self._cuda_graph_requested} enabled={self._cuda_graph_enabled} "
                f"mode={self.cuda_graph_mode} warmup_steps={self.cuda_graph_warmup_steps} "
                f"fallback_eager={self.cuda_graph_fallback_eager}"
            )

        for epoch in range(start_epoch + 1, num_epochs + 1):
            self._current_epoch_for_monitor = int(epoch)
            self._current_routing_temp = self._routing_temperature_for_epoch(epoch)
            (
                self._current_image_loss_weight,
                self._current_gbuffer_image_loss_weight,
            ) = self._aux_image_weights_for_epoch(epoch)
            self._current_lambda_routing_balance = self._routing_balance_weight_for_epoch(epoch)
            self.image_loss_weight = float(self._current_image_loss_weight)
            self._set_image_sampling_context_for_epoch(epoch)
            if self.show_progress:
                self._progress_prefix = f"epoch {epoch}/{num_epochs}"
            train_metrics = self.train_epoch(train_loader)
            history["train"]["total"].append(train_metrics["total"])
            history["train"]["recon"].append(train_metrics["recon"])
            history["train"]["image"].append(train_metrics["image"])
            history["train"]["gbuffer"].append(train_metrics.get("gbuffer", 0.0))
            history["train"]["coeff_l1"].append(train_metrics["coeff_l1"])
            history["train"]["linearity"].append(train_metrics["linearity"])
            history["train"]["spatial"].append(train_metrics["spatial"])

            # Log training metrics to Rerun
            if self.rerun_logger:
                self.rerun_logger.log_scalars(epoch, train_metrics, "train")

            val_metrics = None
            val_metrics_raw = None
            val_metrics_ema = None
            val_profiles_metrics: Dict[str, Dict[str, float]] = {}
            if val_loader is not None:
                # Keep validation/eval sampling deterministic per epoch, even when
                # training uses per-step image sampling.
                self._set_image_sampling_context_for_epoch(epoch)
                run_raw_eval = True
                if self.ema_eval:
                    run_raw_eval = (
                        epoch == 1
                        or epoch == num_epochs
                        or (self.raw_eval_every_epochs > 0 and epoch % self.raw_eval_every_epochs == 0)
                    )
                if run_raw_eval:
                    val_metrics_raw = self.validate_epoch(val_loader)
                if self.ema_eval:
                    with self._use_ema_weights():
                        val_metrics_ema = self.validate_epoch(val_loader)
                    val_metrics = val_metrics_ema
                else:
                    val_metrics = val_metrics_raw if val_metrics_raw is not None else self.validate_epoch(val_loader)

                for key in history["val"]:
                    history["val"][key].append(val_metrics[key])

                # Log validation metrics to Rerun
                if self.rerun_logger:
                    if self.ema_eval and val_metrics_raw is not None:
                        self.rerun_logger.log_scalars(epoch, val_metrics_raw, "val_raw")
                    self.rerun_logger.log_scalars(epoch, val_metrics, "val")
                    if self.ema_eval and val_metrics_ema is not None:
                        self.rerun_logger.log_scalars(epoch, val_metrics_ema, "val_ema")

                if val_profiles_resolved:
                    def _run_profile_eval() -> None:
                        grouped_share_candidates: Dict[tuple[int, Optional[int]], list[Dict[str, Any]]] = {}
                        fallback_profiles: list[Dict[str, Any]] = []
                        for profile in val_profiles_resolved:
                            profile_name = str(profile["name"])
                            profile_kwargs: Dict[str, Any] = {
                                "top_k": int(profile.get("top_k", self.top_k)),
                                "training_soft_routing": bool(profile.get("training_soft_routing", False)),
                                "routing_soft_topk": profile.get("routing_soft_topk", None),
                                "routing_temperature": profile.get("routing_temperature", None),
                            }
                            if (
                                profile_kwargs["top_k"] == int(self.top_k)
                                and bool(profile_kwargs["training_soft_routing"]) is False
                                and profile_kwargs["routing_soft_topk"] is None
                                and profile_kwargs["routing_temperature"] is None
                            ):
                                val_profiles_metrics[profile_name] = dict(val_metrics)
                                continue
                            if (
                                self.enable_soft_profile_sharing
                                and bool(profile_kwargs["training_soft_routing"])
                                and profile_kwargs["routing_temperature"] is not None
                            ):
                                share_key = self._soft_profile_share_key(profile_kwargs)
                                if share_key not in self._disabled_soft_profile_share_keys:
                                    grouped_share_candidates.setdefault(share_key, []).append(
                                        {
                                            "name": profile_name,
                                            "top_k": int(profile_kwargs["top_k"]),
                                            "routing_soft_topk": profile_kwargs["routing_soft_topk"],
                                            "routing_temperature": float(profile_kwargs["routing_temperature"]),
                                        }
                                    )
                                    continue
                            fallback_profiles.append({"name": profile_name, "kwargs": profile_kwargs})

                        for share_key, group_profiles in grouped_share_candidates.items():
                            if len(group_profiles) < 2:
                                fallback_profiles.extend(
                                    [
                                        {
                                            "name": p["name"],
                                            "kwargs": {
                                                "top_k": int(p["top_k"]),
                                                "training_soft_routing": True,
                                                "routing_soft_topk": p["routing_soft_topk"],
                                                "routing_temperature": p["routing_temperature"],
                                            },
                                        }
                                        for p in group_profiles
                                    ]
                                )
                                continue
                            check_equivalence = share_key not in self._verified_soft_profile_share_keys
                            try:
                                grouped_metrics = self.validate_epoch_soft_profile_group(
                                    val_loader,
                                    profiles=group_profiles,
                                    check_equivalence=check_equivalence,
                                )
                                self._verified_soft_profile_share_keys.add(share_key)
                                for p in group_profiles:
                                    name = str(p["name"])
                                    if name in grouped_metrics:
                                        val_profiles_metrics[name] = grouped_metrics[name]
                            except Exception:
                                self._disabled_soft_profile_share_keys.add(share_key)
                                fallback_profiles.extend(
                                    [
                                        {
                                            "name": p["name"],
                                            "kwargs": {
                                                "top_k": int(p["top_k"]),
                                                "training_soft_routing": True,
                                                "routing_soft_topk": p["routing_soft_topk"],
                                                "routing_temperature": p["routing_temperature"],
                                            },
                                        }
                                        for p in group_profiles
                                    ]
                                )

                        for item in fallback_profiles:
                            profile_name = str(item["name"])
                            profile_kwargs = dict(item["kwargs"])
                            if profile_name in val_profiles_metrics:
                                continue
                            if (
                                profile_kwargs["top_k"] == int(self.top_k)
                                and bool(profile_kwargs["training_soft_routing"]) is False
                                and profile_kwargs["routing_soft_topk"] is None
                                and profile_kwargs["routing_temperature"] is None
                            ):
                                profile_metrics = dict(val_metrics)
                            else:
                                profile_metrics = self.validate_epoch(val_loader, **profile_kwargs)
                            val_profiles_metrics[profile_name] = profile_metrics

                        for profile in val_profiles_resolved:
                            profile_name = str(profile["name"])
                            profile_metrics = val_profiles_metrics.get(profile_name, {})
                            if self.rerun_logger:
                                self.rerun_logger.log_scalars(epoch, profile_metrics, f"val/{profile_name}")

                            if "val_profiles" in history and profile_name in history["val_profiles"]:
                                for key in history["val"]:
                                    history["val_profiles"][profile_name][key].append(profile_metrics.get(key, 0.0))

                    if self.ema_eval:
                        with self._use_ema_weights():
                            _run_profile_eval()
                    else:
                        _run_profile_eval()

                # Log visualizations every N epochs
                if self.rerun_logger and epoch % self.rerun_logger.log_frequency == 0:
                    self._log_visualizations(epoch, val_loader)

                if save_best and output_dir is not None:
                    current = val_metrics.get(best_metric, None)
                    if self._metric_is_better(best_metric, current, best_value):
                        best_value = float(current)
                        _save_checkpoint(
                            path=output_dir / "best_model.pt",
                            epoch=epoch,
                            best_value_to_save=float(best_value),
                            metric_name=str(best_metric),
                            save_with_ema_weights=True,
                        )

                if save_best_profiles and output_dir is not None and val_profiles_metrics:
                    for profile in val_profiles_resolved:
                        profile_name = str(profile["name"])
                        profile_metrics = val_profiles_metrics.get(profile_name, {})
                        current_profile_value = profile_metrics.get(best_metric_per_profile, None)
                        if current_profile_value is None:
                            continue
                        previous = best_value_by_profile.get(
                            profile_name,
                            self._metric_init_best_value(best_metric_per_profile),
                        )
                        if self._metric_is_better(best_metric_per_profile, current_profile_value, previous):
                            best_value_by_profile[profile_name] = float(current_profile_value)
                            _save_checkpoint(
                                path=output_dir / _profile_checkpoint_filename(profile_name),
                                epoch=epoch,
                                best_value_to_save=float(current_profile_value),
                                metric_name=str(best_metric_per_profile),
                                profile_name=profile_name,
                                profile_config=profile,
                                save_with_ema_weights=True,
                            )
            in_warmup = self._apply_warmup(epoch)
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    if not in_warmup:
                        if val_metrics is None:
                            scheduler.step(train_metrics.get("total", 0.0))
                        else:
                            scheduler.step(self._metric_for_plateau_scheduler(best_metric, val_metrics))
                elif not in_warmup:
                    scheduler.step()

            if self.rerun_logger:
                self.rerun_logger.log_scalars(epoch, self._current_lr_metrics(), "lr")
                if self.ema_enabled:
                    self.rerun_logger.log_scalars(
                        epoch,
                        {
                            "enabled": 1.0,
                            "decay": float(self.ema_decay),
                            "eval_enabled": 1.0 if self.ema_eval else 0.0,
                            "save_best_enabled": 1.0 if self.ema_save_best else 0.0,
                        },
                        "ema",
                    )

            if epoch_end_callback is not None:
                epoch_end_callback(
                    {
                        "epoch": int(epoch),
                        "num_epochs": int(num_epochs),
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                        "val_metrics_raw": val_metrics_raw,
                        "val_metrics_ema": val_metrics_ema,
                        "best_value": float(best_value),
                        "best_metric": str(best_metric),
                        "val_profiles_metrics": val_profiles_metrics,
                        "best_value_by_profile": dict(best_value_by_profile),
                    }
                )

            if self.show_progress:
                self._progress_prefix = ""

            if log_every and (epoch == 1 or epoch % log_every == 0):
                msg = (
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Train total={train_metrics['total']:.6f}, "
                    f"recon={train_metrics['recon']:.6f}, "
                    f"lambda_img={float(self._current_image_loss_weight):.4g}, "
                    f"lambda_gbuf={float(self._current_gbuffer_image_loss_weight):.4g}, "
                    f"handover={float(self._current_proxy_gbuffer_handover_progress):.3f}, "
                    f"lambda_balance={float(self._current_lambda_routing_balance):.4g}, "
                    f"coeff_l1={train_metrics['coeff_l1']:.6f}, "
                    f"linearity={train_metrics['linearity']:.6f}, "
                    f"gbuffer={train_metrics.get('gbuffer', 0.0):.6f}, "
                    f"eff_img={train_metrics.get('loss_eff_contrib_image', 0.0):.4g}, "
                    f"eff_gbuf={train_metrics.get('loss_eff_contrib_gbuffer', 0.0):.4g}, "
                    f"grad_conflict={train_metrics.get('grad_conflict_rate', 0.0):.3f}, "
                    f"upd_ratio={train_metrics.get('param_update_ratio_shared', 0.0):.3e}, "
                    f"clip_rate={train_metrics.get('grad_clip_trigger_rate', 0.0):.3f}, "
                    f"logit_rms={train_metrics.get('routing_logit_rms', 0.0):.3f}, "
                    f"pulse_rec={train_metrics.get('pulse_recovery_steps_mean', 0.0):.2f}"
                )
                if val_loader is not None:
                    msg += (
                        f" | Val mae={history['val']['mae'][-1]:.6f}, "
                        f"rmse={history['val']['rmse'][-1]:.6f}, "
                        f"img_psnr={history['val']['img_psnr'][-1]:.3f}, "
                        f"superposition={history['val']['superposition'][-1]:.6f}"
                    )
                print(msg)

            if output_dir is not None and int(save_resume_every) > 0 and epoch % int(save_resume_every) == 0:
                _save_checkpoint(
                    path=output_dir / str(resume_checkpoint_name),
                    epoch=epoch,
                    best_value_to_save=float(best_value) if np.isfinite(float(best_value)) else None,
                    metric_name=str(best_metric),
                    save_with_ema_weights=False,
                )

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            _save_checkpoint(
                path=output_dir / "last_model.pt",
                epoch=num_epochs,
                best_value_to_save=float(best_value) if np.isfinite(float(best_value)) else None,
                metric_name=str(best_metric),
                save_with_ema_weights=False,
            )

        self._close_profiler()
        return history
