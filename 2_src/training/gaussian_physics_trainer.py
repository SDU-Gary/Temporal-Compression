"""Shared trainer for Gaussian-Physics model variants."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import os
import random
import re
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


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
        charbonnier_eps: float = 1e-3,
        temporal_weight: float = 0.0,
        temporal_loss_fn: Optional[Callable[[nn.Module], torch.Tensor]] = None,
        top_k: int = 3,
        grad_clip: Optional[float] = 1.0,
        linearity_weight: float = 0.0,
        linearity_aug_pairs: int = 0,
        spatial_weight: float = 0.0,
        spatial_k: int = 1,
        image_loss_weight: float = 0.0,
        image_loss_warmup_epochs: int = 0,
        image_loss_type: str = "mse",
        image_samples: int = 64,
        image_sample_seed: int = 42,
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
        ema_enabled: bool = False,
        ema_decay: float = 0.999,
        ema_eval: bool = False,
        ema_save_best: bool = False,
        rerun_logger=None,  # Optional RerunLogger instance
        show_progress: bool = True,
        amp_mode: str = "off",
        grad_norm_log_every_steps: int = 1,
        raw_eval_every_epochs: int = 1,
        max_train_batches: int = 0,
        max_val_batches: int = 0,
        enable_soft_profile_sharing: bool = False,
        soft_profile_equiv_check_batches: int = 2,
        soft_profile_mae_tolerance: float = 1e-6,
        soft_profile_img_psnr_tolerance: float = 5e-4,
    ):
        self.model = model
        self.device = device
        self.adapter = adapter
        self.recon_loss = recon_loss
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
        self.spatial_weight = spatial_weight
        self.spatial_k = spatial_k
        self.image_loss_weight_target = max(0.0, float(image_loss_weight))
        self.image_loss_warmup_epochs = max(0, int(image_loss_warmup_epochs))
        self._current_image_loss_weight = float(self.image_loss_weight_target)
        # Backward-compatible alias used by existing logs/checkpoints.
        self.image_loss_weight = float(self.image_loss_weight_target)
        self.image_loss_type = str(image_loss_type)
        self.image_samples = max(1, int(image_samples))
        self.image_sample_seed = int(image_sample_seed)
        self.image_loss_space = str(image_loss_space)
        self.proxy_image_loss_mix_mode = str(proxy_image_loss_mix_mode).strip().lower()
        self.proxy_image_log_mix_weight = float(np.clip(float(proxy_image_log_mix_weight), 0.0, 1.0))
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
        self._disabled_soft_profile_share_keys: set[tuple[int, Optional[int]]] = set()
        self._verified_soft_profile_share_keys: set[tuple[int, Optional[int]]] = set()
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
        self._grad_scaler = torch.cuda.amp.GradScaler(enabled=(self.amp_mode == "fp16" and self.device.type == "cuda"))

        if self.image_loss_type not in {"mse", "charbonnier"}:
            raise ValueError(f"Unsupported image_loss_type: {self.image_loss_type}")
        if self.image_loss_space not in {"linear", "srgb"}:
            raise ValueError(f"Unsupported image_loss_space: {self.image_loss_space}")
        if self.proxy_image_loss_mix_mode not in {"linear_log_mix", "linear_only", "log_only"}:
            raise ValueError(f"Unsupported proxy_image_loss_mix_mode: {self.proxy_image_loss_mix_mode}")
        if self.sh_weight_mode != "basis":
            raise ValueError(f"Unsupported sh_weight_mode: {self.sh_weight_mode}")

        if sh_loss_weights is None:
            sh_loss_weights = [3.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5]
        if len(sh_loss_weights) != 9:
            raise ValueError("sh_loss_weights must contain exactly 9 values")
        sh_w9 = torch.tensor(sh_loss_weights, dtype=torch.float32)
        self._sh_loss_weights_27 = torch.cat([sh_w9, sh_w9, sh_w9], dim=0)
        self._image_basis_cache: Dict[tuple[str, str], torch.Tensor] = {}

        self.optimizer = self._build_optimizer(lr=lr, weight_decay=weight_decay)
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

        return torch.optim.Adam(
            optimizer_groups,
            lr=default_lr,
            weight_decay=weight_decay,
        )

    def _autocast_context(self):
        if not self._use_amp or self._amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=True)

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
            try:
                return self.model.forward_with_routing(routing, params, light_mask=mask)
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

    def _build_image_basis(self, ref: torch.Tensor) -> torch.Tensor:
        key = (str(ref.device), str(ref.dtype))
        cached = self._image_basis_cache.get(key)
        if cached is not None:
            return cached

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.image_sample_seed)
        u = torch.rand((self.image_samples,), generator=gen, dtype=torch.float32)
        v = torch.rand((self.image_samples,), generator=gen, dtype=torch.float32)
        z = 2.0 * u - 1.0
        phi = 2.0 * np.pi * v
        r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)

        dirs = torch.stack([x, y, z], dim=-1).to(device=ref.device, dtype=ref.dtype)
        xb, yb, zb = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        basis = torch.stack(
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
        self._image_basis_cache[key] = basis
        return basis

    def _render_sh_to_samples(self, sh: torch.Tensor) -> torch.Tensor:
        basis = self._build_image_basis(sh)
        sh_rgb = torch.stack([sh[..., 0:9], sh[..., 9:18], sh[..., 18:27]], dim=-2)
        image = torch.einsum("...cn,sn->...sc", sh_rgb, basis)
        if self.image_loss_space == "srgb":
            image = self._srgb_encode(self._tone_map_reinhard(image))
        else:
            image = torch.clamp(image, min=0.0)
        return image

    def _image_loss_from_samples(self, pred_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        if self.image_loss_type == "charbonnier":
            return charbonnier_loss(pred_img, target_img, epsilon=self.charbonnier_eps)
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

        pred_img = self._render_sh_to_samples(pred_sh)
        target_img = self._render_sh_to_samples(target_sh)

        mode = self.proxy_image_loss_mix_mode
        linear_loss = zero
        log_loss = zero

        if mode in {"linear_log_mix", "linear_only"}:
            linear_loss = self._image_loss_from_samples(pred_img, target_img)
        if mode in {"linear_log_mix", "log_only"}:
            pred_log = torch.log1p(torch.clamp(pred_img, min=0.0))
            target_log = torch.log1p(torch.clamp(target_img, min=0.0))
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
            "image_linear": torch.tensor(0.0, device=self.device),
            "image_log": torch.tensor(0.0, device=self.device),
            "coeff_l1": torch.tensor(0.0, device=self.device),
            "linearity": torch.tensor(0.0, device=self.device),
            "spatial": torch.tensor(0.0, device=self.device),
            "routing_balance": 0.0,
            "routing_entropy": 0.0,
            "routing_nonzero_ratio": 0.0,
        }
        pre_clip_norms_list = []
        post_clip_norms_list = []
        num_batches = 0

        try:
            loader = train_loader
            if self.show_progress and _tqdm is not None:
                desc = self._progress_prefix + " [train]" if self._progress_prefix else "train"
                loader = _tqdm(train_loader, desc=desc, leave=False, unit="batch")

            for step_idx, batch in enumerate(loader, start=1):
                if self.max_train_batches > 0 and step_idx > self.max_train_batches:
                    break
                unpacked = self.adapter.unpack(batch, self.device)
                if len(unpacked) == 4:
                    positions, params, targets, mask = unpacked
                else:
                    positions, params, targets = unpacked
                    mask = None

                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast_context():
                    routing = self._compute_routing(positions, for_training=True)
                    # TODO(perf): 一个训练 step 里除主前向外，还可能触发 image/temporal/linearity/spatial 等额外前向。
                    # 当正则全部开启时，单 step 的有效前向次数显著增加，是当前主要计算热点。
                    preds = self._forward(positions, params, mask, routing=routing)

                    loss_recon = self._recon_loss(preds, targets)
                    preds_eval, targets_eval = preds, targets
                    if self.adapter.target_inverse is not None:
                        preds_eval, targets_eval = self.adapter.inverse_targets(preds, targets)

                    loss_image, image_loss_components = self._compute_image_loss_components(preds_eval, targets_eval)
                    loss_temporal = self._compute_temporal_loss()
                    loss_linearity = self._compute_linearity_loss(positions, params, mask, preds, routing_full=routing)
                    loss_linearity_aug = self._compute_linearity_aug_loss(positions, params, mask, routing_full=routing)
                    probe_idx = batch.get("probe_idx") if isinstance(batch, dict) else None
                    loss_spatial = self._compute_spatial_loss(preds, params, mask, probe_idx)
                    loss_routing_balance = torch.tensor(0.0, device=self.device)
                    routing_stats = {
                        "routing_balance_loss": 0.0,
                        "routing_entropy": 0.0,
                        "routing_nonzero_ratio": 0.0,
                    }
                    if self._current_lambda_routing_balance > 0.0:
                        loss_routing_balance, routing_stats = self._compute_routing_balance_loss(routing)

                    loss_total = (
                        loss_recon
                        + self._current_image_loss_weight * loss_image
                        + self.temporal_weight * loss_temporal
                        + self.linearity_weight * (loss_linearity + loss_linearity_aug)
                        + self.spatial_weight * loss_spatial
                        + self._current_lambda_routing_balance * loss_routing_balance
                    )

                if self._grad_scaler.is_enabled():
                    self._grad_scaler.scale(loss_total).backward()
                    self._grad_scaler.unscale_(self.optimizer)
                else:
                    loss_total.backward()

                should_log_grad = self.grad_norm_log_every_steps > 0 and (step_idx % self.grad_norm_log_every_steps == 0)
                if should_log_grad:
                    pre_clip_norms = self._compute_grad_norms(prefix="grad_pre_clip")
                    pre_clip_norms_list.append(pre_clip_norms)

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                if should_log_grad:
                    post_clip_norms = self._compute_grad_norms(prefix="grad_post_clip")
                    post_clip_norms_list.append(post_clip_norms)

                if self._grad_scaler.is_enabled():
                    self._grad_scaler.step(self.optimizer)
                    self._grad_scaler.update()
                else:
                    self.optimizer.step()
                self._update_ema()

                totals["total"] += loss_total.detach()
                totals["recon"] += loss_recon.detach()
                totals["image"] += loss_image.detach()
                totals["image_linear"] += image_loss_components["linear"].detach()
                totals["image_log"] += image_loss_components["log"].detach()
                totals["coeff_l1"] += loss_temporal.detach()
                totals["linearity"] += (loss_linearity.detach() + loss_linearity_aug.detach())
                totals["spatial"] += loss_spatial.detach()
                totals["routing_balance"] += float(routing_stats["routing_balance_loss"])
                totals["routing_entropy"] += float(routing_stats["routing_entropy"])
                totals["routing_nonzero_ratio"] += float(routing_stats["routing_nonzero_ratio"])
                num_batches += 1
        finally:
            self._train_routing_mode = False

        # Average loss metrics
        metrics = {}
        denom = max(1, num_batches)
        for k, v in totals.items():
            if torch.is_tensor(v):
                metrics[k] = float((v / denom).detach().item())
            else:
                metrics[k] = float(v / denom)
        metrics["routing_temp"] = float(self._current_routing_temp)
        metrics["image_loss_weight"] = float(self._current_image_loss_weight)
        metrics["routing_balance_lambda"] = float(self._current_lambda_routing_balance)

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
            unpacked = self.adapter.unpack(batch, self.device)
            if len(unpacked) == 4:
                positions, params, targets, mask = unpacked
            else:
                positions, params, targets = unpacked
                mask = None

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
            "train": {"total": [], "recon": [], "image": [], "coeff_l1": [], "linearity": [], "spatial": []},
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

        for epoch in range(start_epoch + 1, num_epochs + 1):
            self._current_routing_temp = self._routing_temperature_for_epoch(epoch)
            self._current_image_loss_weight = self._image_loss_weight_for_epoch(epoch)
            self._current_lambda_routing_balance = self._routing_balance_weight_for_epoch(epoch)
            self.image_loss_weight = float(self._current_image_loss_weight)
            if self.show_progress:
                self._progress_prefix = f"epoch {epoch}/{num_epochs}"
            train_metrics = self.train_epoch(train_loader)
            history["train"]["total"].append(train_metrics["total"])
            history["train"]["recon"].append(train_metrics["recon"])
            history["train"]["image"].append(train_metrics["image"])
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
                    f"lambda_balance={float(self._current_lambda_routing_balance):.4g}, "
                    f"coeff_l1={train_metrics['coeff_l1']:.6f}, "
                    f"linearity={train_metrics['linearity']:.6f}"
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

        return history
