"""Shared trainer for Gaussian-Physics model variants."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
import re
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, Optional

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
        image_loss_type: str = "mse",
        image_samples: int = 64,
        image_sample_seed: int = 42,
        image_loss_space: str = "linear",
        enable_weighted_sh_loss: bool = False,
        sh_loss_weights: Optional[list[float]] = None,
        sh_weight_mode: str = "basis",
        compute_img_metrics_in_val: bool = False,
        compute_superposition_in_val: bool = False,
        param_group_lrs: Optional[Dict[str, float]] = None,
        lambda_routing_balance: float = 0.0,
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
        self.image_loss_weight = max(0.0, float(image_loss_weight))
        self.image_loss_type = str(image_loss_type)
        self.image_samples = max(1, int(image_samples))
        self.image_sample_seed = int(image_sample_seed)
        self.image_loss_space = str(image_loss_space)
        self.enable_weighted_sh_loss = bool(enable_weighted_sh_loss)
        self.sh_weight_mode = str(sh_weight_mode)
        self.compute_img_metrics_in_val = bool(compute_img_metrics_in_val)
        self.compute_superposition_in_val = bool(compute_superposition_in_val)
        self.param_group_lrs = dict(param_group_lrs or {})
        self.lambda_routing_balance = max(0.0, float(lambda_routing_balance))
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
        self._probe_positions = None
        self._probe_neighbors = None
        self._optimizer_group_names: list[str] = []
        self._optimizer_group_base_lrs: list[float] = []
        self._ema_state_dict: Optional[Dict[str, torch.Tensor]] = None

        if self.image_loss_type not in {"mse", "charbonnier"}:
            raise ValueError(f"Unsupported image_loss_type: {self.image_loss_type}")
        if self.image_loss_space not in {"linear", "srgb"}:
            raise ValueError(f"Unsupported image_loss_space: {self.image_loss_space}")
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

        weights, _ = routing  # [B, K']
        if weights.numel() == 0:
            return torch.tensor(0.0, device=self.device), {
                "routing_balance_loss": 0.0,
                "routing_entropy": 0.0,
                "routing_nonzero_ratio": 0.0,
            }

        usage = torch.mean(weights, dim=0)
        usage = usage / (torch.sum(usage) + 1e-8)
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
            image = torch.clamp(image, 0.0, 1.0)
        return image

    def _compute_image_loss(self, pred_sh: torch.Tensor, target_sh: torch.Tensor) -> torch.Tensor:
        if self.image_loss_weight <= 0:
            return torch.tensor(0.0, device=pred_sh.device)
        pred_img = self._render_sh_to_samples(pred_sh)
        target_img = self._render_sh_to_samples(target_sh)
        if self.image_loss_type == "charbonnier":
            return charbonnier_loss(pred_img, target_img, epsilon=self.charbonnier_eps)
        return torch.mean((pred_img - target_img) ** 2)

    def _compute_image_metrics(self, pred_sh: torch.Tensor, target_sh: torch.Tensor) -> Dict[str, float]:
        from utils.unified_metrics import compute_pair_metrics

        pred_img = self._render_sh_to_samples(pred_sh)
        target_img = self._render_sh_to_samples(target_sh)
        metrics = compute_pair_metrics(
            target_img,
            pred_img,
            max_i=1.0,
            clip_unit=True,
            ssim_mode="image",
        )
        return {
            "img_mae": float(metrics["mae"]),
            "img_rmse": float(metrics["rmse"]),
            "img_psnr": float(metrics["psnr"]),
        }

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
            "total": 0.0,
            "recon": 0.0,
            "image": 0.0,
            "coeff_l1": 0.0,
            "linearity": 0.0,
            "spatial": 0.0,
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

            for batch in loader:
                unpacked = self.adapter.unpack(batch, self.device)
                if len(unpacked) == 4:
                    positions, params, targets, mask = unpacked
                else:
                    positions, params, targets = unpacked
                    mask = None

                self.optimizer.zero_grad()
                routing = self._compute_routing(positions, for_training=True)
            # TODO(perf): 一个训练 step 里除主前向外，还可能触发 image/temporal/linearity/spatial 等额外前向。
            # 当正则全部开启时，单 step 的有效前向次数显著增加，是当前主要计算热点。
                preds = self._forward(positions, params, mask, routing=routing)

                loss_recon = self._recon_loss(preds, targets)
                preds_eval, targets_eval = preds, targets
                if self.adapter.target_inverse is not None:
                    preds_eval, targets_eval = self.adapter.inverse_targets(preds, targets)

                loss_image = self._compute_image_loss(preds_eval, targets_eval)
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
                if self.lambda_routing_balance > 0.0:
                    loss_routing_balance, routing_stats = self._compute_routing_balance_loss(routing)

                loss_total = (
                    loss_recon
                    + self.image_loss_weight * loss_image
                    + self.temporal_weight * loss_temporal
                    + self.linearity_weight * (loss_linearity + loss_linearity_aug)
                    + self.spatial_weight * loss_spatial
                    + self.lambda_routing_balance * loss_routing_balance
                )

                loss_total.backward()

            # Record pre-clip gradient norms
                pre_clip_norms = self._compute_grad_norms(prefix="grad_pre_clip")
                pre_clip_norms_list.append(pre_clip_norms)

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            # Record post-clip gradient norms
                post_clip_norms = self._compute_grad_norms(prefix="grad_post_clip")
                post_clip_norms_list.append(post_clip_norms)

                self.optimizer.step()
                self._update_ema()

                totals["total"] += loss_total.item()
                totals["recon"] += loss_recon.item()
                totals["image"] += loss_image.item()
                totals["coeff_l1"] += loss_temporal.item()
                totals["linearity"] += (loss_linearity.item() + loss_linearity_aug.item())
                totals["spatial"] += loss_spatial.item()
                totals["routing_balance"] += float(routing_stats["routing_balance_loss"])
                totals["routing_entropy"] += float(routing_stats["routing_entropy"])
                totals["routing_nonzero_ratio"] += float(routing_stats["routing_nonzero_ratio"])
                num_batches += 1
        finally:
            self._train_routing_mode = False

        # Average loss metrics
        metrics = {k: v / max(1, num_batches) for k, v in totals.items()}
        metrics["routing_temp"] = float(self._current_routing_temp)

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

        sums = {"mae": 0.0, "rmse": 0.0, "charbonnier": 0.0, "superposition": 0.0, "img_mae": 0.0, "img_rmse": 0.0, "img_psnr": 0.0}
        num_batches = 0

        loader = val_loader
        if self.show_progress and _tqdm is not None:
            desc = self._progress_prefix + " [val]" if self._progress_prefix else "val"
            loader = _tqdm(val_loader, desc=desc, leave=False, unit="batch")

        for batch in loader:
            unpacked = self.adapter.unpack(batch, self.device)
            if len(unpacked) == 4:
                positions, params, targets, mask = unpacked
            else:
                positions, params, targets = unpacked
                mask = None

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

            mae = torch.mean(torch.abs(preds_eval - targets_eval)).item()
            rmse = torch.sqrt(torch.mean((preds_eval - targets_eval) ** 2)).item()

            if self.recon_loss == "charbonnier":
                charbonnier = charbonnier_loss(preds_eval, targets_eval, epsilon=self.charbonnier_eps).item()
            else:
                charbonnier = 0.0

            sums["mae"] += mae
            sums["rmse"] += rmse
            sums["charbonnier"] += charbonnier

            if self.compute_img_metrics_in_val:
                img_metrics = self._compute_image_metrics(preds_eval, targets_eval)
                sums["img_mae"] += img_metrics["img_mae"]
                sums["img_rmse"] += img_metrics["img_rmse"]
                sums["img_psnr"] += img_metrics["img_psnr"]

            if self.compute_superposition_in_val and self.linearity_weight > 0:
                sup_err = self._compute_linearity_loss(positions, params, mask, preds).item()
                sums["superposition"] += sup_err
            num_batches += 1

        return {k: v / max(1, num_batches) for k, v in sums.items()}

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
    ) -> Dict[str, Dict[str, list]]:
        history = {
            "train": {"total": [], "recon": [], "image": [], "coeff_l1": [], "linearity": [], "spatial": []},
            "val": {"mae": [], "rmse": [], "charbonnier": [], "superposition": [], "img_mae": [], "img_rmse": [], "img_psnr": []},
        }

        best_value = float("inf")
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
            profile["name"]: float("inf") for profile in val_profiles_resolved
        }

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
        ) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            model_state_to_save = self._model_state_dict_clone()
            raw_state_for_ref = None
            if self.ema_save_best:
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
                    "saved_with_ema_weights": bool(self.ema_save_best),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_metric": best_value_to_save,
                    "best_metric_name": metric_name,
                    "val_profile_name": profile_name,
                    "val_profile_config": profile_config,
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

        scheduler = None
        if self.lr_scheduler == "cosine":
            scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=self.lr_min)
        elif self.lr_scheduler == "plateau":
            scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=20, min_lr=self.lr_min)

        self._init_spatial_neighbors(train_loader)

        for epoch in range(1, num_epochs + 1):
            self._current_routing_temp = self._routing_temperature_for_epoch(epoch)
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
                val_metrics_raw = self.validate_epoch(val_loader)
                if self.ema_eval:
                    with self._use_ema_weights():
                        val_metrics_ema = self.validate_epoch(val_loader)
                    val_metrics = val_metrics_ema
                else:
                    val_metrics = val_metrics_raw

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
                                profile_metrics = dict(val_metrics)
                            else:
                                profile_metrics = self.validate_epoch(val_loader, **profile_kwargs)

                            val_profiles_metrics[profile_name] = profile_metrics
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
                    if current is not None and current < best_value:
                        best_value = current
                        _save_checkpoint(
                            path=output_dir / "best_model.pt",
                            epoch=epoch,
                            best_value_to_save=float(best_value),
                            metric_name=str(best_metric),
                        )

                if save_best_profiles and output_dir is not None and val_profiles_metrics:
                    for profile in val_profiles_resolved:
                        profile_name = str(profile["name"])
                        profile_metrics = val_profiles_metrics.get(profile_name, {})
                        current_profile_value = profile_metrics.get(best_metric_per_profile, None)
                        if current_profile_value is None:
                            continue
                        previous = best_value_by_profile.get(profile_name, float("inf"))
                        if current_profile_value < previous:
                            best_value_by_profile[profile_name] = float(current_profile_value)
                            _save_checkpoint(
                                path=output_dir / _profile_checkpoint_filename(profile_name),
                                epoch=epoch,
                                best_value_to_save=float(current_profile_value),
                                metric_name=str(best_metric_per_profile),
                                profile_name=profile_name,
                                profile_config=profile,
                            )
            in_warmup = self._apply_warmup(epoch)
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    if not in_warmup:
                        if val_metrics is None:
                            scheduler.step(train_metrics.get("total", 0.0))
                        else:
                            scheduler.step(val_metrics.get(best_metric, val_metrics["mae"]))
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

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": num_epochs,
                    "model_state_dict": self._model_state_dict_clone(),
                    "ema_state_dict": self._ema_state_dict,
                    "ema_enabled": bool(self.ema_enabled),
                    "ema_eval": bool(self.ema_eval),
                    "ema_save_best": bool(self.ema_save_best),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "meta": checkpoint_meta,
                },
                output_dir / "last_model.pt",
            )

        return history
