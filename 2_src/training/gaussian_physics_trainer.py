"""Shared trainer for Gaussian-Physics model variants."""

from __future__ import annotations

from dataclasses import dataclass
import os
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

    def unpack(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = batch[self.positions_key].to(device)
        params = batch[self.params_key].to(device)
        targets = batch[self.target_key].to(device)

        if self.params_transform is not None:
            params = self.params_transform(params)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        if self.mask_key is not None and self.mask_key in batch:
            mask = batch[self.mask_key].to(device)
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
        self.rerun_logger = rerun_logger
        self.show_progress = show_progress
        self._progress_prefix = ""
        self._probe_positions = None
        self._probe_neighbors = None

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

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        if recon_loss == "mse":
            self.criterion = nn.MSELoss()
        elif recon_loss == "l1":
            self.criterion = nn.L1Loss()
        elif recon_loss == "charbonnier":
            self.criterion = None
        else:
            raise ValueError(f"Unknown recon_loss: {recon_loss}")

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
        params_flat = params[:, None, ...].expand(B, K, *params.shape[1:]).reshape(B * K, *params.shape[1:])
        if mask is not None:
            mask_flat = mask[:, None, ...].expand(B, K, *mask.shape[1:]).reshape(B * K, *mask.shape[1:])
        else:
            mask_flat = None

        preds_neighbor = self._forward(neighbor_pos_flat, params_flat, mask_flat)
        preds_neighbor = preds_neighbor.reshape(B, K, -1)
        preds_self = preds[:, None, :].expand(B, K, -1)
        return torch.mean(torch.abs(preds_self - preds_neighbor))

    def _forward(self, positions: torch.Tensor, params: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward wrapper that handles optional light masks."""
        if mask is None:
            return self.model(positions, params, top_k=self.top_k)
        try:
            return self.model(positions, params, top_k=self.top_k, light_mask=mask)
        except TypeError:
            # Fallback for models without mask support
            return self.model(positions, params, top_k=self.top_k)

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
        pred_img = self._render_sh_to_samples(pred_sh)
        target_img = self._render_sh_to_samples(target_sh)
        diff = pred_img - target_img
        mae = torch.mean(torch.abs(diff)).item()
        mse = torch.mean(diff * diff)
        rmse = torch.sqrt(mse).item()
        mse_val = mse.item()
        if mse_val <= 1e-12:
            psnr = float("inf")
        else:
            psnr = float(10.0 * np.log10(1.0 / mse_val))
        return {"img_mae": mae, "img_rmse": rmse, "img_psnr": psnr}

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

        preds_a = self._forward(positions_v, params_v, mask_a_v)
        preds_b = self._forward(positions_v, params_v, mask_b_v)

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
        for i, j in pairs:
            pos = positions[i : i + 1]
            params_a = params[i : i + 1]
            params_b = params[j : j + 1]
            mask_a = mask[i : i + 1]
            mask_b = mask[j : j + 1]

            # Extract single light descriptors
            idx_a = torch.argmax(mask_a, dim=-1)
            idx_b = torch.argmax(mask_b, dim=-1)
            light_a = params_a[0, idx_a]
            light_b = params_b[0, idx_b]

            # Build combined params (slots 0 and 1)
            B, N, D = params_a.shape
            combined = torch.zeros((B, N, D), device=params.device, dtype=params.dtype)
            combined_mask = torch.zeros((B, N), device=params.device, dtype=mask.dtype)
            combined[0, 0] = light_a
            combined[0, 1] = light_b
            combined_mask[0, 0] = 1.0
            combined_mask[0, 1] = 1.0

            pred_a = self._forward(pos, params_a, mask_a)
            pred_b = self._forward(pos, params_b, mask_b)
            pred_ab = self._forward(pos, combined, combined_mask)

            if self.adapter.target_inverse is not None:
                pred_a = self.adapter.target_inverse(pred_a)
                pred_b = self.adapter.target_inverse(pred_b)
                pred_ab = self.adapter.target_inverse(pred_ab)

            total += torch.mean(torch.abs(pred_ab - (pred_a + pred_b)))

        return total / num_pairs

    def _compute_grad_norms(self, prefix: str = "") -> Dict[str, float]:
        """Compute gradient norms for all parameter groups.

        Args:
            prefix: Prefix for metric keys (e.g., "grad_pre_clip", "grad_post_clip")

        Returns:
            Dictionary with gradient norms for total and each parameter group
        """
        grad_norms = {}

        # Compute total gradient norm
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms[f'{prefix}/total'] = total_norm

        # Compute per-parameter-group norms
        param_groups = {
            'mu': getattr(self.model, 'mu', None),
            'log_scale': getattr(self.model, 'log_scale', None),
            'U': getattr(self.model, 'U', None),
            'time_coeffs': getattr(self.model, 'time_coeffs', None),
        }

        for name, param in param_groups.items():
            if param is not None and param.grad is not None:
                norm = param.grad.detach().data.norm(2).item()
                grad_norms[f'{prefix}/{name}'] = norm
            elif param is not None:
                grad_norms[f'{prefix}/{name}'] = 0.0

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

        totals = {"total": 0.0, "recon": 0.0, "image": 0.0, "coeff_l1": 0.0, "linearity": 0.0, "spatial": 0.0}
        pre_clip_norms_list = []
        post_clip_norms_list = []
        num_batches = 0

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
            preds = self._forward(positions, params, mask)

            loss_recon = self._recon_loss(preds, targets)
            preds_eval, targets_eval = self.adapter.inverse_targets(preds, targets)
            loss_image = self._compute_image_loss(preds_eval, targets_eval)
            loss_temporal = self._compute_temporal_loss()
            loss_linearity = self._compute_linearity_loss(positions, params, mask, preds)
            loss_linearity_aug = self._compute_linearity_aug_loss(positions, params, mask)
            probe_idx = batch.get("probe_idx") if isinstance(batch, dict) else None
            loss_spatial = self._compute_spatial_loss(preds, params, mask, probe_idx)

            loss_total = (
                loss_recon
                + self.image_loss_weight * loss_image
                + self.temporal_weight * loss_temporal
                + self.linearity_weight * (loss_linearity + loss_linearity_aug)
                + self.spatial_weight * loss_spatial
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

            totals["total"] += loss_total.item()
            totals["recon"] += loss_recon.item()
            totals["image"] += loss_image.item()
            totals["coeff_l1"] += loss_temporal.item()
            totals["linearity"] += (loss_linearity.item() + loss_linearity_aug.item())
            totals["spatial"] += loss_spatial.item()
            num_batches += 1

        # Average loss metrics
        metrics = {k: v / max(1, num_batches) for k, v in totals.items()}

        # Average and add gradient norms
        avg_pre_clip = self._accumulate_grad_norms(pre_clip_norms_list)
        avg_post_clip = self._accumulate_grad_norms(post_clip_norms_list)
        metrics.update(avg_pre_clip)
        metrics.update(avg_post_clip)

        return metrics

    @torch.no_grad()
    def validate_epoch(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
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

            preds = self._forward(positions, params, mask)
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

            img_metrics = self._compute_image_metrics(preds_eval, targets_eval)
            sums["img_mae"] += img_metrics["img_mae"]
            sums["img_rmse"] += img_metrics["img_rmse"]
            sums["img_psnr"] += img_metrics["img_psnr"]

            if self.linearity_weight > 0:
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
        log_every: int = 0,
    ) -> Dict[str, Dict[str, list]]:
        history = {
            "train": {"total": [], "recon": [], "image": [], "coeff_l1": [], "linearity": [], "spatial": []},
            "val": {"mae": [], "rmse": [], "charbonnier": [], "superposition": [], "img_mae": [], "img_rmse": [], "img_psnr": []},
        }

        best_value = float("inf")

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

            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader)
                for key in history["val"]:
                    history["val"][key].append(val_metrics[key])

                # Log validation metrics to Rerun
                if self.rerun_logger:
                    self.rerun_logger.log_scalars(epoch, val_metrics, "val")

                # Log visualizations every N epochs
                if self.rerun_logger and epoch % self.rerun_logger.log_frequency == 0:
                    self._log_visualizations(epoch, val_loader)

                if save_best and output_dir is not None:
                    current = val_metrics.get(best_metric, None)
                    if current is not None and current < best_value:
                        best_value = current
                        output_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": self.model.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict(),
                                "best_metric": best_value,
                                "meta": checkpoint_meta,
                            },
                            output_dir / "best_model.pt",
                        )
                if scheduler is not None:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(val_metrics.get(best_metric, val_metrics["mae"]))
                    else:
                        scheduler.step()
            elif scheduler is not None:
                scheduler.step()

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
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "meta": checkpoint_meta,
                },
                output_dir / "last_model.pt",
            )

        return history
