"""Shared trainer for Gaussian-Physics model variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn


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
    params_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    def unpack(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = batch[self.positions_key].to(device)
        params = batch[self.params_key].to(device)
        targets = batch[self.target_key].to(device)

        if self.params_transform is not None:
            params = self.params_transform(params)

        return positions, params, targets


class GaussianPhysicsTrainer:
    """Standardized trainer for Gaussian-Physics variants."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        adapter: BatchAdapter,
        lr: float = 1e-3,
        recon_loss: str = "mse",
        charbonnier_eps: float = 1e-3,
        temporal_weight: float = 0.0,
        temporal_loss_fn: Optional[Callable[[nn.Module], torch.Tensor]] = None,
        top_k: int = 3,
        grad_clip: Optional[float] = 1.0,
    ):
        self.model = model
        self.device = device
        self.adapter = adapter
        self.recon_loss = recon_loss
        self.charbonnier_eps = charbonnier_eps
        self.temporal_weight = temporal_weight
        self.temporal_loss_fn = temporal_loss_fn
        self.top_k = top_k
        self.grad_clip = grad_clip

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

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

    def _recon_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.recon_loss == "charbonnier":
            return charbonnier_loss(pred, target, epsilon=self.charbonnier_eps)
        return self.criterion(pred, target)

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.train()

        totals = {"total": 0.0, "recon": 0.0, "temporal": 0.0}
        num_batches = 0

        for batch in train_loader:
            positions, params, targets = self.adapter.unpack(batch, self.device)

            self.optimizer.zero_grad()
            preds = self.model(positions, params, top_k=self.top_k)

            loss_recon = self._recon_loss(preds, targets)
            loss_temporal = self._compute_temporal_loss()
            loss_total = loss_recon + self.temporal_weight * loss_temporal

            loss_total.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            totals["total"] += loss_total.item()
            totals["recon"] += loss_recon.item()
            totals["temporal"] += loss_temporal.item()
            num_batches += 1

        return {k: v / max(1, num_batches) for k, v in totals.items()}

    @torch.no_grad()
    def validate_epoch(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.eval()

        sums = {"mae": 0.0, "rmse": 0.0, "charbonnier": 0.0}
        num_batches = 0

        for batch in val_loader:
            positions, params, targets = self.adapter.unpack(batch, self.device)
            preds = self.model(positions, params, top_k=self.top_k)

            mae = torch.mean(torch.abs(preds - targets)).item()
            rmse = torch.sqrt(torch.mean((preds - targets) ** 2)).item()

            if self.recon_loss == "charbonnier":
                charbonnier = charbonnier_loss(preds, targets, epsilon=self.charbonnier_eps).item()
            else:
                charbonnier = 0.0

            sums["mae"] += mae
            sums["rmse"] += rmse
            sums["charbonnier"] += charbonnier
            num_batches += 1

        return {k: v / max(1, num_batches) for k, v in sums.items()}

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        num_epochs: int,
        output_dir: Optional[Path] = None,
        save_best: bool = True,
        best_metric: str = "mae",
        log_every: int = 0,
    ) -> Dict[str, Dict[str, list]]:
        history = {
            "train": {"total": [], "recon": [], "temporal": []},
            "val": {"mae": [], "rmse": [], "charbonnier": []},
        }

        best_value = float("inf")

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            history["train"]["total"].append(train_metrics["total"])
            history["train"]["recon"].append(train_metrics["recon"])
            history["train"]["temporal"].append(train_metrics["temporal"])

            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader)
                for key in history["val"]:
                    history["val"][key].append(val_metrics[key])

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
                            },
                            output_dir / "best_model.pt",
                        )

            if log_every and (epoch == 1 or epoch % log_every == 0):
                msg = (
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Train total={train_metrics['total']:.6f}, "
                    f"recon={train_metrics['recon']:.6f}, "
                    f"temporal={train_metrics['temporal']:.6f}"
                )
                if val_loader is not None:
                    msg += (
                        f" | Val mae={history['val']['mae'][-1]:.6f}, "
                        f"rmse={history['val']['rmse'][-1]:.6f}"
                    )
                print(msg)

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": num_epochs,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                output_dir / "last_model.pt",
            )

        return history
