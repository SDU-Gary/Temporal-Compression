"""Evaluation metrics for multi-temporal lighting compression.

Implements:
    1. PSNR: Peak Signal-to-Noise Ratio
    2. MSE: Mean Squared Error
    3. MAE: Mean Absolute Error
    4. Relative Error: Normalized error metric
"""

import torch
import numpy as np
from typing import Union, Tuple


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Mean Squared Error.

    Args:
        pred: Predicted values
        target: Ground truth values

    Returns:
        mse: Mean squared error
    """
    return torch.mean((pred - target) ** 2).item()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Mean Absolute Error.

    Args:
        pred: Predicted values
        target: Ground truth values

    Returns:
        mae: Mean absolute error
    """
    return torch.mean(torch.abs(pred - target)).item()


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    PSNR = 20 * log10(MAX) - 10 * log10(MSE)
         = 10 * log10(MAX^2 / MSE)

    Args:
        pred: Predicted values [...]
        target: Ground truth values [...]
        max_val: Maximum possible value (default: 1.0 for normalized data)

    Returns:
        psnr: PSNR in dB
    """
    mse_value = torch.mean((pred - target) ** 2).item()

    if mse_value == 0:
        return float('inf')

    psnr_value = 10 * np.log10(max_val ** 2 / mse_value)
    return psnr_value


def relative_error(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute relative error.

    RE = ||pred - target|| / (||target|| + eps)

    Args:
        pred: Predicted values
        target: Ground truth values
        eps: Small constant to avoid division by zero

    Returns:
        re: Relative error
    """
    numerator = torch.norm(pred - target).item()
    denominator = torch.norm(target).item() + eps
    return numerator / denominator


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0
) -> dict:
    """Compute all metrics at once.

    Args:
        pred: Predicted SH coefficients
        target: Ground truth SH coefficients
        max_val: Maximum value for PSNR computation

    Returns:
        metrics: Dictionary with all metric values
    """
    metrics = {
        'mse': mse(pred, target),
        'mae': mae(pred, target),
        'psnr': psnr(pred, target, max_val),
        'relative_error': relative_error(pred, target)
    }

    return metrics


class MetricsTracker:
    """Track metrics over multiple batches/epochs.

    Accumulates metrics and computes running averages.
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()

    def reset(self):
        """Reset all accumulated metrics."""
        self.metrics = {}
        self.counts = {}

    def update(self, metrics_dict: dict, count: int = 1):
        """Update metrics with new values.

        Args:
            metrics_dict: Dictionary of metric name -> value
            count: Number of samples this update represents
        """
        for name, value in metrics_dict.items():
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0

            self.metrics[name] += value * count
            self.counts[name] += count

    def get_average(self) -> dict:
        """Get average of all tracked metrics.

        Returns:
            avg_metrics: Dictionary with averaged metric values
        """
        avg_metrics = {}
        for name in self.metrics:
            if self.counts[name] > 0:
                avg_metrics[name] = self.metrics[name] / self.counts[name]
            else:
                avg_metrics[name] = 0.0

        return avg_metrics

    def __repr__(self) -> str:
        avg = self.get_average()
        return f"MetricsTracker({avg})"


def evaluate_sh_reconstruction(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    max_val: float = 10.0
) -> dict:
    """Evaluate model on a dataset.

    Args:
        model: Compression model
        dataloader: DataLoader for evaluation data
        device: Device to run on
        max_val: Maximum value for PSNR (based on SH coefficient range)

    Returns:
        metrics: Dictionary with average metrics
    """
    model.eval()
    tracker = MetricsTracker()

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            positions = batch['position'].to(device)  # [B, 3]
            sh_gt = batch['sh_gt'].to(device)  # [B, T, 27]
            sun_dirs = batch['sun_dirs'].to(device)  # [B, T, 3]

            # Forward pass
            sh_pred = model(positions, sun_dirs)  # [B, T, 27]

            # Compute metrics
            batch_metrics = compute_all_metrics(sh_pred, sh_gt, max_val)

            # Update tracker
            tracker.update(batch_metrics, count=positions.shape[0])

    # Get averaged metrics
    avg_metrics = tracker.get_average()

    return avg_metrics
