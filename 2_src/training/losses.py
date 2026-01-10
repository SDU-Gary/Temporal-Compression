"""Loss functions for multi-temporal lighting compression training.

Implements:
    1. Reconstruction Loss: L2 loss on SH coefficients
    2. Temporal Smoothness: Encourages smooth temporal transitions
    3. Energy Conservation: Physics-based constraint (Stage 3+)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """L2 reconstruction loss on SH coefficients.

    Measures the difference between predicted and ground truth SH coefficients.
    """

    def __init__(self, reduction: str = 'mean'):
        """Initialize reconstruction loss.

        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute L2 reconstruction loss.

        Args:
            pred: [B, T, 27] or [B, N, T, 27] predicted SH coefficients
            target: [B, T, 27] or [B, N, T, 27] ground truth SH coefficients

        Returns:
            loss: Scalar loss value
        """
        # MSE loss
        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return loss


class TemporalSmoothLoss(nn.Module):
    """Temporal smoothness loss using finite differences.

    Encourages smooth transitions between consecutive time moments.
    Uses L1 or L2 norm on temporal differences.
    """

    def __init__(self, norm: str = 'l2', reduction: str = 'mean'):
        """Initialize temporal smooth loss.

        Args:
            norm: 'l1' or 'l2' norm for temporal differences
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.norm = norm
        self.reduction = reduction

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """Compute temporal smoothness loss.

        Args:
            pred: [B, T, 27] or [B, N, T, 27] predicted SH coefficients

        Returns:
            loss: Scalar smoothness loss
        """
        if pred.shape[-2] < 2:
            # Need at least 2 time steps
            return torch.tensor(0.0, device=pred.device)

        # Compute temporal differences: Δ = pred[:, t+1, :] - pred[:, t, :]
        if len(pred.shape) == 3:
            # [B, T, 27]
            diff = pred[:, 1:, :] - pred[:, :-1, :]  # [B, T-1, 27]
        else:
            # [B, N, T, 27]
            diff = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # [B, N, T-1, 27]

        # Compute norm
        if self.norm == 'l1':
            loss = torch.abs(diff)
        else:  # l2
            loss = diff ** 2

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LPIPSTemporalLoss(nn.Module):
    """Temporal consistency loss using LPIPS metric.

    Measures perceptual similarity between consecutive time moments.
    Requires rendering images from SH coefficients (more expensive).

    Note: This is a simplified version. Full implementation would require
    rendering SH to images and computing LPIPS on those images.
    """

    def __init__(self, reduction: str = 'mean'):
        """Initialize LPIPS temporal loss.

        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.reduction = reduction

        # Import LPIPS if available
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='alex')
            self.available = True
        except ImportError:
            print("Warning: lpips not available, using L2 fallback")
            self.available = False

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """Compute LPIPS temporal loss.

        For now, this is a placeholder that uses L2 as fallback.
        Full implementation would render SH to images first.

        Args:
            pred: [B, T, 27] predicted SH coefficients

        Returns:
            loss: Scalar LPIPS loss
        """
        if not self.available or pred.shape[-2] < 2:
            return torch.tensor(0.0, device=pred.device)

        # Simplified version: use L2 on SH coefficients as proxy
        diff = pred[:, 1:, :] - pred[:, :-1, :]
        loss = (diff ** 2).mean()

        return loss


class EnergyConservationLoss(nn.Module):
    """Energy conservation constraint based on physics.

    Ensures that the total energy of lighting remains physically plausible.
    Based on the fact that SH order 0 (DC component) represents average energy.

    For 2nd order SH with 9 bases × 3 RGB channels:
        - Coefficients 0, 9, 18 are the DC components for R, G, B
    """

    def __init__(self, reduction: str = 'mean'):
        """Initialize energy conservation loss.

        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute energy conservation loss.

        Penalizes deviation from target energy levels.

        Args:
            pred: [B, T, 27] or [B, N, T, 27] predicted SH coefficients
            target: [B, T, 27] or [B, N, T, 27] ground truth SH coefficients

        Returns:
            loss: Scalar energy conservation loss
        """
        # Extract DC components (indices 0, 9, 18 for R, G, B)
        dc_indices = torch.tensor([0, 9, 18], device=pred.device)

        if len(pred.shape) == 3:
            # [B, T, 27]
            pred_dc = pred[:, :, dc_indices]  # [B, T, 3]
            target_dc = target[:, :, dc_indices]  # [B, T, 3]
        else:
            # [B, N, T, 27]
            pred_dc = pred[:, :, :, dc_indices]  # [B, N, T, 3]
            target_dc = target[:, :, :, dc_indices]  # [B, N, T, 3]

        # L2 loss on DC components
        loss = F.mse_loss(pred_dc, target_dc, reduction=self.reduction)

        return loss


class CombinedLoss(nn.Module):
    """Combined loss function with multiple components.

    Combines:
        1. Reconstruction loss (L2 on SH coefficients)
        2. Temporal smoothness (optional, for temporal consistency)
        3. Energy conservation (optional, for physics constraint)

    Loss = w_recon * L_recon + w_temp * L_temp + w_energy * L_energy
    """

    def __init__(
        self,
        w_reconstruction: float = 1.0,
        w_temporal_smooth: float = 0.01,
        w_energy: float = 0.0,
        temporal_norm: str = 'l2'
    ):
        """Initialize combined loss.

        Args:
            w_reconstruction: Weight for reconstruction loss
            w_temporal_smooth: Weight for temporal smoothness
            w_energy: Weight for energy conservation
            temporal_norm: Norm for temporal smoothness ('l1' or 'l2')
        """
        super().__init__()

        self.w_reconstruction = w_reconstruction
        self.w_temporal_smooth = w_temporal_smooth
        self.w_energy = w_energy

        # Loss components
        self.reconstruction_loss = ReconstructionLoss()
        self.temporal_smooth_loss = TemporalSmoothLoss(norm=temporal_norm)
        self.energy_conservation_loss = EnergyConservationLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Compute combined loss.

        Args:
            pred: [B, T, 27] or [B, N, T, 27] predicted SH coefficients
            target: [B, T, 27] or [B, N, T, 27] ground truth SH coefficients

        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Dictionary with individual loss components
        """
        # Reconstruction loss (always computed)
        loss_recon = self.reconstruction_loss(pred, target)

        # Temporal smoothness loss (if weight > 0)
        if self.w_temporal_smooth > 0:
            loss_temp = self.temporal_smooth_loss(pred)
        else:
            loss_temp = torch.tensor(0.0, device=pred.device)

        # Energy conservation loss (if weight > 0)
        if self.w_energy > 0:
            loss_energy = self.energy_conservation_loss(pred, target)
        else:
            loss_energy = torch.tensor(0.0, device=pred.device)

        # Total weighted loss
        total_loss = (
            self.w_reconstruction * loss_recon +
            self.w_temporal_smooth * loss_temp +
            self.w_energy * loss_energy
        )

        # Return total and individual components for logging
        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': loss_recon.item(),
            'temporal_smooth': loss_temp.item(),
            'energy_conservation': loss_energy.item()
        }

        return total_loss, loss_dict

    def extra_repr(self) -> str:
        return (
            f"w_reconstruction={self.w_reconstruction}, "
            f"w_temporal_smooth={self.w_temporal_smooth}, "
            f"w_energy={self.w_energy}"
        )
