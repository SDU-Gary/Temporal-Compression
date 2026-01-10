"""Gaussian Mixture Model for spatial latent code representation.

Based on SIGGRAPH 2025 paper: "Gaussian Compression for Precomputed Indirect Illumination"
Implements 3D Gaussian functions to store base latent codes that vary across space.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, Optional


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix.

    Args:
        q: [K, 4] quaternions (w, x, y, z) - normalized

    Returns:
        R: [K, 3, 3] rotation matrices
    """
    # Normalize quaternion
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Build rotation matrix (from quaternion formula)
    R = torch.stack([
        torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1),
        torch.stack([2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)], dim=-1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)], dim=-1)
    ], dim=1)  # [K, 3, 3]

    return R


class GaussianMixture(nn.Module):
    """3D Gaussian mixture model for storing spatial latent codes.

    Each Gaussian G_j stores:
        - μ_j: 3D position (center)
        - s_j: 3D scale (anisotropic)
        - q_j: 4D quaternion (rotation)
        - F_j: D-dim base latent code (time-invariant)

    Forward pass computes weighted sum of latent codes at query positions:
        F(p) = Σ_j F_j * G_j(p)
        where G_j(p) = exp(-1/2 * (p-μ_j)^T Σ_j^(-1) (p-μ_j))
        and Σ_j = R_j S_j S_j^T R_j^T
    """

    def __init__(
        self,
        num_gaussians: int,
        latent_dim: int,
        init_scale: float = 1.0,
    ):
        """Initialize Gaussian mixture model.

        Args:
            num_gaussians: Number of Gaussian functions (K)
            latent_dim: Dimension of base latent code (D)
            init_scale: Initial scale for Gaussians
        """
        super().__init__()

        self.num_gaussians = num_gaussians
        self.latent_dim = latent_dim

        # Gaussian parameters (following 3DGS convention)
        self.means = nn.Parameter(torch.zeros(num_gaussians, 3))  # μ
        self.scales = nn.Parameter(torch.ones(num_gaussians, 3) * init_scale)  # s (log space)
        self.rotations = nn.Parameter(torch.zeros(num_gaussians, 4))  # q (quaternion)
        self.rotations.data[:, 0] = 1.0  # Initialize to identity rotation

        # Base latent codes (time-invariant)
        self.latent_codes = nn.Parameter(torch.randn(num_gaussians, latent_dim) * 0.1)

    def initialize_from_positions(self, positions: np.ndarray, random_state: int = 42):
        """Initialize Gaussian centers using K-Means clustering on probe positions.

        Args:
            positions: [N, 3] probe positions
            random_state: Random seed for K-Means

        Raises:
            ValueError: If num_gaussians > number of positions
        """
        num_positions = len(positions)

        # Validate that we have enough positions for K-Means
        if self.num_gaussians > num_positions:
            raise ValueError(
                f"Cannot initialize {self.num_gaussians} Gaussians from only {num_positions} positions. "
                f"Either reduce num_gaussians in config (recommended: <= {int(num_positions * 0.8)}) "
                f"or increase the number of probe positions in the dataset."
            )

        # K-Means clustering to find initial centers
        kmeans = KMeans(n_clusters=self.num_gaussians, random_state=random_state, n_init=10)
        kmeans.fit(positions)
        centers = kmeans.cluster_centers_  # [K, 3]

        # Set means
        self.means.data = torch.from_numpy(centers).float()

        # Compute scale based on nearest neighbor distance
        # For each Gaussian, find distance to nearest other Gaussian
        centers_torch = torch.from_numpy(centers).float()
        distances = torch.cdist(centers_torch, centers_torch)  # [K, K]
        distances = distances + torch.eye(self.num_gaussians) * 1e10  # Mask diagonal
        min_distances = distances.min(dim=1)[0]  # [K]

        # Set scale to 1/3 of nearest neighbor distance (3-sigma rule covers 99%)
        # Use log scale for stability during optimization
        self.scales.data = torch.log(min_distances.unsqueeze(1).repeat(1, 3) / 3.0 + 1e-6)

    def get_covariance_matrices(self) -> torch.Tensor:
        """Compute covariance matrices from scale and rotation.

        Returns:
            Σ: [K, 3, 3] covariance matrices
        """
        # Scale matrix S (diagonal)
        S = torch.diag_embed(torch.exp(self.scales))  # [K, 3, 3]

        # Rotation matrix R from quaternion
        R = quaternion_to_rotation_matrix(self.rotations)  # [K, 3, 3]

        # Covariance: Σ = R S S^T R^T
        Sigma = R @ S @ S.transpose(-2, -1) @ R.transpose(-2, -1)  # [K, 3, 3]

        return Sigma

    def gaussian_weights(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Gaussian weights for query positions.

        Args:
            positions: [N, 3] query positions

        Returns:
            weights: [N, K] Gaussian weights at each position
            valid_mask: [N, K] mask indicating which Gaussians influence each position
        """
        N = positions.shape[0]
        K = self.num_gaussians

        # Expand for broadcasting: positions [N, 1, 3], means [1, K, 3]
        pos_expanded = positions.unsqueeze(1)  # [N, 1, 3]
        means_expanded = self.means.unsqueeze(0)  # [1, K, 3]

        # Compute differences
        diff = pos_expanded - means_expanded  # [N, K, 3]

        # Compute covariance matrices and their inverses
        Sigma = self.get_covariance_matrices()  # [K, 3, 3]
        Sigma_inv = torch.inverse(Sigma + torch.eye(3, device=Sigma.device) * 1e-6)  # [K, 3, 3]

        # Compute Mahalanobis distance: (p-μ)^T Σ^(-1) (p-μ)
        # diff: [N, K, 3, 1], Sigma_inv: [1, K, 3, 3]
        diff_expanded = diff.unsqueeze(-1)  # [N, K, 3, 1]
        Sigma_inv_expanded = Sigma_inv.unsqueeze(0)  # [1, K, 3, 3]

        mahal_dist = (diff_expanded.transpose(-2, -1) @ Sigma_inv_expanded @ diff_expanded).squeeze(-1).squeeze(-1)  # [N, K]

        # Compute Gaussian weights: exp(-1/2 * mahal_dist)
        weights = torch.exp(-0.5 * mahal_dist)  # [N, K]

        # Compute influence range: 3-sigma rule (99% confidence)
        # |p - μ| < 3 * max(scale)
        max_scales = torch.exp(self.scales).max(dim=1)[0]  # [K]
        dist_to_centers = torch.norm(diff, dim=-1)  # [N, K]
        valid_mask = dist_to_centers < (3.0 * max_scales.unsqueeze(0))  # [N, K]

        # Apply mask (set weights to 0 outside influence range)
        weights = weights * valid_mask.float()

        return weights, valid_mask

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute spatially-varying latent codes at query positions.

        Args:
            positions: [N, 3] or [B, N, 3] query positions

        Returns:
            latent_features: [N, D] or [B, N, D] base latent codes at each position
        """
        input_shape = positions.shape
        if len(input_shape) == 3:
            # Batch mode: [B, N, 3]
            B, N, _ = input_shape
            positions = positions.reshape(B * N, 3)

        # Compute Gaussian weights
        weights, _ = self.gaussian_weights(positions)  # [N or B*N, K]

        # Weighted sum of latent codes: F(p) = Σ_j F_j * G_j(p)
        latent_features = weights @ self.latent_codes  # [N or B*N, D]

        if len(input_shape) == 3:
            # Reshape back to batch mode
            latent_features = latent_features.reshape(B, N, self.latent_dim)

        return latent_features

    def extra_repr(self) -> str:
        return f"num_gaussians={self.num_gaussians}, latent_dim={self.latent_dim}"
