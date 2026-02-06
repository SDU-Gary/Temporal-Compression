#!/usr/bin/env python3
"""
Gaussian-Physics Compression for 1D Intensity Modulation

Simplified version of gaussian_physics_5D.py for intensity-only variations.
Reduces physics basis from 5D→7D to 1D→2D, achieving 14% parameter reduction.

Architecture:
    SH(p, intensity) = Σ_j G_j(p) × [U_j @ (coeffs_j @ Φ(intensity))]

    where:
    - G_j(p): Gaussian weight at position p
    - U_j: [27, rank] low-rank matrix for Gaussian j
    - coeffs_j: [rank, 2] time coefficients for Gaussian j (reduced from 7)
    - Φ(intensity): [2] physics basis = [intensity, constant]

Parameters (K=30, rank=8):
    - Gaussian positions μ_j: 30 × 3 = 90
    - Gaussian scales s_j: 30 × 3 = 90
    - Low-rank matrices U_j: 30 × 27 × 8 = 6,480
    - Time coefficients coeffs_j: 30 × 8 × 2 = 480
    Total: 7,140 (vs 8,340 for 5D → 14.4% reduction)

Compression Ratio (K=30, rank=8, 343 probes × 12 moments):
    - Naive: 343 × 12 × 27 = 111,132 parameters
    - Compressed: 7,140 parameters
    - Ratio: 111,132 / 7,140 ≈ 15.6×
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans


class SimpleIntensityBasis(nn.Module):
    """1D intensity → 2D physics basis encoder.

    Maps scalar intensity to a 2D basis: [intensity, constant].
    This is the simplest possible physics basis for intensity modulation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, intensity_1D):
        """
        Args:
            intensity_1D: [B, 1] - intensity values in [0, 1]

        Returns:
            basis: [B, 2] - [intensity, constant]
        """
        intensity = intensity_1D[:, 0]  # [B]
        constant = torch.ones_like(intensity)  # [B]

        basis = torch.stack([intensity, constant], dim=-1)  # [B, 2]
        return basis


class GaussianPhysicsCompression1D(nn.Module):
    """Gaussian-Physics Compression for 1D intensity modulation."""

    def __init__(self, num_gaussians=30, rank=8, sh_dim=27):
        """
        Args:
            num_gaussians: Number of Gaussians K
            rank: Low-rank dimension
            sh_dim: SH coefficient dimension (default 27)
        """
        super().__init__()

        self.K = num_gaussians
        self.rank = rank
        self.sh_dim = sh_dim
        self.physics_dim = 2  # [intensity, constant]

        # Gaussian parameters
        self.mu = nn.Parameter(torch.randn(num_gaussians, 3) * 0.1)  # [K, 3] positions
        self.log_scale = nn.Parameter(torch.zeros(num_gaussians, 3))  # [K, 3] log scales

        # Low-rank decomposition: SH = U @ (time_coeffs @ physics_basis)
        self.U = nn.Parameter(torch.randn(num_gaussians, sh_dim, rank) * 0.01)  # [K, 27, rank]
        self.time_coeffs = nn.Parameter(torch.randn(num_gaussians, rank, self.physics_dim) * 0.01)  # [K, rank, 2]

        # Physics basis encoder
        self.physics_encoder = SimpleIntensityBasis()

        print(f"GaussianPhysicsCompression1D initialized:")
        print(f"  Gaussians: {num_gaussians}")
        print(f"  Rank: {rank}")
        print(f"  Physics basis dimension: {self.physics_dim} (1D→2D)")
        print(f"  Total parameters: {self.num_params()}")

    def compute_gaussian_weights(self, positions, top_k=3):
        """Compute Gaussian weights (vectorized version).

        Args:
            positions: [B, 3] query positions
            top_k: Use only nearest k Gaussians

        Returns:
            gaussian_weights: [B, top_k] normalized weights
            topk_indices: [B, top_k] top-k Gaussian indices
        """
        B = positions.shape[0]

        # Find top-k nearest Gaussians for each position
        mu_expanded = self.mu.unsqueeze(0)  # [1, K, 3]
        pos_expanded = positions.unsqueeze(1)  # [B, 1, 3]
        distances = torch.norm(mu_expanded - pos_expanded, dim=-1)  # [B, K]

        # Select top-k
        topk_values, topk_indices = torch.topk(
            distances, k=min(top_k, self.K), largest=False, dim=-1
        )  # [B, top_k]

        # Gather selected Gaussian parameters
        selected_mu = self.mu[topk_indices]  # [B, top_k, 3]
        selected_scale = torch.exp(self.log_scale[topk_indices])  # [B, top_k, 3]

        # Compute Gaussian weights [B, top_k]
        diff = pos_expanded - selected_mu  # [B, top_k, 3]
        weighted_diff = diff / selected_scale  # [B, top_k, 3]
        exponent = -0.5 * torch.sum(weighted_diff ** 2, dim=-1)  # [B, top_k]
        gaussian_weights = torch.exp(exponent)  # [B, top_k]

        # Normalize weights
        weight_sum = torch.sum(gaussian_weights, dim=-1, keepdim=True) + 1e-8  # [B, 1]
        gaussian_weights = gaussian_weights / weight_sum  # [B, top_k]

        return gaussian_weights, topk_indices

    def forward(self, positions, intensity_1D, top_k=3):
        """Forward pass: (p, intensity) → SH coefficients.

        Args:
            positions: [B, 3] probe positions
            intensity_1D: [B, 1] intensity values
            top_k: Number of nearest Gaussians to use

        Returns:
            sh_pred: [B, 27] predicted SH coefficients
        """
        B = positions.shape[0]

        # 1. Encode 1D intensity → 2D physics basis
        physics_basis = self.physics_encoder(intensity_1D)  # [B, 2]

        # 2. Compute Gaussian weights
        gaussian_weights, topk_indices = self.compute_gaussian_weights(
            positions, top_k=top_k
        )  # [B, top_k], [B, top_k]

        # 3. Vectorized low-rank computation
        # U: [K, 27, r]
        # time_coeffs: [K, r, 2]
        # physics_basis: [B, 2]

        U_selected = self.U[topk_indices]  # [B, top_k, 27, r]
        time_coeffs_selected = self.time_coeffs[topk_indices]  # [B, top_k, r, 2]

        # time_features = time_coeffs @ physics_basis
        # [B, top_k, r, 2] @ [B, 2, 1] = [B, top_k, r, 1]
        time_features = torch.matmul(
            time_coeffs_selected,
            physics_basis.unsqueeze(-1).unsqueeze(1).expand(-1, top_k, -1, -1)
        ).squeeze(-1)  # [B, top_k, r]

        # sh_components = U @ time_features
        # [B, top_k, 27, r] @ [B, top_k, r, 1] = [B, top_k, 27, 1]
        sh_components = torch.matmul(
            U_selected,
            time_features.unsqueeze(-1)
        ).squeeze(-1)  # [B, top_k, 27]

        # 4. Weighted sum over Gaussians
        sh_pred = torch.sum(
            gaussian_weights.unsqueeze(-1) * sh_components,
            dim=1
        )  # [B, 27]

        return sh_pred

    def initialize_from_probes(self, probe_positions, method='kmeans', device='cuda'):
        """Initialize Gaussian positions from probe positions.

        Args:
            probe_positions: [N, 3] probe positions (numpy or torch)
            method: Initialization method ('kmeans' or 'random')
            device: Device to move parameters to

        Notes:
            - KMeans: Cluster probe positions into K centers
            - Random: Random subset of probe positions
        """
        if isinstance(probe_positions, torch.Tensor):
            probe_positions = probe_positions.cpu().numpy()

        if method == 'kmeans':
            print(f"Initializing {self.K} Gaussians via KMeans...")
            kmeans = KMeans(n_clusters=self.K, random_state=42, n_init=10)
            kmeans.fit(probe_positions)
            centers = kmeans.cluster_centers_  # [K, 3]

            # Compute initial scales as distance to nearest neighbor
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(centers)
            distances, _ = nbrs.kneighbors(centers)
            scales = distances[:, 1:2]  # [K, 1]
            scales = np.tile(scales, (1, 3))  # [K, 3]

            # Update parameters
            self.mu.data = torch.from_numpy(centers).float().to(device)
            self.log_scale.data = torch.log(torch.from_numpy(scales).float().to(device) + 1e-6)

            print(f"  ✓ Initialized with KMeans")
            print(f"  ✓ Mean scale: {scales.mean():.4f}")

        elif method == 'random':
            print(f"Initializing {self.K} Gaussians via random sampling...")
            indices = np.random.choice(len(probe_positions), self.K, replace=False)
            centers = probe_positions[indices]

            self.mu.data = torch.from_numpy(centers).float().to(device)
            print(f"  ✓ Initialized with random sampling")

        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def num_params(self):
        """Calculate total number of parameters.

        Returns:
            Total parameters: K × (6 + 27*rank + rank*physics_dim)

        Example (K=30, rank=8, physics_dim=2):
            30 × (6 + 216 + 16) = 30 × 238 = 7,140
        """
        return self.K * (6 + self.sh_dim * self.rank + self.rank * self.physics_dim)

    def get_compression_ratio(self, num_probes, num_moments):
        """Calculate compression ratio vs naive storage.

        Args:
            num_probes: Number of probe positions
            num_moments: Number of time moments

        Returns:
            compression_ratio: Naive params / Compressed params
        """
        naive_params = num_probes * num_moments * self.sh_dim
        compressed_params = self.num_params()
        return naive_params / compressed_params


if __name__ == '__main__':  # pragma: no cover
    """Test 1D model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create model
    model = GaussianPhysicsCompression1D(
        num_gaussians=30,
        rank=8
    ).to(device)

    # Test forward pass
    B = 16
    positions = torch.randn(B, 3).to(device)
    intensity = torch.rand(B, 1).to(device)

    sh_pred = model(positions, intensity)
    print(f"\nForward pass test:")
    print(f"  Input: positions {positions.shape}, intensity {intensity.shape}")
    print(f"  Output: SH coefficients {sh_pred.shape}")

    # Test initialization
    probe_positions = np.random.randn(343, 3)
    model.initialize_from_probes(probe_positions, method='kmeans', device=device)

    # Calculate compression ratio
    compression_ratio = model.get_compression_ratio(num_probes=343, num_moments=12)
    print(f"\nCompression ratio (343 probes × 12 moments):")
    print(f"  Naive: {343 * 12 * 27} parameters")
    print(f"  Compressed: {model.num_params()} parameters")
    print(f"  Ratio: {compression_ratio:.2f}×")
