"""Dual-Gaussian Factorized Basis Tucker (FBT) model.

Implements low-rank bi-gaussian transport compression:
    SH(p,t) = Σ_{i,j,r} G_i^probe(p) · G_j^light(l(t)) · U_{ir} · V_{jr}

Architecture:
- Probe space: K_p Gaussians modeling spatial distribution
- Light space: K_l Gaussians modeling light trajectory
- Tucker coupling: Learned factor matrices U[K_p, 27, r], V[K_l, 27, r]
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, Optional


class ProbeGaussianMixture(nn.Module):
    """Gaussian mixture in probe (scene) space.

    Each Gaussian stores a spatial location and scale.
    Outputs blending weights for query positions.
    """

    def __init__(self, num_gaussians: int = 20):
        """Initialize probe Gaussians.

        Args:
            num_gaussians: Number of Gaussians in probe space (K_p)
        """
        super().__init__()

        self.num_gaussians = num_gaussians

        # Gaussian parameters [K_p, 3]
        self.centers = nn.Parameter(torch.randn(num_gaussians, 3) * 0.1)
        self.log_scales = nn.Parameter(torch.zeros(num_gaussians, 3) - 1.0)  # log(sigma)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian weights for query positions.

        Args:
            positions: [B, 3] probe positions (normalized to [-1, 1])

        Returns:
            weights: [B, K_p] Gaussian blending weights (softmax normalized)
        """
        # Compute squared Mahalanobis distance
        # [B, K_p, 3] = [B, 1, 3] - [1, K_p, 3]
        diff = positions.unsqueeze(1) - self.centers.unsqueeze(0)

        # Diagonal covariance: sigma^2
        scales = torch.exp(self.log_scales).unsqueeze(0)  # [1, K_p, 3]

        # Squared distance weighted by scale
        dist_sq = torch.sum(diff ** 2 / (scales ** 2 + 1e-6), dim=-1)  # [B, K_p]

        # Gaussian weights (unnormalized)
        weights_unnorm = torch.exp(-0.5 * dist_sq)  # [B, K_p]

        # Softmax normalization
        weights = torch.softmax(weights_unnorm, dim=-1)

        return weights

    def initialize_from_positions(self, positions: np.ndarray, random_state: int = 42):
        """Initialize Gaussian centers via K-Means clustering.

        Args:
            positions: [N, 3] training probe positions
            random_state: Random seed for K-Means
        """
        if positions.shape[0] < self.num_gaussians:
            raise ValueError(f"Need at least {self.num_gaussians} positions for K-Means")

        # K-Means clustering
        kmeans = KMeans(n_clusters=self.num_gaussians, random_state=random_state, n_init=10)
        kmeans.fit(positions)

        # Set centers
        with torch.no_grad():
            self.centers.copy_(torch.from_numpy(kmeans.cluster_centers_).float())

        print(f"Initialized {self.num_gaussians} probe Gaussians via K-Means")


class LightGaussianMixture(nn.Module):
    """Gaussian mixture in light trajectory space.

    Each Gaussian stores a 3D position in light space.
    Outputs blending weights for query light positions.
    """

    def __init__(self, num_gaussians: int = 10):
        """Initialize light Gaussians.

        Args:
            num_gaussians: Number of Gaussians in light space (K_l)
        """
        super().__init__()

        self.num_gaussians = num_gaussians

        # Gaussian parameters [K_l, 3]
        self.centers = nn.Parameter(torch.randn(num_gaussians, 3) * 0.1)
        self.log_scales = nn.Parameter(torch.zeros(num_gaussians, 3) - 1.0)

    def forward(self, light_positions: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian weights for query light positions.

        Args:
            light_positions: [B, 3] light positions (normalized)

        Returns:
            weights: [B, K_l] Gaussian blending weights (softmax normalized)
        """
        # Compute squared distance
        diff = light_positions.unsqueeze(1) - self.centers.unsqueeze(0)  # [B, K_l, 3]
        scales = torch.exp(self.log_scales).unsqueeze(0)  # [1, K_l, 3]
        dist_sq = torch.sum(diff ** 2 / (scales ** 2 + 1e-6), dim=-1)  # [B, K_l]

        # Gaussian weights
        weights_unnorm = torch.exp(-0.5 * dist_sq)
        weights = torch.softmax(weights_unnorm, dim=-1)

        return weights

    def initialize_from_positions(self, positions: np.ndarray, random_state: int = 42):
        """Initialize Gaussian centers via K-Means clustering.

        Args:
            positions: [L, 3] training light positions
            random_state: Random seed for K-Means
        """
        if positions.shape[0] < self.num_gaussians:
            # If fewer light positions than Gaussians, just use the positions
            with torch.no_grad():
                self.centers.copy_(torch.from_numpy(positions).float())
                # Initialize additional centers with slight noise if needed
                if positions.shape[0] < self.num_gaussians:
                    extra = self.num_gaussians - positions.shape[0]
                    noise = torch.randn(extra, 3) * 0.05
                    base = torch.from_numpy(positions[np.random.choice(positions.shape[0], extra)]).float()
                    self.centers[positions.shape[0]:].copy_(base + noise)
        else:
            # K-Means clustering
            kmeans = KMeans(n_clusters=self.num_gaussians, random_state=random_state, n_init=10)
            kmeans.fit(positions)

            with torch.no_grad():
                self.centers.copy_(torch.from_numpy(kmeans.cluster_centers_).float())

        print(f"Initialized {self.num_gaussians} light Gaussians")


class DualGaussianFBT(nn.Module):
    """Dual-Gaussian Factorized Basis Tucker (FBT) model.

    Formula:
        SH(p, l) = Σ_c Σ_i Σ_j Σ_r G_i^probe(p) · G_j^light(l) · U[i,c,r] · V[j,c,r]

    Where:
        - G_i^probe(p): probe Gaussian i's weight at position p
        - G_j^light(l): light Gaussian j's weight at light position l
        - U[i,c,r]: probe factor matrix (K_p × 27 × rank)
        - V[j,c,r]: light factor matrix (K_l × 27 × rank)
        - c: SH coefficient index (0-26)
        - r: Tucker rank index (0 to rank-1)
    """

    def __init__(
        self,
        num_probe_gaussians: int = 20,
        num_light_gaussians: int = 10,
        tucker_rank: int = 3,
        sh_dim: int = 27
    ):
        """Initialize dual-gaussian FBT model.

        Args:
            num_probe_gaussians: Number of probe space Gaussians (K_p)
            num_light_gaussians: Number of light space Gaussians (K_l)
            tucker_rank: Tucker decomposition rank (r)
            sh_dim: SH coefficients dimension (27 for 2nd order × RGB)
        """
        super().__init__()

        self.num_probe_gaussians = num_probe_gaussians
        self.num_light_gaussians = num_light_gaussians
        self.tucker_rank = tucker_rank
        self.sh_dim = sh_dim

        # Probe and light Gaussian mixtures
        self.probe_gaussians = ProbeGaussianMixture(num_probe_gaussians)
        self.light_gaussians = LightGaussianMixture(num_light_gaussians)

        # Tucker factor matrices
        # U: [K_p, C, R] - probe factors
        # V: [K_l, C, R] - light factors
        self.U = nn.Parameter(torch.randn(num_probe_gaussians, sh_dim, tucker_rank) * 0.1)
        self.V = nn.Parameter(torch.randn(num_light_gaussians, sh_dim, tucker_rank) * 0.1)

    def forward(
        self,
        probe_positions: torch.Tensor,
        light_positions: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass: predict SH coefficients.

        Args:
            probe_positions: [B, 3] probe world positions (normalized)
            light_positions: [B, 3] light world positions (normalized)

        Returns:
            sh_coeffs: [B, 27] predicted SH coefficients
        """
        batch_size = probe_positions.shape[0]

        # Get Gaussian weights
        probe_weights = self.probe_gaussians(probe_positions)  # [B, K_p]
        light_weights = self.light_gaussians(light_positions)  # [B, K_l]

        # Compute SH coefficients via Tucker reconstruction
        sh_coeffs = torch.zeros(batch_size, self.sh_dim, device=probe_positions.device)

        for c in range(self.sh_dim):
            # For each SH coefficient, compute:
            # SH_c = Σ_i Σ_j Σ_r w_i^p · w_j^l · U[i,c,r] · V[j,c,r]

            # [B, K_p, R]
            U_weighted = probe_weights.unsqueeze(2) * self.U[:, c, :].unsqueeze(0)

            # [B, K_l, R]
            V_weighted = light_weights.unsqueeze(2) * self.V[:, c, :].unsqueeze(0)

            # Sum over probe Gaussians
            U_sum = U_weighted.sum(dim=1)  # [B, R]

            # Sum over light Gaussians
            V_sum = V_weighted.sum(dim=1)  # [B, R]

            # Element-wise product and sum over rank
            sh_coeffs[:, c] = (U_sum * V_sum).sum(dim=1)  # [B]

        return sh_coeffs

    def initialize_from_data(
        self,
        probe_positions: np.ndarray,
        light_positions: np.ndarray,
        random_state: int = 42
    ):
        """Initialize Gaussian centers from training data.

        Args:
            probe_positions: [P, 3] all probe positions from dataset
            light_positions: [L, 3] all light positions from dataset
            random_state: Random seed for K-Means
        """
        self.probe_gaussians.initialize_from_positions(probe_positions, random_state)
        self.light_gaussians.initialize_from_positions(light_positions, random_state)

    def get_model_size(self) -> Tuple[int, float]:
        """Calculate model size.

        Returns:
            size_params: Total number of parameters
            size_mb: Model size in megabytes (assuming float32)
        """
        total_params = sum(p.numel() for p in self.parameters())
        size_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per float32
        return total_params, size_mb

    def get_compression_ratio(self, num_probes: int, num_lights: int) -> float:
        """Calculate compression ratio vs uncompressed data.

        Args:
            num_probes: Number of probes in original dataset
            num_lights: Number of light positions in dataset

        Returns:
            ratio: Compression ratio (original_size / compressed_size)
        """
        # Original: num_probes × num_lights × 27 coefficients
        original_params = num_probes * num_lights * self.sh_dim

        # Compressed: our model parameters
        compressed_params, _ = self.get_model_size()

        ratio = original_params / compressed_params
        return ratio

    def extra_repr(self) -> str:
        params, size_mb = self.get_model_size()
        return (
            f"probe_gaussians={self.num_probe_gaussians}, "
            f"light_gaussians={self.num_light_gaussians}, "
            f"tucker_rank={self.tucker_rank}, "
            f"params={params}, "
            f"size={size_mb:.2f}MB"
        )


if __name__ == '__main__':  # pragma: no cover
    """Test dual-gaussian FBT model."""
    # Create model
    model = DualGaussianFBT(
        num_probe_gaussians=20,
        num_light_gaussians=10,
        tucker_rank=3
    )

    print(model)

    # Test forward pass
    batch_size = 4
    probe_pos = torch.randn(batch_size, 3)
    light_pos = torch.randn(batch_size, 3)

    sh_coeffs = model(probe_pos, light_pos)
    print(f"\nForward pass test:")
    print(f"  Input shapes: probe_pos {probe_pos.shape}, light_pos {light_pos.shape}")
    print(f"  Output shape: {sh_coeffs.shape}")
    print(f"  Output range: [{sh_coeffs.min():.4f}, {sh_coeffs.max():.4f}]")

    # Test initialization
    probe_positions_np = np.random.randn(100, 3) * 0.5
    light_positions_np = np.random.randn(12, 3)

    model.initialize_from_data(probe_positions_np, light_positions_np)

    # Test compression ratio
    compression = model.get_compression_ratio(num_probes=343, num_lights=12)
    print(f"\nCompression ratio (343 probes × 12 lights): {compression:.1f}×")
