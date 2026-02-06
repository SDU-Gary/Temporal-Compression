"""
Physics-Guided Light Trajectory (PG-RGLT) Model

Per-probe low-rank factorization for transmission tensor compression.
Avoids cross-space Tucker decomposition by modeling each probe independently.

Formula (per probe p):
    SH(l) = U_p @ (coeffs_p @ Φ(l))

Where:
    - U_p: [27, rank] spatial basis (learned)
    - coeffs_p: [rank, 3] mixing weights (learned)
    - Φ(l): [3] physics basis = [cos(θ_l), sin(θ_l), 1]
    - θ_l: angle of light position on circular trajectory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class PhysicsBasisEncoder(nn.Module):
    """
    Encode light trajectory position using physics-based trigonometric functions.

    For circular trajectory, uses angle-based encoding:
        Φ(light_idx) = [cos(θ), sin(θ), 1]
    where θ = 2π * (light_idx / num_lights)
    """

    def __init__(self, num_light_positions: int = 12):
        super().__init__()
        self.num_light_positions = num_light_positions

        # Precompute angles for each light position
        angles = 2 * np.pi * np.arange(num_light_positions) / num_light_positions
        self.register_buffer('angles', torch.tensor(angles, dtype=torch.float32))

    def forward(self, light_indices: torch.Tensor) -> torch.Tensor:
        """
        Encode light position indices.

        Args:
            light_indices: [B] integer indices in [0, num_light_positions)

        Returns:
            basis: [B, 3] physics basis [cos(θ), sin(θ), 1]
        """
        # Get angles for these light positions
        theta = self.angles[light_indices]  # [B]

        # Compute basis functions
        cos_theta = torch.cos(theta)  # [B]
        sin_theta = torch.sin(theta)  # [B]
        ones = torch.ones_like(theta)  # [B]

        # Stack to [B, 3]
        basis = torch.stack([cos_theta, sin_theta, ones], dim=-1)

        return basis


class PerProbeLowRank(nn.Module):
    """
    Low-rank factorization for a single probe's SH variation across light positions.

    SH(l) = U @ (coeffs @ Φ(l))

    Parameters:
        - U: [sh_dim, rank] = [27, rank]
        - coeffs: [rank, 3]
    Total: 27*rank + rank*3 parameters
    """

    def __init__(
        self,
        sh_dim: int = 27,
        rank: int = 5,
        init_scale: float = 0.1
    ):
        super().__init__()
        self.sh_dim = sh_dim
        self.rank = rank

        # Spatial basis matrix U [sh_dim, rank]
        self.U = nn.Parameter(torch.randn(sh_dim, rank) * init_scale)

        # Mixing coefficients [rank, 3]
        self.coeffs = nn.Parameter(torch.randn(rank, 3) * init_scale)

    def forward(self, physics_basis: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct SH coefficients for given light positions.

        Args:
            physics_basis: [B, 3] physics basis Φ(l)

        Returns:
            sh_coeffs: [B, 27] reconstructed SH coefficients
        """
        # coeffs @ Φ(l)^T: [rank, 3] @ [B, 3]^T = [rank, B]
        # Then transpose to [B, rank]
        temporal_encoding = self.coeffs @ physics_basis.T  # [rank, B]
        temporal_encoding = temporal_encoding.T  # [B, rank]

        # U @ temporal_encoding^T: [27, rank] @ [B, rank]^T = [27, B]
        # Then transpose to [B, 27]
        sh_coeffs = self.U @ temporal_encoding.T  # [27, B]
        sh_coeffs = sh_coeffs.T  # [B, 27]

        return sh_coeffs

    def get_params_count(self) -> int:
        """Return total number of learnable parameters."""
        return self.sh_dim * self.rank + self.rank * 3


class PGRGLT(nn.Module):
    """
    Physics-Guided Light Trajectory (PG-RGLT) model for transmission tensor.

    Stores one PerProbeLowRank module per probe position.

    Forward pass:
        1. Look up probe index
        2. Encode light position with physics basis
        3. Query per-probe low-rank module
        4. Return reconstructed SH coefficients
    """

    def __init__(
        self,
        num_probes: int = 343,
        num_light_positions: int = 12,
        sh_dim: int = 27,
        rank: int = 5,
        init_scale: float = 0.1
    ):
        super().__init__()
        self.num_probes = num_probes
        self.num_light_positions = num_light_positions
        self.sh_dim = sh_dim
        self.rank = rank

        # Physics basis encoder
        self.physics_encoder = PhysicsBasisEncoder(num_light_positions)

        # Per-probe low-rank modules
        self.probe_modules = nn.ModuleList([
            PerProbeLowRank(sh_dim=sh_dim, rank=rank, init_scale=init_scale)
            for _ in range(num_probes)
        ])

    def forward(
        self,
        probe_indices: torch.Tensor,
        light_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct SH coefficients for (probe, light) pairs.

        Args:
            probe_indices: [B] integer indices in [0, num_probes)
            light_indices: [B] integer indices in [0, num_light_positions)

        Returns:
            sh_coeffs: [B, 27] reconstructed SH coefficients
        """
        batch_size = probe_indices.shape[0]

        # Encode light positions
        physics_basis = self.physics_encoder(light_indices)  # [B, 3]

        # Query per-probe modules
        sh_coeffs_list = []
        for i in range(batch_size):
            probe_idx = probe_indices[i].item()
            probe_module = self.probe_modules[probe_idx]

            # Get SH for this (probe, light) pair
            sh_coeff = probe_module(physics_basis[i:i+1])  # [1, 27]
            sh_coeffs_list.append(sh_coeff)

        # Stack results
        sh_coeffs = torch.cat(sh_coeffs_list, dim=0)  # [B, 27]

        return sh_coeffs

    def get_compression_stats(self) -> dict:
        """
        Calculate compression statistics.

        Returns:
            stats: dict with compression metrics
        """
        # Parameters per probe
        params_per_probe = self.sh_dim * self.rank + self.rank * 3

        # Total parameters
        total_params = self.num_probes * params_per_probe

        # Uncompressed size: num_probes * num_lights * sh_dim
        uncompressed = self.num_probes * self.num_light_positions * self.sh_dim

        # Compression ratio
        compression_ratio = uncompressed / total_params

        return {
            'params_per_probe': params_per_probe,
            'total_params': total_params,
            'uncompressed_size': uncompressed,
            'compression_ratio': compression_ratio,
            'rank': self.rank,
            'num_probes': self.num_probes,
            'num_lights': self.num_light_positions
        }

    def initialize_from_svd(
        self,
        transfer_tensor: np.ndarray,
        rank: Optional[int] = None
    ):
        """
        Initialize U and coeffs from SVD of per-probe transmission matrices.

        Args:
            transfer_tensor: [num_probes, num_lights, sh_dim] numpy array
            rank: Optional rank override (default: use self.rank)
        """
        if rank is None:
            rank = self.rank

        num_probes, num_lights, sh_dim = transfer_tensor.shape
        assert num_probes == self.num_probes
        assert num_lights == self.num_light_positions
        assert sh_dim == self.sh_dim

        # For each probe, perform SVD on transmission matrix
        for probe_idx in range(num_probes):
            # Get transmission matrix for this probe: [num_lights, sh_dim]
            transmission_matrix = transfer_tensor[probe_idx]  # [12, 27]

            # SVD: [12, 27] = [12, 12] @ [12] @ [12, 27]
            U_svd, S, Vt = np.linalg.svd(transmission_matrix, full_matrices=False)

            # Take top-rank components
            U_r = Vt[:rank, :].T  # [27, rank] - right singular vectors
            S_r = S[:rank]  # [rank]
            coeffs_r = U_svd[:, :rank] * S_r[None, :]  # [12, rank]

            # Now we need to factor coeffs_r [12, rank] ≈ basis [12, 3] @ weights [3, rank]
            # Get physics basis for all light positions
            light_indices = torch.arange(num_lights, dtype=torch.long)
            physics_basis = self.physics_encoder(light_indices).detach().numpy()  # [12, 3]

            # Least-squares solve: coeffs_r ≈ physics_basis @ weights
            # weights = (basis^T basis)^-1 basis^T coeffs_r
            weights = np.linalg.lstsq(physics_basis, coeffs_r, rcond=None)[0]  # [3, rank]

            # Set parameters (need to transpose weights to match [rank, 3] shape)
            self.probe_modules[probe_idx].U.data = torch.tensor(U_r, dtype=torch.float32)
            self.probe_modules[probe_idx].coeffs.data = torch.tensor(weights.T, dtype=torch.float32)  # [rank, 3]

    def analyze_rank_distribution(self, transfer_tensor: np.ndarray) -> dict:
        """
        Analyze SVD rank distribution across all probes.

        Returns:
            analysis: dict with statistics on singular values
        """
        num_probes, num_lights, sh_dim = transfer_tensor.shape

        singular_values_all = []
        energy_at_rank = {1: [], 2: [], 3: [], 5: [], 10: []}

        for probe_idx in range(num_probes):
            transmission_matrix = transfer_tensor[probe_idx]

            # SVD
            U, S, Vt = np.linalg.svd(transmission_matrix, full_matrices=False)

            singular_values_all.append(S)

            # Compute energy captured at different ranks
            total_energy = (S ** 2).sum()
            for rank in energy_at_rank.keys():
                if rank <= len(S):
                    energy = (S[:rank] ** 2).sum() / total_energy
                    energy_at_rank[rank].append(energy)

        # Aggregate statistics
        analysis = {
            'mean_energy_at_rank': {
                rank: np.mean(energies) for rank, energies in energy_at_rank.items()
            },
            'min_energy_at_rank': {
                rank: np.min(energies) for rank, energies in energy_at_rank.items()
            },
            'max_energy_at_rank': {
                rank: np.max(energies) for rank, energies in energy_at_rank.items()
            }
        }

        return analysis


def test_pg_rglt():
    """Test PG-RGLT model with random data."""
    print("Testing PG-RGLT Model")
    print("=" * 50)

    # Model parameters
    num_probes = 343
    num_lights = 12
    sh_dim = 27
    rank = 5

    # Create model
    model = PGRGLT(
        num_probes=num_probes,
        num_light_positions=num_lights,
        sh_dim=sh_dim,
        rank=rank
    )

    print(f"Model created:")
    print(f"  Probes: {num_probes}")
    print(f"  Light positions: {num_lights}")
    print(f"  SH dimension: {sh_dim}")
    print(f"  Rank: {rank}")
    print()

    # Compression stats
    stats = model.get_compression_stats()
    print("Compression Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    print()

    # Test forward pass
    batch_size = 16
    probe_indices = torch.randint(0, num_probes, (batch_size,))
    light_indices = torch.randint(0, num_lights, (batch_size,))

    print(f"Testing forward pass with batch_size={batch_size}...")
    sh_coeffs = model(probe_indices, light_indices)
    print(f"  Input: probe_indices {probe_indices.shape}, light_indices {light_indices.shape}")
    print(f"  Output: sh_coeffs {sh_coeffs.shape}")
    print()

    # Test SVD initialization
    print("Testing SVD initialization...")
    dummy_tensor = np.random.randn(num_probes, num_lights, sh_dim).astype(np.float32)
    model.initialize_from_svd(dummy_tensor, rank=rank)
    print("  SVD initialization successful")
    print()

    # Test rank analysis
    print("Testing rank analysis...")
    analysis = model.analyze_rank_distribution(dummy_tensor)
    print("  Mean energy captured at different ranks:")
    for rank_val, energy in analysis['mean_energy_at_rank'].items():
        print(f"    Rank {rank_val}: {energy*100:.2f}%")

    print("\n✓ All tests passed!")


if __name__ == '__main__':  # pragma: no cover
    test_pg_rglt()
