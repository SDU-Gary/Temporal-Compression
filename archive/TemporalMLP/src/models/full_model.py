"""Full multi-temporal lighting compression model.

Integrates all components:
    1. GaussianMixture: spatial latent code representation
    2. TemporalMLP: temporal variation modeling
    3. DecoderMLP: SH coefficient reconstruction

Data flow:
    (position, time) → Gaussian → F_base(p) → TemporalMLP → F(p,t) → DecoderMLP → SH(p,t)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional

from .gaussian_mixture import GaussianMixture
from .temporal_mlp import TemporalMLP
from .decoder_mlp import DecoderMLP


class MultiTimeCompressionModel(nn.Module):
    """Full model for multi-temporal lighting compression.

    Architecture:
        1. Gaussian Mixture: K 3D Gaussians storing base latent codes F_j^base
        2. Temporal MLP: [F_j^base, sun_dir(t)] → F_j(t)
        3. Decoder MLP: [F(p,t), p, t] → SH coefficients

    This design allows:
        - Spatial compression via Gaussian mixture
        - Temporal compression via shared temporal MLP
        - High-quality reconstruction via decoder MLP
    """

    def __init__(self, config: Dict):
        """Initialize full model from configuration.

        Args:
            config: Model configuration dict with structure:
                model:
                  num_gaussians: 750
                  latent_dim_base: 6
                  latent_dim_time: 9
                  temporal_mlp:
                    input_dim: 9
                    hidden_dim: 32
                    num_layers: 2
                  decoder_mlp:
                    input_dim: 15
                    hidden_dim: 64
                    num_layers: 2
                    output_dim: 27
        """
        super().__init__()

        # Extract config
        model_config = config['model']
        self.num_gaussians = model_config['num_gaussians']
        self.latent_dim_base = model_config['latent_dim_base']
        self.latent_dim_time = model_config['latent_dim_time']

        # 1. Gaussian Mixture for spatial latent codes
        self.gaussian_mixture = GaussianMixture(
            num_gaussians=self.num_gaussians,
            latent_dim=self.latent_dim_base,
            init_scale=1.0
        )

        # 2. Temporal MLP for time-varying latent codes
        temporal_config = model_config['temporal_mlp']
        self.temporal_mlp = TemporalMLP(
            input_dim=temporal_config['input_dim'],
            hidden_dim=temporal_config['hidden_dim'],
            num_layers=temporal_config['num_layers'],
            output_dim=self.latent_dim_time,
            activation=temporal_config.get('activation', 'relu')
        )

        # 3. Decoder MLP for SH coefficient reconstruction
        decoder_config = model_config['decoder_mlp']
        self.decoder_mlp = DecoderMLP(
            input_dim=decoder_config['input_dim'],
            hidden_dim=decoder_config['hidden_dim'],
            num_layers=decoder_config['num_layers'],
            output_dim=decoder_config['output_dim'],
            activation=decoder_config.get('activation', 'relu')
        )

    def initialize_gaussians(self, positions: np.ndarray, random_state: int = 42):
        """Initialize Gaussian centers from probe positions.

        Args:
            positions: [N, 3] probe positions
            random_state: Random seed for K-Means
        """
        self.gaussian_mixture.initialize_from_positions(positions, random_state)

    def forward(
        self,
        positions: torch.Tensor,
        sun_directions: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass: predict SH coefficients at given positions and times.

        Args:
            positions: [N, 3] or [B, N, 3] query positions (normalized)
            sun_directions: [T, 3] or [B, T, 3] sun directions for each time moment

        Returns:
            sh_coeffs: [N, T, 27] or [B, N, T, 27] predicted SH coefficients
        """
        # Step 1: Get base latent codes from Gaussian mixture
        # F_base(p) = Σ_j F_j^base * G_j(p)
        base_latent = self.gaussian_mixture(positions)  # [N, D_base] or [B, N, D_base]

        # Step 2: Apply temporal MLP to get time-varying latent codes
        # F(p,t) = TemporalMLP([F_base(p), sun_dir(t)])
        time_latent = self.temporal_mlp(base_latent, sun_directions)  # [N, T, D_time] or [B, N, T, D_time]

        # Step 3: Decode to SH coefficients
        # SH(p,t) = DecoderMLP([F(p,t), p, t])
        sh_coeffs = self.decoder_mlp(time_latent, positions, sun_directions)  # [N, T, 27] or [B, N, T, 27]

        return sh_coeffs

    def decompress(
        self,
        positions: torch.Tensor,
        sun_directions: torch.Tensor
    ) -> torch.Tensor:
        """Alias for forward() with clearer name for inference.

        Args:
            positions: [N, 3] query positions
            sun_directions: [T, 3] sun directions

        Returns:
            sh_coeffs: [N, T, 27] decompressed SH coefficients
        """
        return self.forward(positions, sun_directions)

    def get_model_size(self) -> Tuple[int, float]:
        """Calculate model size in bytes and megabytes.

        Returns:
            size_bytes: Total model size in bytes
            size_mb: Total model size in MB
        """
        total_params = sum(p.numel() for p in self.parameters())
        # Assume float32 (4 bytes per parameter)
        size_bytes = total_params * 4
        size_mb = size_bytes / (1024 ** 2)
        return size_bytes, size_mb

    def get_compression_ratio(self, num_probes: int, num_moments: int) -> float:
        """Calculate compression ratio vs uncompressed probe data.

        Args:
            num_probes: Number of probes in original dataset
            num_moments: Number of time moments

        Returns:
            ratio: Compression ratio (original_size / compressed_size)
        """
        # Original size: num_probes × num_moments × 27 SH coeffs × 4 bytes (float32)
        original_size = num_probes * num_moments * 27 * 4

        # Compressed size: our model size
        compressed_size, _ = self.get_model_size()

        ratio = original_size / compressed_size
        return ratio

    def extra_repr(self) -> str:
        size_bytes, size_mb = self.get_model_size()
        return (
            f"num_gaussians={self.num_gaussians}, "
            f"latent_dim_base={self.latent_dim_base}, "
            f"latent_dim_time={self.latent_dim_time}, "
            f"model_size={size_mb:.2f}MB"
        )
