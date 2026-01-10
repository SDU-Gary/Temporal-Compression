"""Decoder MLP for converting latent codes to SH coefficients.

Maps: [F(p,t), position, time_encoding] → SH coefficients
Decodes the compressed latent representation into full lighting information.
"""

import torch
import torch.nn as nn
import numpy as np


class DecoderMLP(nn.Module):
    """MLP decoder that converts latent codes to SH coefficients.

    Input: [latent_code (9-D), position (3-D), time_encoding (3-D)] → 15-D
    Output: SH coefficients (27-D) = 9 bases × 3 RGB channels

    Architecture: 2 hidden layers × 64 neurons with ReLU activation
    """

    def __init__(
        self,
        input_dim: int = 15,  # latent_dim_time (9) + position (3) + time_encoding (3)
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 27,  # 2nd order SH: 9 bases × 3 RGB
        activation: str = 'relu'
    ):
        """Initialize decoder MLP.

        Args:
            input_dim: Input dimension (latent + position + time)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            output_dim: Output dimension (SH coefficients)
            activation: Activation function
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Build MLP layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(self._get_activation(activation))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self._get_activation(activation))

        # Output layer (no activation - SH coefficients can be positive or negative)
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def encode_time(self, sun_direction: torch.Tensor) -> torch.Tensor:
        """Encode time information from sun direction.

        Simple encoding: use sun direction directly as time encoding.
        More sophisticated encodings (e.g., Fourier features) can be added later.

        Args:
            sun_direction: [3] or [T, 3] or [B, T, 3] normalized sun direction

        Returns:
            time_encoding: Same shape as input
        """
        # For now, use sun direction directly as time encoding
        # Can be enhanced with positional encoding in future
        return sun_direction

    def forward(
        self,
        latent_code: torch.Tensor,
        position: torch.Tensor,
        sun_direction: torch.Tensor
    ) -> torch.Tensor:
        """Decode latent codes to SH coefficients.

        Args:
            latent_code: [N, D_time] or [N, T, D_time] or [B, N, T, D_time] time-varying latent
            position: [N, 3] or [B, N, 3] 3D positions (normalized)
            sun_direction: [3] or [T, 3] or [B, T, 3] sun directions for time encoding

        Returns:
            sh_coeffs: [N, 27] or [N, T, 27] or [B, N, T, 27] SH coefficients
        """
        latent_shape = latent_code.shape
        pos_shape = position.shape

        # Case 1: Single time step
        # latent_code: [N, D_time], position: [N, 3], sun_direction: [3]
        if len(latent_shape) == 2:
            N, D_time = latent_shape
            # Encode time
            time_enc = self.encode_time(sun_direction)  # [3]
            time_enc_expanded = time_enc.unsqueeze(0).expand(N, -1)  # [N, 3]

            # Concatenate inputs
            decoder_input = torch.cat([latent_code, position, time_enc_expanded], dim=-1)  # [N, 15]

            # Decode
            sh_coeffs = self.network(decoder_input)  # [N, 27]
            return sh_coeffs

        # Case 2: Multiple time steps
        # latent_code: [N, T, D_time], position: [N, 3], sun_direction: [T, 3]
        elif len(latent_shape) == 3 and len(pos_shape) == 2:
            N, T, D_time = latent_shape
            # Encode time
            time_enc = self.encode_time(sun_direction)  # [T, 3]

            # Expand position for all time steps
            pos_expanded = position.unsqueeze(1).expand(N, T, 3)  # [N, T, 3]
            time_enc_expanded = time_enc.unsqueeze(0).expand(N, T, 3)  # [N, T, 3]

            # Concatenate inputs
            decoder_input = torch.cat([latent_code, pos_expanded, time_enc_expanded], dim=-1)  # [N, T, 15]

            # Decode (flatten N and T for batch processing)
            decoder_input_flat = decoder_input.reshape(N * T, -1)
            sh_coeffs_flat = self.network(decoder_input_flat)  # [N*T, 27]
            sh_coeffs = sh_coeffs_flat.reshape(N, T, self.output_dim)  # [N, T, 27]
            return sh_coeffs

        # Case 3: Batch mode with multiple time steps
        # latent_code: [B, N, T, D_time], position: [B, N, 3], sun_direction: [B, T, 3]
        elif len(latent_shape) == 4:
            B, N, T, D_time = latent_shape
            # Encode time
            time_enc = self.encode_time(sun_direction)  # [B, T, 3]

            # Expand for broadcasting
            pos_expanded = position.unsqueeze(2).expand(B, N, T, 3)  # [B, N, T, 3]
            time_enc_expanded = time_enc.unsqueeze(1).expand(B, N, T, 3)  # [B, N, T, 3]

            # Concatenate inputs
            decoder_input = torch.cat([latent_code, pos_expanded, time_enc_expanded], dim=-1)  # [B, N, T, 15]

            # Decode (flatten for batch processing)
            decoder_input_flat = decoder_input.reshape(B * N * T, -1)
            sh_coeffs_flat = self.network(decoder_input_flat)  # [B*N*T, 27]
            sh_coeffs = sh_coeffs_flat.reshape(B, N, T, self.output_dim)  # [B, N, T, 27]
            return sh_coeffs

        else:
            raise ValueError(
                f"Unsupported input shapes: latent_code {latent_shape}, position {pos_shape}"
            )

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, output_dim={self.output_dim}"
        )
