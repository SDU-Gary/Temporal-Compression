"""Temporal MLP for converting base latent codes to time-varying latent codes.

Maps: [F_j^base, sun_dir(t)] → F_j(t)
This enables the model to capture temporal lighting variations.
"""

import torch
import torch.nn as nn


class TemporalMLP(nn.Module):
    """MLP that adds temporal variation to base latent codes.

    Input: [base_latent (6-D), sun_direction (3-D)] → 9-D
    Output: time_varying_latent (9-D)

    Architecture: 2 hidden layers × 32 neurons with ReLU activation
    """

    def __init__(
        self,
        input_dim: int = 9,  # latent_dim_base (6) + sun_dir (3)
        hidden_dim: int = 32,
        num_layers: int = 2,
        output_dim: int = 9,  # latent_dim_time
        activation: str = 'relu'
    ):
        """Initialize temporal MLP.

        Args:
            input_dim: Input dimension (base_latent + sun_dir)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            output_dim: Output dimension (time-varying latent)
            activation: Activation function ('relu', 'sigmoid', etc.)
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

        # Output layer
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

    def forward(
        self,
        base_latent: torch.Tensor,
        sun_direction: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass to generate time-varying latent codes.

        Args:
            base_latent: [N, D_base] or [B, N, D_base] base latent codes from Gaussians
            sun_direction: [3] or [T, 3] or [B, T, 3] sun direction vectors (normalized)

        Returns:
            time_latent: [N, D_time] or [B, N, T, D_time] time-varying latent codes
        """
        # Handle different input shapes
        base_shape = base_latent.shape
        sun_shape = sun_direction.shape

        # Case 1: Simple case - single position, single time
        # base_latent: [N, D_base], sun_direction: [3]
        if len(base_shape) == 2 and len(sun_shape) == 1:
            N, D_base = base_shape
            # Expand sun_direction to match batch
            sun_expanded = sun_direction.unsqueeze(0).expand(N, -1)  # [N, 3]
            # Concatenate
            mlp_input = torch.cat([base_latent, sun_expanded], dim=-1)  # [N, D_base+3]
            # Forward through network
            output = self.network(mlp_input)  # [N, D_time]
            return output

        # Case 2: Multiple positions, multiple times
        # base_latent: [N, D_base], sun_direction: [T, 3]
        elif len(base_shape) == 2 and len(sun_shape) == 2:
            N, D_base = base_shape
            T, _ = sun_shape
            # Expand for broadcasting
            base_expanded = base_latent.unsqueeze(1).expand(N, T, D_base)  # [N, T, D_base]
            sun_expanded = sun_direction.unsqueeze(0).expand(N, T, 3)  # [N, T, 3]
            # Concatenate
            mlp_input = torch.cat([base_expanded, sun_expanded], dim=-1)  # [N, T, D_base+3]
            # Forward through network (flatten N and T for batch processing)
            mlp_input_flat = mlp_input.reshape(N * T, -1)
            output_flat = self.network(mlp_input_flat)  # [N*T, D_time]
            output = output_flat.reshape(N, T, self.output_dim)  # [N, T, D_time]
            return output

        # Case 3: Batch mode - batch of positions, multiple times
        # base_latent: [B, N, D_base], sun_direction: [B, T, 3]
        elif len(base_shape) == 3 and len(sun_shape) == 3:
            B, N, D_base = base_shape
            _, T, _ = sun_shape
            # Expand for broadcasting
            base_expanded = base_latent.unsqueeze(2).expand(B, N, T, D_base)  # [B, N, T, D_base]
            sun_expanded = sun_direction.unsqueeze(1).expand(B, N, T, 3)  # [B, N, T, 3]
            # Concatenate
            mlp_input = torch.cat([base_expanded, sun_expanded], dim=-1)  # [B, N, T, D_base+3]
            # Forward through network (flatten for batch processing)
            mlp_input_flat = mlp_input.reshape(B * N * T, -1)
            output_flat = self.network(mlp_input_flat)  # [B*N*T, D_time]
            output = output_flat.reshape(B, N, T, self.output_dim)  # [B, N, T, D_time]
            return output

        else:
            raise ValueError(
                f"Unsupported input shapes: base_latent {base_shape}, sun_direction {sun_shape}"
            )

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, output_dim={self.output_dim}"
        )
