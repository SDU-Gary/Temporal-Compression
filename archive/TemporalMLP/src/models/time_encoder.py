"""Time Encoder for TDML (Temporal Dynamics Metric Learning).

Maps raw time t to metric embedding z_t, where ||z_ti - z_tj|| reflects
the dynamic similarity between frames i and j.
"""

import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """Encoder that maps time to metric embedding space.

    This is the core component of TDML. The embedding space has a learned
    metric where distance reflects dynamic similarity (optical flow,
    perceptual distance, etc.).

    Input: t (scalar or [T] tensor) - raw time in [0, T_max]
    Output: z_t ([D_embed]) - metric embedding
    """

    def __init__(
        self,
        embed_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 2,
        activation: str = 'relu',
        use_fourier_features: bool = True,
        num_fourier_freqs: int = 8
    ):
        """Initialize time encoder.

        Args:
            embed_dim: Dimension of output metric embedding
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            activation: Activation function
            use_fourier_features: Whether to use Fourier feature encoding
            num_fourier_freqs: Number of Fourier frequencies (if used)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.use_fourier_features = use_fourier_features
        self.num_fourier_freqs = num_fourier_freqs

        # Input dimension
        if use_fourier_features:
            # Fourier features: [sin(2π*f*t), cos(2π*f*t)] for multiple freqs
            input_dim = 2 * num_fourier_freqs
            # Learnable frequencies
            self.register_parameter(
                'freqs',
                nn.Parameter(torch.randn(num_fourier_freqs) * 10.0)
            )
        else:
            # Raw time scalar
            input_dim = 1

        # Build MLP
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(self._get_activation(activation))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self._get_activation(activation))

        # Output layer
        layers.append(nn.Linear(hidden_dim, embed_dim))

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

    def encode_time(self, t: torch.Tensor) -> torch.Tensor:
        """Encode raw time to Fourier features (if enabled).

        Args:
            t: [...] raw time values

        Returns:
            features: [..., 2*num_fourier_freqs] Fourier features
        """
        if not self.use_fourier_features:
            # Just add channel dimension
            return t.unsqueeze(-1)  # [..., 1]

        # Compute Fourier features: [sin(2π*f*t), cos(2π*f*t)]
        # t: [...], freqs: [F]
        # Output: [..., 2F]
        t_expanded = t.unsqueeze(-1)  # [..., 1]
        freqs_expanded = self.freqs.view(1, -1)  # [1, F]

        # Compute phase: 2π * f * t
        phase = 2.0 * torch.pi * freqs_expanded * t_expanded  # [..., F]

        # Compute sin and cos
        sin_features = torch.sin(phase)  # [..., F]
        cos_features = torch.cos(phase)  # [..., F]

        # Concatenate
        features = torch.cat([sin_features, cos_features], dim=-1)  # [..., 2F]

        return features

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass to generate metric embeddings.

        Args:
            t: Scalar or [...] tensor of time values

        Returns:
            z_t: [..., D_embed] metric embeddings
        """
        # Ensure t is a tensor
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=next(self.parameters()).device)

        # Remember original shape
        original_shape = t.shape

        # Encode time to features
        features = self.encode_time(t)  # [..., input_dim]

        # Flatten for MLP
        features_flat = features.view(-1, features.shape[-1])  # [N, input_dim]

        # Forward through network
        z_flat = self.network(features_flat)  # [N, D_embed]

        # Reshape to original batch dimensions
        z_t = z_flat.view(*original_shape, self.embed_dim)  # [..., D_embed]

        return z_t

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"use_fourier={self.use_fourier_features}, "
            f"num_freqs={self.num_fourier_freqs if self.use_fourier_features else 'N/A'}"
        )


class MetricLearningLoss(nn.Module):
    """Loss for learning the metric structure of time embeddings.

    Constrains ||z_ti - z_tj||^2 to be proportional to dynamic difference
    between frames i and j (measured by optical flow, LPIPS, etc.).
    """

    def __init__(
        self,
        metric_type: str = 'l2',  # 'l2' or 'cosine'
        margin: float = 0.1,
        reduction: str = 'mean'
    ):
        """Initialize metric learning loss.

        Args:
            metric_type: Distance metric in embedding space ('l2' or 'cosine')
            margin: Margin for contrastive learning (optional)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()

        self.metric_type = metric_type
        self.margin = margin
        self.reduction = reduction

    def compute_embedding_distance(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor
    ) -> torch.Tensor:
        """Compute distance in embedding space.

        Args:
            z_i: [B, D] embeddings for time i
            z_j: [B, D] embeddings for time j

        Returns:
            dist: [B] distances
        """
        if self.metric_type == 'l2':
            # Euclidean distance
            return torch.sum((z_i - z_j) ** 2, dim=-1)
        elif self.metric_type == 'cosine':
            # Cosine distance (1 - cosine_similarity)
            cos_sim = torch.sum(z_i * z_j, dim=-1) / (
                torch.norm(z_i, dim=-1) * torch.norm(z_j, dim=-1) + 1e-8
            )
            return 1.0 - cos_sim
        else:
            raise ValueError(f"Unknown metric type: {self.metric_type}")

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        dynamic_diff: torch.Tensor,
        alpha: float = 1.0
    ) -> torch.Tensor:
        """Compute metric learning loss.

        Args:
            z_i: [B, D] embeddings for time i
            z_j: [B, D] embeddings for time j
            dynamic_diff: [B] ground truth dynamic differences (e.g., optical flow norm)
            alpha: Scaling factor for dynamic_diff

        Returns:
            loss: Scalar loss value
        """
        # Compute embedding distance
        embed_dist = self.compute_embedding_distance(z_i, z_j)  # [B]

        # Target distance (scaled dynamic difference)
        target_dist = alpha * dynamic_diff  # [B]

        # MSE loss: (embed_dist - target_dist)^2
        loss = (embed_dist - target_dist) ** 2

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
