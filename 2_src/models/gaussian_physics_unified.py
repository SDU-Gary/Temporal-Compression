#!/usr/bin/env python3
"""
Unified Gaussian-Physics compression with light-set input and dynamic basis modulation.

Core idea:
    SH(p, L) = Σ_j G_j(p) × [U_j(Z) @ (coeffs_j @ Z)]

Where:
  - L is a set of lights, each described by a fixed-length descriptor.
  - Z is a pooled light embedding (sum pooling, linear in intensity).
  - U_j(Z) is FiLM-modulated low-rank basis (dynamic basis).

This model is schema-agnostic: it only depends on descriptor dimension and mask.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class LightSetEncoder(nn.Module):
    """Encode a variable-size set of lights into a single latent vector Z.

    Intensity decoupling ensures linearity:
        Embed(L_i) = intensity_i * MLP(features_i)
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 32,
        intensity_dim: int = 1,
        intensity_offset: int = 0,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.intensity_dim = intensity_dim
        self.intensity_offset = intensity_offset
        self.feature_dim = max(0, input_dim - intensity_dim)
        self._intensity_rgb = intensity_dim == 3

        if intensity_offset < 0 or intensity_dim < 0:
            raise ValueError("intensity_offset and intensity_dim must be non-negative")
        if intensity_offset + intensity_dim > input_dim:
            raise ValueError(
                f"intensity_offset + intensity_dim exceeds input_dim: "
                f"{intensity_offset}+{intensity_dim}>{input_dim}"
            )
        if intensity_dim not in (0, 1, 3):
            raise ValueError("intensity_dim must be 0, 1, or 3")

        if self.feature_dim == 0:
            if self._intensity_rgb:
                self.base_embed = nn.Parameter(torch.randn(3, embed_dim) * 0.01)
            else:
                self.base_embed = nn.Parameter(torch.randn(embed_dim) * 0.01)
            self.mlp = None
        else:
            layers = []
            in_dim = self.feature_dim
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                in_dim = hidden_dim
            out_dim = embed_dim * 3 if self._intensity_rgb else embed_dim
            layers.append(nn.Linear(in_dim, out_dim))
            self.mlp = nn.Sequential(*layers)

    def forward(self, light_desc: torch.Tensor, light_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode light descriptors.

        Args:
            light_desc: [B, N, D] or [B, D] light descriptors.
            light_mask: [B, N] mask (1 for valid light, 0 for padding).

        Returns:
            Z: [B, embed_dim]
        """
        if light_desc.dim() == 2:
            light_desc = light_desc.unsqueeze(1)

        B, N, D = light_desc.shape
        if light_mask is None:
            light_mask = torch.ones(B, N, device=light_desc.device, dtype=light_desc.dtype)

        if self.intensity_dim > 0:
            start = self.intensity_offset
            end = start + self.intensity_dim
            intensity = light_desc[..., start:end]
            features = torch.cat([light_desc[..., :start], light_desc[..., end:]], dim=-1)
        else:
            intensity = None
            features = light_desc

        if self.mlp is None:
            if self._intensity_rgb:
                embed = self.base_embed.view(1, 1, 3, -1).expand(B, N, 3, -1)
            else:
                embed = self.base_embed.view(1, 1, -1).expand(B, N, -1)
        else:
            embed = self.mlp(features)

        # Intensity decoupling for linearity
        if intensity is not None:
            if self._intensity_rgb:
                # embed: [B, N, 3*E] -> [B, N, 3, E]
                if embed.dim() == 3:
                    embed = embed.view(B, N, 3, self.embed_dim)
                embed = embed * intensity.unsqueeze(-1)
                embed = embed.sum(dim=2)
            else:
                embed = embed * intensity

        # Apply mask and sum pooling
        mask = light_mask.unsqueeze(-1)
        embed = embed * mask
        z = embed.sum(dim=1)
        return z


class GaussianPhysicsCompressionUnified(nn.Module):
    """Unified Gaussian-Physics model with dynamic basis modulation."""

    def __init__(
        self,
        num_gaussians: int = 20,
        rank: int = 5,
        sh_dim: int = 27,
        light_dim: int = 5,
        embed_dim: int = 32,
        intensity_dim: int = 1,
        intensity_offset: int = 1,
        film_hidden: int = 64,
        enable_film: bool = True,
    ) -> None:
        super().__init__()

        self.K = num_gaussians
        self.rank = rank
        self.sh_dim = sh_dim
        self.enable_film = enable_film

        # Gaussian parameters
        self.mu = nn.Parameter(torch.randn(num_gaussians, 3) * 0.1)
        self.log_scale = nn.Parameter(torch.zeros(num_gaussians, 3))

        # Low-rank basis and coefficients
        self.U = nn.Parameter(torch.randn(num_gaussians, sh_dim, rank) * 0.1)
        self.coeffs = nn.Parameter(torch.randn(num_gaussians, rank, embed_dim) * 0.1)
        # L0 branch parameters (R,G,B)
        self.U_l0 = nn.Parameter(torch.randn(num_gaussians, 3, rank) * 0.05)
        self.coeffs_l0 = nn.Parameter(torch.randn(num_gaussians, rank, embed_dim) * 0.05)

        # Light encoder
        self.light_encoder = LightSetEncoder(
            input_dim=light_dim,
            embed_dim=embed_dim,
            intensity_dim=intensity_dim,
            intensity_offset=intensity_offset,
        )

        # FiLM modulation for dynamic basis
        self.gamma = nn.Sequential(
            nn.Linear(embed_dim, film_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(film_hidden, rank),
        )
        self.beta = nn.Sequential(
            nn.Linear(embed_dim, film_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(film_hidden, rank),
        )

        # Initialize FiLM to near-identity
        for layer in [self.gamma[-1], self.beta[-1]]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def compute_gaussian_routing(self, positions: torch.Tensor, top_k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        B = positions.shape[0]
        mu_expanded = self.mu.unsqueeze(0)
        pos_expanded = positions.unsqueeze(1)
        # NOTE: 计算平方距离避免开根号，保持数值稳定性和效率。
        diff_all = mu_expanded - pos_expanded
        distances = torch.sum(diff_all * diff_all, dim=-1)

        topk_values, topk_indices = torch.topk(
            distances, k=min(top_k, self.K), largest=False, dim=-1
        )

        selected_mu = self.mu[topk_indices]
        selected_scale = torch.exp(self.log_scale[topk_indices])

        diff = pos_expanded - selected_mu
        weighted_diff = diff / selected_scale
        exponent = -0.5 * torch.sum(weighted_diff ** 2, dim=-1)
        weights = torch.exp(exponent)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        return weights, topk_indices

    def compute_gaussian_weights(self, positions: torch.Tensor, top_k: int = 3):
        return self.compute_gaussian_routing(positions, top_k)

    def forward_with_routing(
        self,
        routing: Tuple[torch.Tensor, torch.Tensor],
        light_params: torch.Tensor,
        light_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gaussian_weights, topk_indices = routing

        # 1) Encode light set
        z = self.light_encoder(light_params, light_mask)  # [B, embed_dim]

        # 2) Select parameters
        selected_U = self.U[topk_indices]  # [B, K', sh_dim, rank]
        selected_coeffs = self.coeffs[topk_indices]  # [B, K', rank, embed_dim]
        selected_U_l0 = self.U_l0[topk_indices]  # [B, K', 3, rank]
        selected_coeffs_l0 = self.coeffs_l0[topk_indices]  # [B, K', rank, embed_dim]

        # 3) Compute rank weights
        # time_weights: [B, K', rank]
        # TODO(perf): 这里与后续多次 einsum 是主要算子热点，适合优先尝试 torch.compile。
        time_weights = torch.einsum('bkrl,bl->bkr', selected_coeffs, z)
        time_weights_l0 = torch.einsum('bkrl,bl->bkr', selected_coeffs_l0, z)

        # 4) Dynamic basis modulation (FiLM)
        if self.enable_film:
            gamma = self.gamma(z).view(-1, 1, 1, self.rank)
            beta = self.beta(z).view(-1, 1, 1, self.rank)
            selected_U = selected_U * (1.0 + gamma) + beta
            selected_U_l0 = selected_U_l0 * (1.0 + gamma) + beta

        # 5) SH contributions and sum
        sh_contrib = torch.einsum('bkdr,bkr->bkd', selected_U, time_weights)
        sh_pred = torch.einsum('bk,bkd->bd', gaussian_weights, sh_contrib)

        # L0 branch (R,G,B)
        l0_contrib = torch.einsum('bkdr,bkr->bkd', selected_U_l0, time_weights_l0)
        l0_pred = torch.einsum('bk,bkd->bd', gaussian_weights, l0_contrib)
        l0_pred = torch.nn.functional.softplus(l0_pred)

        if self.sh_dim >= 27:
            # TODO(perf): clone + 三次列写回会产生额外内存流量；可考虑拼接式构造避免 copy。
            sh_pred = sh_pred.clone()
            sh_pred[:, 0] = l0_pred[:, 0]
            sh_pred[:, 9] = l0_pred[:, 1]
            sh_pred[:, 18] = l0_pred[:, 2]

        return sh_pred

    def forward(
        self,
        positions: torch.Tensor,
        light_params: torch.Tensor,
        top_k: int = 3,
        light_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        routing = self.compute_gaussian_routing(positions, top_k)
        return self.forward_with_routing(routing, light_params, light_mask)

    def set_film_enabled(self, enabled: bool) -> None:
        self.enable_film = enabled

    def compute_temporal_smoothness_loss(self) -> torch.Tensor:
        # Optional: L1 regularization on coeffs to encourage sparsity
        return torch.mean(torch.abs(self.coeffs)) + torch.mean(torch.abs(self.coeffs_l0))

    def init_from_kmeans(self, probe_positions, light_configs, sh_tensor):
        """Initialize Gaussians with KMeans and SVD on SH tensor."""
        import numpy as np
        from sklearn.cluster import KMeans

        P = probe_positions.shape[0]
        if self.K > P:
            self.K = P

        kmeans = KMeans(n_clusters=self.K, random_state=42)
        kmeans.fit(probe_positions)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        self.mu.data = torch.from_numpy(centers).float().to(self.mu.device)

        for k in range(self.K):
            cluster_points = probe_positions[labels == k]
            if len(cluster_points) > 1:
                center = centers[k]
                distances = np.linalg.norm(cluster_points - center, axis=1)
                avg_dist = np.mean(distances) + 1e-6
                self.log_scale.data[k] = torch.log(torch.tensor(avg_dist)).to(self.log_scale.device)
            else:
                self.log_scale.data[k] = torch.log(torch.tensor(0.1)).to(self.log_scale.device)

        # Initialize U with SVD over mean SH per cluster
        for k in range(self.K):
            cluster_mask = labels == k
            cluster_sh = sh_tensor[cluster_mask, :, :]  # [P_k, M, 27]
            if cluster_sh.shape[0] == 0:
                continue

            avg_sh = cluster_sh.mean(dim=0)  # [M, 27]
            U, S, Vt = np.linalg.svd(avg_sh.numpy(), full_matrices=False)

            available_rank = min(avg_sh.shape[0], avg_sh.shape[1])
            effective_rank = min(self.rank, available_rank)

            U_j = Vt[:effective_rank, :].T
            if effective_rank < self.rank:
                padding = np.random.randn(self.sh_dim, self.rank - effective_rank) * 0.01
                U_j = np.concatenate([U_j, padding], axis=1)
            self.U.data[k] = torch.from_numpy(U_j).float().to(self.U.device)

            # Initialize L0 basis (R,G,B) from L0 columns
            l0_cols = avg_sh[:, [0, 9, 18]]
            Ul0, Sl0, Vt0 = np.linalg.svd(l0_cols.numpy(), full_matrices=False)
            eff_l0 = min(self.rank, Vt0.shape[0])
            U_l0 = Vt0[:eff_l0, :].T
            if eff_l0 < self.rank:
                padding = np.random.randn(3, self.rank - eff_l0) * 0.01
                U_l0 = np.concatenate([U_l0, padding], axis=1)
            self.U_l0.data[k] = torch.from_numpy(U_l0).float().to(self.U_l0.device)
