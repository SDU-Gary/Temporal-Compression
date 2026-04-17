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

from typing import Literal, Sequence, Tuple

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
        light_encoder_mode: str = "normal",
        bypass_feature_pairs: Sequence[Sequence[int]] | None = None,
        bypass_feature_norm_mean: Sequence[float] | None = None,
        bypass_feature_norm_std: Sequence[float] | None = None,
        contraction_mode: Literal["legacy", "fused"] = "fused",
    ) -> None:
        super().__init__()

        self.K = num_gaussians
        self.rank = rank
        self.sh_dim = sh_dim
        self.embed_dim = embed_dim
        self.enable_film = enable_film
        self.light_dim = light_dim
        self.contraction_mode = str(contraction_mode).strip().lower()
        if self.contraction_mode not in {"legacy", "fused"}:
            raise ValueError(
                f"Unsupported contraction_mode: {contraction_mode}. "
                "Expected one of {'legacy', 'fused'}."
            )

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
        self.light_encoder_mode = str(light_encoder_mode).strip().lower()
        if self.light_encoder_mode not in {
            "normal",
            "bypass_fixed",
            "bypass_linear",
            "bypass_mlp_16_32",
        }:
            raise ValueError(f"Unsupported light_encoder_mode: {light_encoder_mode}")

        if bypass_feature_pairs is None:
            parsed_pairs = [(0, 5), (1, 6)]
        else:
            parsed_pairs = []
            for item in bypass_feature_pairs:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    raise ValueError(
                        "bypass_feature_pairs must be a sequence of [light_index, feature_index]"
                    )
                light_idx = int(item[0])
                feat_idx = int(item[1])
                if light_idx < 0:
                    raise ValueError("bypass_feature_pairs light_index must be >= 0")
                if feat_idx < 0 or feat_idx >= int(light_dim):
                    raise ValueError(
                        f"bypass_feature_pairs feature_index out of range: {feat_idx}, light_dim={light_dim}"
                    )
                parsed_pairs.append((light_idx, feat_idx))
            if not parsed_pairs:
                raise ValueError("bypass_feature_pairs cannot be empty")
        self.bypass_feature_pairs = tuple(parsed_pairs)
        self._bypass_num_features = len(self.bypass_feature_pairs)

        if bypass_feature_norm_mean is None:
            mean_vals = [0.0] * self._bypass_num_features
        else:
            mean_vals = [float(x) for x in bypass_feature_norm_mean]
            if len(mean_vals) != self._bypass_num_features:
                raise ValueError(
                    f"bypass_feature_norm_mean length mismatch: {len(mean_vals)} "
                    f"!= num_features {self._bypass_num_features}"
                )

        if bypass_feature_norm_std is None:
            std_vals = [1.0] * self._bypass_num_features
        else:
            std_vals = [float(x) for x in bypass_feature_norm_std]
            if len(std_vals) != self._bypass_num_features:
                raise ValueError(
                    f"bypass_feature_norm_std length mismatch: {len(std_vals)} "
                    f"!= num_features {self._bypass_num_features}"
                )
        std_vals = [max(1e-6, abs(x)) for x in std_vals]

        self.bypass_feature_norm_mean = tuple(mean_vals)
        self.bypass_feature_norm_std = tuple(std_vals)
        if self.light_encoder_mode == "bypass_linear":
            self.bypass_proj = nn.Linear(self._bypass_num_features, embed_dim)
            self.bypass_mlp = None
        elif self.light_encoder_mode == "bypass_mlp_16_32":
            self.bypass_proj = None
            self.bypass_mlp = nn.Sequential(
                nn.Linear(self._bypass_num_features, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, embed_dim),
            )
        else:
            self.bypass_proj = None
            self.bypass_mlp = None

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

    def _encode_light_set(
        self,
        light_params: torch.Tensor,
        light_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.light_encoder_mode == "normal":
            return self.light_encoder(light_params, light_mask)

        if light_params.dim() == 2:
            light_params = light_params.unsqueeze(1)
        bsz, n_lights, _ = light_params.shape

        if light_mask is None:
            light_mask = torch.ones(
                (bsz, n_lights),
                device=light_params.device,
                dtype=light_params.dtype,
            )
        else:
            light_mask = light_mask.to(device=light_params.device, dtype=light_params.dtype)

        cols = []
        for light_idx, feat_idx in self.bypass_feature_pairs:
            src_light = min(max(int(light_idx), 0), int(n_lights) - 1)
            vals = light_params[:, src_light, int(feat_idx)]
            vals = vals * light_mask[:, src_light]
            cols.append(vals)
        features = torch.stack(cols, dim=-1)

        mean = torch.tensor(
            self.bypass_feature_norm_mean,
            device=features.device,
            dtype=features.dtype,
        )
        std = torch.tensor(
            self.bypass_feature_norm_std,
            device=features.device,
            dtype=features.dtype,
        )
        features = (features - mean) / std

        if self.light_encoder_mode == "bypass_fixed":
            repeats = (self.embed_dim + self._bypass_num_features - 1) // self._bypass_num_features
            return features.repeat(1, repeats)[:, : self.embed_dim]

        if self.light_encoder_mode == "bypass_linear":
            if self.bypass_proj is None:
                raise RuntimeError("bypass_proj is not initialized for bypass_linear mode")
            return self.bypass_proj(features)

        if self.light_encoder_mode == "bypass_mlp_16_32":
            if self.bypass_mlp is None:
                raise RuntimeError("bypass_mlp is not initialized for bypass_mlp_16_32 mode")
            return self.bypass_mlp(features)

        raise RuntimeError(f"Unsupported light_encoder_mode in _encode_light_set: {self.light_encoder_mode}")

    def _contract_selected_params(
        self,
        gaussian_weights: torch.Tensor,
        z: torch.Tensor,
        selected_u: torch.Tensor,
        selected_coeffs: torch.Tensor,
        selected_u_l0: torch.Tensor,
        selected_coeffs_l0: torch.Tensor,
        contraction_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode = str(contraction_mode).strip().lower()
        if mode == "legacy":
            time_weights = torch.einsum("bkrl,bl->bkr", selected_coeffs, z)
            time_weights_l0 = torch.einsum("bkrl,bl->bkr", selected_coeffs_l0, z)
            sh_contrib = torch.einsum("bkdr,bkr->bkd", selected_u, time_weights)
            l0_contrib = torch.einsum("bkdr,bkr->bkd", selected_u_l0, time_weights_l0)
            sh_pred = torch.einsum("bk,bkd->bd", gaussian_weights, sh_contrib)
            l0_pred = torch.einsum("bk,bkd->bd", gaussian_weights, l0_contrib)
            return sh_pred, l0_pred

        if mode != "fused":
            raise ValueError(f"Unsupported contraction_mode: {contraction_mode}")

        bsz, k_count = int(gaussian_weights.shape[0]), int(gaussian_weights.shape[1])
        rank = int(selected_u.shape[-1])
        embed_dim = int(z.shape[-1])
        sh_dim = int(selected_u.shape[-2])

        coeffs_stack = torch.stack(
            [selected_coeffs, selected_coeffs_l0],
            dim=2,
        )  # [B, K, 2, rank, embed_dim]
        coeffs_flat = coeffs_stack.contiguous().reshape(bsz * k_count * 2, rank, embed_dim)
        z_expand = (
            z[:, None, None, :]
            .expand(bsz, k_count, 2, embed_dim)
            .contiguous()
            .reshape(bsz * k_count * 2, embed_dim, 1)
        )
        time_weights = torch.bmm(coeffs_flat, z_expand).reshape(bsz, k_count, 2, rank)
        sh_time = time_weights[:, :, 0, :]
        l0_time = time_weights[:, :, 1, :]

        sh_contrib = torch.bmm(
            selected_u.contiguous().reshape(bsz * k_count, sh_dim, rank),
            sh_time.contiguous().reshape(bsz * k_count, rank, 1),
        ).reshape(bsz, k_count, sh_dim)
        l0_contrib = torch.bmm(
            selected_u_l0.contiguous().reshape(bsz * k_count, 3, rank),
            l0_time.contiguous().reshape(bsz * k_count, rank, 1),
        ).reshape(bsz, k_count, 3)

        contrib_cat = torch.cat([sh_contrib, l0_contrib], dim=-1)
        pred_cat = torch.bmm(
            gaussian_weights.contiguous().unsqueeze(1),
            contrib_cat.contiguous(),
        ).squeeze(1)
        sh_pred = pred_cat[:, :sh_dim]
        l0_pred = pred_cat[:, sh_dim:]
        return sh_pred, l0_pred

    def compute_gaussian_routing(
        self,
        positions: torch.Tensor,
        top_k: int = 3,
        training_soft_routing: bool = False,
        routing_temperature: float = 1.0,
        routing_soft_topk: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = positions.shape[0]
        mu_expanded = self.mu.unsqueeze(0)
        pos_expanded = positions.unsqueeze(1)
        # NOTE: 计算平方距离避免开根号，保持数值稳定性和效率。
        diff_all = mu_expanded - pos_expanded
        distances = torch.sum(diff_all * diff_all, dim=-1)

        if training_soft_routing:
            if routing_soft_topk is None or int(routing_soft_topk) <= 0:
                # Full-K soft routing: avoid top-k truncation for a smooth dense path.
                selected_mu = self.mu.unsqueeze(0).expand(B, -1, -1)
                selected_scale = torch.exp(self.log_scale).unsqueeze(0).expand(B, -1, -1)
                diff = pos_expanded - selected_mu
                weighted_diff = diff / selected_scale
                exponent = -0.5 * torch.sum(weighted_diff ** 2, dim=-1)
                temperature = max(1e-6, float(routing_temperature))
                logits = exponent / temperature
                weights = torch.softmax(logits, dim=-1)
                all_indices = torch.arange(
                    int(self.K),
                    device=positions.device,
                    dtype=torch.long,
                ).unsqueeze(0).expand(B, -1)
                return weights, all_indices
            soft_k = min(self.K, max(int(routing_soft_topk), int(top_k)))
            topk_values, topk_indices = torch.topk(
                distances, k=soft_k, largest=False, dim=-1
            )
            selected_mu = self.mu[topk_indices]
            selected_scale = torch.exp(self.log_scale[topk_indices])
            diff = pos_expanded - selected_mu
            weighted_diff = diff / selected_scale
            exponent = -0.5 * torch.sum(weighted_diff ** 2, dim=-1)
            temperature = max(1e-6, float(routing_temperature))
            logits = exponent / temperature
            weights = torch.softmax(logits, dim=-1)
            return weights, topk_indices

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

    def compute_gaussian_weights(
        self,
        positions: torch.Tensor,
        top_k: int = 3,
        training_soft_routing: bool = False,
        routing_temperature: float = 1.0,
        routing_soft_topk: int | None = None,
    ):
        return self.compute_gaussian_routing(
            positions,
            top_k=top_k,
            training_soft_routing=training_soft_routing,
            routing_temperature=routing_temperature,
            routing_soft_topk=routing_soft_topk,
        )

    def forward_with_routing(
        self,
        routing: Tuple[torch.Tensor, torch.Tensor],
        light_params: torch.Tensor,
        light_mask: torch.Tensor | None = None,
        *,
        param_select_mode: Literal["gather", "dense_masked"] = "gather",
        contraction_mode: Literal["legacy", "fused"] | None = None,
    ) -> torch.Tensor:
        gaussian_weights, topk_indices = routing

        # 1) Encode light set
        z = self._encode_light_set(light_params, light_mask)  # [B, embed_dim]

        contract_mode = self.contraction_mode if contraction_mode is None else str(contraction_mode).strip().lower()
        if contract_mode not in {"legacy", "fused"}:
            raise ValueError(f"Unsupported contraction_mode: {contraction_mode}")

        mode = str(param_select_mode).strip().lower()
        if mode not in {"gather", "dense_masked"}:
            raise ValueError(f"Unsupported param_select_mode: {param_select_mode}")

        if mode == "dense_masked":
            batch_size = int(gaussian_weights.shape[0])
            dense_weights = torch.zeros(
                batch_size,
                int(self.K),
                dtype=gaussian_weights.dtype,
                device=gaussian_weights.device,
            )
            dense_weights.scatter_add_(1, topk_indices.long(), gaussian_weights)
            selected_u = self.U.unsqueeze(0).expand(batch_size, -1, -1, -1)
            selected_coeffs = self.coeffs.unsqueeze(0).expand(batch_size, -1, -1, -1)
            selected_u_l0 = self.U_l0.unsqueeze(0).expand(batch_size, -1, -1, -1)
            selected_coeffs_l0 = self.coeffs_l0.unsqueeze(0).expand(batch_size, -1, -1, -1)
            active_weights = dense_weights
        else:
            # 2) Select parameters
            selected_U = self.U[topk_indices]  # [B, K', sh_dim, rank]
            selected_coeffs = self.coeffs[topk_indices]  # [B, K', rank, embed_dim]
            selected_U_l0 = self.U_l0[topk_indices]  # [B, K', 3, rank]
            selected_coeffs_l0 = self.coeffs_l0[topk_indices]  # [B, K', rank, embed_dim]
            selected_u = selected_U
            selected_u_l0 = selected_U_l0
            active_weights = gaussian_weights

        # Dynamic basis modulation (FiLM)
        if self.enable_film:
            gamma = self.gamma(z).view(-1, 1, 1, self.rank)
            beta = self.beta(z).view(-1, 1, 1, self.rank)
            selected_u = selected_u * (1.0 + gamma) + beta
            selected_u_l0 = selected_u_l0 * (1.0 + gamma) + beta

        sh_pred, l0_pred = self._contract_selected_params(
            active_weights,
            z,
            selected_u,
            selected_coeffs,
            selected_u_l0,
            selected_coeffs_l0,
            contraction_mode=contract_mode,
        )
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
        training_soft_routing: bool = False,
        routing_temperature: float = 1.0,
        routing_soft_topk: int | None = None,
    ) -> torch.Tensor:
        routing = self.compute_gaussian_routing(
            positions,
            top_k=top_k,
            training_soft_routing=training_soft_routing,
            routing_temperature=routing_temperature,
            routing_soft_topk=routing_soft_topk,
        )
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
