#!/usr/bin/env python3
"""
高斯-物理混合压缩模型 (5D扩展版)

扩展自 gaussian_physics_compression.py:
- 使用 5D→7D 物理基 (ExtendedPhysicsBasis5D)
- 集成 SurveyGo 改进: L1 temporal regularization, Charbonnier loss
- 支持多探针压缩 (目标17×压缩比)

Architecture:
    SH(p, light_params) = Σ_j G_j(p) × [U_j @ (coeffs_j @ Φ_physics(light_params))]

    where:
    - G_j(p): Gaussian weight at position p
    - U_j: [27, rank] low-rank matrix for Gaussian j
    - coeffs_j: [rank, 7] time coefficients for Gaussian j
    - Φ_physics: [7] physics basis from ExtendedPhysicsBasis5D

Parameters (K=20, rank=5):
    - Gaussian positions μ_j: 20 × 3 = 60
    - Gaussian scales s_j: 20 × 3 = 60
    - Low-rank matrices U_j: 20 × 27 × 5 = 2,700
    - Time coefficients coeffs_j: 20 × 5 × 7 = 700
    Total: 3,520
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans

from models.physics_low_rank import ExtendedPhysicsBasis5D


class GaussianPhysicsCompression5D(nn.Module):
    """高斯-物理混合压缩模型 (5D扩展版)"""

    def __init__(self, num_gaussians=20, rank=5, sh_dim=27):
        """
        Args:
            num_gaussians: 高斯数量 K
            rank: 低秩秩数
            sh_dim: SH系数维度（默认27）
        """
        super().__init__()

        self.K = num_gaussians
        self.rank = rank
        self.sh_dim = sh_dim

        # 高斯参数
        self.mu = nn.Parameter(torch.randn(num_gaussians, 3) * 0.1)  # [K, 3] 位置
        self.log_scale = nn.Parameter(torch.zeros(num_gaussians, 3))  # [K, 3] 对数尺度

        # 每个高斯的时空表示
        self.U = nn.Parameter(torch.randn(num_gaussians, sh_dim, rank) * 0.1)  # [K, 27, rank]
        self.time_coeffs = nn.Parameter(torch.randn(num_gaussians, rank, 7) * 0.1)  # [K, rank, 7] (7D physics basis)

        # 物理基编码器 (共享)
        self.physics_encoder = ExtendedPhysicsBasis5D()

        print(f"GaussianPhysicsCompression5D initialized:")
        print(f"  Gaussians: {num_gaussians}")
        print(f"  Rank: {rank}")
        print(f"  Physics basis dimension: 7 (5D→7D)")
        print(f"  Parameters: {self.num_params()}")

    def compute_gaussian_weights(self, positions, top_k=3):
        """计算高斯权重 (矢量化版本)

        Args:
            positions: [B, 3] 查询位置
            top_k: 只使用最近的k个高斯

        Returns:
            gaussian_weights: [B, top_k] 归一化权重
            topk_indices: [B, top_k] top-k高斯索引
        """
        B = positions.shape[0]

        # 找到每个位置的top-k最近高斯
        mu_expanded = self.mu.unsqueeze(0)  # [1, K, 3]
        pos_expanded = positions.unsqueeze(1)  # [B, 1, 3]
        distances = torch.norm(mu_expanded - pos_expanded, dim=-1)  # [B, K]

        # 选择top-k
        topk_values, topk_indices = torch.topk(
            distances, k=min(top_k, self.K), largest=False, dim=-1
        )  # [B, top_k]

        # 收集选中的高斯参数
        selected_mu = self.mu[topk_indices]  # [B, top_k, 3]
        selected_scale = torch.exp(self.log_scale[topk_indices])  # [B, top_k, 3]

        # 计算高斯权重 [B, top_k]
        diff = pos_expanded - selected_mu  # [B, top_k, 3]
        weighted_diff = diff / selected_scale  # [B, top_k, 3]
        exponent = -0.5 * torch.sum(weighted_diff ** 2, dim=-1)  # [B, top_k]
        gaussian_weights = torch.exp(exponent)  # [B, top_k]

        # 归一化权重
        gaussian_weights = gaussian_weights / (gaussian_weights.sum(dim=-1, keepdim=True) + 1e-8)

        return gaussian_weights, topk_indices

    def forward(self, positions, light_params_5D, top_k=3):
        """前向传播

        Args:
            positions: [B, 3] 查询位置
            light_params_5D: [B, 5] 光源参数 [zenith, azimuth, intensity, temp, cloud]
            top_k: 只使用最近的k个高斯（加速计算）

        Returns:
            sh_pred: [B, 27] 预测的SH系数
        """
        B = positions.shape[0]

        # 1. 计算物理基函数 [B, 7]
        physics_basis = self.physics_encoder(light_params_5D)

        # 2. 计算高斯权重和选中的高斯索引
        gaussian_weights, topk_indices = self.compute_gaussian_weights(positions, top_k)  # [B, top_k]

        # 3. 矢量化计算top-k高斯的贡献
        selected_U = self.U[topk_indices]  # [B, top_k, 27, rank]
        selected_coeffs = self.time_coeffs[topk_indices]  # [B, top_k, rank, 7]

        # 计算时间权重 Φ_j(t) = coeffs_j @ physics_basis  [B, top_k, rank]
        # selected_coeffs: [B, top_k, rank, 7]
        # physics_basis: [B, 7]
        # einsum: 'bkrc,bc->bkr'
        time_weights = torch.einsum('bkrc,bc->bkr', selected_coeffs, physics_basis)

        # 计算SH贡献 SH_j = U_j @ Φ_j  [B, top_k, 27]
        # selected_U: [B, top_k, 27, rank]
        # time_weights: [B, top_k, rank]
        # einsum: 'bkdr,bkr->bkd'
        sh_contributions = torch.einsum('bkdr,bkr->bkd', selected_U, time_weights)

        # 加权求和 [B, 27]
        sh_pred = torch.einsum('bk,bkd->bd', gaussian_weights, sh_contributions)

        return sh_pred

    def init_from_kmeans(self, probe_positions, light_configs, sh_tensor):
        """使用K-Means初始化高斯位置和U矩阵

        Args:
            probe_positions: [P, 3] 探针位置
            light_configs: [M, 5] 光源配置 [zenith, azimuth, intensity, temp, cloud]
            sh_tensor: [P, M, 27] SH系数张量 (probe × config × SH)
        """
        P = probe_positions.shape[0]
        M = light_configs.shape[1]

        if self.K > P:
            print(f"Warning: K={self.K} > P={P}, setting K={P}")
            self.K = P

        # 1. K-Means聚类探针位置
        print(f"\n初始化 {self.K} 个高斯...")
        kmeans = KMeans(n_clusters=self.K, random_state=42)
        kmeans.fit(probe_positions)

        cluster_centers = kmeans.cluster_centers_  # [K, 3]
        labels = kmeans.labels_  # [P]

        # 设置高斯中心
        self.mu.data = torch.from_numpy(cluster_centers).float().to(self.mu.device)

        # 2. 计算每个簇的平均最近邻距离，用于初始化scale
        for k in range(self.K):
            cluster_points = probe_positions[labels == k]
            if len(cluster_points) > 1:
                center = cluster_centers[k]
                distances = np.linalg.norm(cluster_points - center, axis=1)
                avg_dist = np.mean(distances) + 1e-6
                self.log_scale.data[k] = torch.log(torch.tensor(avg_dist)).to(self.log_scale.device)
            else:
                self.log_scale.data[k] = torch.log(torch.tensor(0.1)).to(self.log_scale.device)

        # 3. 对每个高斯，使用其簇内点的SH数据进行SVD初始化U
        for k in range(self.K):
            cluster_mask = labels == k
            cluster_sh = sh_tensor[cluster_mask, :, :]  # [P_k, M, 27]

            if cluster_sh.shape[0] == 0:
                print(f"  Warning: Gaussian {k} has no points in cluster")
                continue

            # 平均该簇内的所有探针 → [M, 27]
            avg_sh = cluster_sh.mean(dim=0)  # [M, 27]

            # SVD: avg_sh = U @ S @ Vt
            U, S, Vt = np.linalg.svd(avg_sh.numpy(), full_matrices=False)

            # 实际可用的秩
            available_rank = min(avg_sh.shape[0], avg_sh.shape[1])
            effective_rank = min(self.rank, available_rank)

            # U_j = Vt[:effective_rank].T  → [27, effective_rank]
            U_j = Vt[:effective_rank, :].T

            # 如果effective_rank < self.rank，padding
            if effective_rank < self.rank:
                padding = np.random.randn(27, self.rank - effective_rank) * 0.01
                U_j = np.concatenate([U_j, padding], axis=1)  # [27, self.rank]

            self.U.data[k] = torch.from_numpy(U_j).float().to(self.U.device)

            energy = (S[:effective_rank].sum() / S.sum()) * 100
            print(f"  Gaussian {k}: cluster_size={cluster_mask.sum()}, "
                  f"SVD energy={energy:.2f}% (rank={effective_rank}/{self.rank})")

        print(f"✓ Initialized {self.K} Gaussians from K-Means + SVD")

    def compute_temporal_smoothness_loss(self):
        """计算时间平滑正则化 (SurveyGo改进: L1 temporal)

        对time_coeffs施加L1范数，鼓励稀疏和平滑的时间变化

        Returns:
            loss: 标量正则化损失
        """
        # L1 norm on time_coeffs
        # time_coeffs: [K, rank, 7]
        # 计算相邻维度的差异 (这里7维没有明确的时间顺序，所以用整体L1)
        l1_loss = torch.mean(torch.abs(self.time_coeffs))

        return l1_loss

    def num_params(self):
        """计算参数量"""
        # mu: K*3
        # log_scale: K*3
        # U: K*27*rank
        # time_coeffs: K*rank*7 (5D→7D physics basis)
        return self.K * 3 + self.K * 3 + self.K * 27 * self.rank + self.K * self.rank * 7


def charbonnier_loss(pred, target, epsilon=1e-3):
    """Charbonnier loss (SurveyGo改进: 鲁棒损失函数)

    More robust to outliers than MSE, useful for Monte Carlo noise

    L(x) = sqrt((pred - target)^2 + epsilon^2)

    Args:
        pred: [B, ...] 预测值
        target: [B, ...] ground truth
        epsilon: 平滑参数

    Returns:
        loss: 标量损失
    """
    diff = pred - target
    loss = torch.sqrt(diff ** 2 + epsilon ** 2)
    return torch.mean(loss)


class GaussianPhysics5DTrainer:
    """训练器 (集成SurveyGo改进)"""

    def __init__(
        self,
        model,
        lr=1e-3,
        lambda_temporal=0.001,  # L1 temporal regularization weight
        use_charbonnier=True    # Use Charbonnier loss instead of MSE
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.lambda_temporal = lambda_temporal
        self.use_charbonnier = use_charbonnier

        if use_charbonnier:
            print(f"Using Charbonnier loss (SurveyGo improvement)")
        else:
            self.criterion = nn.MSELoss()

        print(f"Temporal L1 regularization: λ={lambda_temporal} (SurveyGo improvement)")

    def train_epoch(self, positions, light_params_5D, sh_gt):
        """训练一个epoch

        Args:
            positions: [B, 3] 位置
            light_params_5D: [B, 5] 光源参数
            sh_gt: [B, 27] ground truth SH

        Returns:
            loss_dict: 损失字典 (包含总损失和各项分量)
        """
        self.model.train()
        self.optimizer.zero_grad()

        # 前向
        sh_pred = self.model(positions, light_params_5D)

        # 重建损失
        if self.use_charbonnier:
            loss_recon = charbonnier_loss(sh_pred, sh_gt)
        else:
            loss_recon = self.criterion(sh_pred, sh_gt)

        # 时间平滑正则化 (SurveyGo: L1 temporal)
        loss_temporal = self.model.compute_temporal_smoothness_loss()

        # 总损失
        loss_total = loss_recon + self.lambda_temporal * loss_temporal

        # 反向
        loss_total.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return {
            'total': loss_total.item(),
            'recon': loss_recon.item(),
            'temporal': loss_temporal.item()
        }

    def evaluate(self, positions, light_params_5D, sh_gt):
        """评估"""
        self.model.eval()

        with torch.no_grad():
            sh_pred = self.model(positions, light_params_5D)

            mae = torch.mean(torch.abs(sh_pred - sh_gt)).item()
            rmse = torch.sqrt(torch.mean((sh_pred - sh_gt) ** 2)).item()

            # Charbonnier loss
            if self.use_charbonnier:
                charbonnier = charbonnier_loss(sh_pred, sh_gt).item()
            else:
                charbonnier = 0.0

        return {
            'mae': mae,
            'rmse': rmse,
            'charbonnier': charbonnier
        }
