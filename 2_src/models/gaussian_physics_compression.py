#!/usr/bin/env python3
"""
高斯-物理混合压缩模型

空间：高斯混合模型（K个3D高斯）
时间：物理引导的低秩基函数（cos/sin太阳高度角）

Architecture:
    F(p,t) = Σ_j F_j(t) * G_j(p)

    where:
    - G_j(p): Gaussian weight at position p
    - F_j(t) = U_j @ Φ(t)
    - Φ(t) = coeffs_j @ [cos(θ), sin(θ), 1]
    - U_j: [27, rank] spatial basis for Gaussian j
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans


def compute_sun_elevation(hour, latitude=40.0):
    """计算太阳高度角（简化版本）

    Args:
        hour: 小时 (0-24)
        latitude: 纬度（度）

    Returns:
        elevation: 高度角（弧度）
    """
    lat_rad = np.deg2rad(latitude)

    # 简化：假设夏至，太阳赤纬23.5度
    declination = np.deg2rad(23.5)

    # 时角（每小时15度）
    hour_angle = np.deg2rad((hour - 12) * 15)

    # 高度角公式
    sin_elevation = (np.sin(lat_rad) * np.sin(declination) +
                     np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle))

    elevation = np.arcsin(np.clip(sin_elevation, -1, 1))

    return max(0, elevation)  # 负值表示太阳在地平线下


class GaussianPhysicsCompression(nn.Module):
    """高斯-物理混合压缩模型"""

    def __init__(self, num_gaussians=50, rank=5, sh_dim=27):
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
        self.time_coeffs = nn.Parameter(torch.randn(num_gaussians, rank, 3) * 0.1)  # [K, rank, 3]

        print(f"GaussianPhysicsCompression initialized:")
        print(f"  Gaussians: {num_gaussians}")
        print(f"  Rank: {rank}")
        print(f"  Parameters: {self.num_params()}")

    def gaussian_weight(self, positions, gaussian_idx):
        """计算高斯权重

        Args:
            positions: [B, 3] 查询位置
            gaussian_idx: 标量，高斯索引

        Returns:
            weight: [B] 权重
        """
        mu = self.mu[gaussian_idx]  # [3]
        scale = torch.exp(self.log_scale[gaussian_idx])  # [3]

        # 各向异性高斯
        diff = positions - mu  # [B, 3]
        weighted_diff = diff / scale  # [B, 3]
        exponent = -0.5 * torch.sum(weighted_diff ** 2, dim=-1)  # [B]

        weight = torch.exp(exponent)  # [B]

        return weight

    def compute_physics_basis(self, sun_elevation):
        """计算物理基函数

        Args:
            sun_elevation: [B] 太阳高度角（弧度）

        Returns:
            basis: [B, 3] 基函数 [cos(θ), sin(θ), 1]
        """
        basis = torch.stack([
            torch.cos(sun_elevation),
            torch.sin(sun_elevation),
            torch.ones_like(sun_elevation)
        ], dim=-1)  # [B, 3]

        return basis

    def forward(self, positions, sun_elevation, top_k=3):
        """前向传播（矢量化版本）

        Args:
            positions: [B, 3] 查询位置
            sun_elevation: [B] 太阳高度角（弧度）
            top_k: 只使用最近的k个高斯（加速计算）

        Returns:
            sh_pred: [B, 27] 预测的SH系数
        """
        B = positions.shape[0]
        device = positions.device

        # 1. 计算物理基函数 [B, 3]
        basis = self.compute_physics_basis(sun_elevation)

        # 2. 找到每个位置的top-k最近高斯
        mu_expanded = self.mu.unsqueeze(0)  # [1, K, 3]
        pos_expanded = positions.unsqueeze(1)  # [B, 1, 3]
        distances = torch.norm(mu_expanded - pos_expanded, dim=-1)  # [B, K]

        # 选择top-k
        topk_values, topk_indices = torch.topk(distances, k=min(top_k, self.K), largest=False, dim=-1)  # [B, top_k]

        # 3. 矢量化计算top-k高斯的贡献
        # 收集选中的高斯参数
        selected_mu = self.mu[topk_indices]  # [B, top_k, 3]
        selected_scale = torch.exp(self.log_scale[topk_indices])  # [B, top_k, 3]
        selected_U = self.U[topk_indices]  # [B, top_k, 27, rank]
        selected_coeffs = self.time_coeffs[topk_indices]  # [B, top_k, rank, 3]

        # 计算高斯权重 [B, top_k]
        diff = pos_expanded - selected_mu  # [B, top_k, 3]
        weighted_diff = diff / selected_scale  # [B, top_k, 3]
        exponent = -0.5 * torch.sum(weighted_diff ** 2, dim=-1)  # [B, top_k]
        gaussian_weights = torch.exp(exponent)  # [B, top_k]

        # 归一化权重（可选）
        gaussian_weights = gaussian_weights / (gaussian_weights.sum(dim=-1, keepdim=True) + 1e-8)  # [B, top_k]

        # 计算时间权重 Φ_j(t) = coeffs_j @ basis  [B, top_k, rank]
        # selected_coeffs: [B, top_k, rank, 3]
        # basis: [B, 3]
        # einsum: 'bkrc,bc->bkr'
        time_weights = torch.einsum('bkrc,bc->bkr', selected_coeffs, basis)  # [B, top_k, rank]

        # 计算SH贡献 SH_j = U_j @ Φ_j  [B, top_k, 27]
        # selected_U: [B, top_k, 27, rank]
        # time_weights: [B, top_k, rank]
        # einsum: 'bkdr,bkr->bkd'
        sh_contributions = torch.einsum('bkdr,bkr->bkd', selected_U, time_weights)  # [B, top_k, 27]

        # 加权求和 [B, 27]
        # gaussian_weights: [B, top_k]
        # sh_contributions: [B, top_k, 27]
        sh_pred = torch.einsum('bk,bkd->bd', gaussian_weights, sh_contributions)  # [B, 27]

        return sh_pred

    def init_from_kmeans(self, probe_positions, sh_matrix_all_times):
        """使用K-Means初始化高斯位置和U矩阵

        Args:
            probe_positions: [N, 3] 探针位置
            sh_matrix_all_times: [T, N, 27] 所有时刻的SH系数
        """
        N = probe_positions.shape[0]
        T = sh_matrix_all_times.shape[0]

        if self.K > N:
            print(f"Warning: K={self.K} > N={N}, setting K={N}")
            self.K = N

        # 1. K-Means聚类探针位置
        kmeans = KMeans(n_clusters=self.K, random_state=42)
        kmeans.fit(probe_positions)

        cluster_centers = kmeans.cluster_centers_  # [K, 3]
        labels = kmeans.labels_  # [N]

        # 设置高斯中心
        self.mu.data = torch.from_numpy(cluster_centers).float()

        # 2. 计算每个簇的平均最近邻距离，用于初始化scale
        for k in range(self.K):
            cluster_points = probe_positions[labels == k]
            if len(cluster_points) > 1:
                # 计算簇内点的平均距离
                center = cluster_centers[k]
                distances = np.linalg.norm(cluster_points - center, axis=1)
                avg_dist = np.mean(distances) + 1e-6
                self.log_scale.data[k] = torch.log(torch.tensor(avg_dist))
            else:
                self.log_scale.data[k] = torch.log(torch.tensor(0.1))

        # 3. 对每个高斯，使用其簇内点的SH数据进行SVD初始化U
        for k in range(self.K):
            cluster_mask = labels == k
            cluster_sh = sh_matrix_all_times[:, cluster_mask, :]  # [T, N_k, 27]

            if cluster_sh.shape[1] == 0:
                continue

            # 平均该簇内的所有探针
            avg_sh = cluster_sh.mean(dim=1)  # [T, 27]

            # SVD
            avg_sh_np = avg_sh.numpy()
            U, S, Vt = np.linalg.svd(avg_sh_np, full_matrices=False)

            # 实际可用的秩
            available_rank = min(avg_sh_np.shape[0], avg_sh_np.shape[1])
            effective_rank = min(self.rank, available_rank)

            # U_j = Vt[:effective_rank].T
            U_j = Vt[:effective_rank, :].T  # [27, effective_rank]

            # 如果effective_rank < self.rank，需要padding
            if effective_rank < self.rank:
                padding = np.random.randn(27, self.rank - effective_rank) * 0.01
                U_j = np.concatenate([U_j, padding], axis=1)  # [27, self.rank]

            self.U.data[k] = torch.from_numpy(U_j).float()

            energy = (S[:effective_rank].sum() / S.sum()) * 100
            print(f"  Gaussian {k}: cluster_size={cluster_mask.sum()}, SVD energy={energy:.2f}% (rank={effective_rank}/{self.rank})")

        print(f"✓ Initialized {self.K} Gaussians from K-Means + SVD")

    def num_params(self):
        """计算参数量"""
        # mu: K*3
        # log_scale: K*3
        # U: K*27*rank
        # time_coeffs: K*rank*3
        return self.K * 3 + self.K * 3 + self.K * 27 * self.rank + self.K * self.rank * 3


class GaussianPhysicsTrainer:
    """训练器"""

    def __init__(self, model, lr=1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_epoch(self, positions, sun_elevations, sh_gt):
        """训练一个epoch

        Args:
            positions: [B, 3] 位置
            sun_elevations: [B] 太阳高度角
            sh_gt: [B, 27] ground truth SH

        Returns:
            loss: 标量损失
        """
        self.model.train()
        self.optimizer.zero_grad()

        # 前向
        sh_pred = self.model(positions, sun_elevations)

        # 损失
        loss = self.criterion(sh_pred, sh_gt)

        # 反向
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.item()

    def evaluate(self, positions, sun_elevations, sh_gt):
        """评估"""
        self.model.eval()

        with torch.no_grad():
            sh_pred = self.model(positions, sun_elevations)

            mae = torch.mean(torch.abs(sh_pred - sh_gt)).item()
            rmse = torch.sqrt(torch.mean((sh_pred - sh_gt) ** 2)).item()

        return {'mae': mae, 'rmse': rmse}
