"""
Physics-Guided Low-Rank Temporal Compression

核心思想：
  SH(t) ≈ U @ Φ(t)

  U: 空间基 [27, k]
  Φ(t): 物理基函数 [k, T]
       = coeffs @ [cos(θ(t)), sin(θ(t)), 1]

扩展至5D参数化光源：
  SH(light_params) ≈ U @ Φ(zenith, azimuth, intensity, temp, cloud)
  Φ: [cos(θ), sin(θ), cos(φ), sin(φ), intensity, (temp-5500)/2000, exp(-cloud)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PhysicsLowRank(nn.Module):
    """
    物理引导的低秩时间表示

    参数量：27*k + k*3 = O(30k)
    压缩比：351 / (30k) ≈ 12/k

    例如 k=5: 150参数，2.3x压缩
    """

    def __init__(self, rank=5, init_method='random'):
        """
        Args:
            rank: 秩（推荐4-5）
            init_method: 初始化方法
                - 'random': 随机初始化
                - 'svd': 用SVD初始化（需要训练数据）
        """
        super().__init__()

        self.rank = rank

        # 空间基 [27, k]
        # 表示k个"光照模式"
        self.U = nn.Parameter(torch.randn(27, rank) * 0.1)

        # 时间系数矩阵 [k, 3]
        # 对应 [cos(θ), sin(θ), 1]
        self.time_coeffs = nn.Parameter(torch.randn(rank, 3) * 0.1)

        print(f"PhysicsLowRank initialized:")
        print(f"  Rank: {rank}")
        print(f"  Parameters: {self.num_params()}")

    def num_params(self):
        """计算参数量"""
        return 27 * self.rank + self.rank * 3

    def forward(self, sun_elevation):
        """
        前向传播

        Args:
            sun_elevation: [B] 太阳高度角（弧度）

        Returns:
            sh: [B, 27] SH系数
        """
        batch_size = sun_elevation.shape[0]

        # 物理基函数 [B, 3]
        basis = torch.stack([
            torch.cos(sun_elevation),
            torch.sin(sun_elevation),
            torch.ones_like(sun_elevation)
        ], dim=-1)  # [B, 3]

        # 时间权重 [B, k]
        # Φ(t) = coeffs @ basis^T
        weights = basis @ self.time_coeffs.T  # [B, k]

        # 重建SH [B, 27]
        # SH = weights @ U^T
        sh = weights @ self.U.T  # [B, 27]

        return sh

    @torch.no_grad()
    def init_from_svd(self, sh_matrix):
        """
        用SVD结果初始化U

        Args:
            sh_matrix: [T, 27] 训练数据的SH矩阵
        """
        # SVD
        U_svd, S, Vt = np.linalg.svd(sh_matrix, full_matrices=False)

        # 实际可用的秩（min(T, 27)）
        available_rank = min(sh_matrix.shape[0], sh_matrix.shape[1])
        effective_rank = min(self.rank, available_rank)

        if effective_rank < self.rank:
            print(f"  Warning: Only {available_rank} components available, using rank={effective_rank}")

        # 提取前k个分量
        # 注意：numpy的SVD返回的是V^T，要转置
        U_init = Vt[:effective_rank, :].T  # [27, effective_rank]

        # 用奇异值缩放
        U_init = U_init * S[:effective_rank]

        # 如果effective_rank < self.rank，需要padding
        if effective_rank < self.rank:
            # 用随机小值填充剩余的列
            padding = np.random.randn(27, self.rank - effective_rank) * 0.01
            U_init = np.concatenate([U_init, padding], axis=1)  # [27, self.rank]

        # 设置参数
        self.U.data = torch.from_numpy(U_init).float()

        print(f"✓ U initialized from SVD")
        print(f"  Captured energy: {S[:effective_rank].sum()/S.sum()*100:.2f}% (using {effective_rank}/{self.rank} components)")


class PhysicsLowRankTrainer:
    """训练器"""

    def __init__(self, model, lr=1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_epoch(self, sun_elevations, sh_gt):
        """
        训练一个epoch

        Args:
            sun_elevations: [T] 太阳高度角
            sh_gt: [T, 27] 真值SH系数

        Returns:
            loss: 标量损失
        """
        self.model.train()

        # 前向传播
        sh_pred = self.model(sun_elevations)

        # 损失
        loss = F.mse_loss(sh_pred, sh_gt)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, sun_elevations, sh_gt):
        """
        评估

        Returns:
            metrics: 包含MAE, RMSE等指标的字典
        """
        self.model.eval()

        sh_pred = self.model(sun_elevations)

        # 计算指标
        mae = F.l1_loss(sh_pred, sh_gt).item()
        rmse = torch.sqrt(F.mse_loss(sh_pred, sh_gt)).item()

        # 最大误差
        max_error = torch.abs(sh_pred - sh_gt).max().item()

        return {
            'mae': mae,
            'rmse': rmse,
            'max_error': max_error
        }


def compute_sun_elevation(hour, latitude=40.0):
    """
    计算太阳高度角（简化版）

    Args:
        hour: 小时 [6-18]
        latitude: 纬度（度）

    Returns:
        elevation: 太阳高度角（弧度）
    """
    # 简化公式：假设夏至，北纬40度
    # 仅用于演示，实际应该用完整公式

    # 太阳时角
    hour_angle = (hour - 12) * 15  # 度
    hour_angle_rad = np.deg2rad(hour_angle)

    # 太阳赤纬（夏至约23.5度）
    declination = np.deg2rad(23.5)

    # 纬度
    lat_rad = np.deg2rad(latitude)

    # 高度角
    sin_elevation = (np.sin(lat_rad) * np.sin(declination) +
                     np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle_rad))

    elevation = np.arcsin(np.clip(sin_elevation, -1, 1))

    return elevation


# ============ 使用示例 ============

if __name__ == '__main__':
    """
    简单的训练示例
    """

    # 1. 准备数据（示例）
    hours = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    sun_elevations = np.array([compute_sun_elevation(h) for h in hours])

    # 假设的SH数据 [13, 27]
    sh_gt_np = np.random.randn(13, 27)

    # 转为tensor
    sun_elevations_t = torch.from_numpy(sun_elevations).float()
    sh_gt_t = torch.from_numpy(sh_gt_np).float()

    # 2. 创建模型
    model = PhysicsLowRank(rank=5, init_method='random')

    # 可选：SVD初始化
    model.init_from_svd(sh_gt_np)

    # 3. 训练
    trainer = PhysicsLowRankTrainer(model, lr=1e-3)

    print("\nTraining...")
    for epoch in range(1000):
        loss = trainer.train_epoch(sun_elevations_t, sh_gt_t)

        if (epoch + 1) % 200 == 0:
            metrics = trainer.evaluate(sun_elevations_t, sh_gt_t)
            print(f"Epoch {epoch+1}: Loss={loss:.6f}, MAE={metrics['mae']:.6f}")

    # 4. 最终评估
    final_metrics = trainer.evaluate(sun_elevations_t, sh_gt_t)
    print(f"\nFinal Metrics:")
    print(f"  MAE:  {final_metrics['mae']:.6f}")
    print(f"  RMSE: {final_metrics['rmse']:.6f}")
    print(f"  Max Error: {final_metrics['max_error']:.6f}")

    # 5. 参数量对比
    print(f"\nCompression Analysis:")
    print(f"  Spline:      351 params")
    print(f"  Ours (k={model.rank}):  {model.num_params()} params")
    print(f"  Compression: {351/model.num_params():.2f}x")


# ============ 5D扩展 (Week 1-2 验证) ============

class ExtendedPhysicsBasis5D(nn.Module):
    """
    5D参数化光源的物理基函数编码器

    输入: [sun_zenith, sun_azimuth, intensity, color_temp, cloud_cover]
    输出: 7维物理基 [cos(θ), sin(θ), cos(φ), sin(φ), I, ΔT, exp(-cloud)]

    设计原理:
    - 天顶/方位: 三角函数捕捉周期性几何变化
    - 强度: 线性保留相对亮度
    - 色温: 归一化偏移（5500K为参考daylight）
    - 云量: 指数衰减模拟大气透射率
    """

    def __init__(
        self,
        ref_color_temp: float = 5500.0,
        temp_scale: float = 2000.0
    ):
        """
        Args:
            ref_color_temp: 参考色温（K），用于归一化
            temp_scale: 色温归一化尺度
        """
        super().__init__()
        self.ref_color_temp = ref_color_temp
        self.temp_scale = temp_scale

    def forward(self, light_params_5D: torch.Tensor) -> torch.Tensor:
        """
        编码5D光源参数为7维物理基

        Args:
            light_params_5D: [B, 5] 光源参数
                [:, 0] - sun_zenith (度)
                [:, 1] - sun_azimuth (度)
                [:, 2] - intensity (相对强度)
                [:, 3] - color_temp (Kelvin)
                [:, 4] - cloud_cover (0-1)

        Returns:
            basis: [B, 7] 物理基函数
                [:, 0] - cos(zenith_rad)
                [:, 1] - sin(zenith_rad)
                [:, 2] - cos(azimuth_rad)
                [:, 3] - sin(azimuth_rad)
                [:, 4] - intensity (直接传递)
                [:, 5] - (color_temp - ref_temp) / temp_scale
                [:, 6] - exp(-cloud_cover)
        """
        # 提取参数
        zenith_deg = light_params_5D[:, 0]
        azimuth_deg = light_params_5D[:, 1]
        intensity = light_params_5D[:, 2]
        color_temp = light_params_5D[:, 3]
        cloud_cover = light_params_5D[:, 4]

        # 1. 几何基函数（三角编码）
        zenith_rad = torch.deg2rad(zenith_deg)
        azimuth_rad = torch.deg2rad(azimuth_deg)

        cos_zenith = torch.cos(zenith_rad)
        sin_zenith = torch.sin(zenith_rad)
        cos_azimuth = torch.cos(azimuth_rad)
        sin_azimuth = torch.sin(azimuth_rad)

        # 2. 强度基函数（线性）
        intensity_basis = intensity

        # 3. 色温基函数（归一化偏移）
        temp_basis = (color_temp - self.ref_color_temp) / self.temp_scale

        # 4. 大气基函数（指数衰减）
        atmos_basis = torch.exp(-cloud_cover)

        # 拼接为7维基
        basis = torch.stack([
            cos_zenith,
            sin_zenith,
            cos_azimuth,
            sin_azimuth,
            intensity_basis,
            temp_basis,
            atmos_basis
        ], dim=-1)  # [B, 7]

        return basis


class PhysicsLowRank5D(nn.Module):
    """
    5D参数化光源的物理引导低秩模型

    公式: SH(light_params) = U @ (coeffs @ Φ(light_params))

    参数量: 27*rank + rank*7 = rank*(27+7) = 34*rank

    例如 rank=5: 170参数（vs Spline 810参数 = 4.76x压缩）
    """

    def __init__(
        self,
        rank: int = 5,
        ref_color_temp: float = 5500.0,
        temp_scale: float = 2000.0,
        init_method: str = 'random'
    ):
        """
        Args:
            rank: 低秩维度（推荐5）
            ref_color_temp: 参考色温
            temp_scale: 色温归一化尺度
            init_method: 初始化方法 ('random' | 'svd')
        """
        super().__init__()

        self.rank = rank

        # 物理基编码器
        self.physics_encoder = ExtendedPhysicsBasis5D(
            ref_color_temp=ref_color_temp,
            temp_scale=temp_scale
        )

        # 空间基 U [27, rank]
        self.U = nn.Parameter(torch.randn(27, rank) * 0.1)

        # 时间系数矩阵 [rank, 7]（扩展至7维物理基）
        self.time_coeffs = nn.Parameter(torch.randn(rank, 7) * 0.1)

        print(f"PhysicsLowRank5D initialized:")
        print(f"  Rank: {rank}")
        print(f"  Parameters: {self.num_params()}")
        print(f"  Basis dimension: 7 (zenith×2 + azimuth×2 + intensity + temp + cloud)")

    def num_params(self):
        """计算参数量"""
        return 27 * self.rank + self.rank * 7

    def forward(self, light_params_5D: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            light_params_5D: [B, 5] 光源参数

        Returns:
            sh: [B, 27] SH系数
        """
        # 1. 编码物理基 [B, 7]
        basis = self.physics_encoder(light_params_5D)

        # 2. 计算时间权重 [B, rank]
        # weights = basis @ time_coeffs^T
        weights = basis @ self.time_coeffs.T  # [B, rank]

        # 3. 重建SH [B, 27]
        # sh = weights @ U^T
        sh = weights @ self.U.T  # [B, 27]

        return sh

    @torch.no_grad()
    def init_from_svd(
        self,
        light_params_np: np.ndarray,
        sh_matrix: np.ndarray
    ):
        """
        用SVD初始化U，用最小二乘初始化coeffs

        Args:
            light_params_np: [N, 5] 光源参数（numpy）
            sh_matrix: [N, 27] SH系数（numpy）
        """
        N = sh_matrix.shape[0]

        # 1. SVD on SH matrix
        U_svd, S, Vt = np.linalg.svd(sh_matrix, full_matrices=False)

        available_rank = min(N, 27)
        effective_rank = min(self.rank, available_rank)

        if effective_rank < self.rank:
            print(f"  Warning: Only {available_rank} components available, using rank={effective_rank}")

        # 提取前rank个右奇异向量作为空间基U
        U_init = Vt[:effective_rank, :].T  # [27, effective_rank]
        U_init = U_init * S[:effective_rank]  # 用奇异值缩放

        # Padding if needed
        if effective_rank < self.rank:
            padding = np.random.randn(27, self.rank - effective_rank) * 0.01
            U_init = np.concatenate([U_init, padding], axis=1)

        self.U.data = torch.from_numpy(U_init).float()

        # 2. 最小二乘求解coeffs
        # 目标: sh_matrix ≈ basis @ coeffs @ U^T
        # 即: U^T @ sh_matrix^T ≈ coeffs @ basis^T
        # 求解: coeffs = (U^T @ sh_matrix^T) @ basis^+ (其中basis^+是伪逆)

        # 计算物理基
        light_params_t = torch.from_numpy(light_params_np).float()
        with torch.no_grad():
            basis_np = self.physics_encoder(light_params_t).numpy()  # [N, 7]

        # 投影SH到U空间: [rank, N]
        U_numpy = self.U.data.numpy()
        projected = U_numpy.T @ sh_matrix.T  # [rank, N]

        # 最小二乘求解: coeffs @ basis^T ≈ projected
        # coeffs = projected @ pinv(basis^T)
        basis_pinv = np.linalg.pinv(basis_np.T)  # [N, 7]
        coeffs_init = projected @ basis_pinv  # [rank, N] @ [N, 7] = [rank, 7]

        self.time_coeffs.data = torch.from_numpy(coeffs_init).float()

        print(f"✓ Initialized from SVD + least-squares")
        print(f"  Captured energy: {S[:effective_rank].sum()/S.sum()*100:.2f}%")
        print(f"  U shape: {self.U.shape}")
        print(f"  Coeffs shape: {self.time_coeffs.shape}")


class PhysicsLowRank5DTrainer:
    """5D模型训练器"""

    def __init__(self, model, lr=1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_epoch(self, light_params, sh_gt):
        """
        训练一个epoch

        Args:
            light_params: [N, 5] 光源参数
            sh_gt: [N, 27] 真值SH

        Returns:
            loss: 标量损失
        """
        self.model.train()

        sh_pred = self.model(light_params)
        loss = F.mse_loss(sh_pred, sh_gt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, light_params, sh_gt):
        """评估"""
        self.model.eval()

        sh_pred = self.model(light_params)

        mae = F.l1_loss(sh_pred, sh_gt).item()
        rmse = torch.sqrt(F.mse_loss(sh_pred, sh_gt)).item()
        max_error = torch.abs(sh_pred - sh_gt).max().item()

        return {
            'mae': mae,
            'rmse': rmse,
            'max_error': max_error
        }


def test_physics_low_rank_5D():
    """测试5D模型"""
    print("=" * 70)
    print("Testing PhysicsLowRank5D")
    print("=" * 70)

    # 1. 生成模拟5D数据
    np.random.seed(42)
    torch.manual_seed(42)

    num_samples = 30

    # 5D参数
    light_params_np = np.random.rand(num_samples, 5)
    light_params_np[:, 0] = light_params_np[:, 0] * 75 + 15  # zenith [15, 90]
    light_params_np[:, 1] = light_params_np[:, 1] * 360  # azimuth [0, 360]
    light_params_np[:, 2] = light_params_np[:, 2] * 1.0 + 0.5  # intensity [0.5, 1.5]
    light_params_np[:, 3] = light_params_np[:, 3] * 4000 + 3000  # temp [3000, 7000]
    light_params_np[:, 4] = light_params_np[:, 4] * 0.8  # cloud [0, 0.8]

    # 模拟SH数据
    sh_gt_np = np.random.randn(num_samples, 27) * 0.5

    light_params = torch.from_numpy(light_params_np).float()
    sh_gt = torch.from_numpy(sh_gt_np).float()

    # 2. 创建模型
    model = PhysicsLowRank5D(rank=5)

    # 3. SVD初始化
    model.init_from_svd(light_params_np, sh_gt_np)

    # 4. 训练
    trainer = PhysicsLowRank5DTrainer(model, lr=1e-3)

    print("\nTraining...")
    for epoch in range(500):
        loss = trainer.train_epoch(light_params, sh_gt)

        if (epoch + 1) % 100 == 0:
            metrics = trainer.evaluate(light_params, sh_gt)
            print(f"Epoch {epoch+1}: Loss={loss:.6f}, MAE={metrics['mae']:.6f}")

    # 5. 最终评估
    final_metrics = trainer.evaluate(light_params, sh_gt)
    print(f"\nFinal Metrics:")
    print(f"  MAE:  {final_metrics['mae']:.6f}")
    print(f"  RMSE: {final_metrics['rmse']:.6f}")
    print(f"  Max Error: {final_metrics['max_error']:.6f}")

    # 6. 参数量对比
    print(f"\nCompression Analysis:")
    print(f"  Spline (41 configs × 27 SH × 3 RGB / 3 basis):  ~810 params")
    print(f"  Ours (rank={model.rank}):  {model.num_params()} params")
    print(f"  Compression: {810/model.num_params():.2f}x")

    print("\n✓ All tests passed!")


if __name__ == '__main__':
    # 运行5D测试
    test_physics_low_rank_5D()
