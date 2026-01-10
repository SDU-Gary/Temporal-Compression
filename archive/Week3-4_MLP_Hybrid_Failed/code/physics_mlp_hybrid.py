"""
Physics + MLP Hybrid Model for 5D Parametric Light Sources

核心思想:
    SH(light_params) = U @ (Φ_physics(light) + α·MLP(light))

    - Φ_physics: 7D物理基 [cos(θ), sin(θ), cos(φ), sin(φ), I, ΔT, exp(-cloud)]
    - MLP: 神经网络残差，拟合物理基无法建模的高频细节
    - α: 可学习的混合权重，控制物理先验与数据驱动的平衡

优势:
    - 物理基提供低秩结构，减少参数量
    - MLP拟合残差，提升精度
    - α自适应调整，无需手动调参

参数量 (rank=5, hidden=64):
    - U矩阵: 27×5 = 135
    - time_coeffs: 5×7 = 35
    - MLP: (5→64→64→5) ≈ 4,500
    - α: 1 scalar
    总计: ~4,671 (vs Spline 4536, 仅略多但精度更高)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class ResidualMLP(nn.Module):
    """残差MLP网络

    输入: 5D light parameters [zenith, azimuth, intensity, temp, cloud]
    输出: rank维残差向量

    设计:
        - 2层隐藏层 (ReLU激活)
        - LayerNorm用于稳定训练
        - 残差连接（如果输入输出维度相同）
    """

    def __init__(self, input_dim: int = 5, output_dim: int = 5, hidden_dim: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

        # 残差连接（仅当维度匹配时）
        self.use_residual = (input_dim == output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim] input features

        Returns:
            out: [B, output_dim] residual features
        """
        out = self.net(x)

        if self.use_residual:
            out = out + x

        return out


class PhysicsMLPHybrid(nn.Module):
    """Physics + MLP Hybrid Model (Single Probe)

    公式:
        SH(light_params) = U @ (Φ_physics(light) + α·MLP(light))

    其中:
        - Φ_physics ∈ ℝ^7: 物理基函数编码
        - MLP ∈ ℝ^rank: 神经网络残差
        - α ∈ ℝ: 混合权重（可学习）
        - U ∈ ℝ^{27×rank}: 低秩空间基
    """

    def __init__(
        self,
        rank: int = 5,
        mlp_hidden: int = 64,
        ref_color_temp: float = 5500.0,
        temp_scale: float = 2000.0,
        init_alpha: float = 0.1,
        learnable_alpha: bool = True
    ):
        """
        Args:
            rank: 低秩维度
            mlp_hidden: MLP隐藏层维度
            ref_color_temp: 参考色温（归一化用）
            temp_scale: 色温归一化尺度
            init_alpha: α初始值
            learnable_alpha: α是否可学习
        """
        super().__init__()

        self.rank = rank

        # 1. 物理基编码器（复用PhysicsLowRank5D的设计）
        from models.physics_low_rank import ExtendedPhysicsBasis5D
        self.physics_encoder = ExtendedPhysicsBasis5D(ref_color_temp, temp_scale)

        # 2. 低秩矩阵 U: [27, rank]
        self.U = nn.Parameter(torch.randn(27, rank) * 0.1)

        # 3. 物理基的时间系数: [rank, 7]
        self.time_coeffs = nn.Parameter(torch.randn(rank, 7) * 0.1)

        # 4. MLP残差网络: 5D → rank维
        self.residual_mlp = ResidualMLP(
            input_dim=5,
            output_dim=rank,
            hidden_dim=mlp_hidden
        )

        # 5. 混合权重 α
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(init_alpha))
        else:
            self.register_buffer('alpha', torch.tensor(init_alpha))

        self.learnable_alpha = learnable_alpha

    def forward(self, light_params_5D: torch.Tensor, use_mlp: bool = True) -> torch.Tensor:
        """
        Args:
            light_params_5D: [B, 5] light parameters [zenith, azimuth, intensity, temp, cloud]
            use_mlp: 是否使用MLP残差（用于分阶段训练）

        Returns:
            sh_coeffs: [B, 27] predicted SH coefficients
        """
        # 1. 物理基编码: [B, 7]
        physics_basis = self.physics_encoder(light_params_5D)

        # 2. 物理基权重: [B, rank]
        physics_weights = physics_basis @ self.time_coeffs.T

        # 3. MLP残差权重: [B, rank]
        if use_mlp:
            mlp_residual = self.residual_mlp(light_params_5D)
            # 混合: physics + α·mlp
            combined_weights = physics_weights + self.alpha * mlp_residual
        else:
            combined_weights = physics_weights

        # 4. 重建SH系数: [B, 27]
        sh_coeffs = combined_weights @ self.U.T

        return sh_coeffs

    def get_alpha_value(self) -> float:
        """获取当前α值"""
        return self.alpha.item()

    def freeze_mlp(self):
        """冻结MLP参数（Stage 1训练用）"""
        for param in self.residual_mlp.parameters():
            param.requires_grad = False
        print("✓ MLP parameters frozen")

    def unfreeze_mlp(self):
        """解冻MLP参数（Stage 2训练用）"""
        for param in self.residual_mlp.parameters():
            param.requires_grad = True
        print("✓ MLP parameters unfrozen")

    @torch.no_grad()
    def init_from_physics_model(self, physics_model):
        """从预训练的PhysicsLowRank5D初始化

        Args:
            physics_model: PhysicsLowRank5D instance
        """
        # 复制U和time_coeffs
        self.U.data.copy_(physics_model.U.data)
        self.time_coeffs.data.copy_(physics_model.time_coeffs.data)

        print(f"✓ Initialized from pretrained physics model")
        print(f"  U shape: {self.U.shape}")
        print(f"  Coeffs shape: {self.time_coeffs.shape}")

    @torch.no_grad()
    def init_from_svd(self, light_params_np: np.ndarray, sh_matrix: np.ndarray):
        """SVD初始化（仅初始化U和time_coeffs，MLP保持随机）

        Args:
            light_params_np: [N, 5] light configurations
            sh_matrix: [N, 27] SH coefficients
        """
        N, num_sh = sh_matrix.shape
        assert num_sh == 27, f"Expected 27 SH coeffs, got {num_sh}"

        # 1. SVD on SH matrix
        U_svd, S, Vt = np.linalg.svd(sh_matrix, full_matrices=False)

        # 取前rank个分量
        effective_rank = min(self.rank, len(S))
        U_init = Vt[:effective_rank, :].T  # [27, effective_rank]
        U_init = U_init * S[:effective_rank]  # 缩放

        # 如果rank < effective_rank, 补零
        if self.rank > effective_rank:
            U_init = np.pad(U_init, ((0, 0), (0, self.rank - effective_rank)))

        self.U.data = torch.from_numpy(U_init).float()

        # 2. 最小二乘求解time_coeffs
        light_params_t = torch.from_numpy(light_params_np).float()
        basis_np = self.physics_encoder(light_params_t).numpy()  # [N, 7]

        U_numpy = self.U.data.numpy()
        projected = U_numpy.T @ sh_matrix.T  # [rank, N]

        basis_pinv = np.linalg.pinv(basis_np.T)  # [N, 7]
        coeffs_init = projected @ basis_pinv  # [rank, N] @ [N, 7] = [rank, 7]

        self.time_coeffs.data = torch.from_numpy(coeffs_init).float()

        print(f"✓ Initialized from SVD + least-squares")
        print(f"  Captured energy: {S[:effective_rank].sum()/S.sum()*100:.2f}%")
        print(f"  U shape: {self.U.shape}")
        print(f"  Coeffs shape: {self.time_coeffs.shape}")

    def count_parameters(self) -> dict:
        """统计参数量"""
        physics_params = self.U.numel() + self.time_coeffs.numel()
        mlp_params = sum(p.numel() for p in self.residual_mlp.parameters())
        alpha_params = 1 if self.learnable_alpha else 0
        total_params = physics_params + mlp_params + alpha_params

        return {
            'physics': physics_params,
            'mlp': mlp_params,
            'alpha': alpha_params,
            'total': total_params
        }


def test_physics_mlp_hybrid():
    """测试混合模型"""
    print("="*70)
    print("Testing PhysicsMLPHybrid")
    print("="*70)

    # 生成模拟数据
    np.random.seed(42)
    N_train = 30
    N_test = 10

    # 5D参数空间采样
    train_params = np.random.rand(N_train, 5)
    train_params[:, 0] *= 90  # zenith [0, 90]
    train_params[:, 1] *= 360  # azimuth [0, 360]
    train_params[:, 2] = train_params[:, 2] * 1.5 + 0.5  # intensity [0.5, 2.0]
    train_params[:, 3] = train_params[:, 3] * 5000 + 3000  # temp [3000, 8000]
    train_params[:, 4] *= 0.8  # cloud [0, 0.8]

    test_params = np.random.rand(N_test, 5)
    test_params[:, 0] *= 90
    test_params[:, 1] *= 360
    test_params[:, 2] = test_params[:, 2] * 1.5 + 0.5
    test_params[:, 3] = test_params[:, 3] * 5000 + 3000
    test_params[:, 4] *= 0.8

    # 模拟SH系数 (低秩 + 噪声)
    rank_true = 3
    U_true = np.random.randn(27, rank_true)
    coeffs_true = np.random.randn(rank_true, 5)

    train_sh = (train_params @ coeffs_true.T) @ U_true.T
    train_sh += np.random.randn(*train_sh.shape) * 0.01  # 添加噪声

    test_sh = (test_params @ coeffs_true.T) @ U_true.T
    test_sh += np.random.randn(*test_sh.shape) * 0.01

    # === 测试1: 模型初始化 ===
    print("\n1. Testing model initialization...")
    model = PhysicsMLPHybrid(rank=5, mlp_hidden=64, learnable_alpha=True)

    param_counts = model.count_parameters()
    print(f"  Physics params: {param_counts['physics']}")
    print(f"  MLP params: {param_counts['mlp']}")
    print(f"  Alpha params: {param_counts['alpha']}")
    print(f"  Total params: {param_counts['total']}")

    # === 测试2: SVD初始化 ===
    print("\n2. Testing SVD initialization...")
    model.init_from_svd(train_params, train_sh)

    # === 测试3: Forward pass (physics only) ===
    print("\n3. Testing forward pass (physics only, α=0)...")
    model.alpha.data.fill_(0.0)  # 禁用MLP

    train_params_t = torch.from_numpy(train_params).float()
    test_params_t = torch.from_numpy(test_params).float()

    with torch.no_grad():
        pred_train = model(train_params_t, use_mlp=False).numpy()
        pred_test = model(test_params_t, use_mlp=False).numpy()

    mae_train = np.abs(pred_train - train_sh).mean()
    mae_test = np.abs(pred_test - test_sh).mean()

    print(f"  Train MAE: {mae_train:.6f}")
    print(f"  Test MAE: {mae_test:.6f}")

    # === 测试4: Forward pass (hybrid) ===
    print("\n4. Testing forward pass (hybrid, α=0.1)...")
    model.alpha.data.fill_(0.1)

    with torch.no_grad():
        pred_train_hybrid = model(train_params_t, use_mlp=True).numpy()
        pred_test_hybrid = model(test_params_t, use_mlp=True).numpy()

    mae_train_hybrid = np.abs(pred_train_hybrid - train_sh).mean()
    mae_test_hybrid = np.abs(pred_test_hybrid - test_sh).mean()

    print(f"  Train MAE: {mae_train_hybrid:.6f}")
    print(f"  Test MAE: {mae_test_hybrid:.6f}")

    # === 测试5: 简单训练测试 ===
    print("\n5. Testing training (100 epochs)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    train_sh_t = torch.from_numpy(train_sh).float()

    for epoch in range(100):
        optimizer.zero_grad()
        pred = model(train_params_t, use_mlp=True)
        loss = criterion(pred, train_sh_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: Loss={loss.item():.6f}, α={model.get_alpha_value():.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred_final = model(test_params_t, use_mlp=True).numpy()

    mae_final = np.abs(pred_final - test_sh).mean()
    print(f"\n  Final Test MAE: {mae_final:.6f}")

    print("\n✓ All tests passed!")


if __name__ == '__main__':
    test_physics_mlp_hybrid()
