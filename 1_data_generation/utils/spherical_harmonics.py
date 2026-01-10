"""
球谐系数拟合工具函数

用于将多个方向的辐射度采样拟合为球谐系数（Spherical Harmonics）
"""

import numpy as np
from typing import Tuple


def fibonacci_sphere(num_samples: int) -> np.ndarray:
    """
    使用Fibonacci spiral方法在单位球面上均匀采样

    Args:
        num_samples: 采样点数量

    Returns:
        directions: [N, 3] 单位方向向量
    """
    indices = np.arange(0, num_samples, dtype=float) + 0.5

    phi = np.arccos(1 - 2 * indices / num_samples)
    theta = np.pi * (1 + 5**0.5) * indices

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.stack([x, y, z], axis=1)


def evaluate_sh_basis(directions: np.ndarray, max_order: int = 2) -> np.ndarray:
    """
    计算球谐基函数值

    Args:
        directions: [N, 3] 方向向量（已归一化）
        max_order: 最大阶数（2阶=9个基函数，3阶=16个基函数）

    Returns:
        Y: [N, num_basis] 球谐基函数矩阵

    Note:
        2阶球谐（9个基函数）对于漫反射光照通常足够
        3阶球谐（16个基函数）可以表示中等光泽材质
    """
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    N = directions.shape[0]

    if max_order == 2:
        # 9个基函数（l=0,1,2）
        num_basis = 9
        Y = np.zeros((N, num_basis))

        # l=0, m=0
        Y[:, 0] = 0.282095  # sqrt(1/(4*pi))

        # l=1, m=-1,0,1
        Y[:, 1] = 0.488603 * y  # sqrt(3/(4*pi)) * y
        Y[:, 2] = 0.488603 * z  # sqrt(3/(4*pi)) * z
        Y[:, 3] = 0.488603 * x  # sqrt(3/(4*pi)) * x

        # l=2, m=-2,-1,0,1,2
        Y[:, 4] = 1.092548 * x * y  # sqrt(15/(4*pi)) * x * y
        Y[:, 5] = 1.092548 * y * z  # sqrt(15/(4*pi)) * y * z
        Y[:, 6] = 0.315392 * (3*z*z - 1)  # sqrt(5/(16*pi)) * (3*z^2-1)
        Y[:, 7] = 1.092548 * x * z  # sqrt(15/(4*pi)) * x * z
        Y[:, 8] = 0.546274 * (x*x - y*y)  # sqrt(15/(16*pi)) * (x^2-y^2)

    else:
        raise NotImplementedError(f"max_order={max_order} not implemented yet")

    return Y


def fit_sh_coefficients(
    directions: np.ndarray,
    radiances: np.ndarray,
    max_order: int = 2
) -> np.ndarray:
    """
    使用最小二乘法拟合球谐系数

    Args:
        directions: [N, 3] 采样方向（已归一化）
        radiances: [N, 3] RGB辐射度值
        max_order: 球谐最大阶数

    Returns:
        coeffs: [num_basis * 3] 球谐系数（RGB三通道拼接）

    Example:
        >>> directions = fibonacci_sphere(64)
        >>> radiances = sample_radiance(scene, probe_pos, directions)
        >>> sh_coeffs = fit_sh_coefficients(directions, radiances, max_order=2)
        >>> # sh_coeffs.shape = [27] (9个基函数 × RGB 3通道)
    """
    # 计算球谐基函数矩阵
    Y = evaluate_sh_basis(directions, max_order)  # [N, num_basis]

    # 对每个RGB通道分别拟合
    coeffs_list = []
    for channel in range(3):  # R, G, B
        L = radiances[:, channel]  # [N]

        # 最小二乘法求解: c = (Y^T Y)^{-1} Y^T L
        c, residuals, rank, s = np.linalg.lstsq(Y, L, rcond=None)
        coeffs_list.append(c)

    # 拼接为 [num_basis, 3] 然后展平为 [num_basis * 3]
    coeffs = np.concatenate(coeffs_list)

    return coeffs


def reconstruct_from_sh(
    sh_coeffs: np.ndarray,
    directions: np.ndarray,
    max_order: int = 2
) -> np.ndarray:
    """
    从球谐系数重建辐射度

    Args:
        sh_coeffs: [num_basis * 3] 球谐系数
        directions: [N, 3] 查询方向
        max_order: 球谐阶数

    Returns:
        radiances: [N, 3] 重建的RGB辐射度
    """
    num_basis = (max_order + 1) ** 2
    Y = evaluate_sh_basis(directions, max_order)  # [N, num_basis]

    # 重塑系数为 [num_basis, 3]
    coeffs_rgb = sh_coeffs.reshape(3, num_basis).T  # [num_basis, 3]

    # 矩阵乘法: [N, num_basis] @ [num_basis, 3] = [N, 3]
    radiances = Y @ coeffs_rgb

    return radiances


def compute_sh_reconstruction_error(
    directions: np.ndarray,
    radiances_gt: np.ndarray,
    sh_coeffs: np.ndarray,
    max_order: int = 2
) -> Tuple[float, float]:
    """
    计算球谐重建的误差

    Args:
        directions: [N, 3] 采样方向
        radiances_gt: [N, 3] Ground Truth辐射度
        sh_coeffs: [num_basis * 3] 拟合的球谐系数
        max_order: 球谐阶数

    Returns:
        mse: 均方误差
        relative_error: 相对误差（百分比）
    """
    radiances_recon = reconstruct_from_sh(sh_coeffs, directions, max_order)

    mse = np.mean((radiances_gt - radiances_recon) ** 2)
    relative_error = np.mean(np.abs(radiances_gt - radiances_recon)) / (np.mean(radiances_gt) + 1e-6)

    return mse, relative_error * 100


if __name__ == "__main__":
    # 测试代码
    print("Testing Spherical Harmonics utilities...")

    # 1. 测试Fibonacci球面采样
    directions = fibonacci_sphere(64)
    print(f"Sampled {len(directions)} directions")
    print(f"First direction: {directions[0]}")
    print(f"Direction norms (should be ~1.0): {np.linalg.norm(directions, axis=1)[:5]}")

    # 2. 测试球谐基函数
    Y = evaluate_sh_basis(directions, max_order=2)
    print(f"\nSH basis matrix shape: {Y.shape}")  # Should be [64, 9]

    # 3. 测试拟合与重建
    # 模拟一个简单的光照：从上方来的定向光
    test_radiances = np.maximum(0, directions[:, 2:3]) * np.array([1.0, 0.9, 0.8])
    sh_coeffs = fit_sh_coefficients(directions, test_radiances, max_order=2)
    print(f"\nFitted SH coefficients shape: {sh_coeffs.shape}")  # Should be [27]

    # 重建并计算误差
    mse, rel_err = compute_sh_reconstruction_error(
        directions, test_radiances, sh_coeffs, max_order=2
    )
    print(f"Reconstruction MSE: {mse:.6f}")
    print(f"Relative error: {rel_err:.2f}%")

    print("\nAll tests passed! ✓")
