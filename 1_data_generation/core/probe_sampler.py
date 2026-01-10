"""
探针位置采样模块

提供场景中探针位置的采样策略
"""

import numpy as np
from typing import Tuple


def sample_probe_positions(
    num_probes: int,
    scene_bounds: Tuple[float, float, float, float, float, float]
) -> np.ndarray:
    """
    在场景边界内采样探针位置

    使用均匀网格采样策略，在场景的3D边界框内生成探针位置

    Args:
        num_probes: 探针数量
        scene_bounds: (x_min, x_max, y_min, y_max, z_min, z_max) 场景边界

    Returns:
        probes: [N, 3] 探针位置数组

    Example:
        >>> bounds = (-5, 5, -5, 5, 0, 10)
        >>> probes = sample_probe_positions(100, bounds)
        >>> probes.shape
        (100, 3)
    """
    x_min, x_max, y_min, y_max, z_min, z_max = scene_bounds

    # 计算每维度的采样点数（立方根）
    n_per_dim = int(np.cbrt(num_probes))

    # 在每个维度上均匀采样
    x = np.linspace(x_min, x_max, n_per_dim)
    y = np.linspace(y_min, y_max, n_per_dim)
    z = np.linspace(z_min, z_max, n_per_dim)

    # 创建3D网格
    xx, yy, zz = np.meshgrid(x, y, z)
    probes = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

    # 取前num_probes个（如果网格点数超过需求）
    probes = probes[:num_probes]

    return probes


def sample_probe_positions_random(
    num_probes: int,
    scene_bounds: Tuple[float, float, float, float, float, float],
    seed: int = 42
) -> np.ndarray:
    """
    随机采样探针位置（替代方案）

    Args:
        num_probes: 探针数量
        scene_bounds: (x_min, x_max, y_min, y_max, z_min, z_max)
        seed: 随机种子

    Returns:
        probes: [N, 3] 探针位置数组
    """
    np.random.seed(seed)
    x_min, x_max, y_min, y_max, z_min, z_max = scene_bounds

    probes = np.random.uniform(
        low=[x_min, y_min, z_min],
        high=[x_max, y_max, z_max],
        size=(num_probes, 3)
    )

    return probes
