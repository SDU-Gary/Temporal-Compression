"""
球谐系数烘焙模块

在探针位置烘焙球谐系数，用于表示该点的入射光照
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

# 添加utils到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

import numpy as np
import mitsuba as mi
from spherical_harmonics import fibonacci_sphere, fit_sh_coefficients

if TYPE_CHECKING:
    from mitsuba import Scene


def bake_sh_at_probe(
    scene: 'Scene',
    probe_position: np.ndarray,
    num_samples: int = 64,
    spp: int = 256
) -> np.ndarray:
    """
    在探针位置烘焙球谐系数

    通过在探针周围采样多个方向的入射辐射度，
    并拟合球谐系数来表示该点的光照环境

    Args:
        scene: Mitsuba场景对象
        probe_position: [3] 探针位置坐标
        num_samples: 采样方向数量（建议32-128）
        spp: 每像素采样数（samples per pixel，影响渲染质量）

    Returns:
        sh_coeffs: [27] 球谐系数
                   前9个：R通道的9个基函数系数
                   中9个：G通道的9个基函数系数
                   后9个：B通道的9个基函数系数

    Example:
        >>> scene = mi.load_file('scene.xml')
        >>> probe_pos = np.array([0.0, 0.0, 1.0])
        >>> sh = bake_sh_at_probe(scene, probe_pos, num_samples=64, spp=128)
        >>> sh.shape
        (27,)
    """
    # 1. 使用Fibonacci球采样生成均匀分布的方向
    directions = fibonacci_sphere(num_samples)

    # 2. 为每个方向渲染入射辐射度
    radiances = []

    for direction in directions:
        # 创建朝向该方向的微小视野相机（模拟单方向采样）
        sensor = mi.load_dict({
            'type': 'perspective',
            'fov': 1.0,  # 很小的视野角，相当于单个方向
            'to_world': mi.ScalarTransform4f.look_at(
                origin=probe_position,
                target=probe_position + direction,
                up=[0, 0, 1]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': 1,
                'height': 1,
                'rfilter': {'type': 'box'}
            }
        })

        # 渲染该方向的辐射度
        image = mi.render(scene, sensor=sensor, spp=spp)
        radiance = np.array(image).flatten()  # [3] RGB

        radiances.append(radiance)

    radiances = np.array(radiances)  # [num_samples, 3]

    # 3. 拟合2阶球谐系数（9个基函数）
    sh_coeffs = fit_sh_coefficients(directions, radiances, max_order=2)

    return sh_coeffs


def batch_bake_sh(
    scene: 'Scene',
    probe_positions: np.ndarray,
    num_samples: int = 64,
    spp: int = 256,
    show_progress: bool = True
) -> np.ndarray:
    """
    批量烘焙多个探针的球谐系数

    Args:
        scene: Mitsuba场景对象
        probe_positions: [N, 3] N个探针位置
        num_samples: 每个探针的采样方向数
        spp: 每像素采样数
        show_progress: 是否显示进度条

    Returns:
        sh_coeffs_array: [N, 27] N个探针的球谐系数
    """
    num_probes = len(probe_positions)
    sh_coeffs_list = []

    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(probe_positions, desc="Baking SH coefficients")
    else:
        iterator = probe_positions

    for probe_pos in iterator:
        sh_coeffs = bake_sh_at_probe(scene, probe_pos, num_samples, spp)
        sh_coeffs_list.append(sh_coeffs)

    return np.array(sh_coeffs_list)
