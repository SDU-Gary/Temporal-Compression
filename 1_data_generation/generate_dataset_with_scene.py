"""
使用外部场景文件生成多时刻光照数据集

支持加载Benedikt Bitterli场景库或其他Mitsuba 3兼容场景
"""

import sys
import os
from pathlib import Path

# 添加utils到Python路径
sys.path.insert(0, str(Path(__file__).parent / "utils"))

import numpy as np
import mitsuba as mi
from tqdm import tqdm
import json

from spherical_harmonics import (
    fibonacci_sphere,
    fit_sh_coefficients,
)
from sun_position import (
    calculate_sun_direction_simple,
)


# 设置Mitsuba变体（使用CUDA加速）
try:
    mi.set_variant('cuda_ad_rgb')
    print("Using CUDA variant")
except:
    mi.set_variant('llvm_ad_rgb')
    print("CUDA not available, using LLVM variant")


def load_and_modify_scene(scene_path: Path, sun_direction: np.ndarray) -> mi.Scene:
    """
    加载外部场景并修改太阳光方向

    Args:
        scene_path: 场景XML文件路径
        sun_direction: [3] 太阳方向向量

    Returns:
        scene: 修改后的Mitsuba场景对象
    """
    print(f"Loading scene from: {scene_path}")

    # 加载场景XML文件
    scene_dict = mi.load_file(str(scene_path))

    # TODO: 根据场景实际情况修改光源
    # 这里需要分析场景XML，找到directional光源并修改方向
    # 或者添加新的定向光源

    print(f"✓ Scene loaded: {scene_dict}")
    return scene_dict


def analyze_scene_bounds(scene_path: Path) -> tuple:
    """
    分析场景边界框，确定探针采样范围

    Args:
        scene_path: 场景XML文件路径

    Returns:
        bounds: (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    # 加载场景
    scene = mi.load_file(str(scene_path))

    # 获取场景边界框
    bbox = scene.bbox()

    x_min, y_min, z_min = bbox.min
    x_max, y_max, z_max = bbox.max

    print(f"\nScene bounding box:")
    print(f"  X: [{x_min:.2f}, {x_max:.2f}]")
    print(f"  Y: [{y_min:.2f}, {y_max:.2f}]")
    print(f"  Z: [{z_min:.2f}, {z_max:.2f}]")
    print(f"  Volume: {(x_max-x_min)*(y_max-y_min)*(z_max-z_min):.2f} m³")

    return (x_min, x_max, y_min, y_max, z_min, z_max)


def sample_probe_positions(num_probes: int, scene_bounds: tuple) -> np.ndarray:
    """
    在场景中采样探针位置

    Args:
        num_probes: 探针数量
        scene_bounds: (x_min, x_max, y_min, y_max, z_min, z_max)

    Returns:
        probes: [N, 3] 探针位置
    """
    x_min, x_max, y_min, y_max, z_min, z_max = scene_bounds

    # 均匀网格采样
    n_per_dim = int(np.cbrt(num_probes))
    x = np.linspace(x_min, x_max, n_per_dim)
    y = np.linspace(y_min, y_max, n_per_dim)
    z = np.linspace(z_min, z_max, n_per_dim)

    xx, yy, zz = np.meshgrid(x, y, z)
    probes = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

    # 取前num_probes个
    probes = probes[:num_probes]

    return probes


def bake_sh_at_probe(
    scene: mi.Scene,
    probe_position: np.ndarray,
    num_samples: int = 64,
    spp: int = 256
) -> np.ndarray:
    """
    在探针位置烘焙球谐系数

    Args:
        scene: Mitsuba场景对象
        probe_position: [3] 探针位置
        num_samples: 采样方向数量
        spp: 每像素采样数（samples per pixel）

    Returns:
        sh_coeffs: [27] 球谐系数（9基函数 × RGB 3通道）
    """
    # 1. 生成采样方向
    directions = fibonacci_sphere(num_samples)

    # 2. 为每个方向渲染一个像素
    radiances = []

    for direction in directions:
        # 创建朝向该方向的传感器（1像素）
        sensor = mi.load_dict({
            'type': 'perspective',
            'fov': 1.0,  # 很小的视野，相当于采样单个方向
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

        # 渲染
        image = mi.render(scene, sensor=sensor, spp=spp)
        radiance = np.array(image).flatten()  # [3] RGB

        radiances.append(radiance)

    radiances = np.array(radiances)  # [N, 3]

    # 3. 拟合球谐系数
    sh_coeffs = fit_sh_coefficients(directions, radiances, max_order=2)

    return sh_coeffs


def generate_moment_data(
    moment_id: int,
    sun_direction: np.ndarray,
    scene_path: Path,
    probe_positions: np.ndarray,
    output_dir: Path,
    num_gt_images: int = 10,
    num_sh_samples: int = 64,
    spp: int = 256
):
    """
    生成单个时刻的数据

    Args:
        moment_id: 时刻ID（0-23对应小时）
        sun_direction: 太阳方向
        scene_path: 场景文件路径
        probe_positions: [N, 3] 探针位置
        output_dir: 输出目录
        num_gt_images: Ground Truth图像数量
        num_sh_samples: 球谐采样方向数
        spp: 每像素采样数
    """
    moment_dir = output_dir / f"moment_{moment_id:02d}"
    moment_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Generating data for Hour {moment_id}:00 ===")
    print(f"Sun direction: {sun_direction}")

    # 1. 加载场景
    # 注意：这里简化处理，实际需要修改场景中的光源方向
    scene = mi.load_file(str(scene_path))

    # TODO: 实际使用时需要修改场景中的定向光源方向
    # 这可能需要：
    # 1. 解析XML
    # 2. 找到<emitter type="directional">
    # 3. 修改direction参数
    # 4. 重新加载场景

    # 2. 烘焙球谐系数
    print(f"Baking SH coefficients for {len(probe_positions)} probes...")
    sh_coeffs_list = []

    for i, probe_pos in enumerate(tqdm(probe_positions, desc="Baking SH")):
        sh_coeffs = bake_sh_at_probe(scene, probe_pos, num_sh_samples, spp=spp)
        sh_coeffs_list.append(sh_coeffs)

    sh_coeffs_array = np.array(sh_coeffs_list)  # [N, 27]

    # 保存球谐系数
    np.savez_compressed(
        moment_dir / "sh_coeffs.npz",
        coeffs=sh_coeffs_array,
        sun_dir=sun_direction
    )
    print(f"Saved SH coefficients: {sh_coeffs_array.shape}")

    # 3. 生成Ground Truth图像
    print(f"Rendering {num_gt_images} GT images...")
    images_dir = moment_dir / "images"
    images_dir.mkdir(exist_ok=True)

    camera_params = []

    # 获取场景边界框
    bbox = scene.bbox()
    center = (bbox.min + bbox.max) / 2
    radius = np.linalg.norm(bbox.max - bbox.min) / 2

    for img_id in range(num_gt_images):
        # 随机相机位置（在场景周围）
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi/3)  # 避免从正上方看
        r = radius * np.random.uniform(1.5, 2.5)

        camera_pos = center + np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ])

        # 看向场景中心附近
        look_at = center + np.random.uniform(-radius*0.3, radius*0.3, 3)

        sensor = mi.load_dict({
            'type': 'perspective',
            'fov': 50,
            'to_world': mi.ScalarTransform4f.look_at(
                origin=camera_pos,
                target=look_at,
                up=[0, 0, 1]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': 512,
                'height': 512,
            }
        })

        # 渲染
        image = mi.render(scene, sensor=sensor, spp=spp)

        # 保存EXR（HDR）
        mi.util.write_bitmap(str(images_dir / f"{img_id:03d}.exr"), image)

        # 保存PNG（预览）
        mi.util.write_bitmap(str(images_dir / f"{img_id:03d}.png"), image)

        # 记录相机参数
        camera_params.append({
            'image_id': img_id,
            'camera_pos': camera_pos.tolist(),
            'look_at': look_at.tolist(),
        })

    # 保存相机参数
    with open(moment_dir / "camera_params.json", 'w') as f:
        json.dump(camera_params, f, indent=2)

    # 保存太阳方向
    with open(moment_dir / "sun_direction.txt", 'w') as f:
        f.write(f"{sun_direction[0]:.6f} {sun_direction[1]:.6f} {sun_direction[2]:.6f}\n")

    print(f"✓ Moment {moment_id} completed")


def main():
    """主函数"""
    # 配置参数
    config = {
        'scene_path': Path(__file__).parent / 'scenes' / 'cornell-box' / 'scene.xml',  # 修改为实际场景路径
        'num_probes': 100,  # 快速原型：100探针
        'num_moments': 2,   # 快速原型：2个时刻
        'num_gt_images': 5,
        'num_sh_samples': 32,
        'spp': 128,
        'output_dir': Path(__file__).parent / 'output' / 'external_scene_test'
    }

    print("="*60)
    print("Multi-Moment Dataset Generation with External Scene")
    print("="*60)
    print(f"Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("="*60)

    # 检查场景文件是否存在
    if not config['scene_path'].exists():
        print(f"\n✗ Scene file not found: {config['scene_path']}")
        print(f"\nPlease:")
        print(f"  1. Download scenes using download_scenes.py")
        print(f"  2. Or update 'scene_path' to point to your scene XML file")
        return

    # 创建输出目录
    config['output_dir'].mkdir(parents=True, exist_ok=True)

    # 1. 分析场景边界
    print("\n[1/4] Analyzing scene bounds...")
    scene_bounds = analyze_scene_bounds(config['scene_path'])

    # 2. 采样探针位置
    print("\n[2/4] Sampling probe positions...")
    probe_positions = sample_probe_positions(config['num_probes'], scene_bounds)
    print(f"Sampled {len(probe_positions)} probe positions")

    # 保存探针位置
    np.savez_compressed(
        config['output_dir'] / "probes.npz",
        positions=probe_positions
    )

    # 3. 生成太阳轨迹
    print("\n[3/4] Generating sun trajectory...")
    selected_hours = [12, 18]
    sun_trajectory = []

    for hour in selected_hours:
        sun_dir = calculate_sun_direction_simple(hour)
        sun_trajectory.append((hour, sun_dir))
        print(f"  Hour {hour:02d}:00 - Sun altitude: {np.degrees(np.arcsin(sun_dir[1])):.1f}°")

    # 4. 为每个时刻生成数据
    print("\n[4/4] Generating multi-moment data...")

    for idx, (hour, sun_dir) in enumerate(sun_trajectory):
        generate_moment_data(
            moment_id=hour,
            sun_direction=sun_dir,
            scene_path=config['scene_path'],
            probe_positions=probe_positions,
            output_dir=config['output_dir'],
            num_gt_images=config['num_gt_images'],
            num_sh_samples=config['num_sh_samples'],
            spp=config['spp']
        )

    print("\n" + "="*60)
    print("Dataset generation completed! ✓")
    print(f"Output directory: {config['output_dir']}")
    print("="*60)


if __name__ == "__main__":
    main()
