"""
多时刻光照数据集生成主脚本

使用Mitsuba 3渲染引擎生成：
1. 探针位置
2. 每个时刻的球谐系数（通过采样多个方向拟合）
3. Ground Truth渲染图像
4. 太阳方向信息

支持命令行参数和外部场景文件
"""

import sys
import os
from pathlib import Path
import argparse
from datetime import datetime

# 添加utils和core到Python路径
sys.path.insert(0, str(Path(__file__).parent / "utils"))
sys.path.insert(0, str(Path(__file__).parent / "core"))

import numpy as np
import mitsuba as mi
from tqdm import tqdm
import json

from spherical_harmonics import (
    fibonacci_sphere,
    fit_sh_coefficients,
    compute_sh_reconstruction_error
)
from sun_position import (
    calculate_sun_direction_simple,
    generate_24hour_sun_trajectory
)

# 导入新的core模块
from lighting_modifier import LightingModifier
from probe_sampler import sample_probe_positions as sample_probe_positions_core
from sh_baker import bake_sh_at_probe as bake_sh_at_probe_core
from scene_utils import (
    analyze_scene_bounds,
    discover_scenes,
    print_scene_summary,
    print_scene_bounds_info
)


# 设置Mitsuba变体（使用CUDA加速）
try:
    mi.set_variant('cuda_ad_rgb')
    print("Using CUDA variant")
except:
    mi.set_variant('llvm_ad_rgb')
    print("CUDA not available, using LLVM variant")


def create_simple_outdoor_scene(sun_direction: np.ndarray) -> dict:
    """
    创建简单的室外场景（原型验证用）

    Args:
        sun_direction: [3] 太阳方向向量

    Returns:
        scene_dict: Mitsuba场景字典
    """
    scene_dict = {
        'type': 'scene',

        # 路径追踪积分器
        'integrator': {
            'type': 'path',
            'max_depth': 8,  # 最大路径长度
        },

        # 定向光源（太阳）
        'sun': {
            'type': 'directional',
            'direction': sun_direction.tolist(),
            'irradiance': {
                'type': 'rgb',
                'value': [2.0, 1.9, 1.7],  # 暖色太阳光
            }
        },

        # 天空环境光（简化版）
        'sky': {
            'type': 'constant',
            'radiance': {
                'type': 'rgb',
                'value': [0.4, 0.5, 0.6],  # 蓝色天空
            }
        },

        # 地面
        'ground': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.scale([50, 50, 1]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': [0.6, 0.6, 0.6],  # 灰色地面
                }
            }
        },

        # 立方体建筑物
        'building': {
            'type': 'cube',
            'to_world': mi.ScalarTransform4f.translate([0, 0, 2.5]).scale(2.5),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': [0.8, 0.7, 0.6],  # 米色建筑
                }
            }
        },
    }

    return scene_dict


def sample_probe_positions(num_probes: int, scene_bounds: tuple = (-10, 10, -10, 10, 0.5, 8)) -> np.ndarray:
    """
    在场景中采样探针位置

    Args:
        num_probes: 探针数量
        scene_bounds: (x_min, x_max, y_min, y_max, z_min, z_max)

    Returns:
        probes: [N, 3] 探针位置
    """
    x_min, x_max, y_min, y_max, z_min, z_max = scene_bounds

    # 均匀网格采样（初期简单策略）
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

    # 1. 创建场景
    scene_dict = create_simple_outdoor_scene(sun_direction)
    scene = mi.load_dict(scene_dict)

    # 2. 烘焙球谐系数
    print(f"Baking SH coefficients for {len(probe_positions)} probes...")
    sh_coeffs_list = []

    for i, probe_pos in enumerate(tqdm(probe_positions, desc="Baking SH")):
        sh_coeffs = bake_sh_at_probe(scene, probe_pos, num_sh_samples, spp=spp)
        sh_coeffs_list.append(sh_coeffs)

        # 每10个探针保存一次（避免内存溢出）
        if (i + 1) % 10 == 0:
            pass  # 可以添加中间保存逻辑

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

    for img_id in range(num_gt_images):
        # 随机相机位置和朝向
        camera_pos = np.random.uniform([-8, -8, 2], [8, 8, 6])
        look_at = np.random.uniform([-3, -3, 0], [3, 3, 4])

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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Multi-moment lighting dataset generation for neural compression',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 基本用法（使用外部场景）
  python generate_dataset.py --scene scenes/cornell-box/scene.xml --output output/cornell_data

  # 指定详细参数
  python generate_dataset.py \\
    --scene scenes/classroom/scene.xml \\
    --output output/classroom_data \\
    --num-probes 500 \\
    --sun-hours 6 12 18 \\
    --spp 256

  # 列出可用场景
  python generate_dataset.py --list-scenes

  # 使用内置场景（legacy模式）
  python generate_dataset.py
        """
    )

    # 场景发现
    parser.add_argument(
        '--list-scenes', action='store_true',
        help='列出所有可用场景并退出'
    )

    # 场景选择
    parser.add_argument(
        '--scene', type=Path,
        help='场景XML文件路径 (例如: scenes/classroom/scene.xml)'
    )

    # 输出目录
    parser.add_argument(
        '--output', type=Path,
        help='输出目录路径'
    )

    # 探针参数
    parser.add_argument(
        '--num-probes', type=int, default=100,
        help='探针数量 (默认: 100)'
    )

    # 时刻参数
    parser.add_argument(
        '--num-moments', type=int, default=2,
        help='时刻数量（如果未指定--sun-hours则均匀分布）(默认: 2)'
    )

    parser.add_argument(
        '--sun-hours', type=int, nargs='+',
        help='指定太阳小时数 (例如: 6 12 18)。如果指定，将覆盖--num-moments'
    )

    # 渲染参数
    parser.add_argument(
        '--spp', type=int, default=128,
        help='每像素采样数 (默认: 128)'
    )

    parser.add_argument(
        '--num-sh-samples', type=int, default=32,
        help='球谐采样方向数 (默认: 32)'
    )

    parser.add_argument(
        '--num-gt-images', type=int, default=5,
        help='每时刻Ground Truth图像数量 (默认: 5)'
    )

    # 场景边界（可选，如果未指定则自动检测）
    parser.add_argument(
        '--scene-bounds', type=float, nargs=6, metavar=('X_MIN', 'X_MAX', 'Y_MIN', 'Y_MAX', 'Z_MIN', 'Z_MAX'),
        help='手动指定场景边界 (如果未指定则自动检测)'
    )

    return parser.parse_args()


def save_config(args, output_path):
    """保存配置到JSON文件"""
    config = {
        'scene': str(args.scene) if args.scene else 'built-in',
        'num_probes': args.num_probes,
        'num_moments': args.num_moments,
        'sun_hours': args.sun_hours if args.sun_hours else None,
        'spp': args.spp,
        'num_sh_samples': args.num_sh_samples,
        'num_gt_images': args.num_gt_images,
        'scene_bounds': args.scene_bounds if hasattr(args, 'scene_bounds') else None,
        'timestamp': datetime.now().isoformat(),
        'generated_by': 'generate_dataset.py (CLI mode)'
    }

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to: {output_path}")


def cli_main(args):
    """新的CLI模式主函数"""

    # 场景发现功能
    if args.list_scenes:
        scenes_dir = Path(__file__).parent / 'scenes'
        if not scenes_dir.exists():
            print(f"场景目录不存在: {scenes_dir}")
            print("请先下载场景或创建scenes/目录")
            return 1

        scenes = discover_scenes(scenes_dir)
        print_scene_summary(scenes)

        # 打印每个场景的详细信息
        for scene_info in scenes:
            print_scene_bounds_info(scene_info['path'])

        return 0

    # 验证必需参数
    if not args.scene:
        print("错误: 必须指定 --scene 参数（或使用 --list-scenes 查看可用场景）")
        print("提示: 运行不带参数以使用legacy模式（内置场景）")
        return 1

    if not args.output:
        print("错误: 必须指定 --output 参数")
        return 1

    if not args.scene.exists():
        print(f"错误: 场景文件不存在: {args.scene}")
        return 1

    # 创建输出目录
    args.output.mkdir(parents=True, exist_ok=True)

    # 保存配置
    save_config(args, args.output / 'config.json')

    print("="*60)
    print("Multi-Moment Dataset Generation (CLI Mode)")
    print("="*60)
    print(f"Scene: {args.scene}")
    print(f"Output: {args.output}")
    print(f"Probes: {args.num_probes}")
    print(f"SPP: {args.spp}")
    print("="*60)

    # 1. 加载场景并分析边界
    print("\n[1/4] Analyzing scene...")
    temp_scene = mi.load_file(str(args.scene))

    if args.scene_bounds:
        scene_bounds = tuple(args.scene_bounds)
        print(f"Using manual scene bounds: {scene_bounds}")
    else:
        scene_bounds = analyze_scene_bounds(temp_scene)
        print(f"Auto-detected scene bounds: {scene_bounds}")

    # 2. 采样探针位置
    print("\n[2/4] Sampling probe positions...")
    probe_positions = sample_probe_positions_core(args.num_probes, scene_bounds)
    print(f"Sampled {len(probe_positions)} probe positions")

    # 保存探针位置
    np.savez_compressed(
        args.output / "probes.npz",
        positions=probe_positions
    )

    # 3. 确定时刻列表
    print("\n[3/4] Determining time moments...")
    if args.sun_hours:
        hours = args.sun_hours
        print(f"Using specified hours: {hours}")
    else:
        hours = np.linspace(0, 23, args.num_moments, dtype=int).tolist()
        print(f"Using {args.num_moments} uniformly distributed hours: {hours}")

    sun_trajectory = []
    for hour in hours:
        sun_dir = calculate_sun_direction_simple(hour)
        sun_trajectory.append((hour, sun_dir))
        print(f"  Hour {hour:02d}:00 - Sun altitude: {np.degrees(np.arcsin(sun_dir[1])):.1f}°")

    # 4. 为每个时刻生成数据
    print(f"\n[4/4] Generating data for {len(sun_trajectory)} moments...")

    for idx, (hour, sun_dir) in enumerate(sun_trajectory):
        print(f"\n=== Processing moment {idx+1}/{len(sun_trajectory)}: Hour {hour:02d}:00 ===")

        # 使用光源修改器
        scene = LightingModifier.modify_scene_lighting(args.scene, sun_dir)

        # 生成该时刻的数据
        moment_dir = args.output / f"moment_{hour:02d}"
        moment_dir.mkdir(parents=True, exist_ok=True)

        # 烘焙SH系数
        print(f"Baking SH coefficients for {len(probe_positions)} probes...")
        sh_coeffs_list = []

        for i, probe_pos in enumerate(tqdm(probe_positions, desc="Baking SH")):
            sh_coeffs = bake_sh_at_probe_core(scene, probe_pos, args.num_sh_samples, args.spp)
            sh_coeffs_list.append(sh_coeffs)

        sh_coeffs_array = np.array(sh_coeffs_list)

        # 保存SH系数
        np.savez_compressed(
            moment_dir / "sh_coeffs.npz",
            coeffs=sh_coeffs_array,
            sun_dir=sun_dir
        )

        # 生成GT图像
        print(f"Rendering {args.num_gt_images} GT images...")
        images_dir = moment_dir / "images"
        images_dir.mkdir(exist_ok=True)

        bbox = scene.bbox()
        center = (bbox.min + bbox.max) / 2
        radius = np.linalg.norm(bbox.max - bbox.min) / 2

        camera_params = []

        for img_id in range(args.num_gt_images):
            # 随机相机位置
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi/3)
            r = radius * np.random.uniform(1.5, 2.5)

            camera_pos = center + np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ])

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

            image = mi.render(scene, sensor=sensor, spp=args.spp)

            mi.util.write_bitmap(str(images_dir / f"{img_id:03d}.exr"), image)
            mi.util.write_bitmap(str(images_dir / f"{img_id:03d}.png"), image)

            camera_params.append({
                'image_id': img_id,
                'camera_pos': np.array(camera_pos).tolist(),
                'look_at': np.array(look_at).tolist(),
            })

        # 保存元数据
        with open(moment_dir / "camera_params.json", 'w') as f:
            json.dump(camera_params, f, indent=2)

        with open(moment_dir / "sun_direction.txt", 'w') as f:
            f.write(f"{sun_dir[0]:.6f} {sun_dir[1]:.6f} {sun_dir[2]:.6f}\n")

        print(f"✓ Moment {hour:02d}:00 completed")

    print("\n" + "="*60)
    print("Dataset generation completed! ✓")
    print(f"Output directory: {args.output}")
    print("="*60)

    return 0


def legacy_main():
    """原有的硬编码配置模式（向后兼容）"""
    print("\n" + "="*60)
    print("WARNING: Running in LEGACY mode")
    print("="*60)
    print("You are using the old hardcoded configuration.")
    print("For better flexibility, use CLI mode with --help for options.")
    print("="*60 + "\n")

    # 原有的硬编码配置
    config = {
        'num_probes': 100,
        'num_moments': 2,
        'num_gt_images': 5,
        'num_sh_samples': 32,
        'spp': 128,
        'output_dir': Path(__file__).parent / 'output' / 'quick_prototype'
    }

    print("="*60)
    print("Multi-Moment Lighting Dataset Generation")
    print("="*60)
    print(f"Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("="*60)

    # 创建输出目录
    config['output_dir'].mkdir(parents=True, exist_ok=True)

    # 1. 采样探针位置
    print("\n[1/3] Sampling probe positions...")
    probe_positions = sample_probe_positions(config['num_probes'])
    print(f"Sampled {len(probe_positions)} probe positions")

    # 保存探针位置
    np.savez_compressed(
        config['output_dir'] / "probes.npz",
        positions=probe_positions
    )

    # 2. 生成太阳轨迹
    print("\n[2/3] Generating sun trajectory...")
    selected_hours = [12, 18]
    sun_trajectory = []

    for hour in selected_hours:
        sun_dir = calculate_sun_direction_simple(hour)
        sun_trajectory.append((hour, sun_dir))
        print(f"  Hour {hour:02d}:00 - Sun altitude: {np.degrees(np.arcsin(sun_dir[1])):.1f}°")

    # 3. 为每个时刻生成数据
    print("\n[3/3] Generating multi-moment data...")

    for idx, (hour, sun_dir) in enumerate(sun_trajectory):
        generate_moment_data(
            moment_id=hour,
            sun_direction=sun_dir,
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

    # 打印数据集结构
    print("\nDataset structure:")
    print(f"{config['output_dir']}/")
    print(f"├── probes.npz")
    for hour in selected_hours:
        print(f"├── moment_{hour:02d}/")
        print(f"│   ├── sh_coeffs.npz")
        print(f"│   ├── sun_direction.txt")
        print(f"│   ├── camera_params.json")
        print(f"│   └── images/")
        print(f"│       ├── 000.exr")
        print(f"│       ├── 000.png")
        print(f"│       └── ...")

    return 0


def main():
    """主入口：检测CLI模式或legacy模式"""
    if len(sys.argv) > 1:
        # CLI模式
        args = parse_args()
        return cli_main(args)
    else:
        # Legacy模式（向后兼容）
        return legacy_main()


if __name__ == "__main__":
    main()
