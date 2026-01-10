"""
数据集完整性和质量检查脚本

检查项目：
1. 文件完整性检查
2. 数据形状和类型验证
3. 数值范围合理性
4. 球谐系数统计分析
5. 图像质量评估
6. 探针分布可视化
"""

import sys
from pathlib import Path
import numpy as np
import json
from typing import Dict, List, Tuple
import warnings

# 添加utils到路径
sys.path.insert(0, str(Path(__file__).parent / "utils"))


class DatasetValidator:
    """数据集验证器"""

    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        self.errors = []
        self.warnings = []
        self.stats = {}

    def validate(self) -> Dict:
        """执行完整验证"""
        print("="*70)
        print(f"数据集验证报告: {self.dataset_path.name}")
        print("="*70)

        # 1. 文件完整性
        print("\n[1/7] 检查文件完整性...")
        self.check_file_completeness()

        # 2. 配置文件
        print("\n[2/7] 验证配置文件...")
        self.check_config()

        # 3. 探针数据
        print("\n[3/7] 验证探针数据...")
        self.check_probes()

        # 4. 时刻数据
        print("\n[4/7] 验证时刻数据...")
        self.check_moments()

        # 5. 球谐系数
        print("\n[5/7] 分析球谐系数...")
        self.analyze_sh_coefficients()

        # 6. 图像数据
        print("\n[6/7] 检查图像数据...")
        self.check_images()

        # 7. 数据一致性
        print("\n[7/7] 验证数据一致性...")
        self.check_consistency()

        # 生成报告
        self.print_summary()

        return {
            'dataset_path': str(self.dataset_path),
            'errors': self.errors,
            'warnings': self.warnings,
            'stats': self.stats,
            'status': 'PASS' if len(self.errors) == 0 else 'FAIL'
        }

    def check_file_completeness(self):
        """检查必需文件是否存在"""
        required_files = ['config.json', 'probes.npz']

        for filename in required_files:
            filepath = self.dataset_path / filename
            if not filepath.exists():
                self.errors.append(f"缺失必需文件: {filename}")
            else:
                print(f"  ✓ {filename}")

        # 检查时刻目录
        moment_dirs = sorted(self.dataset_path.glob('moment_*'))
        if len(moment_dirs) == 0:
            self.errors.append("没有找到任何时刻数据目录")
        else:
            print(f"  ✓ 发现 {len(moment_dirs)} 个时刻目录")
            self.stats['num_moments'] = len(moment_dirs)

    def check_config(self):
        """验证配置文件"""
        config_path = self.dataset_path / 'config.json'

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            print(f"  场景: {config.get('scene', 'N/A')}")
            print(f"  探针数: {config.get('num_probes', 'N/A')}")
            print(f"  时刻数: {config.get('num_moments', 'N/A')}")
            print(f"  SPP: {config.get('spp', 'N/A')}")
            print(f"  SH采样: {config.get('num_sh_samples', 'N/A')}")
            print(f"  GT图像数: {config.get('num_gt_images', 'N/A')}")
            print(f"  生成时间: {config.get('timestamp', 'N/A')}")

            self.stats['config'] = config

        except Exception as e:
            self.errors.append(f"无法读取config.json: {e}")

    def check_probes(self):
        """验证探针数据"""
        probes_path = self.dataset_path / 'probes.npz'

        try:
            data = np.load(probes_path)
            positions = data['positions']

            print(f"  探针数量: {len(positions)}")
            print(f"  数据形状: {positions.shape}")
            print(f"  数据类型: {positions.dtype}")

            # 检查维度
            if positions.shape[1] != 3:
                self.errors.append(f"探针位置维度错误: 期望3，实际{positions.shape[1]}")

            # 统计边界
            x_min, y_min, z_min = positions.min(axis=0)
            x_max, y_max, z_max = positions.max(axis=0)

            print(f"  探针分布范围:")
            print(f"    X: [{x_min:.2f}, {x_max:.2f}] (宽度: {x_max-x_min:.2f})")
            print(f"    Y: [{y_min:.2f}, {y_max:.2f}] (深度: {y_max-y_min:.2f})")
            print(f"    Z: [{z_min:.2f}, {z_max:.2f}] (高度: {z_max-z_min:.2f})")

            volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
            density = len(positions) / volume if volume > 0 else 0

            print(f"  覆盖体积: {volume:.2f} m³")
            print(f"  探针密度: {density:.4f} 探针/m³")

            self.stats['num_probes'] = len(positions)
            self.stats['probe_bounds'] = {
                'x': (float(x_min), float(x_max)),
                'y': (float(y_min), float(y_max)),
                'z': (float(z_min), float(z_max))
            }
            self.stats['probe_density'] = float(density)

        except Exception as e:
            self.errors.append(f"无法读取探针数据: {e}")

    def check_moments(self):
        """验证时刻数据完整性"""
        moment_dirs = sorted(self.dataset_path.glob('moment_*'))

        for moment_dir in moment_dirs:
            moment_id = moment_dir.name.split('_')[1]
            print(f"\n  检查时刻 {moment_id}:00:")

            # 检查必需文件
            required = ['sh_coeffs.npz', 'sun_direction.txt', 'camera_params.json', 'images']
            missing = []

            for item in required:
                path = moment_dir / item
                if not path.exists():
                    missing.append(item)

            if missing:
                self.errors.append(f"时刻{moment_id}缺失: {', '.join(missing)}")
            else:
                print(f"    ✓ 文件完整")

            # 检查图像数量
            images_dir = moment_dir / 'images'
            if images_dir.exists():
                exr_files = list(images_dir.glob('*.exr'))
                png_files = list(images_dir.glob('*.png'))
                print(f"    ✓ {len(exr_files)} EXR 图像, {len(png_files)} PNG 图像")

                if len(exr_files) != len(png_files):
                    self.warnings.append(f"时刻{moment_id}: EXR和PNG数量不匹配")

    def analyze_sh_coefficients(self):
        """分析球谐系数统计"""
        moment_dirs = sorted(self.dataset_path.glob('moment_*'))

        all_stats = []

        for moment_dir in moment_dirs:
            moment_id = moment_dir.name.split('_')[1]
            sh_path = moment_dir / 'sh_coeffs.npz'

            if not sh_path.exists():
                continue

            try:
                data = np.load(sh_path)
                sh_coeffs = data['coeffs']
                sun_dir = data['sun_dir']

                # 基本统计
                stats = {
                    'moment': moment_id,
                    'shape': sh_coeffs.shape,
                    'mean': float(sh_coeffs.mean()),
                    'std': float(sh_coeffs.std()),
                    'min': float(sh_coeffs.min()),
                    'max': float(sh_coeffs.max()),
                    'sun_direction': sun_dir.tolist(),
                    'sun_altitude': float(np.degrees(np.arcsin(sun_dir[1])))
                }

                # 检查异常值
                if np.any(np.isnan(sh_coeffs)):
                    self.errors.append(f"时刻{moment_id}: SH系数包含NaN")

                if np.any(np.isinf(sh_coeffs)):
                    self.errors.append(f"时刻{moment_id}: SH系数包含Inf")

                # 检查形状
                expected_shape = (self.stats.get('num_probes', 0), 27)
                if sh_coeffs.shape != expected_shape:
                    self.errors.append(
                        f"时刻{moment_id}: SH系数形状错误 "
                        f"(期望{expected_shape}, 实际{sh_coeffs.shape})"
                    )

                all_stats.append(stats)

                print(f"\n  时刻 {moment_id}:00:")
                print(f"    形状: {sh_coeffs.shape}")
                print(f"    太阳方位: [{sun_dir[0]:.3f}, {sun_dir[1]:.3f}, {sun_dir[2]:.3f}]")
                print(f"    太阳高度角: {stats['sun_altitude']:.1f}°")
                print(f"    SH统计: 均值={stats['mean']:.6f}, 标准差={stats['std']:.6f}")
                print(f"           最小值={stats['min']:.6f}, 最大值={stats['max']:.6f}")

            except Exception as e:
                self.errors.append(f"时刻{moment_id}无法分析SH系数: {e}")

        self.stats['sh_statistics'] = all_stats

        # 跨时刻分析
        if len(all_stats) >= 2:
            print("\n  跨时刻分析:")
            means = [s['mean'] for s in all_stats]
            stds = [s['std'] for s in all_stats]
            altitudes = [s['sun_altitude'] for s in all_stats]

            print(f"    太阳高度角范围: {min(altitudes):.1f}° ~ {max(altitudes):.1f}°")
            print(f"    SH均值范围: {min(means):.6f} ~ {max(means):.6f}")
            print(f"    SH标准差范围: {min(stds):.6f} ~ {max(stds):.6f}")

            # 检查时刻间变化
            mean_variation = (max(means) - min(means)) / (np.mean(means) + 1e-10)
            if mean_variation < 0.01:
                self.warnings.append(
                    f"不同时刻的SH系数变化很小 ({mean_variation*100:.2f}%)，"
                    "可能光源修改未生效"
                )

    def check_images(self):
        """检查图像文件"""
        moment_dirs = sorted(self.dataset_path.glob('moment_*'))

        total_exr = 0
        total_png = 0
        total_size = 0

        for moment_dir in moment_dirs:
            images_dir = moment_dir / 'images'
            if not images_dir.exists():
                continue

            exr_files = list(images_dir.glob('*.exr'))
            png_files = list(images_dir.glob('*.png'))

            total_exr += len(exr_files)
            total_png += len(png_files)

            # 计算大小
            for f in exr_files + png_files:
                total_size += f.stat().st_size

        print(f"  总EXR图像: {total_exr}")
        print(f"  总PNG图像: {total_png}")
        print(f"  图像总大小: {total_size / 1024 / 1024:.2f} MB")

        self.stats['total_images'] = total_exr + total_png
        self.stats['image_size_mb'] = total_size / 1024 / 1024

        # 抽样检查图像尺寸（如果安装了OpenEXR）
        try:
            import OpenEXR
            import Imath

            # 随机抽取一个EXR文件检查
            moment_dirs = list(self.dataset_path.glob('moment_*/images/*.exr'))
            if moment_dirs:
                sample_exr = moment_dirs[0]
                exr_file = OpenEXR.InputFile(str(sample_exr))
                header = exr_file.header()
                dw = header['dataWindow']
                width = dw.max.x - dw.min.x + 1
                height = dw.max.y - dw.min.y + 1

                print(f"  图像分辨率: {width}x{height} (抽样自 {sample_exr.name})")
                self.stats['image_resolution'] = (width, height)
        except ImportError:
            print("  (未安装OpenEXR，跳过EXR文件解析)")
        except Exception as e:
            self.warnings.append(f"无法读取EXR文件: {e}")

    def check_consistency(self):
        """检查数据一致性"""
        # 检查探针数与SH系数数量一致性
        if 'num_probes' in self.stats and 'sh_statistics' in self.stats:
            for stat in self.stats['sh_statistics']:
                if stat['shape'][0] != self.stats['num_probes']:
                    self.errors.append(
                        f"时刻{stat['moment']}: 探针数不一致 "
                        f"(probes.npz有{self.stats['num_probes']}个, "
                        f"SH系数有{stat['shape'][0]}个)"
                    )

        # 检查配置与实际数据一致性
        if 'config' in self.stats:
            config = self.stats['config']

            # 检查时刻数
            if 'num_moments' in self.stats:
                actual_moments = self.stats['num_moments']
                if 'sun_hours' in config:
                    expected_moments = len(config['sun_hours'])
                    if actual_moments != expected_moments:
                        self.warnings.append(
                            f"时刻数不匹配: 配置{expected_moments}, 实际{actual_moments}"
                        )

        print("  ✓ 数据一致性检查完成")

    def print_summary(self):
        """打印验证摘要"""
        print("\n" + "="*70)
        print("验证摘要")
        print("="*70)

        # 基本信息
        print("\n📊 数据集概览:")
        print(f"  路径: {self.dataset_path}")
        print(f"  探针数: {self.stats.get('num_probes', 'N/A')}")
        print(f"  时刻数: {self.stats.get('num_moments', 'N/A')}")
        print(f"  总图像数: {self.stats.get('total_images', 'N/A')}")
        print(f"  数据大小: {self.stats.get('image_size_mb', 0):.2f} MB")

        if 'probe_density' in self.stats:
            print(f"  探针密度: {self.stats['probe_density']:.4f} 探针/m³")

        # 错误和警告
        print(f"\n🔍 检查结果:")
        print(f"  ❌ 错误数: {len(self.errors)}")
        print(f"  ⚠️  警告数: {len(self.warnings)}")

        if self.errors:
            print("\n错误详情:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")

        if self.warnings:
            print("\n警告详情:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")

        # 最终判定
        print("\n" + "="*70)
        if len(self.errors) == 0:
            print("✅ 验证通过！数据集质量良好，可用于训练。")
        else:
            print("❌ 验证失败！请修复上述错误后重新生成数据。")
        print("="*70 + "\n")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='验证数据集完整性和质量')
    parser.add_argument('dataset_path', type=Path, help='数据集路径')
    parser.add_argument('--save-report', type=Path, help='保存验证报告为JSON')

    args = parser.parse_args()

    # 执行验证
    validator = DatasetValidator(args.dataset_path)
    report = validator.validate()

    # 保存报告
    if args.save_report:
        with open(args.save_report, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n报告已保存到: {args.save_report}")

    # 返回状态码
    return 0 if report['status'] == 'PASS' else 1


if __name__ == "__main__":
    sys.exit(main())
