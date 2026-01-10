"""
球谐系数质量深度分析

检查：
1. SH系数的能量分布
2. 各阶球谐基函数的贡献
3. RGB通道一致性
4. 重建误差估算
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "utils"))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from spherical_harmonics import reconstruct_from_sh, fibonacci_sphere

def analyze_sh_order_contribution(sh_coeffs: np.ndarray):
    """分析各阶球谐基函数的能量贡献"""
    # sh_coeffs shape: (N, 27) = (N, 9 bases × 3 RGB)

    # 重组为 (N, 3, 9)
    sh_reshaped = sh_coeffs.reshape(-1, 3, 9)

    # 各阶能量: order 0 (1个), order 1 (3个), order 2 (5个)
    order_0_energy = np.mean(sh_reshaped[:, :, 0:1]**2)  # L=0: 1个基
    order_1_energy = np.mean(sh_reshaped[:, :, 1:4]**2)  # L=1: 3个基
    order_2_energy = np.mean(sh_reshaped[:, :, 4:9]**2)  # L=2: 5个基

    total_energy = order_0_energy + order_1_energy + order_2_energy

    return {
        'order_0': float(order_0_energy),
        'order_1': float(order_1_energy),
        'order_2': float(order_2_energy),
        'total': float(total_energy),
        'order_0_ratio': float(order_0_energy / (total_energy + 1e-10)),
        'order_1_ratio': float(order_1_energy / (total_energy + 1e-10)),
        'order_2_ratio': float(order_2_energy / (total_energy + 1e-10))
    }

def analyze_rgb_consistency(sh_coeffs: np.ndarray):
    """分析RGB三通道的一致性"""
    sh_reshaped = sh_coeffs.reshape(-1, 3, 9)

    r_channel = sh_reshaped[:, 0, :]  # (N, 9)
    g_channel = sh_reshaped[:, 1, :]
    b_channel = sh_reshaped[:, 2, :]

    # 计算通道间相关性
    corr_rg = np.corrcoef(r_channel.flatten(), g_channel.flatten())[0, 1]
    corr_rb = np.corrcoef(r_channel.flatten(), b_channel.flatten())[0, 1]
    corr_gb = np.corrcoef(g_channel.flatten(), b_channel.flatten())[0, 1]

    return {
        'r_g_correlation': float(corr_rg),
        'r_b_correlation': float(corr_rb),
        'g_b_correlation': float(corr_gb),
        'avg_correlation': float((corr_rg + corr_rb + corr_gb) / 3)
    }

def estimate_reconstruction_quality(sh_coeffs: np.ndarray, num_samples: int = 100):
    """估算重建质量（采样部分探针）"""
    sample_indices = np.random.choice(len(sh_coeffs), min(num_samples, len(sh_coeffs)), replace=False)

    test_directions = fibonacci_sphere(64)

    energies = []
    non_negative_ratio = []

    for idx in sample_indices:
        sh = sh_coeffs[idx]
        reconstructed = reconstruct_from_sh(sh, test_directions, max_order=2)

        # 计算总能量
        energy = reconstructed.mean()
        energies.append(energy)

        # 非负比例
        non_neg = (reconstructed >= 0).sum() / reconstructed.size
        non_negative_ratio.append(non_neg)

    return {
        'mean_energy': float(np.mean(energies)),
        'std_energy': float(np.std(energies)),
        'min_energy': float(np.min(energies)),
        'max_energy': float(np.max(energies)),
        'non_negative_ratio': float(np.mean(non_negative_ratio))
    }

def main():
    dataset_path = Path('output/method_test_v1')

    print("="*70)
    print("球谐系数质量深度分析")
    print("="*70)

    moment_dirs = sorted(dataset_path.glob('moment_*'))

    all_results = {}

    for moment_dir in moment_dirs:
        moment_id = moment_dir.name.split('_')[1]
        sh_path = moment_dir / 'sh_coeffs.npz'

        if not sh_path.exists():
            continue

        data = np.load(sh_path)
        sh_coeffs = data['coeffs']
        sun_dir = data['sun_dir']

        print(f"\n{'='*70}")
        print(f"时刻 {moment_id}:00 (太阳高度: {np.degrees(np.arcsin(sun_dir[1])):.1f}°)")
        print('='*70)

        # 1. 各阶能量分析
        print("\n[1] 球谐阶数能量贡献:")
        order_analysis = analyze_sh_order_contribution(sh_coeffs)
        print(f"  Order 0 (常数): {order_analysis['order_0']:.6f} ({order_analysis['order_0_ratio']*100:.2f}%)")
        print(f"  Order 1 (线性): {order_analysis['order_1']:.6f} ({order_analysis['order_1_ratio']*100:.2f}%)")
        print(f"  Order 2 (二次): {order_analysis['order_2']:.6f} ({order_analysis['order_2_ratio']*100:.2f}%)")
        print(f"  总能量: {order_analysis['total']:.6f}")

        # 2. RGB一致性分析
        print("\n[2] RGB通道一致性:")
        rgb_analysis = analyze_rgb_consistency(sh_coeffs)
        print(f"  R-G 相关性: {rgb_analysis['r_g_correlation']:.4f}")
        print(f"  R-B 相关性: {rgb_analysis['r_b_correlation']:.4f}")
        print(f"  G-B 相关性: {rgb_analysis['g_b_correlation']:.4f}")
        print(f"  平均相关性: {rgb_analysis['avg_correlation']:.4f}")

        # 3. 重建质量
        print("\n[3] 重建质量估算 (采样100个探针):")
        recon_analysis = estimate_reconstruction_quality(sh_coeffs)
        print(f"  平均辐射能量: {recon_analysis['mean_energy']:.4f} ± {recon_analysis['std_energy']:.4f}")
        print(f"  能量范围: [{recon_analysis['min_energy']:.4f}, {recon_analysis['max_energy']:.4f}]")
        print(f"  非负值比例: {recon_analysis['non_negative_ratio']*100:.2f}%")

        # 4. 异常值检测
        print("\n[4] 异常值检测:")
        nan_count = np.isnan(sh_coeffs).sum()
        inf_count = np.isinf(sh_coeffs).sum()
        very_large = (np.abs(sh_coeffs) > 100).sum()

        print(f"  NaN值: {nan_count}")
        print(f"  Inf值: {inf_count}")
        print(f"  极大值(|x|>100): {very_large} ({very_large/sh_coeffs.size*100:.3f}%)")

        all_results[moment_id] = {
            'order_contribution': order_analysis,
            'rgb_consistency': rgb_analysis,
            'reconstruction': recon_analysis,
            'anomalies': {
                'nan': int(nan_count),
                'inf': int(inf_count),
                'very_large': int(very_large)
            }
        }

    # 跨时刻总结
    print("\n" + "="*70)
    print("跨时刻质量总结")
    print("="*70)

    order_0_ratios = [r['order_contribution']['order_0_ratio'] for r in all_results.values()]
    order_1_ratios = [r['order_contribution']['order_1_ratio'] for r in all_results.values()]
    order_2_ratios = [r['order_contribution']['order_2_ratio'] for r in all_results.values()]

    print(f"\n球谐阶数能量占比 (平均):")
    print(f"  Order 0: {np.mean(order_0_ratios)*100:.2f}% ± {np.std(order_0_ratios)*100:.2f}%")
    print(f"  Order 1: {np.mean(order_1_ratios)*100:.2f}% ± {np.std(order_1_ratios)*100:.2f}%")
    print(f"  Order 2: {np.mean(order_2_ratios)*100:.2f}% ± {np.std(order_2_ratios)*100:.2f}%")

    rgb_corrs = [r['rgb_consistency']['avg_correlation'] for r in all_results.values()]
    print(f"\nRGB通道相关性 (平均): {np.mean(rgb_corrs):.4f} ± {np.std(rgb_corrs):.4f}")

    energies = [r['reconstruction']['mean_energy'] for r in all_results.values()]
    print(f"\n辐射能量范围: {min(energies):.4f} ~ {max(energies):.4f}")
    print(f"  变化幅度: {(max(energies) - min(energies)) / (np.mean(energies) + 1e-10) * 100:.2f}%")

    # 质量判定
    print("\n" + "="*70)
    print("质量判定")
    print("="*70)

    issues = []

    # 检查能量分布
    if np.mean(order_0_ratios) > 0.8:
        issues.append("⚠ Order 0能量过高，可能场景光照过于均匀")

    # 检查RGB一致性
    if np.mean(rgb_corrs) < 0.7:
        issues.append("⚠ RGB通道相关性较低，可能有色彩异常")

    # 检查时刻间变化
    if (max(energies) - min(energies)) / (np.mean(energies) + 1e-10) < 0.05:
        issues.append("⚠ 不同时刻能量变化很小(<5%)，光源修改可能未生效")

    # 检查异常值
    total_anomalies = sum(r['anomalies']['nan'] + r['anomalies']['inf'] for r in all_results.values())
    if total_anomalies > 0:
        issues.append(f"❌ 发现{total_anomalies}个数值异常（NaN/Inf）")

    if not issues:
        print("✅ 所有质量指标正常，数据可用于训练！")
    else:
        print("发现以下问题：")
        for issue in issues:
            print(f"  {issue}")

    print("\n" + "="*70)

    # 保存结果
    import json
    with open(dataset_path / 'sh_quality_analysis.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n分析结果已保存到: {dataset_path / 'sh_quality_analysis.json'}")

if __name__ == "__main__":
    main()
