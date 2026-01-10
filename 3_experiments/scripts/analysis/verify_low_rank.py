#!/usr/bin/env python3
"""
验证低秩假设：检查SH系数矩阵的秩

这个脚本分析Cornell Box和House场景的SH系数，
验证是否存在低秩结构，为低秩时间压缩方法提供理论依据。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_all_sh(dataset_path):
    """
    加载数据集中所有时刻的SH系数

    Args:
        dataset_path: 数据集路径

    Returns:
        sh_matrix: [T, 27] 所有时刻的SH系数
        hours: [T] 对应的小时数
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # 查找所有moment目录
    moment_dirs = sorted(dataset_path.glob('moment_*'))

    if not moment_dirs:
        raise ValueError(f"No moment directories found in {dataset_path}")

    sh_list = []
    hours_list = []

    for moment_dir in moment_dirs:
        # 提取小时
        hour_str = moment_dir.name.split('_')[1]
        hour = int(hour_str)

        # 加载SH系数
        sh_file = moment_dir / 'sh_coeffs.npz'

        if not sh_file.exists():
            print(f"Warning: {sh_file} not found, skipping")
            continue

        data = np.load(sh_file)
        sh = data['coeffs']  # [1, 27] 单探针

        sh_list.append(sh[0])  # 取第一个探针
        hours_list.append(hour)

    if not sh_list:
        raise ValueError(f"No SH data loaded from {dataset_path}")

    sh_matrix = np.array(sh_list)  # [T, 27]
    hours = np.array(hours_list)

    print(f"  Loaded {len(hours)} time steps: {hours.tolist()}")
    print(f"  SH matrix shape: {sh_matrix.shape}")

    return sh_matrix, hours


def analyze_low_rank(sh_matrix, scene_name):
    """
    分析SH矩阵的低秩性质

    Args:
        sh_matrix: [T, 27] SH系数矩阵
        scene_name: 场景名称

    Returns:
        analysis: 包含SVD结果和统计信息的字典
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {scene_name}")
    print(f"{'='*60}")

    # SVD分解
    U, S, Vt = np.linalg.svd(sh_matrix, full_matrices=False)

    # 归一化奇异值
    S_normalized = S / S.sum()

    # 累积能量
    cumulative_energy = np.cumsum(S) / S.sum()

    # 统计
    total_rank = len(S)

    analysis = {
        'U': U,
        'S': S,
        'Vt': Vt,
        'S_normalized': S_normalized,
        'cumulative_energy': cumulative_energy,
        'total_rank': total_rank
    }

    # 打印统计信息
    print(f"\nRank Statistics:")
    print(f"  Total possible rank: {total_rank}")
    print(f"  Top singular values (normalized):")
    for i in range(min(10, total_rank)):
        print(f"    σ_{i+1}: {S_normalized[i]:.4f} ({S_normalized[i]*100:.2f}%)")

    print(f"\nCumulative Energy:")
    thresholds = [0.90, 0.95, 0.99, 0.999]
    for threshold in thresholds:
        rank_needed = np.argmax(cumulative_energy >= threshold) + 1
        actual_energy = cumulative_energy[rank_needed - 1]
        print(f"  {threshold*100:.1f}%: rank {rank_needed} (actual: {actual_energy*100:.2f}%)")

    # 关键指标
    rank_95 = np.argmax(cumulative_energy >= 0.95) + 1
    rank_99 = np.argmax(cumulative_energy >= 0.99) + 1

    print(f"\n✓ Effective Rank (95%): {rank_95}")
    print(f"✓ Effective Rank (99%): {rank_99}")

    # 判断低秩性
    if rank_95 <= 5:
        print(f"✓✓✓ EXCELLENT: 前{rank_95}个分量就能保留95%能量！")
        print("    → 低秩假设强烈成立，方法极具潜力！")
    elif rank_95 <= 8:
        print(f"✓✓ GOOD: 前{rank_95}个分量保留95%能量")
        print("    → 低秩假设成立，方法可行")
    elif rank_95 <= total_rank * 0.5:
        print(f"✓ OK: 前{rank_95}个分量保留95%能量")
        print("    → 有一定低秩性，可以尝试")
    else:
        print(f"✗ 需要{rank_95}个分量（占{rank_95/total_rank*100:.1f}%）")
        print("    → 低秩假设较弱")

    return analysis


def plot_svd_analysis(analyses, output_path):
    """
    可视化SVD分析结果

    Args:
        analyses: 场景分析结果字典
        output_path: 输出图片路径
    """
    num_scenes = len(analyses)
    fig, axes = plt.subplots(num_scenes, 3, figsize=(15, 5*num_scenes))

    if num_scenes == 1:
        axes = axes.reshape(1, -1)

    for idx, (scene_name, analysis) in enumerate(analyses.items()):
        S_normalized = analysis['S_normalized']
        cumulative_energy = analysis['cumulative_energy']
        rank_95 = np.argmax(cumulative_energy >= 0.95) + 1

        # 1. 奇异值分布（线性尺度）
        ax1 = axes[idx, 0]
        ax1.plot(range(1, len(S_normalized)+1), S_normalized, 'o-', linewidth=2, markersize=6)
        ax1.axvline(rank_95, color='r', linestyle='--', alpha=0.5, label=f'Rank-{rank_95} (95%)')
        ax1.set_xlabel('Singular Value Index', fontsize=11)
        ax1.set_ylabel('Normalized Singular Value', fontsize=11)
        ax1.set_title(f'{scene_name}: Singular Value Distribution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. 奇异值分布（对数尺度）
        ax2 = axes[idx, 1]
        ax2.semilogy(range(1, len(S_normalized)+1), S_normalized, 'o-', linewidth=2, markersize=6)
        ax2.axvline(rank_95, color='r', linestyle='--', alpha=0.5, label=f'Rank-{rank_95} (95%)')
        ax2.set_xlabel('Singular Value Index', fontsize=11)
        ax2.set_ylabel('Normalized Singular Value (log)', fontsize=11)
        ax2.set_title(f'{scene_name}: Singular Values (Log Scale)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 3. 累积能量
        ax3 = axes[idx, 2]
        ax3.plot(range(1, len(cumulative_energy)+1), cumulative_energy, 'o-', linewidth=2, markersize=6)
        ax3.axhline(0.90, color='orange', linestyle='--', alpha=0.7, label='90%')
        ax3.axhline(0.95, color='red', linestyle='--', alpha=0.7, label='95%')
        ax3.axhline(0.99, color='purple', linestyle='--', alpha=0.7, label='99%')
        ax3.axvline(rank_95, color='r', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Number of Components', fontsize=11)
        ax3.set_ylabel('Cumulative Energy', fontsize=11)
        ax3.set_title(f'{scene_name}: Energy Retention', fontsize=12, fontweight='bold')
        ax3.set_ylim([0.5, 1.02])
        ax3.grid(True, alpha=0.3)
        ax3.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")


def compare_compression_ratios(analyses):
    """
    对比不同秩选择下的压缩比

    Args:
        analyses: 场景分析结果字典
    """
    print(f"\n{'='*60}")
    print("COMPRESSION RATIO ANALYSIS")
    print(f"{'='*60}")

    for scene_name, analysis in analyses.items():
        cumulative_energy = analysis['cumulative_energy']

        print(f"\n{scene_name}:")
        print(f"  Baseline (Spline): 351 params (27 * 13 hours)")
        print(f"\n  Low-Rank Method (Physics Basis):")

        for threshold, desc in [(0.95, "95%"), (0.99, "99%")]:
            rank = np.argmax(cumulative_energy >= threshold) + 1

            # 物理基：U(27×k) + coeffs(k×3)
            params_physics = 27 * rank + rank * 3
            compression_ratio_physics = 351 / params_physics

            # 神经基：U(27×k) + MLP(2→32→k)
            params_neural = 27 * rank + (2*32 + 32 + 32*rank)
            compression_ratio_neural = 351 / params_neural

            print(f"\n    Rank {rank} (Energy: {threshold*100}%):")
            print(f"      Physics basis: {params_physics} params → {compression_ratio_physics:.2f}x compression")
            print(f"      Neural basis:  {params_neural} params → {compression_ratio_neural:.2f}x compression")

            if compression_ratio_physics > 2:
                print(f"      ✓✓ Physics basis 超越2x压缩！")
            elif compression_ratio_physics > 1.5:
                print(f"      ✓ Physics basis 有明显压缩优势")
            else:
                print(f"      △ Physics basis 压缩优势较小")


def main():
    """主函数"""

    # 数据集路径
    datasets = {
        'Cornell Box': Path('/home/kyrie/毕设/data_generation/output/level2_tpe/cornell-box_test'),
        'House': Path('/home/kyrie/毕设/data_generation/output/level2_tpe/house_p0')
    }

    # 输出目录
    output_dir = Path('/home/kyrie/毕设/multi_time_compression/output/low_rank_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("LOW-RANK VERIFICATION FOR TEMPORAL COMPRESSION")
    print("="*60)

    # 分析所有场景
    analyses = {}

    for scene_name, dataset_path in datasets.items():
        try:
            print(f"\nLoading {scene_name}...")
            sh_matrix, hours = load_all_sh(dataset_path)

            analysis = analyze_low_rank(sh_matrix, scene_name)
            analyses[scene_name] = analysis

        except Exception as e:
            print(f"✗ Error processing {scene_name}: {e}")
            continue

    if not analyses:
        print("\n✗ No data loaded successfully!")
        return

    # 可视化
    plot_path = output_dir / 'svd_analysis.png'
    plot_svd_analysis(analyses, plot_path)

    # 压缩比分析
    compare_compression_ratios(analyses)

    # 最终结论
    print(f"\n{'='*60}")
    print("FINAL CONCLUSION")
    print(f"{'='*60}")

    all_excellent = all(
        np.argmax(a['cumulative_energy'] >= 0.95) + 1 <= 5
        for a in analyses.values()
    )

    all_good = all(
        np.argmax(a['cumulative_energy'] >= 0.95) + 1 <= 8
        for a in analyses.values()
    )

    if all_excellent:
        print("\n🎉🎉🎉 BREAKTHROUGH! 🎉🎉🎉")
        print("所有场景都显示出极强的低秩性！")
        print("→ 低秩时间压缩方法极具潜力")
        print("→ 建议立即实现物理基版本")
    elif all_good:
        print("\n✓✓ EXCELLENT!")
        print("所有场景都证实了低秩假设")
        print("→ 低秩时间压缩方法可行")
        print("→ 建议从物理基开始尝试")
    else:
        print("\n✓ FEASIBLE")
        print("数据显示一定程度的低秩性")
        print("→ 可以尝试低秩方法")
        print("→ 可能需要更高的秩(k>8)")

    print(f"\n✓ All results saved to: {output_dir}")
    print(f"✓ Review the plots in: {plot_path}")


if __name__ == '__main__':
    main()
