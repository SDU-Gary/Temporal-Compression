"""
SVD验证脚本: 5D参数化光源低秩假设验证

目标:
    对每个探针的SH矩阵 [41 configs, 27 SH coeffs] 进行SVD分析
    统计rank=5能量占比, 决定是否满足低秩假设 (>90%)

输出:
    - 奇异值衰减曲线可视化
    - 各秩能量占比统计 (均值/最小值/最大值)
    - Go/No-Go决策建议

用法:
    python scripts/verify_5D_low_rank.py --data_dir ../data_generation/output/5D_parametric_validation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json


def load_parametric_dataset(data_dir: Path):
    """加载5D参数化光源数据集

    Returns:
        probe_positions: [N_probes, 3]
        light_configs: [N_configs, 5] - [zenith, azimuth, intensity, temp, cloud]
        sh_tensor: [N_probes, N_configs, 27]
        metadata: dict
    """
    tensor_file = data_dir / 'parametric_tensor.npz'
    metadata_file = data_dir / 'metadata.json'

    if not tensor_file.exists():
        raise FileNotFoundError(f"Tensor file not found: {tensor_file}")

    # 加载NPZ数据
    data = np.load(tensor_file)
    probe_positions = data['probe_positions']  # [N, 3]
    light_configs = data['light_configs']  # [M, 5]
    sh_tensor = data['tensor']  # [N, M, 27]

    # 加载元数据
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    print(f"Loaded dataset from {data_dir}")
    print(f"  Probes: {probe_positions.shape[0]}")
    print(f"  Configs: {light_configs.shape[0]}")
    print(f"  SH order: 3 (27 coefficients)")
    print(f"  Tensor shape: {sh_tensor.shape}")

    return probe_positions, light_configs, sh_tensor, metadata


def analyze_svd_per_probe(sh_tensor: np.ndarray, ranks_to_test=[1, 3, 5, 8, 10]):
    """对每个探针执行SVD分析

    Args:
        sh_tensor: [N_probes, N_configs, 27]
        ranks_to_test: 要分析的秩列表

    Returns:
        results: dict with keys:
            - singular_values: [N_probes, min(N_configs, 27)]
            - energy_by_rank: [N_probes, len(ranks_to_test)]
            - mean_energy: [len(ranks_to_test)]
            - min_energy: [len(ranks_to_test)]
            - max_energy: [len(ranks_to_test)]
    """
    N_probes, N_configs, N_sh = sh_tensor.shape
    max_rank = min(N_configs, N_sh)

    # 存储所有探针的奇异值
    all_singular_values = []
    energy_by_rank = np.zeros((N_probes, len(ranks_to_test)))

    print(f"\nPerforming SVD on {N_probes} probes...")

    for i in range(N_probes):
        # 对当前探针的SH矩阵进行SVD: [N_configs, 27]
        sh_matrix = sh_tensor[i]  # [N_configs, 27]

        U, S, Vt = np.linalg.svd(sh_matrix, full_matrices=False)
        all_singular_values.append(S)

        # 计算总能量
        total_energy = np.sum(S**2)

        # 计算各秩的累积能量占比
        if total_energy > 1e-12:  # 跳过零能量探针
            for j, rank in enumerate(ranks_to_test):
                if rank <= len(S):
                    captured_energy = np.sum(S[:rank]**2)
                    energy_by_rank[i, j] = captured_energy / total_energy
                else:
                    energy_by_rank[i, j] = 1.0
        else:
            # 零能量探针, 标记为NaN (后续统计时排除)
            energy_by_rank[i, :] = np.nan

        if (i + 1) % 25 == 0:
            print(f"  Processed {i+1}/{N_probes} probes")

    # 统计结果 (排除NaN值)
    singular_values = np.array(all_singular_values)  # [N_probes, max_rank]
    mean_energy = np.nanmean(energy_by_rank, axis=0)
    min_energy = np.nanmin(energy_by_rank, axis=0)
    max_energy = np.nanmax(energy_by_rank, axis=0)

    # 统计有效探针数
    valid_probes = np.sum(~np.isnan(energy_by_rank[:, 0]))
    print(f"  Valid probes: {valid_probes}/{N_probes} (excluded {N_probes - valid_probes} zero-energy probes)")

    results = {
        'singular_values': singular_values,
        'energy_by_rank': energy_by_rank,
        'ranks_tested': ranks_to_test,
        'mean_energy': mean_energy,
        'min_energy': min_energy,
        'max_energy': max_energy
    }

    return results


def visualize_svd_results(results: dict, output_dir: Path):
    """可视化SVD分析结果

    生成两张图:
        1. 奇异值衰减曲线 (所有探针平均)
        2. 不同秩的能量占比箱线图
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    singular_values = results['singular_values']
    energy_by_rank = results['energy_by_rank']
    ranks_tested = results['ranks_tested']

    # === 图1: 奇异值衰减曲线 ===
    fig, ax = plt.subplots(figsize=(10, 6))

    # 计算平均奇异值和标准差
    mean_sv = singular_values.mean(axis=0)
    std_sv = singular_values.std(axis=0)

    x = np.arange(len(mean_sv)) + 1
    ax.plot(x, mean_sv, 'b-', linewidth=2, label='Mean')
    ax.fill_between(x, mean_sv - std_sv, mean_sv + std_sv, alpha=0.3, label='±1 Std')

    # 标记测试的秩
    for rank in ranks_tested:
        if rank <= len(mean_sv):
            ax.axvline(rank, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(rank, mean_sv.max() * 0.9, f'r={rank}',
                   rotation=90, va='bottom', ha='right', fontsize=9)

    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Singular Value', fontsize=12)
    ax.set_title('Singular Value Decay (averaged over all probes)', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plot_path = output_dir / 'singular_value_decay.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()

    # === 图2: 能量占比箱线图 ===
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = np.arange(len(ranks_tested))
    bp = ax.boxplot([energy_by_rank[:, i] for i in range(len(ranks_tested))],
                     positions=positions,
                     widths=0.6,
                     patch_artist=True,
                     showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

    # 设置箱体颜色
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    # 添加90%阈值线
    ax.axhline(0.9, color='green', linestyle='--', linewidth=2,
              label='90% threshold (Go/No-Go)')

    ax.set_xticks(positions)
    ax.set_xticklabels([f'Rank {r}' for r in ranks_tested])
    ax.set_ylabel('Energy Captured', fontsize=12)
    ax.set_xlabel('Rank', fontsize=12)
    ax.set_title('Energy Capture Distribution Across Probes', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    ax.legend()

    plot_path = output_dir / 'energy_boxplot.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()


def print_statistics_report(results: dict):
    """打印统计报告"""
    ranks_tested = results['ranks_tested']
    mean_energy = results['mean_energy']
    min_energy = results['min_energy']
    max_energy = results['max_energy']

    print("\n" + "="*70)
    print("SVD Energy Capture Statistics")
    print("="*70)
    print(f"{'Rank':<8} {'Mean':<12} {'Min':<12} {'Max':<12} {'Status':<10}")
    print("-"*70)

    for i, rank in enumerate(ranks_tested):
        mean = mean_energy[i] * 100
        min_val = min_energy[i] * 100
        max_val = max_energy[i] * 100

        # 判断是否满足阈值
        if rank == 5:
            status = "✓ PASS" if mean >= 90.0 else "✗ FAIL"
        else:
            status = ""

        print(f"{rank:<8} {mean:>6.2f}%     {min_val:>6.2f}%     {max_val:>6.2f}%     {status}")

    print("="*70)


def make_decision(results: dict, threshold: float = 0.9):
    """做出Go/No-Go决策

    Args:
        results: SVD分析结果
        threshold: 能量阈值 (默认90%)

    Returns:
        decision: 'GO' or 'NO-GO'
        recommendation: 建议文本
    """
    ranks_tested = results['ranks_tested']
    mean_energy = results['mean_energy']
    min_energy = results['min_energy']

    # 查找rank=5的索引
    try:
        rank5_idx = ranks_tested.index(5)
        rank5_mean = mean_energy[rank5_idx]
        rank5_min = min_energy[rank5_idx]
    except ValueError:
        return 'NO-GO', "Rank 5 was not tested"

    print("\n" + "="*70)
    print("Go/No-Go Decision (Week 2 Milestone)")
    print("="*70)

    # 判断标准
    pass_mean = rank5_mean >= threshold
    pass_min = rank5_min >= threshold * 0.8  # 最小值允许低20%

    if pass_mean and pass_min:
        decision = 'GO'
        recommendation = f"""
✓ GO - 低秩假设成立

理由:
  - Rank-5平均能量: {rank5_mean*100:.2f}% (>= {threshold*100:.0f}%)
  - Rank-5最小能量: {rank5_min*100:.2f}% (>= {threshold*0.8*100:.0f}%)

建议:
  - 继续使用rank=5进行Week 3-4混合模型训练
  - 预期压缩比: ~4.76× (170 params vs 810 Spline)
  - 预期精度: MAE <0.04 (基于文档8经验)
"""
    elif rank5_mean >= threshold * 0.85:
        decision = 'CONDITIONAL GO'
        recommendation = f"""
⚠ CONDITIONAL GO - 低秩假设基本成立, 但需调整

理由:
  - Rank-5平均能量: {rank5_mean*100:.2f}% (略低于{threshold*100:.0f}%)
  - Rank-5最小能量: {rank5_min*100:.2f}%

建议:
  - 方案A: 增加秩至8 (参数: 34*8=272, 压缩比降至2.98×)
  - 方案B: 添加二次交互项 intensity×exp(-cloud) (秩保持5, 基维度7→8)
  - 方案C: 继续rank=5, 但降低MAE目标至0.05
"""
    else:
        decision = 'NO-GO'
        recommendation = f"""
✗ NO-GO - 低秩假设显著偏离预期

理由:
  - Rank-5平均能量: {rank5_mean*100:.2f}% (<< {threshold*100:.0f}%)
  - Rank-5最小能量: {rank5_min*100:.2f}%

应急预案:
  1. 检查数据质量: 是否存在NaN/Inf, SH系数范围是否合理
  2. 增加秩至10-12 (但压缩比显著降低)
  3. 提前进入Week 3 MLP阶段, 用神经网络拟合残差
  4. 重新考虑5D参数化策略 (是否需要降维至3D几何参数?)
"""

    print(recommendation)
    print("="*70)

    return decision, recommendation


def main():
    parser = argparse.ArgumentParser(description='Verify 5D low-rank hypothesis via SVD')
    parser.add_argument('--data_dir', type=str,
                       default='../data_generation/output/5D_parametric_validation',
                       help='Path to 5D parametric dataset')
    parser.add_argument('--output_dir', type=str,
                       default='./svd_analysis',
                       help='Output directory for plots and reports')
    parser.add_argument('--ranks', nargs='+', type=int, default=[1, 3, 5, 8, 10],
                       help='Ranks to test')
    parser.add_argument('--threshold', type=float, default=0.9,
                       help='Energy threshold for Go/No-Go decision')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # 1. 加载数据
    probe_positions, light_configs, sh_tensor, metadata = load_parametric_dataset(data_dir)

    # 2. SVD分析
    results = analyze_svd_per_probe(sh_tensor, ranks_to_test=args.ranks)

    # 3. 可视化
    visualize_svd_results(results, output_dir)

    # 4. 打印统计报告
    print_statistics_report(results)

    # 5. 做出决策
    decision, recommendation = make_decision(results, threshold=args.threshold)

    # 6. 保存结果
    report_path = output_dir / 'svd_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("5D Parametric Light: SVD Low-Rank Analysis Report\n")
        f.write("="*70 + "\n\n")
        f.write(f"Dataset: {data_dir}\n")
        f.write(f"Probes: {probe_positions.shape[0]}\n")
        f.write(f"Configs: {light_configs.shape[0]}\n\n")

        f.write("Energy Capture Statistics:\n")
        f.write("-"*70 + "\n")
        for i, rank in enumerate(results['ranks_tested']):
            mean = results['mean_energy'][i] * 100
            min_val = results['min_energy'][i] * 100
            max_val = results['max_energy'][i] * 100
            f.write(f"Rank {rank}: Mean={mean:.2f}%, Min={min_val:.2f}%, Max={max_val:.2f}%\n")

        f.write("\n" + "="*70 + "\n")
        f.write(f"Decision: {decision}\n")
        f.write("="*70 + "\n")
        f.write(recommendation)

    print(f"\n✓ Report saved: {report_path}")
    print(f"\n✓ SVD verification complete!")


if __name__ == '__main__':
    main()
