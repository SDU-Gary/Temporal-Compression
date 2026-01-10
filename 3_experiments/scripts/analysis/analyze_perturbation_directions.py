"""
实验C: 扰动方向分析

分析TPE优化得到的扰动向量ε(t)的方向特性，验证是否具有物理意义。

关键问题：
1. 扰动方向是否一致？（不是随机的）
2. 扰动方向是否与太阳运动相关？
3. 扰动的主方向是什么？（PCA分析）

Usage:
    python analyze_perturbation_directions.py --results <path_to_validation_results.npz>
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse


def analyze_perturbation_directions(
    perturbations: np.ndarray,  # [T, 3]
    sun_dirs: np.ndarray,       # [T, 3]
    hours: np.ndarray,          # [T]
    reference_idx: int,
    scene_name: str = "unknown"
):
    """
    全面分析扰动方向的物理意义
    """
    T = len(perturbations)

    print(f"\n{'='*70}")
    print(f"PERTURBATION DIRECTION ANALYSIS")
    print(f"{'='*70}")
    print(f"\nScene: {scene_name}")
    print(f"Time steps: {T}")
    print(f"Reference: hour {hours[reference_idx]} (index {reference_idx})")

    # =========================================================================
    # 1. 方向归一化
    # =========================================================================

    pert_norms = np.linalg.norm(perturbations, axis=1, keepdims=True)
    pert_dirs = perturbations / (pert_norms + 1e-8)

    # 过滤掉参考时刻（扰动为0）
    valid_mask = pert_norms.squeeze() > 1e-6
    valid_indices = np.where(valid_mask)[0]

    print(f"\nValid perturbation vectors: {valid_mask.sum()}/{T}")

    # =========================================================================
    # 2. 方向一致性分析
    # =========================================================================

    print(f"\n{'='*70}")
    print(f"1. DIRECTION CONSISTENCY")
    print(f"{'='*70}")

    # Cosine similarity矩阵
    cos_sim_matrix = pert_dirs @ pert_dirs.T

    # 只考虑有效向量间的相似度
    valid_cos_sims = []
    for i in valid_indices:
        for j in valid_indices:
            if i < j:
                valid_cos_sims.append(cos_sim_matrix[i, j])

    if valid_cos_sims:
        mean_cos_sim = np.mean(valid_cos_sims)
        std_cos_sim = np.std(valid_cos_sims)
        min_cos_sim = np.min(valid_cos_sims)

        print(f"\nCosine Similarity between directions:")
        print(f"  Mean: {mean_cos_sim:.4f}")
        print(f"  Std:  {std_cos_sim:.4f}")
        print(f"  Min:  {min_cos_sim:.4f}")

        if mean_cos_sim > 0.9:
            print(f"\n  ✓✓✓ HIGHLY CONSISTENT directions (mean cos > 0.9)")
            print(f"      → Perturbations are NOT random!")
            print(f"      → Strong evidence for physical meaning")
        elif mean_cos_sim > 0.7:
            print(f"\n  ✓✓ MODERATELY CONSISTENT directions (mean cos > 0.7)")
            print(f"     → Perturbations have some structure")
        elif mean_cos_sim > 0.5:
            print(f"\n  ✓ WEAKLY CONSISTENT directions (mean cos > 0.5)")
            print(f"    → Some pattern but noisy")
        else:
            print(f"\n  ✗ INCONSISTENT directions (mean cos < 0.5)")
            print(f"    → Perturbations may be random or optimization failed")

    # =========================================================================
    # 3. 与太阳方向的相关性
    # =========================================================================

    print(f"\n{'='*70}")
    print(f"2. CORRELATION WITH SUN MOVEMENT")
    print(f"{'='*70}")

    # 太阳方向相对于参考时刻的变化
    sun_diffs = sun_dirs - sun_dirs[reference_idx:reference_idx+1]
    sun_diff_norms = np.linalg.norm(sun_diffs, axis=1, keepdims=True)
    sun_diff_dirs = sun_diffs / (sun_diff_norms + 1e-8)

    # 计算扰动方向与太阳方向变化的夹角
    angles = []
    for idx in valid_indices:
        cos_angle = np.dot(pert_dirs[idx], sun_diff_dirs[idx])
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        angles.append(angle)

    angles = np.array(angles)
    angles_deg = np.degrees(angles)

    print(f"\nAngle between perturbation and sun direction change:")
    print(f"  Mean: {angles_deg.mean():.2f}°")
    print(f"  Std:  {angles_deg.std():.2f}°")
    print(f"  Range: [{angles_deg.min():.2f}°, {angles_deg.max():.2f}°]")

    # 解释
    mean_angle = angles_deg.mean()
    if mean_angle < 30:
        print(f"\n  ✓✓✓ ALIGNED with sun movement (< 30°)")
        print(f"      → Perturbations follow sun path")
        print(f"      → Physical interpretation: 'Move probe to compensate sun'")
    elif mean_angle < 60:
        print(f"\n  ✓✓ CORRELATED with sun movement (30° - 60°)")
        print(f"     → Perturbations partially follow sun")
    elif 120 < mean_angle < 150:
        print(f"\n  ✓✓ OPPOSITE to sun movement (120° - 150°)")
        print(f"     → Perturbations in reverse direction")
        print(f"     → Also physically meaningful (shadow compensation)")
    elif 150 < mean_angle:
        print(f"\n  ✓✓✓ ANTI-ALIGNED with sun movement (> 150°)")
        print(f"      → Strong reverse correlation")
    else:
        print(f"\n  ✗ ORTHOGONAL to sun movement (60° - 120°)")
        print(f"    → No clear sun correlation")
        print(f"    → May indicate scene geometry effects")

    # =========================================================================
    # 4. PCA主方向提取
    # =========================================================================

    print(f"\n{'='*70}")
    print(f"3. PRINCIPAL COMPONENT ANALYSIS")
    print(f"{'='*70}")

    # 对所有有效扰动做PCA
    pca = PCA(n_components=3)
    pca.fit(perturbations[valid_mask])

    principal_direction = pca.components_[0]
    explained_variance = pca.explained_variance_ratio_

    print(f"\nPrincipal directions and explained variance:")
    for i in range(3):
        print(f"  PC{i+1}: direction=[{pca.components_[i][0]:+.3f}, {pca.components_[i][1]:+.3f}, {pca.components_[i][2]:+.3f}], "
              f"variance={explained_variance[i]:.4f} ({explained_variance[i]*100:.1f}%)")

    if explained_variance[0] > 0.8:
        print(f"\n  ✓✓✓ DOMINANT SINGLE DIRECTION (PC1 > 80%)")
        print(f"      → Perturbations are essentially 1-dimensional")
        print(f"      → Can use direction-constrained TPE optimization")
        print(f"      → Principal direction: [{principal_direction[0]:+.3f}, {principal_direction[1]:+.3f}, {principal_direction[2]:+.3f}]")
    elif explained_variance[0] > 0.6:
        print(f"\n  ✓✓ STRONG PRINCIPAL DIRECTION (PC1 > 60%)")
        print(f"     → Mostly 1D with some variation")
        print(f"     → Direction constraint may help")
    else:
        print(f"\n  ✓ MULTI-DIMENSIONAL (PC1 < 60%)")
        print(f"    → Perturbations span multiple directions")
        print(f"    → Scene geometry may require 3D freedom")

    # =========================================================================
    # 5. 时间演化可视化
    # =========================================================================

    print(f"\n{'='*70}")
    print(f"4. TEMPORAL EVOLUTION")
    print(f"{'='*70}")

    # 打印每个时刻的扰动信息
    print(f"\nPerturbation details by time:")
    print(f"{'Hour':>6} {'||ε||':>8} {'Direction (x, y, z)':>25} {'∠Sun':>8}")
    print(f"{'-'*60}")

    for i, hour in enumerate(hours):
        norm = pert_norms[i, 0]
        if norm > 1e-6:
            angle_to_sun = angles_deg[list(valid_indices).index(i)] if i in valid_indices else 0
            print(f"{hour:6d} {norm:8.3f}m  [{pert_dirs[i][0]:+.3f}, {pert_dirs[i][1]:+.3f}, {pert_dirs[i][2]:+.3f}]  {angle_to_sun:7.1f}°")
        else:
            print(f"{hour:6d} {'0.000m':>8}  [  REF: reference time, ε=0  ]        -")

    # =========================================================================
    # 6. 可视化（保存图表）
    # =========================================================================

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 6.1 扰动向量3D图
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    for i in valid_indices:
        ax.quiver(0, 0, 0,
                  perturbations[i, 0], perturbations[i, 1], perturbations[i, 2],
                  color='blue', alpha=0.6, arrow_length_ratio=0.1)

    # 添加主方向
    scale = np.max(pert_norms) * 1.2
    ax.quiver(0, 0, 0,
              principal_direction[0]*scale,
              principal_direction[1]*scale,
              principal_direction[2]*scale,
              color='red', linewidth=3, arrow_length_ratio=0.1,
              label='Principal Direction')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Perturbation Vectors ε(t)')
    ax.legend()

    # 6.2 扰动范数 vs 太阳角度变化
    axes[0, 1].scatter(sun_diff_norms[valid_mask], pert_norms[valid_mask])
    axes[0, 1].set_xlabel('||Δsun_dir|| (unit)')
    axes[0, 1].set_ylabel('||ε|| (m)')
    axes[0, 1].set_title(f'Perturbation Norm vs Sun Movement (corr={np.corrcoef(sun_diff_norms[valid_mask].squeeze(), pert_norms[valid_mask].squeeze())[0,1]:.3f})')
    axes[0, 1].grid(True)

    # 6.3 扰动方向与太阳方向夹角
    axes[1, 0].plot(hours[valid_indices], angles_deg, 'o-')
    axes[1, 0].axhline(y=30, color='g', linestyle='--', label='Aligned threshold (30°)')
    axes[1, 0].axhline(y=150, color='g', linestyle='--', label='Anti-aligned threshold (150°)')
    axes[1, 0].set_xlabel('Hour')
    axes[1, 0].set_ylabel('Angle to Sun Direction Change (°)')
    axes[1, 0].set_title('Perturbation-Sun Alignment over Time')
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # 6.4 方向一致性热图
    im = axes[1, 1].imshow(cos_sim_matrix[valid_mask][:, valid_mask],
                          cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 1].set_title('Direction Cosine Similarity Matrix')
    axes[1, 1].set_xlabel('Time step')
    axes[1, 1].set_ylabel('Time step')
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    output_path = Path(__file__).parent / f'perturbation_direction_analysis_{scene_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")

    # =========================================================================
    # 7. 总结与建议
    # =========================================================================

    print(f"\n{'='*70}")
    print(f"SUMMARY AND RECOMMENDATIONS")
    print(f"{'='*70}")

    consistency_score = mean_cos_sim if valid_cos_sims else 0
    sun_alignment_score = 1.0 - abs(mean_angle - 0) / 180  # 0° or 180° is best
    pca_score = explained_variance[0]

    overall_score = (consistency_score + sun_alignment_score + pca_score) / 3

    print(f"\nOverall Direction Quality Scores:")
    print(f"  Consistency:    {consistency_score:.3f} / 1.0")
    print(f"  Sun Alignment:  {sun_alignment_score:.3f} / 1.0")
    print(f"  PCA Dominance:  {pca_score:.3f} / 1.0")
    print(f"  ----------------------------------")
    print(f"  Overall:        {overall_score:.3f} / 1.0")

    if overall_score > 0.75:
        print(f"\n  ✓✓✓ EXCELLENT direction quality!")
        print(f"      → Perturbations have STRONG physical meaning")
        print(f"      → TPE hypothesis is VALID")
        print(f"      → Recommendation: Use direction-constrained TPE (Experiment G)")
    elif overall_score > 0.5:
        print(f"\n  ✓✓ GOOD direction quality")
        print(f"     → Perturbations show physical patterns")
        print(f"     → TPE is promising but needs refinement")
        print(f"     → Recommendation: Continue with Experiment F (real field test)")
    else:
        print(f"\n  ✗ POOR direction quality")
        print(f"    → Perturbations lack clear physical meaning")
        print(f"    → Current TPE optimization may have failed")
        print(f"    → Recommendation: Improve optimization (better initialization/regularization)")

    print(f"\n{'='*70}\n")

    # 返回分析结果
    return {
        'mean_cosine_similarity': consistency_score,
        'angle_to_sun_mean_deg': mean_angle if len(angles_deg) > 0 else 0,
        'angle_to_sun_std_deg': angles_deg.std() if len(angles_deg) > 0 else 0,
        'principal_direction': principal_direction,
        'explained_variance_ratio': explained_variance[0],
        'overall_score': overall_score,
        'angles_deg': angles_deg
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze perturbation direction patterns from TPE validation'
    )

    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to validation results file (tpe_validation_results.npz)'
    )

    args = parser.parse_args()

    # 加载validation结果
    results_file = Path(args.results)

    if not results_file.exists():
        print(f"ERROR: Results file not found: {results_file}")
        print(f"Please run validation first to generate results")
        sys.exit(1)

    print(f"Loading validation results from: {results_file}")

    data = np.load(results_file, allow_pickle=True)

    # 提取数据
    perturbations = data['perturbations']  # [T, 3]
    sun_dirs = data['sun_dirs']  # [T, 3]
    hours = data['hours']  # [T]
    reference_idx = int(data['reference_idx'])
    scene_name = str(data['scene_name'])

    print(f"✓ Loaded data: {len(perturbations)} time steps")

    # 运行分析
    results = analyze_perturbation_directions(
        perturbations,
        sun_dirs,
        hours,
        reference_idx,
        scene_name
    )

    # 保存分析结果
    output_file = results_file.parent / 'direction_analysis_results.npz'
    np.savez_compressed(output_file, **results)

    print(f"\n✓ Analysis results saved to: {output_file}")


if __name__ == "__main__":
    main()
