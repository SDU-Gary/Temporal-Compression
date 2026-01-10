#!/usr/bin/env python3
"""
训练物理引导的低秩时间压缩模型

对比：
  - Spline (baseline)
  - PhysicsLowRank (ours)
"""

import numpy as np
import torch
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.physics_low_rank import PhysicsLowRank, PhysicsLowRankTrainer, compute_sun_elevation
from scipy.interpolate import CubicSpline


def load_dataset(dataset_path):
    """加载数据集"""
    dataset_path = Path(dataset_path)

    moment_dirs = sorted(dataset_path.glob('moment_*'))

    sh_list = []
    hours_list = []

    for moment_dir in moment_dirs:
        hour = int(moment_dir.name.split('_')[1])
        sh_file = moment_dir / 'sh_coeffs.npz'

        if sh_file.exists():
            data = np.load(sh_file)
            sh = data['coeffs'][0]  # [27]
            sh_list.append(sh)
            hours_list.append(hour)

    return np.array(sh_list), np.array(hours_list)


class SplineBaseline:
    """Spline baseline for comparison"""

    def __init__(self):
        self.splines = None

    def fit(self, hours, sh_gt):
        """拟合"""
        self.splines = []
        for i in range(27):
            spline = CubicSpline(hours, sh_gt[:, i])
            self.splines.append(spline)

    def predict(self, hours_query):
        """预测"""
        predictions = []
        for spline in self.splines:
            predictions.append(spline(hours_query))
        return np.stack(predictions, axis=-1)

    def evaluate(self, hours, sh_gt):
        """评估"""
        sh_pred = self.predict(hours)
        mae = np.mean(np.abs(sh_pred - sh_gt))
        rmse = np.sqrt(np.mean((sh_pred - sh_gt) ** 2))
        return {'mae': mae, 'rmse': rmse}


def train_and_evaluate(scene_name, dataset_path, rank=5):
    """训练并评估"""

    print(f"\n{'='*70}")
    print(f"Scene: {scene_name}")
    print(f"{'='*70}")

    # 1. 加载数据
    print("\n[1/5] Loading data...")
    sh_matrix, hours = load_dataset(dataset_path)
    print(f"  Loaded {len(hours)} time steps: {hours.tolist()}")
    print(f"  SH matrix shape: {sh_matrix.shape}")

    # 2. 计算太阳高度角
    print("\n[2/5] Computing sun elevations...")
    sun_elevations = np.array([compute_sun_elevation(h) for h in hours])
    print(f"  Elevation range: [{np.rad2deg(sun_elevations.min()):.1f}°, {np.rad2deg(sun_elevations.max()):.1f}°]")

    # 转为tensor
    sun_elevations_t = torch.from_numpy(sun_elevations).float()
    sh_gt_t = torch.from_numpy(sh_matrix).float()

    # 3. Baseline: Spline
    print("\n[3/5] Training Spline baseline...")
    spline = SplineBaseline()
    spline.fit(hours, sh_matrix)
    spline_metrics = spline.evaluate(hours, sh_matrix)
    print(f"  Spline MAE:  {spline_metrics['mae']:.6f}")
    print(f"  Spline RMSE: {spline_metrics['rmse']:.6f}")

    # 4. 训练PhysicsLowRank
    print(f"\n[4/5] Training PhysicsLowRank (rank={rank})...")

    model = PhysicsLowRank(rank=rank)

    # SVD初始化
    model.init_from_svd(sh_matrix)

    trainer = PhysicsLowRankTrainer(model, lr=1e-3)

    # 训练
    print("  Training for 2000 epochs...")
    for epoch in range(2000):
        loss = trainer.train_epoch(sun_elevations_t, sh_gt_t)

        if (epoch + 1) % 500 == 0:
            metrics = trainer.evaluate(sun_elevations_t, sh_gt_t)
            print(f"    Epoch {epoch+1}: Loss={loss:.6f}, MAE={metrics['mae']:.6f}")

    # 最终评估
    final_metrics = trainer.evaluate(sun_elevations_t, sh_gt_t)

    print(f"\n  ✓ Training complete!")
    print(f"    MAE:  {final_metrics['mae']:.6f}")
    print(f"    RMSE: {final_metrics['rmse']:.6f}")

    # 5. 对比
    print(f"\n[5/5] Comparison:")
    print(f"\n  {'Method':<20} {'MAE':<12} {'RMSE':<12} {'Params':<10} {'Compression'}")
    print(f"  {'-'*70}")

    spline_params = 27 * len(hours)
    our_params = model.num_params()

    print(f"  {'Spline':<20} {spline_metrics['mae']:<12.6f} {spline_metrics['rmse']:<12.6f} {spline_params:<10} 1.00x")
    print(f"  {'PhysicsLowRank':<20} {final_metrics['mae']:<12.6f} {final_metrics['rmse']:<12.6f} {our_params:<10} {spline_params/our_params:.2f}x")

    # 判断
    mae_ratio = final_metrics['mae'] / spline_metrics['mae']

    print(f"\n  MAE Ratio (Ours/Spline): {mae_ratio:.3f}")

    if mae_ratio < 1.05:
        print(f"  ✓✓✓ EXCELLENT: 精度与Spline相当，但压缩{spline_params/our_params:.2f}x!")
    elif mae_ratio < 1.2:
        print(f"  ✓✓ GOOD: 精度接近Spline，压缩{spline_params/our_params:.2f}x")
    elif mae_ratio < 1.5:
        print(f"  ✓ OK: 精度可接受，压缩{spline_params/our_params:.2f}x")
    else:
        print(f"  △ 精度损失较大，可能需要增加rank")

    return {
        'scene': scene_name,
        'spline': spline_metrics,
        'ours': final_metrics,
        'compression': spline_params / our_params,
        'model': model
    }


def visualize_results(results, output_path):
    """可视化结果"""

    scene_names = [r['scene'] for r in results]
    spline_mae = [r['spline']['mae'] for r in results]
    ours_mae = [r['ours']['mae'] for r in results]
    compression = [r['compression'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MAE对比
    ax1 = axes[0]
    x = np.arange(len(scene_names))
    width = 0.35

    ax1.bar(x - width/2, spline_mae, width, label='Spline', alpha=0.8)
    ax1.bar(x + width/2, ours_mae, width, label='PhysicsLowRank', alpha=0.8)

    ax1.set_ylabel('MAE', fontsize=11)
    ax1.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scene_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 压缩比
    ax2 = axes[1]
    colors = ['green' if c > 2 else 'orange' for c in compression]
    ax2.bar(scene_names, compression, color=colors, alpha=0.8)
    ax2.axhline(2.0, color='red', linestyle='--', alpha=0.7, label='2x target')

    ax2.set_ylabel('Compression Ratio', fontsize=11)
    ax2.set_title('Compression Ratio', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")


def main():
    """主函数"""

    print("="*70)
    print("PHYSICS-GUIDED LOW-RANK TEMPORAL COMPRESSION")
    print("="*70)

    # 数据集
    datasets = {
        'Cornell Box': Path('/home/kyrie/毕设/data_generation/output/level2_tpe/cornell-box_test'),
        'House': Path('/home/kyrie/毕设/data_generation/output/level2_tpe/house_p0')
    }

    # 输出目录
    output_dir = Path('/home/kyrie/毕设/multi_time_compression/output/physics_low_rank')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 训练所有场景
    all_results = []

    for scene_name, dataset_path in datasets.items():
        try:
            # Cornell Box用rank=5，House用rank=4（基于SVD分析）
            rank = 5 if 'Cornell' in scene_name else 4

            result = train_and_evaluate(scene_name, dataset_path, rank=rank)
            all_results.append(result)

        except Exception as e:
            print(f"\n✗ Error training {scene_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        print("\n✗ No results generated!")
        return

    # 可视化
    plot_path = output_dir / 'comparison.png'
    visualize_results(all_results, plot_path)

    # 总结
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    all_good = all(r['ours']['mae'] / r['spline']['mae'] < 1.1 for r in all_results)
    all_compressed = all(r['compression'] > 2.0 for r in all_results)

    if all_good and all_compressed:
        print("\n🎉🎉🎉 SUCCESS! 🎉🎉🎉")
        print("物理引导的低秩方法在所有场景都：")
        print("  ✓ 精度与Spline相当")
        print("  ✓ 实现2x+压缩")
        print("\n→ 这是毕设的核心贡献！")
    elif all_good:
        print("\n✓✓ GOOD!")
        print("精度与Spline相当，压缩比可以进一步优化")
    else:
        print("\n✓ FEASIBLE")
        print("方法基本可行，可能需要调整rank或添加正则化")

    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == '__main__':
    main()
