#!/usr/bin/env python3
"""
正确的训练评估：使用train/test split

Train: 奇数小时 [7, 9, 11, 13, 15, 17]
Test:  偶数小时 [6, 8, 10, 12, 14, 16, 18]
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


def split_train_test(hours, sh_matrix):
    """奇偶分割"""
    train_mask = hours % 2 == 1  # 奇数小时
    test_mask = hours % 2 == 0   # 偶数小时

    hours_train = hours[train_mask]
    sh_train = sh_matrix[train_mask]

    hours_test = hours[test_mask]
    sh_test = sh_matrix[test_mask]

    return hours_train, sh_train, hours_test, sh_test


class SplineBaseline:
    """Spline baseline"""

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
    print("\n[1/6] Loading data...")
    sh_matrix, hours = load_dataset(dataset_path)
    print(f"  Loaded {len(hours)} time steps: {hours.tolist()}")

    # 2. Train/Test split
    print("\n[2/6] Splitting train/test...")
    hours_train, sh_train, hours_test, sh_test = split_train_test(hours, sh_matrix)
    print(f"  Train: {hours_train.tolist()} ({len(hours_train)} hours)")
    print(f"  Test:  {hours_test.tolist()} ({len(hours_test)} hours)")

    # 3. 计算太阳高度角
    print("\n[3/6] Computing sun elevations...")
    sun_train = np.array([compute_sun_elevation(h) for h in hours_train])
    sun_test = np.array([compute_sun_elevation(h) for h in hours_test])

    # Tensor
    sun_train_t = torch.from_numpy(sun_train).float()
    sh_train_t = torch.from_numpy(sh_train).float()
    sun_test_t = torch.from_numpy(sun_test).float()
    sh_test_t = torch.from_numpy(sh_test).float()

    # 4. Baseline: Spline
    print("\n[4/6] Training Spline baseline...")
    spline = SplineBaseline()
    spline.fit(hours_train, sh_train)

    spline_train = spline.evaluate(hours_train, sh_train)
    spline_test = spline.evaluate(hours_test, sh_test)

    print(f"  Train MAE: {spline_train['mae']:.6f}")
    print(f"  Test MAE:  {spline_test['mae']:.6f}")

    # 5. 训练PhysicsLowRank
    print(f"\n[5/6] Training PhysicsLowRank (rank={rank})...")

    model = PhysicsLowRank(rank=rank)
    model.init_from_svd(sh_train)  # 只用训练数据初始化

    trainer = PhysicsLowRankTrainer(model, lr=1e-3)

    # 训练
    print("  Training for 2000 epochs...")
    best_test_mae = float('inf')
    patience = 0

    for epoch in range(2000):
        loss = trainer.train_epoch(sun_train_t, sh_train_t)

        if (epoch + 1) % 500 == 0:
            train_metrics = trainer.evaluate(sun_train_t, sh_train_t)
            test_metrics = trainer.evaluate(sun_test_t, sh_test_t)

            print(f"    Epoch {epoch+1}:")
            print(f"      Train: Loss={loss:.6f}, MAE={train_metrics['mae']:.6f}")
            print(f"      Test:  MAE={test_metrics['mae']:.6f}")

            # Early stopping
            if test_metrics['mae'] < best_test_mae:
                best_test_mae = test_metrics['mae']
                patience = 0
            else:
                patience += 1

    # 最终评估
    train_metrics = trainer.evaluate(sun_train_t, sh_train_t)
    test_metrics = trainer.evaluate(sun_test_t, sh_test_t)

    print(f"\n  ✓ Training complete!")
    print(f"    Train MAE: {train_metrics['mae']:.6f}")
    print(f"    Test MAE:  {test_metrics['mae']:.6f}")

    # 6. 对比
    print(f"\n[6/6] Comparison on Test Set:")
    print(f"\n  {'Method':<20} {'Test MAE':<12} {'Test RMSE':<12} {'Params':<10} {'Compression'}")
    print(f"  {'-'*70}")

    spline_params = 27 * len(hours_train)
    our_params = model.num_params()

    print(f"  {'Spline':<20} {spline_test['mae']:<12.6f} {spline_test['rmse']:<12.6f} {spline_params:<10} 1.00x")
    print(f"  {'PhysicsLowRank':<20} {test_metrics['mae']:<12.6f} {test_metrics['rmse']:<12.6f} {our_params:<10} {spline_params/our_params:.2f}x")

    # 判断
    mae_ratio = test_metrics['mae'] / spline_test['mae']

    print(f"\n  MAE Ratio (Ours/Spline): {mae_ratio:.3f}")

    if mae_ratio < 1.05:
        print(f"  🎉🎉🎉 EXCELLENT: 精度与Spline相当，但压缩{spline_params/our_params:.2f}x!")
    elif mae_ratio < 1.2:
        print(f"  ✓✓ GOOD: 精度接近Spline (误差增加{(mae_ratio-1)*100:.1f}%)，压缩{spline_params/our_params:.2f}x")
    elif mae_ratio < 1.5:
        print(f"  ✓ OK: 精度可接受 (误差增加{(mae_ratio-1)*100:.1f}%)，压缩{spline_params/our_params:.2f}x")
    else:
        print(f"  △ 精度损失较大 (误差增加{(mae_ratio-1)*100:.1f}%)，可能需要增加rank")

    # 分析数值范围
    sh_scale = np.abs(sh_test).mean()
    relative_error = test_metrics['mae'] / sh_scale
    print(f"\n  SH值的平均幅度: {sh_scale:.6f}")
    print(f"  相对误差: {relative_error*100:.2f}%")

    return {
        'scene': scene_name,
        'spline_train': spline_train,
        'spline_test': spline_test,
        'ours_train': train_metrics,
        'ours_test': test_metrics,
        'compression': spline_params / our_params,
        'mae_ratio': mae_ratio,
        'model': model
    }


def visualize_results(results, output_path):
    """可视化结果"""

    scene_names = [r['scene'] for r in results]
    spline_mae = [r['spline_test']['mae'] for r in results]
    ours_mae = [r['ours_test']['mae'] for r in results]
    compression = [r['compression'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MAE对比
    ax1 = axes[0]
    x = np.arange(len(scene_names))
    width = 0.35

    ax1.bar(x - width/2, spline_mae, width, label='Spline', alpha=0.8, color='blue')
    ax1.bar(x + width/2, ours_mae, width, label='PhysicsLowRank', alpha=0.8, color='green')

    ax1.set_ylabel('Test MAE', fontsize=11)
    ax1.set_title('Accuracy Comparison (Test Set)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scene_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 压缩比
    ax2 = axes[1]
    colors = ['green' if c >= 2 else 'orange' for c in compression]
    ax2.bar(scene_names, compression, color=colors, alpha=0.8)
    ax2.axhline(2.0, color='red', linestyle='--', alpha=0.7, label='2x target')

    ax2.set_ylabel('Compression Ratio', fontsize=11)
    ax2.set_title('Parameter Compression', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")


def main():
    """主函数"""

    print("="*70)
    print("PHYSICS-GUIDED LOW-RANK TEMPORAL COMPRESSION")
    print("Proper Evaluation with Train/Test Split")
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
            # Cornell Box用rank=5，House用rank=4
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
    plot_path = output_dir / 'comparison_proper.png'
    visualize_results(all_results, plot_path)

    # 总结
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    for result in all_results:
        print(f"\n{result['scene']}:")
        print(f"  Spline Test MAE:  {result['spline_test']['mae']:.6f}")
        print(f"  Ours Test MAE:    {result['ours_test']['mae']:.6f}")
        print(f"  MAE Ratio:        {result['mae_ratio']:.3f}")
        print(f"  Compression:      {result['compression']:.2f}x")

    all_excellent = all(r['mae_ratio'] < 1.1 for r in all_results)
    all_good = all(r['mae_ratio'] < 1.2 for r in all_results)
    all_compressed = all(r['compression'] >= 2.0 for r in all_results)

    print(f"\n{'='*70}")

    if all_excellent and all_compressed:
        print("\n🎉🎉🎉 BREAKTHROUGH! 🎉🎉🎉")
        print("物理引导的低秩方法：")
        print("  ✓ 精度与Spline几乎相同 (<5%误差)")
        print("  ✓ 实现2x+参数压缩")
        print("\n→ 这就是你毕设的核心创新！")
    elif all_good and all_compressed:
        print("\n✓✓ EXCELLENT!")
        print("物理引导的低秩方法：")
        print("  ✓ 精度接近Spline (<20%误差)")
        print("  ✓ 实现2x+参数压缩")
        print("\n→ 方法成功，可以作为毕设主要贡献")
    elif all_compressed:
        print("\n✓ GOOD")
        print("实现了目标压缩，精度有待优化")
    else:
        print("\n△ 需要进一步调整")

    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == '__main__':
    main()
