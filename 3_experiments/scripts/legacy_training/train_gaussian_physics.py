#!/usr/bin/env python3
"""
训练高斯-物理混合压缩模型

空间：高斯混合（K个3D高斯）
时间：物理引导低秩基函数

对比：
  - Baseline: 独立存储 (N*T*27 参数)
  - PhysicsOnly: 全局单一 U 矩阵 (150 参数)
  - GaussianPhysics: 混合方法 (K*(6+27*rank+rank*3) 参数)
"""

import numpy as np
import torch
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.gaussian_physics_compression import (
    GaussianPhysicsCompression,
    compute_sun_elevation
)
from models.physics_low_rank import PhysicsLowRank, PhysicsLowRankTrainer
from scipy.interpolate import CubicSpline
from training import BatchAdapter, GaussianPhysicsTrainer


def load_multiprobe_dataset(dataset_path):
    """加载多探针数据集

    Returns:
        probe_positions: [N, 3]
        sh_all_times: [T, N, 27]
        hours: [T]
    """
    dataset_path = Path(dataset_path)

    # 加载探针位置
    probes_file = dataset_path / 'probes.npz'
    probe_data = np.load(probes_file)
    probe_positions = probe_data['positions']  # [N, 3]

    N = len(probe_positions)

    # 加载所有时刻的SH系数
    moment_dirs = sorted(dataset_path.glob('moment_*'))

    sh_list = []
    hours_list = []

    for moment_dir in moment_dirs:
        hour = int(moment_dir.name.split('_')[1])
        sh_file = moment_dir / 'sh_coeffs.npz'

        if sh_file.exists():
            data = np.load(sh_file)
            sh = data['coeffs']  # [N, 27]

            if sh.shape[0] != N:
                print(f"Warning: {moment_dir.name} has {sh.shape[0]} probes, expected {N}")
                continue

            sh_list.append(sh)
            hours_list.append(hour)

    sh_all_times = np.array(sh_list)  # [T, N, 27]
    hours = np.array(hours_list)  # [T]

    return probe_positions, sh_all_times, hours


def create_train_test_data(probe_positions, sh_all_times, hours):
    """创建训练/测试数据

    策略：奇数小时训练，偶数小时测试

    Returns:
        train/test dictionaries with positions, sh, hours, sun_elevations
    """
    N = probe_positions.shape[0]
    T = len(hours)

    # 奇偶分割
    train_mask = hours % 2 == 1
    test_mask = hours % 2 == 0

    hours_train = hours[train_mask]
    hours_test = hours[test_mask]

    sh_train = sh_all_times[train_mask]  # [T_train, N, 27]
    sh_test = sh_all_times[test_mask]    # [T_test, N, 27]

    # 计算太阳高度角
    sun_train = np.array([compute_sun_elevation(h) for h in hours_train])  # [T_train]
    sun_test = np.array([compute_sun_elevation(h) for h in hours_test])    # [T_test]

    # 展平为 (T*N) 样本
    # positions: 每个时刻重复N次
    def flatten_data(sh_times, sun_times):
        """
        Args:
            sh_times: [T, N, 27]
            sun_times: [T]

        Returns:
            positions: [T*N, 3]
            sun_elevations: [T*N]
            sh: [T*N, 27]
        """
        T = sh_times.shape[0]
        N = sh_times.shape[1]

        # 位置：每个时刻重复N次
        positions = np.tile(probe_positions, (T, 1))  # [T*N, 3]

        # 太阳高度角：每个时刻重复N次
        sun_elevations = np.repeat(sun_times, N)  # [T*N]

        # SH：展平
        sh = sh_times.reshape(T * N, 27)  # [T*N, 27]

        return positions, sun_elevations, sh

    pos_train, sun_elev_train, sh_flat_train = flatten_data(sh_train, sun_train)
    pos_test, sun_elev_test, sh_flat_test = flatten_data(sh_test, sun_test)

    train_data = {
        'positions': pos_train,
        'sun_elevations': sun_elev_train,
        'sh': sh_flat_train,
        'hours': hours_train,
        'sh_matrix': sh_train  # [T_train, N, 27] for init
    }

    test_data = {
        'positions': pos_test,
        'sun_elevations': sun_elev_test,
        'sh': sh_flat_test,
        'hours': hours_test
    }

    return train_data, test_data


class SplineBaseline:
    """Spline baseline: 每个探针独立插值"""

    def __init__(self, probe_positions, sh_train_matrix, hours_train):
        """
        Args:
            probe_positions: [N, 3]
            sh_train_matrix: [T_train, N, 27]
            hours_train: [T_train]
        """
        self.probe_positions = probe_positions
        self.N = len(probe_positions)

        # 为每个探针的每个SH系数构建spline
        self.splines = []
        for n in range(self.N):
            probe_splines = []
            for i in range(27):
                spline = CubicSpline(hours_train, sh_train_matrix[:, n, i])
                probe_splines.append(spline)
            self.splines.append(probe_splines)

    def predict(self, positions, hours_query):
        """
        Args:
            positions: [B, 3] 查询位置
            hours_query: [B] 查询小时

        Returns:
            sh_pred: [B, 27]
        """
        B = len(positions)
        sh_pred = np.zeros((B, 27))

        for b in range(B):
            pos = positions[b]
            hour = hours_query[b]

            # 找到最近的探针
            distances = np.linalg.norm(self.probe_positions - pos, axis=1)
            nearest_idx = np.argmin(distances)

            # 使用该探针的spline预测
            for i in range(27):
                sh_pred[b, i] = self.splines[nearest_idx][i](hour)

        return sh_pred

    def evaluate(self, positions, hours_query, sh_gt):
        """评估"""
        sh_pred = self.predict(positions, hours_query)
        mae = np.mean(np.abs(sh_pred - sh_gt))
        rmse = np.sqrt(np.mean((sh_pred - sh_gt) ** 2))
        return {'mae': mae, 'rmse': rmse}


class FlatGaussianPhysicsDataset(Dataset):
    """Flat dataset for Gaussian-Physics training from pre-flattened arrays."""

    def __init__(self, positions: np.ndarray, sun_elevations: np.ndarray, sh: np.ndarray):
        self.positions = torch.from_numpy(positions).float()
        self.sun_elevations = torch.from_numpy(sun_elevations).float()
        self.sh = torch.from_numpy(sh).float()

    def __len__(self) -> int:
        return self.sh.shape[0]

    def __getitem__(self, idx: int):
        return {
            'probe_position': self.positions[idx],
            'sun_elevation': self.sun_elevations[idx],
            'sh_coeffs': self.sh[idx]
        }


def train_and_evaluate(dataset_path, num_gaussians=50, rank=5):
    """训练并评估"""

    print(f"\n{'='*70}")
    print(f"GAUSSIAN-PHYSICS HYBRID COMPRESSION")
    print(f"{'='*70}")

    # 1. 加载数据
    print("\n[1/7] Loading multi-probe dataset...")
    probe_positions, sh_all_times, hours = load_multiprobe_dataset(dataset_path)

    N = probe_positions.shape[0]
    T = len(hours)

    print(f"  Probes: {N}")
    print(f"  Time moments: {T} ({hours.tolist()})")
    print(f"  SH shape: {sh_all_times.shape}")

    # 2. Train/Test split
    print("\n[2/7] Splitting train/test...")
    train_data, test_data = create_train_test_data(probe_positions, sh_all_times, hours)

    print(f"  Train: {len(train_data['hours'])} moments × {N} probes = {len(train_data['sh'])} samples")
    print(f"  Test:  {len(test_data['hours'])} moments × {N} probes = {len(test_data['sh'])} samples")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 转为tensor
    pos_train_t = torch.from_numpy(train_data['positions']).float().to(device)
    sun_train_t = torch.from_numpy(train_data['sun_elevations']).float().to(device)
    sh_train_t = torch.from_numpy(train_data['sh']).float().to(device)

    pos_test_t = torch.from_numpy(test_data['positions']).float().to(device)
    sun_test_t = torch.from_numpy(test_data['sun_elevations']).float().to(device)
    sh_test_t = torch.from_numpy(test_data['sh']).float().to(device)

    # 3. Baseline: Spline
    print("\n[3/7] Training Spline baseline...")
    spline = SplineBaseline(probe_positions, train_data['sh_matrix'], train_data['hours'])

    # 需要把positions和sun映射回hours
    # 简化：直接在测试数据上计算hours
    # positions: [T_test*N, 3]
    # 每N个样本对应一个小时
    hours_test_repeated = np.repeat(test_data['hours'], N)

    spline_test = spline.evaluate(test_data['positions'], hours_test_repeated, test_data['sh'])
    print(f"  Spline Test MAE: {spline_test['mae']:.6f}")

    # 4. PhysicsOnly baseline (全局单一U矩阵)
    print(f"\n[4/7] Training PhysicsOnly baseline (rank={rank})...")

    # 使用所有探针的平均SH来训练全局模型
    sh_avg = train_data['sh_matrix'].mean(axis=1)  # [T_train, 27]

    physics_only = PhysicsLowRank(rank=rank)
    physics_only.init_from_svd(sh_avg)
    physics_only.to(device)

    trainer_physics = PhysicsLowRankTrainer(physics_only, lr=1e-3)

    # 训练 - 只用唯一的时刻，不是重复N次的样本
    sun_train_unique = np.array([compute_sun_elevation(h) for h in train_data['hours']])  # [T_train]
    sun_unique_t = torch.from_numpy(sun_train_unique).float().to(device)
    sh_avg_t = torch.from_numpy(sh_avg).float().to(device)

    for epoch in range(1000):
        loss = trainer_physics.train_epoch(sun_unique_t, sh_avg_t)

        if (epoch + 1) % 200 == 0:
            metrics = trainer_physics.evaluate(sun_unique_t, sh_avg_t)
            print(f"    Epoch {epoch+1}: Loss={loss:.6f}, MAE={metrics['mae']:.6f}")

    # 评估：使用全局模型预测每个测试样本
    # 全局模型对所有位置使用相同的SH预测（忽略空间变化）
    with torch.no_grad():
        sh_pred_global = physics_only(sun_test_t).detach().cpu().numpy()

    physics_mae = np.mean(np.abs(sh_pred_global - test_data['sh']))
    physics_rmse = np.sqrt(np.mean((sh_pred_global - test_data['sh']) ** 2))

    print(f"  PhysicsOnly Test MAE: {physics_mae:.6f}")
    print(f"  (Note: PhysicsOnly uses global average, ignoring spatial variation)")

    # 5. GaussianPhysics混合模型
    print(f"\n[5/7] Training GaussianPhysics (K={num_gaussians}, rank={rank})...")

    model = GaussianPhysicsCompression(num_gaussians=num_gaussians, rank=rank).to(device)

    # K-Means + SVD 初始化
    model.init_from_kmeans(probe_positions, torch.from_numpy(train_data['sh_matrix']).float())

    adapter = BatchAdapter(
        positions_key='probe_position',
        params_key='sun_elevation',
        target_key='sh_coeffs',
        params_transform=lambda p: p.squeeze(-1) if p.dim() > 1 else p,
    )
    trainer = GaussianPhysicsTrainer(
        model=model,
        device=device,
        adapter=adapter,
        lr=1e-3,
        recon_loss="mse",
        temporal_weight=0.0,
        top_k=3,
        grad_clip=1.0,
    )

    train_dataset = FlatGaussianPhysicsDataset(
        train_data['positions'], train_data['sun_elevations'], train_data['sh']
    )
    test_dataset = FlatGaussianPhysicsDataset(
        test_data['positions'], test_data['sun_elevations'], test_data['sh']
    )
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)

    # 训练
    print("  Training for 1000 epochs...")
    best_test_mae = float('inf')

    for epoch in range(1000):
        train_metrics = trainer.train_epoch(train_loader)

        if (epoch + 1) % 250 == 0:
            train_eval = trainer.validate_epoch(train_loader)
            test_metrics = trainer.validate_epoch(test_loader)

            print(f"    Epoch {epoch+1}:")
            print(f"      Train: Loss={train_metrics['total']:.6f}, MAE={train_eval['mae']:.6f}")
            print(f"      Test:  MAE={test_metrics['mae']:.6f}")

            if test_metrics['mae'] < best_test_mae:
                best_test_mae = test_metrics['mae']

    # 最终评估
    final_metrics = trainer.validate_epoch(test_loader)

    print(f"\n  ✓ Training complete!")
    print(f"    Test MAE:  {final_metrics['mae']:.6f}")
    print(f"    Test RMSE: {final_metrics['rmse']:.6f}")

    # 6. 对比
    print(f"\n[6/7] Comparison on Test Set:")
    print(f"\n  {'Method':<25} {'Test MAE':<12} {'Test RMSE':<12} {'Params':<15} {'Compression'}")
    print(f"  {'-'*80}")

    # 参数量计算
    baseline_params = N * T * 27  # 所有探针所有时刻独立存储
    spline_params = N * 27 * len(train_data['hours'])  # Spline存储训练时刻
    physics_only_params = physics_only.num_params()
    gaussian_physics_params = model.num_params()

    print(f"  {'Baseline (full storage)':<25} {'-':<12} {'-':<12} {baseline_params:<15} 1.00x")
    print(f"  {'Spline':<25} {spline_test['mae']:<12.6f} {spline_test['rmse']:<12.6f} {spline_params:<15} {baseline_params/spline_params:.2f}x")
    print(f"  {'PhysicsOnly (global)':<25} {physics_mae:<12.6f} {physics_rmse:<12.6f} {physics_only_params:<15} {baseline_params/physics_only_params:.2f}x")
    print(f"  {'GaussianPhysics (ours)':<25} {final_metrics['mae']:<12.6f} {final_metrics['rmse']:<12.6f} {gaussian_physics_params:<15} {baseline_params/gaussian_physics_params:.2f}x")

    # 7. 总结
    print(f"\n[7/7] Summary:")

    mae_ratio_vs_spline = final_metrics['mae'] / spline_test['mae']
    compression_ratio = baseline_params / gaussian_physics_params

    print(f"\n  MAE Ratio (Ours/Spline): {mae_ratio_vs_spline:.3f}")
    print(f"  Compression Ratio: {compression_ratio:.2f}x")

    if mae_ratio_vs_spline < 1.1 and compression_ratio > 10:
        print(f"\n  🎉🎉🎉 BREAKTHROUGH!")
        print(f"  混合方法：精度接近Spline，压缩比 {compression_ratio:.0f}x!")
    elif mae_ratio_vs_spline < 1.5 and compression_ratio > 5:
        print(f"\n  ✓✓ EXCELLENT!")
        print(f"  混合方法：精度可接受，压缩比 {compression_ratio:.0f}x")
    else:
        print(f"\n  ✓ GOOD")
        print(f"  方法有效，可能需要调整参数")

    return {
        'spline': spline_test,
        'physics_only': {'mae': physics_mae, 'rmse': physics_rmse},
        'gaussian_physics': final_metrics,
        'params': {
            'baseline': baseline_params,
            'spline': spline_params,
            'physics_only': physics_only_params,
            'gaussian_physics': gaussian_physics_params
        }
    }


def main():
    """主函数"""

    print("="*70)
    print("GAUSSIAN-PHYSICS HYBRID COMPRESSION EXPERIMENT")
    print("="*70)

    # 数据集
    dataset_path = Path('/home/kyrie/毕设/data_generation/output/method_test_v1')

    if not dataset_path.exists():
        print(f"\n✗ Dataset not found: {dataset_path}")
        return

    # 输出目录
    output_dir = Path('/home/kyrie/毕设/multi_time_compression/output/gaussian_physics')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 训练
    try:
        result = train_and_evaluate(
            dataset_path,
            num_gaussians=20,  # 减少高斯数量加速训练
            rank=5
        )

        print(f"\n✓ Experiment completed!")
        print(f"\n✓ Results saved to: {output_dir}")

    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
