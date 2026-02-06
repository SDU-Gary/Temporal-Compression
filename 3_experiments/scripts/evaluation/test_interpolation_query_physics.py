"""
Week 4.1: Physics-only内插查询测试

目标：验证Physics-only模型在训练范围内随机配置上的泛化能力
成功标准：内插MAE / 训练MAE < 1.5
"""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt
from datetime import datetime

from models.physics_low_rank import PhysicsLowRank5D
from data.transfer_tensor_dataset import TransferTensorDataset5D
from tools.manifest_utils import load_manifest


def resolve_data_root(data_root: str | None, manifest_path: str | None) -> str:
    if data_root:
        return data_root
    if not manifest_path:
        raise ValueError("data_root is required when manifest is not provided")
    manifest = load_manifest(manifest_path)
    output_dir = manifest.get("output_dir")
    if not output_dir:
        raise ValueError("manifest missing output_dir")
    return str(output_dir)


def sample_interpolation_configs(
    n_samples: int = 20,
    train_range: Dict[str, Tuple[float, float]] = None,
    seed: int = 42
) -> np.ndarray:
    """
    在训练范围内生成随机5D配置（内插）

    Args:
        n_samples: 采样数量
        train_range: 训练范围字典
        seed: 随机种子

    Returns:
        configs [N, 5]: [zenith, azimuth, intensity, temp, cloud]
    """
    if train_range is None:
        train_range = {
            'zenith': (10.0, 80.0),      # degrees
            'azimuth': (0.0, 360.0),     # degrees
            'intensity': (0.5, 1.5),     # multiplier
            'color_temp': (3000, 7000),  # Kelvin
            'cloud_cover': (0.0, 0.8)    # ratio
        }

    np.random.seed(seed)
    configs = []

    for _ in range(n_samples):
        config = [
            np.random.uniform(*train_range['zenith']),
            np.random.uniform(*train_range['azimuth']),
            np.random.uniform(*train_range['intensity']),
            np.random.uniform(*train_range['color_temp']),
            np.random.uniform(*train_range['cloud_cover'])
        ]
        configs.append(config)

    return np.array(configs)


def compute_interpolation_error(
    model: PhysicsLowRank5D,
    test_configs: np.ndarray,
    gt_sh_data: np.ndarray,
    device: torch.device
) -> Dict[str, float]:
    """
    计算内插查询误差

    Args:
        model: 训练好的Physics模型
        test_configs: [N, 5] 测试配置
        gt_sh_data: [N, 27] ground truth SH系数
        device: 计算设备

    Returns:
        metrics: 误差统计
    """
    model.eval()

    with torch.no_grad():
        configs_t = torch.from_numpy(test_configs).float().to(device)
        gt_t = torch.from_numpy(gt_sh_data).float().to(device)

        # 预测
        pred_sh = model(configs_t)  # [N, 27]

        # 计算误差
        mae = torch.mean(torch.abs(pred_sh - gt_t)).item()
        mse = torch.mean((pred_sh - gt_t) ** 2).item()
        rmse = np.sqrt(mse)

        # Per-config误差
        per_config_mae = torch.mean(torch.abs(pred_sh - gt_t), dim=1).cpu().numpy()

        # Uncertainty estimation (prediction variance)
        pred_var = torch.var(pred_sh, dim=1).mean().item()

    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'per_config_mae': per_config_mae.tolist(),
        'prediction_variance': pred_var
    }

    return metrics


def load_ground_truth_sh(
    configs: np.ndarray,
    data_dir: Path,
    probe_idx: int = 0
) -> np.ndarray:
    """
    加载或渲染ground truth SH系数

    注意：这里假设已有预渲染数据。实际应用中需要调用Mitsuba渲染。
    为了快速验证，我们从现有数据集中查找最接近的配置。

    Args:
        configs: [N, 5] 查询配置
        data_dir: 数据目录
        probe_idx: 探针索引

    Returns:
        sh_coeffs: [N, 27] ground truth SH系数
    """
    # 加载完整数据集获取ground truth (合并train+val+test)
    all_configs = []
    all_sh = []

    for split in ['train', 'val', 'test']:
        dataset = TransferTensorDataset5D(
            data_root=data_dir,
            split=split,
            normalize_params=False  # Keep original scale for nearest neighbor search
        )

        # 从当前split提取配置和SH (仅从指定探针)
        for i in range(len(dataset)):
            sample = dataset[i]
            # Only use data from the specified probe
            if sample['probe_idx'].item() == probe_idx:
                all_configs.append(sample['light_params'].numpy())
                all_sh.append(sample['sh_coeffs'].numpy())

    all_configs = np.array(all_configs)  # [M, 5]
    all_sh = np.array(all_sh)  # [M, 27]

    # 找最近邻（简化版，实际应该渲染）
    sh_results = []
    for config in configs:
        # 计算L2距离（需要归一化）
        config_norm = config / np.array([80, 360, 1.5, 7000, 0.8])
        all_configs_norm = all_configs / np.array([80, 360, 1.5, 7000, 0.8])

        distances = np.linalg.norm(all_configs_norm - config_norm, axis=1)
        nearest_idx = np.argmin(distances)

        sh_results.append(all_sh[nearest_idx])

    return np.array(sh_results)


def visualize_results(
    test_configs: np.ndarray,
    per_config_mae: List[float],
    train_mae: float,
    output_dir: Path
):
    """可视化测试结果"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 绘制per-config MAE分布
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. MAE分布直方图
    axes[0, 0].hist(per_config_mae, bins=15, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(train_mae, color='r', linestyle='--', label=f'Train MAE={train_mae:.4f}')
    axes[0, 0].axvline(np.mean(per_config_mae), color='g', linestyle='--',
                       label=f'Test Mean={np.mean(per_config_mae):.4f}')
    axes[0, 0].set_xlabel('MAE')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Interpolation Query MAE Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. Config参数vs MAE (zenith)
    axes[0, 1].scatter(test_configs[:, 0], per_config_mae, alpha=0.6)
    axes[0, 1].set_xlabel('Sun Zenith (deg)')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('MAE vs Sun Zenith')
    axes[0, 1].grid(alpha=0.3)

    # 3. Config参数vs MAE (intensity)
    axes[1, 0].scatter(test_configs[:, 2], per_config_mae, alpha=0.6, color='orange')
    axes[1, 0].set_xlabel('Intensity')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('MAE vs Intensity')
    axes[1, 0].grid(alpha=0.3)

    # 4. Config参数vs MAE (color_temp)
    axes[1, 1].scatter(test_configs[:, 3], per_config_mae, alpha=0.6, color='green')
    axes[1, 1].set_xlabel('Color Temperature (K)')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('MAE vs Color Temperature')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'interpolation_query_analysis.png', dpi=150)
    print(f"可视化结果已保存至: {output_dir / 'interpolation_query_analysis.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Physics-only interpolation query test")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--checkpoint", default="archive/Week3-4_MLP_Hybrid_Failed/checkpoints/stage1_best.pt")
    parser.add_argument("--output", default="experiments/week4_interpolation_query")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-idx", type=int, default=0)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--train-mae", type=float, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = Path(resolve_data_root(args.data_root, args.manifest))
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output)

    print("=" * 80)
    print("Week 4.1: Physics-only内插查询测试")
    print("=" * 80)

    # 1. 生成测试配置
    print("\n[1/5] 生成20个内插查询配置...")
    test_configs = sample_interpolation_configs(n_samples=args.n_samples, seed=args.seed)
    print(f"配置范围验证:")
    print(f"  Zenith: [{test_configs[:, 0].min():.1f}, {test_configs[:, 0].max():.1f}] deg")
    print(f"  Azimuth: [{test_configs[:, 1].min():.1f}, {test_configs[:, 1].max():.1f}] deg")
    print(f"  Intensity: [{test_configs[:, 2].min():.2f}, {test_configs[:, 2].max():.2f}]")
    print(f"  Color Temp: [{test_configs[:, 3].min():.0f}, {test_configs[:, 3].max():.0f}] K")
    print(f"  Cloud: [{test_configs[:, 4].min():.2f}, {test_configs[:, 4].max():.2f}]")

    # 2. 加载模型
    print("\n[2/5] 加载Physics-only模型...")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint不存在: {checkpoint_path}")
        print("请确保Week 3-4 Stage 1训练已完成")
        return

    model = PhysicsLowRank5D(rank=args.rank)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract only physics components from hybrid checkpoint
    # (Stage 1 was trained with PhysicsMLPHybrid but MLP frozen)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                      if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    print(f"模型已加载: {checkpoint_path}")
    print(f"  训练轮数: {checkpoint.get('epoch', 'N/A')}")
    train_mae = args.train_mae if args.train_mae is not None else checkpoint.get('val_mae', 0.0391)
    print(f"  训练Val MAE: {train_mae:.4f}")

    # 3. 获取ground truth
    print("\n[3/5] 获取ground truth SH系数...")
    print("  注意: 当前使用最近邻近似，实际应重新渲染")
    gt_sh = load_ground_truth_sh(test_configs, data_dir, probe_idx=args.probe_idx)
    print(f"  Ground truth shape: {gt_sh.shape}")

    # 4. 计算内插误差
    print("\n[4/5] 计算内插查询误差...")
    metrics = compute_interpolation_error(model, test_configs, gt_sh, device)

    test_mae = metrics['mae']
    degradation_ratio = test_mae / train_mae

    print(f"\n内插查询结果:")
    print(f"  Test MAE: {test_mae:.4f}")
    print(f"  Train MAE: {train_mae:.4f}")
    print(f"  退化比例: {degradation_ratio:.2f}× (目标<1.5×)")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  Prediction Variance: {metrics['prediction_variance']:.6f}")

    # Per-config统计
    per_config_mae = np.array(metrics['per_config_mae'])
    print(f"\nPer-config MAE统计:")
    print(f"  Mean: {per_config_mae.mean():.4f}")
    print(f"  Std: {per_config_mae.std():.4f}")
    print(f"  Min: {per_config_mae.min():.4f}")
    print(f"  Max: {per_config_mae.max():.4f}")
    print(f"  Median: {np.median(per_config_mae):.4f}")

    # 5. 保存结果
    print("\n[5/5] 保存结果...")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'timestamp': datetime.now().isoformat(),
        'test_configs': test_configs.tolist(),
        'train_mae': train_mae,
        'test_mae': test_mae,
        'degradation_ratio': degradation_ratio,
        'rmse': metrics['rmse'],
        'prediction_variance': metrics['prediction_variance'],
        'per_config_mae': metrics['per_config_mae'],
        'per_config_stats': {
            'mean': float(per_config_mae.mean()),
            'std': float(per_config_mae.std()),
            'min': float(per_config_mae.min()),
            'max': float(per_config_mae.max()),
            'median': float(np.median(per_config_mae))
        },
        'success': degradation_ratio < 1.5
    }

    with open(output_dir / 'interpolation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 可视化
    visualize_results(test_configs, per_config_mae, train_mae, output_dir)

    # Go/No-Go决策
    print("\n" + "=" * 80)
    if results['success']:
        print("✓ 内插查询测试通过 (退化比例 {:.2f}× < 1.5×)".format(degradation_ratio))
        print("  建议: 继续Week 4.2外推查询测试")
    else:
        print("✗ 内插查询测试失败 (退化比例 {:.2f}× >= 1.5×)".format(degradation_ratio))
        print("  建议: 检查模型训练或增加训练数据")
    print("=" * 80)

    print(f"\n结果已保存至: {output_dir}")


if __name__ == '__main__':  # pragma: no cover
    main()
