"""
Week 4.2: Physics-only外推查询测试

目标：验证Physics-only模型在训练范围外的泛化能力
成功标准：外推MAE / 训练MAE < 3.0
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from typing import Dict, List
import json
import matplotlib.pyplot as plt
from datetime import datetime

from models.physics_low_rank import PhysicsLowRank5D
from data.transfer_tensor_dataset import TransferTensorDataset5D


def get_extrapolation_configs() -> np.ndarray:
    """
    定义5个超出训练范围的配置（外推）

    训练范围:
        zenith: [10, 80] deg
        azimuth: [0, 360] deg
        intensity: [0.5, 1.5]
        color_temp: [3000, 7000] K
        cloud_cover: [0.0, 0.8]

    Returns:
        configs [5, 5]: [zenith, azimuth, intensity, temp, cloud]
    """
    configs = [
        # 1. 低强度外推 (0.3 < 0.5 train min)
        [45.0, 180.0, 0.3, 5500.0, 0.5],

        # 2. 高强度外推 (2.0 > 1.5 train max)
        [30.0, 90.0, 2.0, 5500.0, 0.5],

        # 3. 低色温外推 (2500K < 3000K train min)
        [45.0, 180.0, 1.0, 2500.0, 0.3],

        # 4. 高色温外推 (8500K > 7000K train max)
        [60.0, 270.0, 1.0, 8500.0, 0.3],

        # 5. 高云量外推 (0.9 > 0.8 train max)
        [75.0, 315.0, 1.0, 5500.0, 0.9]
    ]

    return np.array(configs)


def compute_extrapolation_error(
    model: PhysicsLowRank5D,
    test_configs: np.ndarray,
    gt_sh_data: np.ndarray,
    device: torch.device
) -> Dict[str, float]:
    """
    计算外推查询误差

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

        # Per-coefficient误差 (分析哪些SH系数外推性能差)
        per_coeff_mae = torch.mean(torch.abs(pred_sh - gt_t), dim=0).cpu().numpy()

        # Relative error
        relative_error = torch.abs(pred_sh - gt_t) / (torch.abs(gt_t) + 1e-6)
        relative_mae = torch.mean(relative_error).item()

    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'per_config_mae': per_config_mae.tolist(),
        'per_coeff_mae': per_coeff_mae.tolist(),
        'relative_mae': relative_mae
    }

    return metrics


def load_ground_truth_sh_extrapolation(
    configs: np.ndarray,
    data_dir: Path,
    probe_idx: int = 0
) -> np.ndarray:
    """
    获取外推配置的ground truth SH系数

    方法：最近邻近似（简化版）
    理想：用Mitsuba重新渲染

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
            normalize_params=False
        )

        # 从当前split提取配置和SH (仅从指定探针)
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample['probe_idx'].item() == probe_idx:
                all_configs.append(sample['light_params'].numpy())
                all_sh.append(sample['sh_coeffs'].numpy())

    all_configs = np.array(all_configs)  # [M, 5]
    all_sh = np.array(all_sh)  # [M, 27]

    # 参数归一化权重（因为各参数量纲不同）
    param_weights = np.array([1.0, 1.0/360, 1.0, 1.0/4000, 1.0])

    # 找最近邻（加权L2距离）
    sh_results = []
    for config in configs:
        config_weighted = config * param_weights
        all_configs_weighted = all_configs * param_weights

        distances = np.linalg.norm(all_configs_weighted - config_weighted, axis=1)
        nearest_idx = np.argmin(distances)

        sh_results.append(all_sh[nearest_idx])

    return np.array(sh_results)


def visualize_extrapolation_results(
    test_configs: np.ndarray,
    per_config_mae: List[float],
    train_mae: float,
    output_dir: Path
):
    """可视化外推测试结果"""
    output_dir.mkdir(parents=True, exist_ok=True)

    config_labels = [
        'Low Intensity\n(0.3 < 0.5)',
        'High Intensity\n(2.0 > 1.5)',
        'Low Temp\n(2500K < 3000K)',
        'High Temp\n(8500K > 7000K)',
        'High Cloud\n(0.9 > 0.8)'
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Per-config MAE条形图
    x_pos = np.arange(len(config_labels))
    axes[0, 0].bar(x_pos, per_config_mae, alpha=0.7, edgecolor='black')
    axes[0, 0].axhline(train_mae, color='r', linestyle='--',
                       label=f'Train MAE={train_mae:.4f}')
    axes[0, 0].axhline(train_mae * 3.0, color='orange', linestyle='--',
                       label=f'3× Threshold={train_mae*3.0:.4f}')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(config_labels, fontsize=9)
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].set_title('Extrapolation Query MAE by Configuration')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3, axis='y')

    # 2. MAE vs 配置参数 (intensity)
    intensities = test_configs[:, 2]
    axes[0, 1].scatter(intensities, per_config_mae, s=100, alpha=0.6, edgecolor='black')
    axes[0, 1].axvspan(0.5, 1.5, alpha=0.2, color='green', label='Train range')
    axes[0, 1].set_xlabel('Intensity')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('MAE vs Intensity (Extrapolation)')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 3. MAE vs 色温
    temps = test_configs[:, 3]
    axes[1, 0].scatter(temps, per_config_mae, s=100, alpha=0.6,
                       color='orange', edgecolor='black')
    axes[1, 0].axvspan(3000, 7000, alpha=0.2, color='green', label='Train range')
    axes[1, 0].set_xlabel('Color Temperature (K)')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('MAE vs Color Temperature (Extrapolation)')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # 4. 统计对比
    mean_mae = np.mean(per_config_mae)
    degradation = mean_mae / train_mae

    stats_text = f"""外推查询统计:

Mean MAE: {mean_mae:.4f}
Train MAE: {train_mae:.4f}
退化比例: {degradation:.2f}×

Per-config MAE:
  Min: {np.min(per_config_mae):.4f}
  Max: {np.max(per_config_mae):.4f}
  Std: {np.std(per_config_mae):.4f}

成功标准: <3.0×
结果: {'PASS ✓' if degradation < 3.0 else 'FAIL ✗'}"""

    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                    verticalalignment='center')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'extrapolation_query_analysis.png', dpi=150)
    print(f"可视化结果已保存至: {output_dir / 'extrapolation_query_analysis.png'}")
    plt.close()


def main():
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = Path('../data_generation/output/5D_parametric_validation')
    checkpoint_path = Path('archive/Week3-4_MLP_Hybrid_Failed/checkpoints/stage1_best.pt')
    output_dir = Path('experiments/week4_extrapolation_query')

    print("=" * 80)
    print("Week 4.2: Physics-only外推查询测试")
    print("=" * 80)

    # 1. 定义外推配置
    print("\n[1/5] 定义5个超范围外推配置...")
    test_configs = get_extrapolation_configs()

    print("外推配置详情:")
    config_names = ['Low Intensity', 'High Intensity', 'Low Temp', 'High Temp', 'High Cloud']
    for i, (name, config) in enumerate(zip(config_names, test_configs)):
        print(f"  {i+1}. {name:15s}: zenith={config[0]:5.1f}°, azimuth={config[1]:6.1f}°, "
              f"intensity={config[2]:.2f}, temp={config[3]:.0f}K, cloud={config[4]:.2f}")

    # 2. 加载模型
    print("\n[2/5] 加载Physics-only模型...")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint不存在: {checkpoint_path}")
        return

    model = PhysicsLowRank5D(rank=5)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract only physics components from hybrid checkpoint
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                      if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)

    print(f"模型已加载: {checkpoint_path}")
    print(f"  训练轮数: {checkpoint.get('epoch', 'N/A')}")
    train_mae = checkpoint.get('val_mae', 0.0391)
    print(f"  训练Val MAE: {train_mae:.4f}")

    # 3. 获取ground truth
    print("\n[3/5] 获取ground truth SH系数...")
    print("  注意: 当前使用最近邻近似，实际应重新渲染")
    gt_sh = load_ground_truth_sh_extrapolation(test_configs, data_dir, probe_idx=0)
    print(f"  Ground truth shape: {gt_sh.shape}")

    # 4. 计算外推误差
    print("\n[4/5] 计算外推查询误差...")
    metrics = compute_extrapolation_error(model, test_configs, gt_sh, device)

    test_mae = metrics['mae']
    degradation_ratio = test_mae / train_mae

    print(f"\n外推查询结果:")
    print(f"  Test MAE: {test_mae:.4f}")
    print(f"  Train MAE: {train_mae:.4f}")
    print(f"  退化比例: {degradation_ratio:.2f}× (目标<3.0×)")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  Relative MAE: {metrics['relative_mae']:.4f}")

    # Per-config分析
    per_config_mae = np.array(metrics['per_config_mae'])
    print(f"\nPer-config MAE:")
    for i, (name, mae) in enumerate(zip(config_names, per_config_mae)):
        status = "✓" if mae < train_mae * 3.0 else "✗"
        print(f"  {status} {name:15s}: {mae:.4f} ({mae/train_mae:.2f}×)")

    print(f"\n统计:")
    print(f"  Mean: {per_config_mae.mean():.4f}")
    print(f"  Std: {per_config_mae.std():.4f}")
    print(f"  Min: {per_config_mae.min():.4f} ({per_config_mae.min()/train_mae:.2f}×)")
    print(f"  Max: {per_config_mae.max():.4f} ({per_config_mae.max()/train_mae:.2f}×)")

    # 5. 保存结果
    print("\n[5/5] 保存结果...")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'timestamp': datetime.now().isoformat(),
        'test_configs': test_configs.tolist(),
        'config_names': config_names,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'degradation_ratio': degradation_ratio,
        'rmse': metrics['rmse'],
        'relative_mae': metrics['relative_mae'],
        'per_config_mae': metrics['per_config_mae'],
        'per_config_stats': {
            'mean': float(per_config_mae.mean()),
            'std': float(per_config_mae.std()),
            'min': float(per_config_mae.min()),
            'max': float(per_config_mae.max())
        },
        'success': degradation_ratio < 3.0
    }

    with open(output_dir / 'extrapolation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 可视化
    visualize_extrapolation_results(test_configs, per_config_mae, train_mae, output_dir)

    # Go/No-Go决策
    print("\n" + "=" * 80)
    if results['success']:
        print(f"✓ 外推查询测试通过 (退化比例 {degradation_ratio:.2f}× < 3.0×)")
        print("  建议: 继续Week 4.3查询延迟测试")
    else:
        print(f"✗ 外推查询测试失败 (退化比例 {degradation_ratio:.2f}× >= 3.0×)")
        print("  建议: 增加物理基维度或重新审视外推策略")
    print("=" * 80)

    print(f"\n结果已保存至: {output_dir}")


if __name__ == '__main__':
    main()
