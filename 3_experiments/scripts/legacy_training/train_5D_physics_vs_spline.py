"""
5D参数化光源: 物理基 vs Spline基线对比实验

目标:
    验证物理引导低秩模型在5D参数空间的优势
    对比方法: Physics Low-Rank (170 params) vs Spline Interpolation (~810 params)

验收标准 (Week 2 Go/No-Go):
    - Physics Test MAE < 1.2× Spline MAE
    - 参数量: Physics << Spline (至少3×压缩)

用法:
    python scripts/train_5D_physics_vs_spline.py \
        --data_dir ../data_generation/output/5D_parametric_validation \
        --rank 5 \
        --epochs 1000 \
        --lr 1e-3
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.physics_low_rank import PhysicsLowRank5D
from data.transfer_tensor_dataset import create_dataloaders_5D


class SplineBaseline5D:
    """Spline插值基线 (5D → 27D输出)

    使用scipy.interpolate.LinearNDInterpolator进行5D插值
    参数量: 训练样本数 × 27 (实际存储所有训练数据点)
    """

    def __init__(self):
        self.interpolators = []  # 27个SH系数, 每个独立插值
        self.num_params = 0

    def fit(self, light_params: np.ndarray, sh_matrix: np.ndarray):
        """训练Spline插值器

        Args:
            light_params: [N, 5] light configurations
            sh_matrix: [N, 27] SH coefficients
        """
        N, num_sh = sh_matrix.shape
        self.interpolators = []

        # 每个SH系数独立建立插值器
        for i in range(num_sh):
            interpolator = LinearNDInterpolator(light_params, sh_matrix[:, i])
            self.interpolators.append(interpolator)

        # 参数量: 每个插值器存储所有训练点
        self.num_params = N * (5 + 1) * num_sh  # (input + output) × SH coeffs
        print(f"Spline interpolators created:")
        print(f"  Training points: {N}")
        print(f"  SH coefficients: {num_sh}")
        print(f"  Effective parameters: {self.num_params}")

    def predict(self, light_params: np.ndarray) -> np.ndarray:
        """预测SH系数

        Args:
            light_params: [M, 5] query light configurations

        Returns:
            sh_pred: [M, 27] predicted SH coefficients
        """
        M = light_params.shape[0]
        sh_pred = np.zeros((M, len(self.interpolators)))

        for i, interp in enumerate(self.interpolators):
            sh_pred[:, i] = interp(light_params)

        # 处理外推导致的NaN (用0填充)
        nan_mask = np.isnan(sh_pred)
        if nan_mask.any():
            print(f"  Warning: {nan_mask.sum()} NaN values in predictions (extrapolation)")
            sh_pred[nan_mask] = 0.0

        return sh_pred


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """计算回归指标

    Args:
        pred: [N, 27] predictions
        gt: [N, 27] ground truth

    Returns:
        metrics: dict with MAE, RMSE, Max Error
    """
    mae = np.abs(pred - gt).mean()
    rmse = np.sqrt(((pred - gt) ** 2).mean())
    max_error = np.abs(pred - gt).max()

    return {
        'MAE': mae,
        'RMSE': rmse,
        'Max Error': max_error
    }


def train_physics_model(
    model: PhysicsLowRank5D,
    train_light_params: np.ndarray,
    train_sh_matrix: np.ndarray,
    test_light_params: np.ndarray,
    test_sh_matrix: np.ndarray,
    epochs: int = 1000,
    lr: float = 1e-3,
    device: str = 'cuda'
):
    """训练物理引导低秩模型

    Args:
        model: PhysicsLowRank5D model
        train_light_params: [N_train, 5]
        train_sh_matrix: [N_train, 27]
        test_light_params: [N_test, 5]
        test_sh_matrix: [N_test, 27]
        epochs: Number of training epochs
        lr: Learning rate
        device: 'cuda' or 'cpu'

    Returns:
        train_losses: List of training losses
        test_losses: List of test losses
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    # Convert to tensors
    train_params_t = torch.from_numpy(train_light_params).float().to(device)
    train_sh_t = torch.from_numpy(train_sh_matrix).float().to(device)
    test_params_t = torch.from_numpy(test_light_params).float().to(device)
    test_sh_t = torch.from_numpy(test_sh_matrix).float().to(device)

    train_losses = []
    test_losses = []

    print(f"\nTraining Physics Low-Rank Model...")
    print(f"  Epochs: {epochs}, LR: {lr}, Device: {device}")

    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()

        pred = model(train_params_t)
        loss = criterion(pred, train_sh_t)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # Evaluate
        if (epoch + 1) % 100 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(test_params_t)
                test_loss = criterion(test_pred, test_sh_t).item()
                test_losses.append(test_loss)

            print(f"Epoch {epoch+1:4d}: Train Loss={loss.item():.6f}, Test Loss={test_loss:.6f}")

    return train_losses, test_losses


def visualize_comparison(results: dict, output_dir: Path):
    """可视化对比结果

    生成两张图:
        1. 训练曲线对比 (Physics)
        2. 测试指标对比柱状图 (Physics vs Spline)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # === 图1: 训练曲线 ===
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(results['physics_train_losses'], label='Train Loss', linewidth=2)
    epochs_test = np.linspace(0, len(results['physics_train_losses']),
                             len(results['physics_test_losses']))
    ax.plot(epochs_test, results['physics_test_losses'],
           label='Test Loss', linewidth=2, linestyle='--')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MAE Loss', fontsize=12)
    ax.set_title('Physics Low-Rank Training Curve', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plot_path = output_dir / 'training_curve.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()

    # === 图2: 指标对比柱状图 ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics_names = ['MAE', 'RMSE', 'Max Error']
    physics_metrics = results['physics_metrics']
    spline_metrics = results['spline_metrics']

    for i, (ax, metric_name) in enumerate(zip(axes, metrics_names)):
        physics_val = physics_metrics[metric_name]
        spline_val = spline_metrics[metric_name]

        x = [0, 1]
        heights = [physics_val, spline_val]
        colors = ['steelblue', 'coral']
        labels = ['Physics\nLow-Rank', 'Spline\nInterpolation']

        bars = ax.bar(x, heights, color=colors, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} Comparison', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, height in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10)

        # 添加胜者标记
        if physics_val < spline_val:
            ax.text(0, physics_val * 1.1, '✓', ha='center', fontsize=20, color='green')

    plt.tight_layout()
    plot_path = output_dir / 'metrics_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()


def print_comparison_report(results: dict):
    """打印对比报告"""
    physics_metrics = results['physics_metrics']
    spline_metrics = results['spline_metrics']
    physics_params = results['physics_params']
    spline_params = results['spline_params']

    print("\n" + "="*70)
    print("5D Parametric Light: Physics vs Spline Comparison")
    print("="*70)

    print(f"\nModel Parameters:")
    print(f"  Physics Low-Rank (rank={results['rank']}): {physics_params} params")
    print(f"  Spline Interpolation: {spline_params} params")
    print(f"  Compression Ratio: {spline_params / physics_params:.2f}×")

    print(f"\nTest Metrics:")
    print(f"  {'Metric':<15} {'Physics':<12} {'Spline':<12} {'Ratio':<10} {'Winner':<10}")
    print("-"*70)

    for metric_name in ['MAE', 'RMSE', 'Max Error']:
        p_val = physics_metrics[metric_name]
        s_val = spline_metrics[metric_name]
        ratio = p_val / s_val
        winner = 'Physics ✓' if p_val < s_val else 'Spline'

        print(f"  {metric_name:<15} {p_val:<12.6f} {s_val:<12.6f} {ratio:<10.3f} {winner:<10}")

    print("="*70)

    # Week 2 Go/No-Go Decision
    mae_ratio = physics_metrics['MAE'] / spline_metrics['MAE']
    compression_ratio = spline_params / physics_params

    print(f"\nWeek 2 Go/No-Go Decision:")
    print("-"*70)

    # 标准1: MAE比值
    mae_pass = mae_ratio < 1.2
    print(f"  1. Physics MAE / Spline MAE = {mae_ratio:.3f}")
    print(f"     Target: <1.2 → {'✓ PASS' if mae_pass else '✗ FAIL'}")

    # 标准2: 压缩比
    compression_pass = compression_ratio >= 3.0
    print(f"  2. Compression Ratio = {compression_ratio:.2f}×")
    print(f"     Target: ≥3.0× → {'✓ PASS' if compression_pass else '✗ FAIL'}")

    # 综合决策
    if mae_pass and compression_pass:
        decision = "✓ GO"
        recommendation = """
建议: 继续Week 3-4混合模型训练
  - 当前物理基已显示优势, MAE优于Spline
  - 添加MLP残差应能进一步提升10-15%精度
  - 预期混合模型: MAE <0.04, 压缩比~4.76×
"""
    elif mae_pass:
        decision = "⚠ CONDITIONAL GO"
        recommendation = f"""
建议: 继续, 但需调整参数
  - MAE达标, 但压缩比偏低 ({compression_ratio:.2f}× < 3.0×)
  - 可能原因: Spline插值器参数计算方式需优化
  - 物理基本身质量良好, 可进入Week 3
"""
    else:
        decision = "✗ NO-GO"
        recommendation = """
应急预案:
  1. 检查训练是否收敛 (增加epochs至3000)
  2. 调整学习率 (尝试1e-4或5e-3)
  3. 增加秩至8 (基于SVD 99.6%能量, rank=8应接近100%)
  4. 检查数据质量和归一化策略
"""

    print(f"\nDecision: {decision}")
    print(recommendation)
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Compare Physics Low-Rank vs Spline Baseline (5D)')
    parser.add_argument('--data_dir', type=str,
                       default='../data_generation/output/5D_parametric_validation',
                       help='Path to 5D parametric dataset')
    parser.add_argument('--rank', type=int, default=5,
                       help='Rank for physics low-rank model')
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='./comparison_5D_physics_vs_spline',
                       help='Output directory for results')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    device = args.device if torch.cuda.is_available() else 'cpu'

    print("="*70)
    print("5D Parametric Light: Physics vs Spline Comparison")
    print("="*70)
    print(f"Data: {args.data_dir}")
    print(f"Rank: {args.rank}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.lr}")
    print(f"Device: {device}")
    print("="*70)

    # === 1. 加载数据 ===
    from data.transfer_tensor_dataset import TransferTensorDataset5D

    train_dataset = TransferTensorDataset5D(
        data_root=args.data_dir,
        split='train',
        normalize_probes=False,  # 不需要probe位置
        normalize_params=False  # 保持原始参数用于物理基
    )

    test_dataset = TransferTensorDataset5D(
        data_root=args.data_dir,
        split='test',
        normalize_probes=False,
        normalize_params=False
    )

    # 提取训练/测试数据
    train_light_params = train_dataset.get_unnormalized_light_configs()  # [28, 5]
    train_sh_matrix = train_dataset.tensor_subset.mean(axis=0)  # [28, 27] - 所有探针平均

    test_light_params = test_dataset.get_unnormalized_light_configs()  # [7, 5]
    test_sh_matrix = test_dataset.tensor_subset.mean(axis=0)  # [7, 27]

    print(f"\nDataset splits:")
    print(f"  Train configs: {train_light_params.shape[0]}")
    print(f"  Test configs: {test_light_params.shape[0]}")

    # === 2. 训练Physics Low-Rank模型 ===
    physics_model = PhysicsLowRank5D(rank=args.rank, init_method='random')

    # SVD初始化 (可选, 加速收敛)
    physics_model.init_from_svd(train_light_params, train_sh_matrix)

    physics_train_losses, physics_test_losses = train_physics_model(
        physics_model,
        train_light_params,
        train_sh_matrix,
        test_light_params,
        test_sh_matrix,
        epochs=args.epochs,
        lr=args.lr,
        device=device
    )

    # 测试集预测
    physics_model.eval()
    with torch.no_grad():
        test_params_t = torch.from_numpy(test_light_params).float().to(device)
        physics_pred = physics_model(test_params_t).cpu().numpy()

    physics_metrics = compute_metrics(physics_pred, test_sh_matrix)
    physics_params = sum(p.numel() for p in physics_model.parameters())

    # === 3. 训练Spline Baseline ===
    print(f"\nTraining Spline Interpolation Baseline...")
    spline_model = SplineBaseline5D()
    spline_model.fit(train_light_params, train_sh_matrix)

    spline_pred = spline_model.predict(test_light_params)
    spline_metrics = compute_metrics(spline_pred, test_sh_matrix)
    spline_params = spline_model.num_params

    # === 4. 汇总结果 ===
    results = {
        'rank': args.rank,
        'physics_params': physics_params,
        'spline_params': spline_params,
        'physics_metrics': physics_metrics,
        'spline_metrics': spline_metrics,
        'physics_train_losses': physics_train_losses,
        'physics_test_losses': physics_test_losses
    }

    # === 5. 可视化 ===
    visualize_comparison(results, output_dir)

    # === 6. 打印报告 ===
    print_comparison_report(results)

    # === 7. 保存结果 ===
    results_path = output_dir / 'comparison_results.json'
    # Convert numpy types to Python native types for JSON
    json_results = {
        'rank': int(args.rank),
        'physics_params': int(physics_params),
        'spline_params': int(spline_params),
        'physics_metrics': {k: float(v) for k, v in physics_metrics.items()},
        'spline_metrics': {k: float(v) for k, v in spline_metrics.items()}
    }

    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n✓ Results saved: {results_path}")
    print(f"\n✓ Comparison complete!")


if __name__ == '__main__':
    main()
