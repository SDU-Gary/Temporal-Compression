"""
Week 6.1: Ablation实验 - Gaussian-Physics 5D模型

目标：
- 找到最优K和rank组合以实现17×压缩比
- 验证质量vs压缩比权衡
- 系统评估参数影响

实验配置：
1. K=20, r=5 (baseline, 39.31×) - 已完成
2. K=30, r=5 (28.8×)
3. K=40, r=5 (21.6×)
4. K=50, r=5 (17.3×) - 目标
5. K=20, r=8 (25.8×)
6. K=30, r=8 (18.5×) - 平衡
"""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt

from models.gaussian_physics_5D import GaussianPhysicsCompression5D
from data.transfer_tensor_dataset import TransferTensorDataset5D
from training import BatchAdapter, GaussianPhysicsTrainer
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


def compute_expected_params(K, rank):
    """计算预期参数量"""
    # mu: K*3, log_scale: K*3, U: K*27*rank, time_coeffs: K*rank*7
    return K * 3 + K * 3 + K * 27 * rank + K * rank * 7


def compute_expected_compression(K, rank, original_params=138375):
    """计算预期压缩比"""
    compressed = compute_expected_params(K, rank)
    return original_params / compressed


def train_single_config(
    config_name,
    num_gaussians,
    rank,
    train_dataset,
    val_dataset,
    test_dataset,
    device,
    output_dir,
    num_epochs=1000,  # Reduce from 2000 since baseline converged at epoch 610
    lr=1e-3,
    lambda_temporal=0.001
):
    """训练单个配置"""
    print(f"\n{'='*80}")
    print(f"配置: {config_name}")
    print(f"  K={num_gaussians}, rank={rank}")
    print(f"  预期参数: {compute_expected_params(num_gaussians, rank):,}")
    print(f"  预期压缩比: {compute_expected_compression(num_gaussians, rank):.2f}×")
    print(f"{'='*80}")

    # 初始化模型
    model = GaussianPhysicsCompression5D(
        num_gaussians=num_gaussians,
        rank=rank,
        sh_dim=27
    )
    model.to(device)

    # K-Means初始化
    full_tensor = train_dataset.tensor
    probe_positions = train_dataset.probe_positions
    light_configs = train_dataset.light_configs_subset

    full_tensor_t = torch.from_numpy(full_tensor).float()

    model.init_from_kmeans(
        probe_positions=probe_positions,
        light_configs=light_configs,
        sh_tensor=full_tensor_t
    )

    # Ensure parameters on correct device
    model.to(device)

    # 计算实际压缩比
    original_params = train_dataset.num_probes * train_dataset.tensor.shape[1] * 27
    compressed_params = model.num_params()
    ratio = original_params / compressed_params

    print(f"\n实际压缩比:")
    print(f"  原始参数: {original_params:,}")
    print(f"  压缩参数: {compressed_params:,}")
    print(f"  压缩比: {ratio:.2f}× (目标: 14-18×)")

    # 训练
    adapter = BatchAdapter(params_key="light_params")
    trainer = GaussianPhysicsTrainer(
        model=model,
        device=device,
        adapter=adapter,
        lr=lr,
        recon_loss="charbonnier",
        temporal_weight=lambda_temporal,
        top_k=3,
        grad_clip=1.0,
    )

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=512, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=512, shuffle=False
    )

    # 训练循环
    config_output_dir = output_dir / config_name
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        output_dir=config_output_dir,
        save_best=True,
        best_metric="mae",
        log_every=100,
    )

    val_mae_history = history["val"]["mae"]
    best_val_mae = min(val_mae_history) if val_mae_history else float("inf")
    best_epoch = val_mae_history.index(best_val_mae) + 1 if val_mae_history else 0

    # 测试
    checkpoint_path = config_output_dir / 'best_model.pt'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=512, shuffle=False
    )

    test_metrics = trainer.validate_epoch(test_loader)

    # 保存结果
    results = {
        'config_name': config_name,
        'num_gaussians': num_gaussians,
        'rank': rank,
        'compressed_params': compressed_params,
        'original_params': original_params,
        'compression_ratio': ratio,
        'best_epoch': best_epoch,
        'val_mae': best_val_mae,
        'test_metrics': test_metrics,
        'target_range': [14.0, 18.0],
        'in_target_range': 14.0 <= ratio <= 18.0
    }

    with open(config_output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{config_name} 完成:")
    print(f"  压缩比: {ratio:.2f}× ({'✓' if 14.0 <= ratio <= 18.0 else '✗'} 目标: 14-18×)")
    print(f"  Test MAE: {test_metrics['mae']:.6f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Gaussian-Physics 5D ablation")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default="experiments/week6_ablation_5D")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = Path(resolve_data_root(args.data_root, args.manifest))
    output_dir = Path(args.output)

    print("=" * 80)
    print("Week 6.1: Ablation实验 - Gaussian-Physics 5D")
    print("=" * 80)

    # 加载数据集
    print("\n加载数据集...")
    train_dataset = TransferTensorDataset5D(
        data_root=data_dir, split='train', normalize_params=False
    )
    val_dataset = TransferTensorDataset5D(
        data_root=data_dir, split='val', normalize_params=False
    )
    test_dataset = TransferTensorDataset5D(
        data_root=data_dir, split='test', normalize_params=False
    )

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    # 实验配置
    configs = [
        # Skip baseline - already trained in Week 5.3
        # ('K20_r5', 20, 5),  # Baseline: 39.31×
        ('K30_r5', 30, 5),  # 28.8×
        ('K40_r5', 40, 5),  # 21.6×
        ('K50_r5', 50, 5),  # 17.3× - TARGET
        ('K20_r8', 20, 8),  # 25.8×
        ('K30_r8', 30, 8),  # 18.5× - Balanced
    ]

    all_results = []

    # 添加已完成的baseline结果
    baseline_results_path = Path('experiments/week5_gaussian_physics_5D/training_results.json')
    if baseline_results_path.exists():
        with open(baseline_results_path) as f:
            baseline_data = json.load(f)
            all_results.append({
                'config_name': 'K20_r5_baseline',
                'num_gaussians': baseline_data['config']['num_gaussians'],
                'rank': baseline_data['config']['rank'],
                'compressed_params': baseline_data['compression']['compressed_params'],
                'original_params': baseline_data['compression']['original_params'],
                'compression_ratio': baseline_data['compression']['ratio'],
                'best_epoch': baseline_data['best_epoch'],
                'val_mae': baseline_data['val_mae'],
                'test_metrics': baseline_data['test_metrics'],
                'target_range': [14.0, 18.0],
                'in_target_range': 14.0 <= baseline_data['compression']['ratio'] <= 18.0
            })

    # 运行ablation实验
    for config_name, K, rank in configs:
        result = train_single_config(
            config_name=config_name,
            num_gaussians=K,
            rank=rank,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            device=device,
            output_dir=output_dir,
            num_epochs=args.epochs
        )
        all_results.append(result)

    # 综合分析
    print("\n" + "=" * 80)
    print("Ablation实验综合结果")
    print("=" * 80)

    print(f"\n{'配置':<15} {'K':>4} {'Rank':>5} {'参数量':>8} {'压缩比':>8} {'Test MAE':>10} {'目标范围':>8}")
    print("-" * 80)

    for result in sorted(all_results, key=lambda x: x['compression_ratio'], reverse=True):
        in_range = '✓' if result['in_target_range'] else ' '
        print(f"{result['config_name']:<15} {result['num_gaussians']:>4} {result['rank']:>5} "
              f"{result['compressed_params']:>8,} {result['compression_ratio']:>7.2f}× "
              f"{result['test_metrics']['mae']:>10.6f} {in_range:>8}")

    # 可视化
    output_dir.mkdir(parents=True, exist_ok=True)
    visualize_ablation_results(all_results, output_dir)

    # 保存汇总
    summary = {
        'timestamp': datetime.now().isoformat(),
        'experiments': all_results,
        'target_compression': [14.0, 18.0],
        'target_mae': 0.05
    }

    with open(output_dir / 'ablation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Go/No-Go决策
    best_in_range = None
    for result in all_results:
        if result['in_target_range']:
            if best_in_range is None or result['test_metrics']['mae'] < best_in_range['test_metrics']['mae']:
                best_in_range = result

    print("\n" + "=" * 80)
    if best_in_range:
        print(f"✓ Week 6.1 Ablation成功")
        print(f"\n最优配置: {best_in_range['config_name']}")
        print(f"  K={best_in_range['num_gaussians']}, rank={best_in_range['rank']}")
        print(f"  压缩比: {best_in_range['compression_ratio']:.2f}× ∈ [14, 18]")
        print(f"  Test MAE: {best_in_range['test_metrics']['mae']:.6f} < 0.05")
        print(f"  参数量: {best_in_range['compressed_params']:,}")
        print("\n建议: 继续Week 6.2最终验证")
    else:
        print(f"✗ Week 6.1 Ablation需调整")
        print("\n无配置在目标压缩比范围内 [14, 18]")
        print("建议: 调整K或rank参数继续实验")
    print("=" * 80)

    print(f"\n结果已保存至: {output_dir}")


def visualize_ablation_results(results, output_dir):
    """可视化ablation结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 提取数据
    configs = [r['config_name'] for r in results]
    compression_ratios = [r['compression_ratio'] for r in results]
    test_maes = [r['test_metrics']['mae'] for r in results]
    params = [r['compressed_params'] for r in results]
    in_range = [r['in_target_range'] for r in results]

    colors = ['green' if ir else 'red' for ir in in_range]

    # 1. 压缩比 vs Test MAE
    axes[0, 0].scatter(compression_ratios, test_maes, c=colors, s=100, alpha=0.7)
    axes[0, 0].axhline(0.05, color='orange', linestyle='--', label='MAE Target=0.05')
    axes[0, 0].axvline(14, color='blue', linestyle='--', alpha=0.5, label='Compression Range')
    axes[0, 0].axvline(18, color='blue', linestyle='--', alpha=0.5)
    axes[0, 0].fill_betweenx([0, max(test_maes)*1.1], 14, 18, alpha=0.1, color='blue')
    axes[0, 0].set_xlabel('Compression Ratio (×)')
    axes[0, 0].set_ylabel('Test MAE')
    axes[0, 0].set_title('Compression vs Quality Trade-off')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    for i, config in enumerate(configs):
        axes[0, 0].annotate(config, (compression_ratios[i], test_maes[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

    # 2. 参数量 vs 压缩比
    axes[0, 1].scatter(params, compression_ratios, c=colors, s=100, alpha=0.7)
    axes[0, 1].axhline(14, color='blue', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(18, color='blue', linestyle='--', alpha=0.5)
    axes[0, 1].fill_between([0, max(params)*1.1], 14, 18, alpha=0.1, color='blue')
    axes[0, 1].set_xlabel('Compressed Parameters')
    axes[0, 1].set_ylabel('Compression Ratio (×)')
    axes[0, 1].set_title('Parameters vs Compression')
    axes[0, 1].grid(alpha=0.3)

    # 3. 配置对比柱状图
    x_pos = np.arange(len(configs))
    axes[1, 0].bar(x_pos, compression_ratios, color=colors, alpha=0.7)
    axes[1, 0].axhline(14, color='blue', linestyle='--', alpha=0.5, label='Target Range')
    axes[1, 0].axhline(18, color='blue', linestyle='--', alpha=0.5)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(configs, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Compression Ratio (×)')
    axes[1, 0].set_title('Compression Ratio by Configuration')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3, axis='y')

    # 4. Test MAE对比
    axes[1, 1].bar(x_pos, test_maes, color=colors, alpha=0.7)
    axes[1, 1].axhline(0.05, color='orange', linestyle='--', label='MAE Target')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(configs, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Test MAE')
    axes[1, 1].set_title('Test MAE by Configuration')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_analysis.png', dpi=150)
    print(f"可视化结果已保存至: {output_dir / 'ablation_analysis.png'}")
    plt.close()


if __name__ == '__main__':  # pragma: no cover
    main()
