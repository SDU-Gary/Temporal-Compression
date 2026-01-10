"""
Week 5.3: 训练多探针Gaussian-Physics模型

目标：
- 训练baseline (K=20, rank=5)
- 验证17×压缩比
- 集成SurveyGo改进 (Charbonnier loss + L1 temporal regularization)

数据：125探针 × 41配置 (5D参数化数据集)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import json
from datetime import datetime
import matplotlib.pyplot as plt

from models.gaussian_physics_5D import GaussianPhysicsCompression5D
from data.transfer_tensor_dataset import TransferTensorDataset5D
from training import BatchAdapter, GaussianPhysicsTrainer


def compute_compression_ratio(model, dataset):
    """计算压缩比

    Args:
        model: 训练好的模型
        dataset: 数据集

    Returns:
        ratio: 压缩比
        original_params: 原始参数量
        compressed_params: 压缩后参数量
    """
    # 原始参数量: P × M × 27
    num_probes = dataset.num_probes
    num_configs = dataset.tensor.shape[1]  # 总配置数（不是split后的）
    original_params = num_probes * num_configs * 27

    # 压缩参数量
    compressed_params = model.num_params()

    ratio = original_params / compressed_params

    return ratio, original_params, compressed_params


def plot_training_curve(train_losses, val_losses, output_dir):
    """绘制训练曲线"""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = range(1, len(train_losses['total']) + 1)

    # 1. 总损失
    axes[0, 0].plot(epochs, train_losses['total'], label='Train', alpha=0.7)
    axes[0, 0].plot(epochs, val_losses['mae'], label='Val MAE', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss / MAE')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_yscale('log')

    # 2. 重建损失
    axes[0, 1].plot(epochs, train_losses['recon'], label='Recon Loss', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].set_title('Reconstruction Loss (Charbonnier)')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_yscale('log')

    # 3. 时间正则化
    axes[1, 0].plot(epochs, train_losses['temporal'], label='Temporal L1', color='orange', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Temporal L1 Loss')
    axes[1, 0].set_title('Temporal Smoothness Regularization (SurveyGo)')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_yscale('log')

    # 4. 验证指标
    axes[1, 1].plot(epochs, val_losses['mae'], label='MAE', alpha=0.7)
    axes[1, 1].plot(epochs, val_losses['rmse'], label='RMSE', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Error')
    axes[1, 1].set_title('Validation Metrics')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curve.png', dpi=150)
    print(f"训练曲线已保存至: {output_dir / 'training_curve.png'}")
    plt.close()


def main():
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = Path('../data_generation/output/5D_parametric_validation')
    output_dir = Path('experiments/week5_gaussian_physics_5D')

    # 训练超参数
    num_gaussians = 20
    rank = 5
    lr = 1e-3
    lambda_temporal = 0.001
    num_epochs = 2000
    batch_size = 512

    print("=" * 80)
    print("Week 5.3: 训练多探针Gaussian-Physics模型")
    print("=" * 80)

    # 1. 加载数据
    print("\n[1/6] 加载数据集...")
    train_dataset = TransferTensorDataset5D(
        data_root=data_dir,
        split='train',
        normalize_params=False  # 保持原始尺度
    )
    val_dataset = TransferTensorDataset5D(
        data_root=data_dir,
        split='val',
        normalize_params=False
    )
    test_dataset = TransferTensorDataset5D(
        data_root=data_dir,
        split='test',
        normalize_params=False
    )

    print(f"数据集加载完成:")
    print(f"  Train: {len(train_dataset)} samples ({train_dataset.num_probes} probes × {train_dataset.num_configs} configs)")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # 2. 初始化模型
    print(f"\n[2/6] 初始化模型 (K={num_gaussians}, rank={rank})...")
    model = GaussianPhysicsCompression5D(
        num_gaussians=num_gaussians,
        rank=rank,
        sh_dim=27
    )
    model.to(device)

    # 3. K-Means初始化
    print(f"\n[3/6] K-Means初始化高斯...")
    # 加载完整数据集用于初始化
    full_tensor = train_dataset.tensor  # [P, M_split, 27]
    probe_positions = train_dataset.probe_positions  # [P, 3]
    light_configs = train_dataset.light_configs_subset  # [M_split, 5]

    # 转换为张量
    full_tensor_t = torch.from_numpy(full_tensor).float()
    probe_positions_np = probe_positions
    light_configs_np = light_configs

    model.init_from_kmeans(
        probe_positions=probe_positions_np,
        light_configs=light_configs_np,
        sh_tensor=full_tensor_t
    )

    # Ensure all parameters are on correct device after K-Means initialization
    model.to(device)

    # 4. 计算压缩比
    print(f"\n[4/6] 计算压缩比...")
    ratio, original_params, compressed_params = compute_compression_ratio(model, train_dataset)
    print(f"压缩比分析:")
    print(f"  原始参数量: {original_params:,} (125 probes × 41 configs × 27 SH)")
    print(f"  压缩参数量: {compressed_params:,}")
    print(f"  压缩比: {ratio:.2f}× (目标: 17×)")

    # 5. 训练
    print(f"\n[5/6] 开始训练 ({num_epochs} epochs)...")
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

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        output_dir=output_dir,
        save_best=True,
        best_metric="mae",
        log_every=100,
    )

    train_losses = history["train"]
    val_losses = history["val"]
    best_val_mae = min(val_losses["mae"]) if val_losses["mae"] else float("inf")
    best_epoch = val_losses["mae"].index(best_val_mae) + 1 if val_losses["mae"] else 0

    # 6. 测试
    print(f"\n[6/6] 测试最佳模型...")
    checkpoint_path = output_dir / 'best_model.pt'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    test_metrics = trainer.validate_epoch(test_loader)

    # 保存结果
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_gaussians': num_gaussians,
            'rank': rank,
            'lr': lr,
            'lambda_temporal': lambda_temporal,
            'num_epochs': num_epochs,
            'batch_size': batch_size
        },
        'compression': {
            'original_params': original_params,
            'compressed_params': compressed_params,
            'ratio': ratio,
            'target_ratio': 17.0,
            'success': ratio >= 14.0 and ratio <= 18.0
        },
        'best_epoch': best_epoch,
        'val_mae': best_val_mae,
        'test_metrics': test_metrics,
        'improvements': {
            'charbonnier_loss': True,
            'l1_temporal_regularization': True,
            'lambda_temporal': lambda_temporal
        }
    }

    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 绘制训练曲线
    plot_training_curve(train_losses, val_losses, output_dir)

    # 打印最终结果
    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)
    print(f"\n最佳模型 (Epoch {best_epoch}):")
    print(f"  Val MAE: {best_val_mae:.6f}")
    print(f"  Test MAE: {test_metrics['mae']:.6f}")
    print(f"  Test RMSE: {test_metrics['rmse']:.6f}")
    print(f"  Test Charbonnier: {test_metrics['charbonnier']:.6f}")

    print(f"\n压缩性能:")
    print(f"  压缩比: {ratio:.2f}× (目标: 14-18×)")
    print(f"  原始参数: {original_params:,}")
    print(f"  压缩参数: {compressed_params:,}")

    print(f"\nSurveyGo改进:")
    print(f"  ✓ Charbonnier loss (鲁棒损失函数)")
    print(f"  ✓ L1 temporal regularization (λ={lambda_temporal})")

    # Go/No-Go决策
    success = (14.0 <= ratio <= 18.0) and (test_metrics['mae'] < 0.05)

    print("\n" + "=" * 80)
    if success:
        print(f"✓ Week 5训练成功")
        print(f"  压缩比: {ratio:.2f}× ∈ [14, 18]")
        print(f"  Test MAE: {test_metrics['mae']:.6f} < 0.05")
        print("  建议: 继续Week 6 Ablation实验")
    else:
        print(f"✗ Week 5训练需改进")
        if not (14.0 <= ratio <= 18.0):
            print(f"  压缩比: {ratio:.2f}× 不在目标范围 [14, 18]")
        if test_metrics['mae'] >= 0.05:
            print(f"  Test MAE: {test_metrics['mae']:.6f} >= 0.05")
        print("  建议: 调整K或rank参数")
    print("=" * 80)

    print(f"\n结果已保存至: {output_dir}")


if __name__ == '__main__':
    main()
