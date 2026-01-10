"""
Stage 1训练: 纯物理基预训练

策略:
    - 冻结MLP参数
    - α=0 (禁用MLP残差)
    - 仅训练U和time_coeffs
    - 目标: 为Stage 2提供良好的初始化

验收标准:
    - Test MAE <0.05 (基于Week 2的0.0135, 应该轻松达标)
    - 训练稳定, 无梯度爆炸

输出:
    - 保存训练好的模型checkpoint
    - 训练曲线可视化
    - 性能报告

用法:
    python scripts/train_stage1_physics_only.py \
        --data_dir ../data_generation/output/5D_parametric_validation \
        --rank 5 \
        --epochs 2000 \
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
import sys

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from models.physics_mlp_hybrid import PhysicsMLPHybrid
from data.transfer_tensor_dataset import TransferTensorDataset5D


def train_physics_only(
    model: PhysicsMLPHybrid,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    device: str,
    save_dir: Path
):
    """Stage 1训练循环

    Args:
        model: PhysicsMLPHybrid (MLP冻结)
        train_loader: Training dataloader
        val_loader: Validation dataloader
        epochs: Number of epochs
        lr: Learning rate
        device: 'cuda' or 'cpu'
        save_dir: Directory to save checkpoints
    """
    model = model.to(device)

    # 冻结MLP, 设置α=0
    model.freeze_mlp()
    model.alpha.data.fill_(0.0)
    model.alpha.requires_grad = False
    print(f"✓ MLP frozen, α=0")

    # 优化器 (仅优化U和time_coeffs)
    optimizer = optim.Adam([
        {'params': [model.U], 'lr': lr},
        {'params': [model.time_coeffs], 'lr': lr}
    ])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200
    )

    criterion = nn.L1Loss()

    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 500

    print(f"\nStage 1 Training Started:")
    print(f"  Epochs: {epochs}")
    print(f"  LR: {lr}")
    print(f"  Device: {device}")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_losses = []

        for batch in train_loader:
            light_params = batch['light_params'].to(device)
            sh_gt = batch['sh_coeffs'].to(device)

            optimizer.zero_grad()

            # Forward (use_mlp=False 显式禁用MLP)
            sh_pred = model(light_params, use_mlp=False)

            loss = criterion(sh_pred, sh_gt)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                light_params = batch['light_params'].to(device)
                sh_gt = batch['sh_coeffs'].to(device)

                sh_pred = model(light_params, use_mlp=False)
                loss = criterion(sh_pred, sh_gt)

                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        history['val_loss'].append(avg_val_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # LR scheduler
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss
            }
            torch.save(checkpoint, save_dir / 'stage1_best.pt')

        else:
            patience_counter += 1

        # Logging
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}: Train Loss={avg_train_loss:.6f}, "
                  f"Val Loss={avg_val_loss:.6f}, "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}, "
                  f"Patience={patience_counter}/{max_patience}")

        # Early stop
        if patience_counter >= max_patience:
            print(f"\n✓ Early stopping at epoch {epoch+1}")
            break

    return history


def visualize_training(history: dict, output_dir: Path):
    """可视化训练曲线"""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax = axes[0]
    ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(history['val_loss'], label='Val Loss', linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MAE Loss', fontsize=12)
    ax.set_title('Stage 1: Physics-Only Training', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Learning rate
    ax = axes[1]
    ax.plot(history['learning_rate'], linewidth=2, color='coral')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'stage1_training_curve.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()


def evaluate_model(model, test_loader, device):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for batch in test_loader:
            light_params = batch['light_params'].to(device)
            sh_gt = batch['sh_coeffs']

            sh_pred = model(light_params, use_mlp=False).cpu()

            all_preds.append(sh_pred)
            all_gts.append(sh_gt)

    preds = torch.cat(all_preds, dim=0).numpy()
    gts = torch.cat(all_gts, dim=0).numpy()

    # Metrics
    mae = np.abs(preds - gts).mean()
    rmse = np.sqrt(((preds - gts) ** 2).mean())
    max_error = np.abs(preds - gts).max()

    return {
        'MAE': mae,
        'RMSE': rmse,
        'Max Error': max_error,
        'predictions': preds,
        'ground_truth': gts
    }


def main():
    parser = argparse.ArgumentParser(description='Stage 1: Physics-Only Pre-training')
    parser.add_argument('--data_dir', type=str,
                       default='../data_generation/output/5D_parametric_validation')
    parser.add_argument('--rank', type=int, default=5)
    parser.add_argument('--mlp_hidden', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./stage1_physics_only')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else 'cpu'

    print("="*70)
    print("Stage 1: Physics-Only Pre-training")
    print("="*70)
    print(f"Data: {args.data_dir}")
    print(f"Rank: {args.rank}")
    print(f"MLP Hidden: {args.mlp_hidden}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.lr}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device: {device}")
    print("="*70)

    # === 1. 加载数据 ===
    from data.transfer_tensor_dataset import create_dataloaders_5D

    train_loader, val_loader, test_loader = create_dataloaders_5D(
        data_root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        normalize_probes=False,
        normalize_params=False
    )

    # === 2. 创建模型 ===
    model = PhysicsMLPHybrid(
        rank=args.rank,
        mlp_hidden=args.mlp_hidden,
        learnable_alpha=False  # Stage 1不学习α
    )

    param_counts = model.count_parameters()
    print(f"\nModel Parameters:")
    print(f"  Physics: {param_counts['physics']}")
    print(f"  MLP (frozen): {param_counts['mlp']}")
    print(f"  Total: {param_counts['total']}")

    # === 3. 训练 ===
    history = train_physics_only(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir=output_dir
    )

    # === 4. 加载最佳模型并评估 ===
    print(f"\nLoading best model for final evaluation...")
    checkpoint = torch.load(output_dir / 'stage1_best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate_model(model, test_loader, device)

    print(f"\nStage 1 Final Results:")
    print(f"  Test MAE: {test_metrics['MAE']:.6f}")
    print(f"  Test RMSE: {test_metrics['RMSE']:.6f}")
    print(f"  Max Error: {test_metrics['Max Error']:.6f}")

    # Go/No-Go Decision
    mae_threshold = 0.05
    pass_test = test_metrics['MAE'] < mae_threshold

    print(f"\nStage 1 Validation:")
    print(f"  Target: Test MAE <{mae_threshold}")
    print(f"  Result: {'✓ PASS' if pass_test else '✗ FAIL'}")

    if pass_test:
        print(f"\n✓ Stage 1 complete - Ready for Stage 2 hybrid training")
    else:
        print(f"\n⚠ Stage 1 MAE higher than expected - Consider:")
        print(f"    - Increase training epochs")
        print(f"    - Adjust learning rate")
        print(f"    - Increase rank")

    # === 5. 可视化 ===
    visualize_training(history, output_dir)

    # === 6. 保存结果 ===
    results = {
        'stage': 1,
        'rank': args.rank,
        'mlp_hidden': args.mlp_hidden,
        'epochs_trained': len(history['train_loss']),
        'final_train_loss': float(history['train_loss'][-1]),
        'final_val_loss': float(history['val_loss'][-1]),
        'test_mae': float(test_metrics['MAE']),
        'test_rmse': float(test_metrics['RMSE']),
        'test_max_error': float(test_metrics['Max Error']),
        'pass_validation': pass_test
    }

    with open(output_dir / 'stage1_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved: {output_dir / 'stage1_results.json'}")
    print(f"\n✓ Stage 1 training complete!")


if __name__ == '__main__':
    main()
