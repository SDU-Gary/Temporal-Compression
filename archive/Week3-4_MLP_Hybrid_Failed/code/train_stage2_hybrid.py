"""
Stage 2训练: Physics + MLP混合模型

策略:
    - 加载Stage 1的checkpoint (预训练的U和time_coeffs)
    - 解冻MLP参数
    - α warmup: 前500 epochs从0线性增至可学习
    - 差分学习率: MLP慢学 (1e-5), U微调 (1e-4)
    - 目标: 混合模型优于纯物理基>10%

验收标准 (Week 4 Go/No-Go):
    - Test MAE <0.04
    - vs Stage 1改进 >10%
    - α值收敛至合理范围 (0.05-0.3)

用法:
    python scripts/train_stage2_hybrid.py \
        --stage1_checkpoint ./stage1_physics_only/stage1_best.pt \
        --data_dir ../data_generation/output/5D_parametric_validation \
        --epochs 1000 \
        --warmup_epochs 200
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
from data.transfer_tensor_dataset import create_dataloaders_5D


class AlphaWarmupScheduler:
    """α线性warmup调度器

    前warmup_epochs个epoch, α从0线性增长到init_alpha
    然后α变为可学习参数
    """

    def __init__(self, model, warmup_epochs: int, target_alpha: float = 0.1):
        self.model = model
        self.warmup_epochs = warmup_epochs
        self.target_alpha = target_alpha
        self.current_epoch = 0

    def step(self):
        """每个epoch调用一次"""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            alpha_value = (self.current_epoch / self.warmup_epochs) * self.target_alpha
            self.model.alpha.data.fill_(alpha_value)
            self.model.alpha.requires_grad = False
        else:
            # 进入可学习阶段
            if self.current_epoch == self.warmup_epochs:
                self.model.alpha.requires_grad = True
                print(f"\n✓ Alpha warmup complete, now learnable (initial value: {self.target_alpha:.4f})")

        self.current_epoch += 1

    def get_alpha(self) -> float:
        return self.model.alpha.item()


def train_hybrid(
    model: PhysicsMLPHybrid,
    train_loader,
    val_loader,
    epochs: int,
    lr_mlp: float,
    lr_physics: float,
    warmup_epochs: int,
    device: str,
    save_dir: Path
):
    """Stage 2训练循环

    Args:
        model: PhysicsMLPHybrid (从Stage 1 checkpoint初始化)
        train_loader: Training dataloader
        val_loader: Validation dataloader
        epochs: Number of epochs
        lr_mlp: MLP learning rate (低)
        lr_physics: Physics parameters learning rate (高)
        warmup_epochs: Alpha warmup epochs
        device: 'cuda' or 'cpu'
        save_dir: Directory to save checkpoints
    """
    model = model.to(device)

    # 解冻MLP
    model.unfreeze_mlp()

    # 差分学习率优化器
    optimizer = optim.Adam([
        {'params': [model.U, model.time_coeffs], 'lr': lr_physics},
        {'params': model.residual_mlp.parameters(), 'lr': lr_mlp},
        # α在warmup阶段后会添加
    ])

    # Alpha warmup scheduler
    alpha_scheduler = AlphaWarmupScheduler(model, warmup_epochs=warmup_epochs, target_alpha=0.1)

    # LR scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=150
    )

    criterion = nn.L1Loss()

    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'alpha_values': [],
        'lr_mlp': [],
        'lr_physics': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 300

    print(f"\nStage 2 Hybrid Training Started:")
    print(f"  Epochs: {epochs}")
    print(f"  LR MLP: {lr_mlp}")
    print(f"  LR Physics: {lr_physics}")
    print(f"  Warmup Epochs: {warmup_epochs}")
    print(f"  Device: {device}")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(epochs):
        # Alpha warmup
        alpha_scheduler.step()

        # 如果α刚变为可学习, 添加到优化器
        if epoch == warmup_epochs:
            optimizer.add_param_group({'params': [model.alpha], 'lr': 1e-3})

        # --- Train ---
        model.train()
        train_losses = []

        for batch in train_loader:
            light_params = batch['light_params'].to(device)
            sh_gt = batch['sh_coeffs'].to(device)

            optimizer.zero_grad()

            # Forward (use_mlp=True 启用混合模式)
            sh_pred = model(light_params, use_mlp=True)

            loss = criterion(sh_pred, sh_gt)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

                sh_pred = model(light_params, use_mlp=True)
                loss = criterion(sh_pred, sh_gt)

                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        history['val_loss'].append(avg_val_loss)
        history['alpha_values'].append(alpha_scheduler.get_alpha())
        history['lr_mlp'].append(optimizer.param_groups[1]['lr'])
        history['lr_physics'].append(optimizer.param_groups[0]['lr'])

        # LR scheduler
        lr_scheduler.step(avg_val_loss)

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
                'train_loss': avg_train_loss,
                'alpha': alpha_scheduler.get_alpha()
            }
            torch.save(checkpoint, save_dir / 'stage2_best.pt')

        else:
            patience_counter += 1

        # Logging
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}: Train={avg_train_loss:.6f}, "
                  f"Val={avg_val_loss:.6f}, "
                  f"α={alpha_scheduler.get_alpha():.4f}, "
                  f"LR_MLP={optimizer.param_groups[1]['lr']:.2e}, "
                  f"Patience={patience_counter}/{max_patience}")

        # Early stop
        if patience_counter >= max_patience:
            print(f"\n✓ Early stopping at epoch {epoch+1}")
            break

    return history


def visualize_training(history: dict, stage1_mae: float, output_dir: Path):
    """可视化训练曲线"""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss curve
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(history['val_loss'], label='Val Loss', linewidth=2, linestyle='--')

    # 添加Stage 1基线
    ax.axhline(stage1_mae, color='red', linestyle=':', linewidth=2,
              label=f'Stage 1 MAE ({stage1_mae:.4f})')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MAE Loss', fontsize=12)
    ax.set_title('Stage 2: Hybrid Training', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Alpha evolution
    ax = axes[0, 1]
    ax.plot(history['alpha_values'], linewidth=2, color='purple')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Alpha Value', fontsize=12)
    ax.set_title('Mixing Coefficient α', fontsize=14)
    ax.grid(True, alpha=0.3)

    # 3. Learning rates
    ax = axes[1, 0]
    ax.plot(history['lr_mlp'], label='LR MLP', linewidth=2, color='coral')
    ax.plot(history['lr_physics'], label='LR Physics', linewidth=2, color='steelblue')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Differential Learning Rates', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Improvement over Stage 1
    ax = axes[1, 1]
    improvement = [(stage1_mae - val_loss) / stage1_mae * 100
                  for val_loss in history['val_loss']]
    ax.plot(improvement, linewidth=2, color='green')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.axhline(10, color='red', linestyle=':', linewidth=2,
              label='10% Target')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Improvement over Stage 1 (%)', fontsize=12)
    ax.set_title('Relative Improvement', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plot_path = output_dir / 'stage2_training_curves.png'
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

            sh_pred = model(light_params, use_mlp=True).cpu()

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
        'Max Error': max_error
    }


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Hybrid Model Training')
    parser.add_argument('--stage1_checkpoint', type=str,
                       default='./stage1_physics_only/stage1_best.pt')
    parser.add_argument('--data_dir', type=str,
                       default='../data_generation/output/5D_parametric_validation')
    parser.add_argument('--rank', type=int, default=5)
    parser.add_argument('--mlp_hidden', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--warmup_epochs', type=int, default=200)
    parser.add_argument('--lr_mlp', type=float, default=1e-5)
    parser.add_argument('--lr_physics', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./stage2_hybrid')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else 'cpu'

    print("="*70)
    print("Stage 2: Hybrid Model Training")
    print("="*70)
    print(f"Stage 1 Checkpoint: {args.stage1_checkpoint}")
    print(f"Data: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Warmup Epochs: {args.warmup_epochs}")
    print(f"LR MLP: {args.lr_mlp}")
    print(f"LR Physics: {args.lr_physics}")
    print(f"Device: {device}")
    print("="*70)

    # === 1. 加载数据 ===
    train_loader, val_loader, test_loader = create_dataloaders_5D(
        data_root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        normalize_probes=False,
        normalize_params=False
    )

    # === 2. 创建模型并加载Stage 1 checkpoint ===
    model = PhysicsMLPHybrid(
        rank=args.rank,
        mlp_hidden=args.mlp_hidden,
        learnable_alpha=True
    )

    # 加载Stage 1权重
    if Path(args.stage1_checkpoint).exists():
        print(f"\nLoading Stage 1 checkpoint...")
        checkpoint = torch.load(args.stage1_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        stage1_test_mae = checkpoint.get('val_loss', 0.0)  # 用val_loss作为近似
        print(f"✓ Loaded Stage 1 checkpoint (Val MAE: {stage1_test_mae:.6f})")
    else:
        print(f"⚠ Stage 1 checkpoint not found: {args.stage1_checkpoint}")
        print(f"  Training from scratch (not recommended)")
        stage1_test_mae = float('inf')

    param_counts = model.count_parameters()
    print(f"\nModel Parameters:")
    print(f"  Physics: {param_counts['physics']}")
    print(f"  MLP: {param_counts['mlp']}")
    print(f"  Alpha: {param_counts['alpha']}")
    print(f"  Total: {param_counts['total']}")

    # === 3. 训练 ===
    history = train_hybrid(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr_mlp=args.lr_mlp,
        lr_physics=args.lr_physics,
        warmup_epochs=args.warmup_epochs,
        device=device,
        save_dir=output_dir
    )

    # === 4. 加载最佳模型并评估 ===
    print(f"\nLoading best model for final evaluation...")
    checkpoint = torch.load(output_dir / 'stage2_best.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate_model(model, test_loader, device)

    print(f"\nStage 2 Final Results:")
    print(f"  Test MAE: {test_metrics['MAE']:.6f}")
    print(f"  Test RMSE: {test_metrics['RMSE']:.6f}")
    print(f"  Max Error: {test_metrics['Max Error']:.6f}")
    print(f"  Final α: {checkpoint['alpha']:.4f}")

    # Week 4 Go/No-Go Decision
    mae_target = 0.04
    improvement_target = 0.10  # 10%

    if stage1_test_mae < float('inf'):
        improvement = (stage1_test_mae - test_metrics['MAE']) / stage1_test_mae
        print(f"\nImprovement over Stage 1:")
        print(f"  Stage 1 MAE: {stage1_test_mae:.6f}")
        print(f"  Stage 2 MAE: {test_metrics['MAE']:.6f}")
        print(f"  Improvement: {improvement*100:.2f}%")
    else:
        improvement = 0.0

    pass_mae = test_metrics['MAE'] < mae_target
    pass_improvement = improvement > improvement_target

    print(f"\nStage 2 Validation:")
    print(f"  1. Test MAE <{mae_target}: {'✓ PASS' if pass_mae else '✗ FAIL'}")
    print(f"  2. Improvement >{improvement_target*100}%: {'✓ PASS' if pass_improvement else '✗ FAIL'}")

    if pass_mae and pass_improvement:
        print(f"\n✓ Stage 2 complete - Ready for ablation & query validation (Week 4)")
    elif pass_mae:
        print(f"\n⚠ MAE达标但改进不足 - MLP贡献有限, 考虑:")
        print(f"    - 增加MLP hidden至128")
        print(f"    - 延长warmup至500 epochs")
    else:
        print(f"\n⚠ 需要调整超参数:")
        print(f"    - 增加training epochs")
        print(f"    - 调整学习率")

    # === 5. 可视化 ===
    visualize_training(history, stage1_test_mae, output_dir)

    # === 6. 保存结果 ===
    results = {
        'stage': 2,
        'rank': args.rank,
        'mlp_hidden': args.mlp_hidden,
        'epochs_trained': len(history['train_loss']),
        'warmup_epochs': args.warmup_epochs,
        'final_train_loss': float(history['train_loss'][-1]),
        'final_val_loss': float(history['val_loss'][-1]),
        'final_alpha': float(checkpoint['alpha']),
        'test_mae': float(test_metrics['MAE']),
        'test_rmse': float(test_metrics['RMSE']),
        'stage1_mae': float(stage1_test_mae),
        'improvement_pct': float(improvement * 100),
        'pass_validation': pass_mae and pass_improvement
    }

    with open(output_dir / 'stage2_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved: {output_dir / 'stage2_results.json'}")
    print(f"\n✓ Stage 2 training complete!")


if __name__ == '__main__':
    main()
