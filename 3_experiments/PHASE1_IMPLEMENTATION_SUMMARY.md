# Phase 1 实验实现总结

## 实现状态

### ✅ 已完成

1. **实验脚本** (`3_experiments/phase1_optimization_experiments.py`)
   - Stage 1: 4 个色彩空间对齐实验 (λ_image: 0.1, 0.5, 1.0, 2.0)
   - Stage 2: 3 个学习率优化实验 (lr: 5e-4, 1e-3, 2e-3 + warmup + cosine)
   - Stage 3: 3 个数据增强实验 (intensity, position, dropout)
   - 自动化训练 → 评估 → benchmark 流程
   - 阶段报告生成

2. **Warmup 机制** (`2_src/training/gaussian_physics_trainer.py`)
   - 新增 `warmup_epochs` 参数 (line 78)
   - 实现线性 warmup 逻辑 (lines 819-836)
   - 集成到 `train.py` (line 307)

3. **数据增强模块** (`2_src/data/augmentations.py`)
   - `augment_intensity()`: RGB × [0.9, 1.1]
   - `augment_position()`: Gaussian noise (σ=0.02)
   - `augment_dropout()`: 随机丢弃 10-20% 光源
   - `create_augmentation_transform()`: 配置驱动的增强函数工厂

4. **快速启动指南** (`3_experiments/PHASE1_QUICKSTART.md`)
   - 详细执行步骤
   - 并行执行示例
   - 监控指标说明
   - 风险与备选方案

### ⚠️ 待集成

1. **数据增强集成到 BatchAdapter**
   - 当前 `augmentations.py` 已实现函数
   - 需要在 `train.py` 中创建 `BatchAdapter` 时注入 `params_transform`
   - 示例代码:
     ```python
     from data.augmentations import create_augmentation_transform

     adapter = BatchAdapter(
         params_key="light_params",
         mask_key="light_mask",
         params_transform=create_augmentation_transform(config['training'])
     )
     ```

2. **Stage 4 架构微调**
   - Attention on Z
   - 残差连接
   - 更深 FiLM 网络
   - 需要修改 `GaussianPhysicsCompressionUnified` 模型

---

## 关键修改

### 1. `gaussian_physics_trainer.py`

**新增参数** (line 78):
```python
warmup_epochs: int = 0,
```

**初始化** (lines 107-108):
```python
self.warmup_epochs = warmup_epochs
self.base_lr = lr
```

**Warmup 逻辑** (lines 819-836):
```python
if scheduler is not None:
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(val_metrics.get(best_metric, val_metrics["mae"]))
    else:
        # Apply warmup if in warmup phase
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            warmup_lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
        else:
            scheduler.step()
elif scheduler is not None:
    # Apply warmup if in warmup phase
    if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
        warmup_lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr
    else:
        scheduler.step()
```

### 2. `train.py`

**新增参数传递** (line 307):
```python
warmup_epochs=getattr(args, 'warmup_epochs', 0),
```

### 3. `augmentations.py`

**核心函数**:
- `augment_intensity(light_params, training, scale_range=(0.9, 1.1))`
- `augment_position(probe_pos, training, std=0.02)`
- `augment_dropout(light_params, light_mask, training, p=0.15)`
- `create_augmentation_transform(config)`: 配置驱动的增强函数工厂

---

## 配置示例

### Stage 1: 色彩空间对齐

```yaml
training:
  lambda_image: 0.5  # 0.1, 0.5, 1.0, 2.0
  image_samples: 1024  # 512 for λ=0.1, 1024 for others
  image_loss_space: "srgb"  # 关键：对齐评估空间
  image_loss_type: "charbonnier"
  enable_weighted_sh_loss: true
  sh_loss_weights: [1.0, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0]
  sh_weight_mode: "basis"
```

### Stage 2: 学习率优化

```yaml
training:
  lr: 1e-3  # 5e-4, 1e-3, 2e-3
  lr_scheduler: "cosine"
  lr_min: 1e-5
  warmup_epochs: 100  # 100, 100, 200
  epochs: 3000  # 延长训练
```

### Stage 3: 数据增强

```yaml
training:
  augment_intensity: true
  augment_position: true  # false for E3a
  augment_dropout: 0.15  # 0.0 for E3a/E3b
```

---

## 执行命令

### Stage 1 (立即可执行)
```bash
cd /home/kyrie/毕设
source venv/bin/activate
python 3_experiments/phase1_optimization_experiments.py --stage 1
```

### Stage 2 (需要 Stage 1 最佳配置)
```bash
python 3_experiments/phase1_optimization_experiments.py \
  --stage 2 \
  --base-config 3_experiments/configs/temp_E1b_srgb_lambda05.yaml
```

### Stage 3 (需要 Stage 2 最佳配置)
```bash
python 3_experiments/phase1_optimization_experiments.py \
  --stage 3 \
  --base-config 3_experiments/configs/temp_E2b_lr1e3_warmup100.yaml
```

---

## 预期结果

| Stage | 目标 PSNR | 提升 | 累计 PSNR |
|-------|-----------|------|-----------|
| Baseline | 23.66 dB | - | 23.66 dB |
| Stage 1 | +3-5 dB | 色彩空间对齐 | 27-29 dB |
| Stage 2 | +2-3 dB | 学习率优化 | 29-32 dB |
| Stage 3 | +1-2 dB | 数据增强 | 30-33 dB |
| Stage 4 | +2-3 dB | 架构微调 | 32-35 dB |

**最终目标**: 35+ dB sRGB PSNR

---

## 监控指标

### 关键指标
- `/val/img_psnr`: 图像 PSNR (sRGB 空间) - **最重要**
- `benchmark_metrics.mean_psnr`: 最终评估指标

### 辅助指标
- `/train/image_loss`: 图像渲染损失
- `/val/mae`: SH 系数 MAE
- `/val/superposition`: 线性叠加误差

---

## 文件清单

### 新增文件
- `3_experiments/phase1_optimization_experiments.py` (主实验脚本)
- `2_src/data/augmentations.py` (数据增强模块)
- `3_experiments/PHASE1_QUICKSTART.md` (快速启动指南)
- `3_experiments/PHASE1_IMPLEMENTATION_SUMMARY.md` (本文档)

### 修改文件
- `2_src/training/gaussian_physics_trainer.py` (warmup 机制)
- `3_experiments/scripts/train.py` (warmup 参数传递)

---

## 下一步行动

1. **立即执行**: 启动 Stage 1
   ```bash
   python 3_experiments/phase1_optimization_experiments.py --stage 1
   ```

2. **监控训练**: 检查 `/val/img_psnr` 是否持续上升

3. **完成集成** (可选，Stage 3 前):
   - 在 `train.py` 中集成数据增强到 `BatchAdapter`

4. **Stage 1 完成后**:
   - 分析结果，选择最佳 λ_image
   - 启动 Stage 2

5. **迭代优化**: 根据每个 Stage 结果调整后续参数
