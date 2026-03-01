# Phase 1 优化实验快速启动指南

## 目标

将 K30_r8 基线从 23.66 dB 提升至 35+ dB sRGB PSNR

## 实验流程

### Stage 1: 色彩空间对齐 (预期 +3-5 dB)

**核心假设**: 训练在线性空间优化，评估在 sRGB 空间，导致 tone mapping 误差未被优化

**执行**:
```bash
cd /home/kyrie/毕设
source venv/bin/activate
python 3_experiments/phase1_optimization_experiments.py --stage 1
```

**4 个实验** (可并行):
- E1a: λ_image=0.1, 512 samples
- E1b: λ_image=0.5, 1024 samples
- E1c: λ_image=1.0, 1024 samples
- E1d: λ_image=2.0, 1024 samples

**时间**: 4 × 4.5h = 18h (串行) 或 4.5h (并行)

**成功标准**: sRGB PSNR ≥ 27 dB

**并行执行** (推荐):
```bash
# 终端 1
tmux new -s train_E1a
cd /home/kyrie/毕设 && source venv/bin/activate
python 3_experiments/scripts/train.py --config 3_experiments/configs/temp_E1a_srgb_lambda01.yaml

# 终端 2
tmux new -s train_E1b
cd /home/kyrie/毕设 && source venv/bin/activate
python 3_experiments/scripts/train.py --config 3_experiments/configs/temp_E1b_srgb_lambda05.yaml

# 终端 3
tmux new -s train_E1c
cd /home/kyrie/毕设 && source venv/bin/activate
python 3_experiments/scripts/train.py --config 3_experiments/configs/temp_E1c_srgb_lambda10.yaml

# 终端 4
tmux new -s train_E1d
cd /home/kyrie/毕设 && source venv/bin/activate
python 3_experiments/scripts/train.py --config 3_experiments/configs/temp_E1d_srgb_lambda20.yaml
```

**监控训练**:
```bash
# 检查 Rerun 日志中的 /val/img_psnr 指标
# 目标: 验证集 img_psnr 持续上升
# 警告: 如果 img_psnr 不变但 SH MAE 下降，说明 λ_image 过小
```

**分析结果**:
```bash
# 查看报告
cat 3_experiments/results/phase1_stage1_final_*.json

# 选择最佳 λ_image (预期 0.5-1.0)
# 记录最佳配置路径，用于 Stage 2
```

---

### Stage 2: 学习率与调度优化 (预期 +2-3 dB)

**前提**: Stage 1 完成，选择最佳配置

**核心假设**: 当前 lr=1e-3 + 无调度导致收敛不充分，引入 warmup + cosine 可提升精细收敛

**执行**:
```bash
# 假设 Stage 1 最佳配置为 E1b_srgb_lambda05
python 3_experiments/phase1_optimization_experiments.py \
  --stage 2 \
  --base-config 3_experiments/configs/temp_E1b_srgb_lambda05.yaml
```

**3 个实验**:
- E2a: lr=5e-4, warmup=100, epochs=3000
- E2b: lr=1e-3, warmup=100, epochs=3000
- E2c: lr=2e-3, warmup=200, epochs=3000

**时间**: 3 × 6h = 18h (串行) 或 6h (并行)

**成功标准**: sRGB PSNR ≥ 30 dB

**实现状态**: ✅ Warmup 机制已实现
- `2_src/training/gaussian_physics_trainer.py`: 新增 `warmup_epochs` 参数
- `3_experiments/scripts/train.py`: 支持 `warmup_epochs` 配置

---

### Stage 3: 数据增强 (预期 +1-2 dB)

**前提**: Stage 2 完成，选择最佳配置

**核心假设**: 当前训练数据有限 (6 moments × 900 probes = 5400 samples)，增强可提升泛化

**执行**:
```bash
# 假设 Stage 2 最佳配置为 E2b_lr1e3_warmup100
python 3_experiments/phase1_optimization_experiments.py \
  --stage 3 \
  --base-config 3_experiments/configs/temp_E2b_lr1e3_warmup100.yaml
```

**3 个实验**:
- E3a: 仅强度抖动 (RGB × [0.9, 1.1])
- E3b: 强度抖动 + 位置噪声 (σ=0.02)
- E3c: 全部增强 (强度 + 位置 + dropout 15%)

**时间**: 3 × 4.5h = 13.5h (串行) 或 4.5h (并行)

**成功标准**: sRGB PSNR ≥ 32 dB

**实现状态**: ✅ 数据增强模块已实现
- `2_src/data/augmentations.py`: 增强函数库
- 集成到 `BatchAdapter.params_transform` (待完成)

---

### Stage 4: 架构微调 (预期 +2-3 dB)

**前提**: Stage 1-3 累计达到 30+ dB

**候选改进**:
1. Attention on Z (embed_dim=32 → 64)
2. 残差连接 (skip connection)
3. 更深 FiLM 网络 (2 层 MLP)

**时间**: 3 × 5h = 15h (串行) 或 5h (并行)

**成功标准**: sRGB PSNR ≥ 35 dB

**实现状态**: ❌ 待实现

---

## 风险与备选方案

### 风险 1: Stage 1 提升不足 (<27 dB)

**备选**:
- 增加 `lambda_image` 至 2.0-5.0
- 增加 `image_samples` 至 2048
- 切换 `image_loss_type` 为 "l1"

### 风险 2: Stage 2 未达 30 dB

**备选**:
- 延长 `epochs` 至 5000
- 调整 `warmup_epochs` 至 300-500
- 尝试 `lr_scheduler: "plateau"`

### 风险 3: Stage 3 未达 32 dB

**备选**:
- 跳过 Stage 4，重新审视架构
- 增加更强的增强 (scale_range: [0.8, 1.2])
- 尝试 mixup 增强

---

## 监控指标

### 训练过程
- `/train/total_loss`: 总损失
- `/train/recon_loss`: SH 重建损失
- `/train/image_loss`: 图像渲染损失 (关键)

### 验证过程
- `/val/mae`: SH 系数 MAE (线性空间)
- `/val/img_psnr`: 图像 PSNR (sRGB 空间，关键)
- `/val/superposition`: 线性叠加误差

### Benchmark
- `mean_psnr`: 平均 sRGB PSNR (最终指标)
- `mean_ssim`: 平均 SSIM

---

## 文件清单

### 新增文件
- `3_experiments/phase1_optimization_experiments.py`: 主实验脚本
- `2_src/data/augmentations.py`: 数据增强模块
- `3_experiments/PHASE1_QUICKSTART.md`: 本文档

### 修改文件
- `2_src/training/gaussian_physics_trainer.py`:
  - 新增 `warmup_epochs` 参数
  - 实现 warmup 逻辑
- `3_experiments/scripts/train.py`:
  - 支持 `warmup_epochs` 配置

---

## 下一步行动

1. **立即执行**: 启动 Stage 1
   ```bash
   python 3_experiments/phase1_optimization_experiments.py --stage 1
   ```

2. **监控训练**: 检查 `/val/img_psnr` 是否持续上升

3. **Stage 1 完成后**:
   - 分析 4 个实验的 sRGB PSNR
   - 选择最佳 λ_image
   - 启动 Stage 2

4. **迭代优化**: 根据每个 Stage 结果调整后续参数
