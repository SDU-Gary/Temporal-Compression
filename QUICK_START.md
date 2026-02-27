# Quick Start Guide - Light Probe Compression Experiments

## Current Status (2026-02-14)

**Experiment A (K50_r16_plain)：训练已完成**

- **训练**：已跑满 2000 epoch，`best_model.pt` / `last_model.pt` 已生成。
- **SH 系数评估**：已运行 `eval.py`，结果见下。
- **渲染图像 PSNR**：尚未跑 `benchmark_realtime_pipeline.py`，需在本机有 Falcor 时执行 `auto_eval_k50_r16_plain.sh` 获取。

### 实验 A 已得结果（SH 系数，test split）

| 指标 | K50_r16_plain (A) | K30_r8 基线 | K50_r16（失败版） |
|------|-------------------|-------------|-------------------|
| MAE  | **0.1530**         | 0.1455      | 0.1487            |
| RMSE | **0.2583**        | 0.2575      | 0.2751            |

结论（仅就 SH 系数）：K50_r16_plain 的 MAE 略高于 K30_r8，RMSE 与 K30_r8 接近、优于失败版 K50_r16。**是否达标需看渲染图像 PSNR**（见下「若训练已完成」）。

---

## When You Return

### Check Training Status（其他实验时用）
```bash
tmux attach -t train_k50_r16_plain
# Press Ctrl+B then D to detach
```

### If Training Completed（补跑实验 A 的渲染 PSNR 与建议）
```bash
cd /home/kyrie/毕设
bash 3_experiments/auto_eval_k50_r16_plain.sh
```

该脚本会：
1. 使用已有 `eval.json`（SH 已评估）
2. 运行 **benchmark_realtime_pipeline.py** 得到渲染图像 PSNR/SSIM
3. 与基线对比并给出下一步实验建议

### Expected Outcomes（以渲染 PSNR 为准）

**If PSNR >= 25 dB** → SUCCESS
- Capacity increase helps
- Launch Experiment B (weighted_fixed)

**If PSNR 20-25 dB** → MARGINAL
- Slight improvement
- Consider alternatives

**If PSNR < 20 dB** → FAILURE
- Need architecture changes

## Background

### Problem
- Training showed 40 dB PSNR (on SH coefficients)
- Evaluation showed 10 dB PSNR (on rendered images)
- Target: 35+ dB

### Root Cause
K50_r16 with "optimizations" FAILED (15.6 dB):
- Weighted loss [3.0, 1.0, 1.0, 1.0, 0.5×5] was backwards
- Image loss (64 dirs, linear) misaligned with eval
- More complexity made things worse

### Solution
Test capacity increase alone (K50_r16_plain):
- 50 Gaussians, rank 16
- No scaler, no weighted loss, no image loss
- Clean baseline to isolate capacity effect

## 后续试验方向（根据 QUICK_START 与当前结果）

### 1. 先补全实验 A 的渲染 PSNR（推荐）

在**有 Falcor 环境**的机器上执行：

```bash
cd /home/kyrie/毕设
bash 3_experiments/auto_eval_k50_r16_plain.sh
```

根据输出的 **sRGB PSNR** 走下面的决策树。

### 2. 按渲染 PSNR 的决策树

| 渲染 PSNR | 判定 | 建议下一步 |
|-----------|------|------------|
| **≥ 25 dB** | SUCCESS | 启动 **实验 B**：K50_r16_weighted_fixed（修正 SH 权重 + 无 image loss） |
| **20–25 dB** | MARGINAL | 可谨慎尝试实验 B，或优先考虑其他结构（attention、层次化、不同基函数） |
| **< 20 dB** | FAILURE | 仅加容量不足，需**架构级改动**（见下） |

### 3. 实验 B / C 启动命令（当 PSNR 达标或边际时）

**实验 B：K50_r16_weighted_fixed**（修正权重 [1.0, 1.5, 1.5, 1.5, 2.0×5]，无 image loss）

```bash
mkdir -p 3_experiments/results/bistro_clean_v2/unified_set_K50_r16_weighted_fixed
tmux new-session -d -s train_weighted_fixed "source venv/bin/activate && python 3_experiments/scripts/train.py --config 3_experiments/configs/bistro_clean_train_k50_r16_weighted_fixed.yaml 2>&1 | tee 3_experiments/results/bistro_clean_v2/unified_set_K50_r16_weighted_fixed/train_\$(date +%Y%m%d_%H%M%S).log"
```

**实验 C：K50_r16_full_fixed**（在 B 基础上 + sRGB image loss，256 samples，λ=0.01，目标 35+ dB）

```bash
mkdir -p 3_experiments/results/bistro_clean_v2/unified_set_K50_r16_full_fixed
tmux new-session -d -s train_full_fixed "source venv/bin/activate && python 3_experiments/scripts/train.py --config 3_experiments/configs/bistro_clean_train_k50_r16_full_fixed.yaml 2>&1 | tee 3_experiments/results/bistro_clean_v2/unified_set_K50_r16_full_fixed/train_\$(date +%Y%m%d_%H%M%S).log"
```

### 4. 若 PSNR < 20 dB（架构级方向）

- 尝试 **attention** 或 **层次化** 结构（如 4×3 cascaded volumes）。
- 调整 **基函数** 或 **SH 阶数**，使训练目标与评估更一致。
- 检查 **线性/尺度**：训练与评估的 irradiance vs radiance、线性 vs sRGB 是否一致。

---

## Key Files

- **Experiment configs**: `3_experiments/configs/bistro_clean_train_k50_r16_*.yaml`
- **Auto-eval script**: `3_experiments/auto_eval_k50_r16_plain.sh`
- **Detailed status**: `3_experiments/EXPERIMENT_STATUS.md`
- **This guide**: `QUICK_START.md`

## Monitoring

**Check progress**:
```bash
tmux capture-pane -t train_k50_r16_plain -p | tail -20
```

**GPU usage**:
```bash
watch -n 2 nvidia-smi
```

**Training log**:
```bash
tail -f 3_experiments/results/bistro_clean_v2/unified_set_K50_r16_plain/train_*.log
```

## Contact Info

Training runs in tmux - persists across SSH disconnects.
Check back in ~8-10 hours to evaluate results.
