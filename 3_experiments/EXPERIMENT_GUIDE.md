# 单一变量控制实验 - 使用指南

## 实验设计

### 实验目标
通过单一变量控制，系统性地验证：
1. 容量提升是否有效
2. 正则化是否有效
3. 两者组合的协同效应

### 实验配置

| 实验 | K | r | Scaler | Weighted Loss | Image Loss | 说明 |
|------|---|---|--------|---------------|------------|------|
| **A. K30_r8 baseline** | 30 | 8 | ✗ | ✗ | ✗ | 纯净基线 |
| **B. K50_r16 capacity** | 50 | 16 | ✗ | ✗ | ✗ | 单一变量：容量↑ |
| **C. K30_r8 regularized** | 30 | 8 | ✗ | ✓ | ✓ | 单一变量：正则化↑ |
| **D. K50_r16 full** | 50 | 16 | ✗ | ✓ | ✓ | 组合：容量↑+正则化↑ |

### 正则化配置（实验 C 和 D）

**修正的 Weighted SH Loss**:
```yaml
sh_loss_weights: [1.0, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0]
# L0 (DC): 1.0 - 基准权重
# L1: 1.5 - 提升方向性权重
# L2: 2.0 - 进一步提升高频方向性
```

**修正的 Image Loss**:
```yaml
lambda_image: 0.01          # 降低权重（从 0.1 → 0.01）
image_loss_space: srgb      # 匹配评估空间（从 linear → srgb）
image_samples: 256          # 增加采样（从 64 → 256）
```

---

## 使用方法

### 方法 1：运行所有实验（推荐）

```bash
cd /home/kyrie/毕设
bash 3_experiments/run_experiments.sh
```

预计时间：**32-48 小时**（4 个实验 × 8-12 小时/实验）

### 方法 2：运行特定实验

```bash
# 仅运行基线和容量实验
bash 3_experiments/run_experiments.sh --experiments A_K30_r8_baseline B_K50_r16_capacity

# 仅运行正则化实验
bash 3_experiments/run_experiments.sh --experiments C_K30_r8_regularized D_K50_r16_full
```

### 方法 3：使用 tmux 后台运行

```bash
# 创建 tmux 会话
tmux new-session -d -s controlled_exp "cd /home/kyrie/毕设 && bash 3_experiments/run_experiments.sh"

# 查看进度
tmux attach -t controlled_exp

# 分离会话：Ctrl+B 然后 D
```

---

## 输出结果

### 1. 训练输出

每个实验的输出目录：
```
3_experiments/results/bistro_clean_v2/
├── exp_A_K30_r8_baseline/
│   ├── best_model.pt          # 最佳模型
│   ├── last_model.pt          # 最后模型
│   ├── training.log           # 训练日志
│   ├── eval.json              # 评估指标
│   └── benchmark/
│       └── benchmark_summary.json  # 渲染指标
├── exp_B_K50_r16_capacity/
├── exp_C_K30_r8_regularized/
└── exp_D_K50_r16_full/
```

### 2. 对比报告

```
3_experiments/results/
└── experiment_results_final_YYYYMMDD_HHMMSS.json
```

包含所有实验的完整指标和对比分析。

---

## 指标说明

### SH 系数指标（eval.json）

- **MAE**: Mean Absolute Error（越小越好）
- **RMSE**: Root Mean Squared Error（越小越好）

### 渲染图像指标（benchmark_summary.json）

- **mean_psnr**: sRGB 空间 PSNR（越大越好，目标 >35 dB）
- **mean_ssim**: 结构相似性（越大越好，范围 0-1）
- **mean_linear_psnr**: 线性空间 PSNR（参考）

---

## 预期结果

### 假设 1：容量提升有效
- **B vs A**: PSNR 提升 2-5 dB
- 说明：更大的模型有更强的表达能力

### 假设 2：正则化有效
- **C vs A**: PSNR 提升 3-8 dB
- 说明：修正的损失函数改善了训练

### 假设 3：正协同效应
- **D vs A**: PSNR 提升 > (B-A) + (C-A)
- 说明：容量和正则化相互增强

### 假设 4：达到目标
- **D**: PSNR ≥ 35 dB
- 说明：组合方案达到论文目标

---

## 监控进度

### 查看当前实验状态

```bash
# 查看最新的中间结果
ls -lt 3_experiments/results/experiment_results_interim_*.json | head -1

# 查看特定实验的训练日志
tail -f 3_experiments/results/bistro_clean_v2/exp_A_K30_r8_baseline/training.log
```

### 查看 GPU 使用

```bash
watch -n 2 nvidia-smi
```

---

## 故障排除

### 问题 1：训练失败

**检查**：
```bash
cat 3_experiments/results/bistro_clean_v2/exp_*/training.log | grep -i error
```

**常见原因**：
- CUDA 内存不足：降低 batch_size
- 数据集路径错误：检查 data_root

### 问题 2：Benchmark 失败

**检查**：
```bash
cat 3_experiments/results/bistro_clean_v2/exp_*/benchmark/benchmark.log | grep -i error
```

**常见原因**：
- Falcor 路径错误：检查 FALCOR_CONFIG
- Vulkan 驱动问题：检查 VK_ICD_FILENAMES

### 问题 3：中断后恢复

脚本会保存中间结果，但不支持自动恢复。如需恢复：

1. 检查哪些实验已完成：
```bash
ls 3_experiments/results/bistro_clean_v2/exp_*/best_model.pt
```

2. 运行未完成的实验：
```bash
bash 3_experiments/run_experiments.sh --experiments [未完成的实验ID]
```

---

## 高级选项

### 仅运行评估和 Benchmark（跳过训练）

如果模型已训练完成，仅需重新评估：

```bash
python 3_experiments/run_controlled_experiments.py --skip-training
```

### 仅运行训练和评估（跳过 Benchmark）

如果不需要渲染图像指标：

```bash
python 3_experiments/run_controlled_experiments.py --skip-benchmark
```

---

## 下一步

实验完成后：

1. **查看对比报告**：
   ```bash
   cat 3_experiments/results/experiment_results_final_*.json | python -m json.tool
   ```

2. **分析结果**：
   - 哪个单一变量效果最显著？
   - 是否存在协同效应？
   - 是否达到 35 dB 目标？

3. **根据结果决定**：
   - 如果 D 达到目标 → 论文实验完成 ✓
   - 如果接近但未达到 → 微调超参数
   - 如果远未达到 → 需要架构改进

---

## 联系信息

脚本位置：
- 主脚本：`3_experiments/run_controlled_experiments.py`
- 启动脚本：`3_experiments/run_experiments.sh`
- 配置文件：`3_experiments/configs/bistro_clean_train*.yaml`
