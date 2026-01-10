# 实验记录索引 / Experiment Index

本目录包含完整的实验记录和分析文档，用于论文写作和后续实验参考。

---

## 阶段1：数据集扩充与过拟合解决

### 关键文档
- **`SOLUTIONS_OVERFITTING.md`** - 过拟合问题的5种解决方案
- **`OVERFITTING_ANALYSIS.md`** - 过拟合理论分析
- **`DATA_GENERATION_PARAMS.md`** - 数据生成参数详解

### 关键发现
- ✅ 数据量不足导致过拟合 (343 probes → params/sample ratio 10:1)
- ✅ 扩充数据集至1,728 probes解决问题 (ratio降至2:1)
- ✅ Train-Val gap从20 dB降至2.8 dB

### 结论
数据量是防止过拟合的关键因素，模型容量必须匹配数据规模。

---

## 阶段2：性能瓶颈分析（当前）

### 主要文档

#### 📄 完整实验记录（论文用）
**`EXPERIMENT_LOG_Phase2_Performance_Analysis.md`**
- 42页完整英文记录
- 包含所有实验数据、图表、统计分析
- 直接可用于论文Methods、Results、Discussion章节
- 包含论文图表建议和LaTeX表格代码

#### 📄 中文摘要（快速查阅）
**`实验记录_阶段2_性能分析_中文摘要.md`**
- 简洁的中文总结
- 快速查找关键数据和结论
- 包含实验流程和要点

#### 📄 诊断报告（首次尝试）
**`DIAGNOSTIC_REPORT_20251208.md`**
- 首次temporal_smooth假设的详细分析
- 虽然假设被后续实验推翻，但诊断方法有价值
- 保留作为研究过程记录

### 关键实验

| 实验 | 配置文件 | 训练PSNR | 验证PSNR | Gap | 状态 |
|-----|---------|---------|---------|-----|------|
| baseline_2k | `configs/baseline_2k.yaml` | 31.1 | 28.3 | 2.8 | ✅ 基线 |
| V2 (降低正则化) | `configs/baseline_2k_v2.yaml` | 31.37 | 28.53 | 2.84 | ✅ 完成 |
| V3 (激进优化) | `configs/baseline_2k_v3.yaml` | 32.56 | 28.7 | 3.86 | ⚠️ 轻微过拟合 |
| K-NN Baseline | - | - | 11.09 | - | ✅ 诊断工具 |

### 关键发现

#### 🔴 主要发现：性能受数据空间不连续性限制

**证据链**:
1. **K-NN基线**: 11.09 dB (空间插值极差)
2. **空间相关性**: ρ = 0.275 (极弱)
3. **相对SH变化**: 65.5% (极大)
4. **模型性能**: 28.53 dB (远超基线17 dB)

**结论**:
- 模型已学到有效latent表示
- 当前性能(28-31 dB)接近数据质量上限(32-35 dB估计)
- 进一步提升需改进数据生成质量而非模型架构

#### 🔴 被否定的假设

| 假设 | 实验验证 | 结果 |
|-----|---------|------|
| temporal_smooth过强(21.5%) | V2: 降至0.02 | +0.23 dB (❌) |
| 模型容量不足 | V3: 增加容量 | 过拟合 (❌) |
| 学习率不优 | V3: 提升25% | 无改善 (❌) |
| Early stopping过早 | V3: 训练更久 | Plateau (❌) |

### 诊断工具和脚本

#### K-NN Baseline评估
```bash
python scripts/knn_baseline.py --k 5 --p 2
```
- **目的**: 评估数据质量上限
- **输出**: PSNR统计、最近邻距离
- **结果**: 11.09 dB → 说明空间结构极弱

#### 空间平滑性分析
```bash
python scripts/analyze_data_smoothness.py
```
- **目的**: 量化空间连续性
- **输出**: 相关系数、距离-差异分析、可视化
- **结果**: ρ=0.275 → 解释K-NN为何表现差

#### 训练诊断
```bash
python scripts/quick_diagnose.py
```
- **目的**: Loss组成分析
- **输出**: 各loss分量占比、训练趋势

### 生成的数据和图表

**位置**: `experiments/`
- `data_smoothness_analysis.png` - 空间距离vs SH差异散点图
- `knn_baseline_results.npz` - K-NN原始评估数据
- `baseline_2k/logs/` - 完整训练日志
- `baseline_2k/checkpoints/` - 模型checkpoint

### 论文写作资源

#### 可用数据
- ✅ 3个配置的完整训练曲线
- ✅ K-NN vs 模型性能对比
- ✅ 空间平滑性统计分析
- ✅ Loss组成分解

#### 推荐图表（见完整记录第8节）
1. **训练曲线对比** - 展示不同配置下的收敛
2. **K-NN vs 模型性能** - 柱状图展示17 dB差距
3. **空间距离 vs SH差异** - 散点图说明弱相关性
4. **距离分bin分析** - 线图展示高变化

#### 可用表格（LaTeX代码已提供）
1. **实验结果总结** - 所有配置的PSNR对比
2. **空间平滑性指标** - 相关性、差异统计
3. **K-NN性能分布** - 百分位数分析

---

## 脚本工具索引

### 训练相关
- `scripts/train.py` - 主训练脚本
- `scripts/diagnose_training.py` - 完整诊断（需checkpoint）
- `scripts/quick_diagnose.py` - 快速诊断（仅需日志）

### 评估相关
- `scripts/knn_baseline.py` - K-NN基线评估 ⭐
- `scripts/analyze_data_smoothness.py` - 空间平滑性分析 ⭐

### 数据生成
- `../data_generation/scripts/generate_dataset.py` - 生成训练数据

---

## 配置文件索引

### 当前使用的配置

#### 推荐配置（论文用）
**`configs/baseline_2k.yaml`** 或 **`configs/baseline_2k_v3.yaml`**
- baseline_2k: 更稳定，gap 2.8 dB
- baseline_2k_v3: 训练集更高，但轻微过拟合

#### 其他配置（参考）
- `configs/baseline.yaml` - 小数据集配置（343 probes）
- `configs/baseline_regularized.yaml` - 强正则化版本
- `configs/tiny_model.yaml` - 极小模型（~2k参数）

### 配置模板
```yaml
# 关键参数
temporal_smooth: 0.05        # 时间平滑权重
num_gaussians: 250           # Gaussian数量
optimizer:
  gaussians: {lr: 0.008}     # Gaussian学习率
  mlp: {lr: 0.0008}          # MLP学习率
early_stopping:
  patience: 20               # 容忍步数
```

---

## Checkpoint索引

### 最佳模型

**验证集最高PSNR**: 28.74 dB (baseline_2k, step 3000)
**位置**: `experiments/baseline_2k/checkpoints/best_model.pt`

**训练集最高PSNR**: 32.56 dB (baseline_2k_v3)
**位置**: `experiments/baseline_2k_v3/checkpoints/best_model.pt`

### Checkpoint结构
```python
checkpoint = {
    'step': 训练步数,
    'model_state_dict': {
        'gaussian_mixture.*': Gaussian参数,
        'temporal_mlp.*': TemporalMLP参数,
        'decoder_mlp.*': DecoderMLP参数
    },
    'optimizer_*': 优化器状态,
    'best_val_psnr': 最佳验证PSNR
}
```

---

## 数据集索引

### dataset_2k (主要使用)
**位置**: `../data_generation/output/dataset_2k/`
**规格**:
- Probes: 1,728 (12³)
- Time moments: 6
- Total samples: 10,368
- SPP: 128
- SH samples: 16

**统计**:
- 空间相关性: ρ = 0.275 (弱)
- 相对SH变化: 65.5% (高)

### method_test_v1 (小数据集，参考)
**位置**: `../data_generation/output/method_test_v1/`
**规格**:
- Probes: 343 (7³)
- Time moments: 6
- Total samples: 2,058

---

## 下一步实验建议

### 如果接受当前性能 (28-31 dB)
✅ **无需进一步实验**
- 准备论文写作
- 使用现有数据和分析
- 强调方法在空间不连续数据上的有效性

### 如果需要更高性能 (34-36 dB)
🔧 **改进数据生成**

#### 实验4：高质量数据
```bash
# 生成新数据集
cd ../data_generation
python scripts/generate_dataset.py \
  --scene scenes/house.xml \
  --output output/dataset_2k_highqual \
  --num-probes 1728 \
  --sun-hours 6 9 12 15 18 21 \
  --spp 1024 \              # 提升至1024
  --num-sh-samples 64       # 提升至64

# 训练
cd ../multi_time_compression
python scripts/train.py --config configs/baseline_2k_highqual.yaml
```

**预期结果**: 验证集PSNR 32-34 dB

#### 实验5：增加MLP容量
```yaml
# configs/baseline_2k_highqual_large.yaml
temporal_mlp:
  hidden_dim: 128    # 从32增加
decoder_mlp:
  hidden_dim: 256    # 从64增加
```

**预期额外提升**: +1-2 dB

**综合预期**: 34-36 dB

---

## 文档历史

| 日期 | 阶段 | 主要发现 | 文档 |
|------|------|---------|------|
| 2025-12-08早 | Phase 1 | 数据扩充解决过拟合 | OVERFITTING_ANALYSIS.md |
| 2025-12-08中 | Phase 2初 | temporal_smooth假设 | DIAGNOSTIC_REPORT_20251208.md |
| 2025-12-08晚 | Phase 2终 | 数据空间不连续性瓶颈 | EXPERIMENT_LOG_Phase2_*.md |

---

## 快速参考

### 关键数字（论文用）

| 指标 | 值 | 说明 |
|-----|---|------|
| K-NN PSNR | 11.09 dB | 空间插值基线 |
| 模型PSNR | 28.53 dB | 当前最佳验证集 |
| 性能提升 | +17.44 dB | 模型 vs K-NN |
| 空间相关性 | ρ=0.275 | 极弱 |
| 相对SH变化 | 65.5% | 极大 |
| 估计上限 | 32-35 dB | 数据质量天花板 |
| 参数量 | 12,612 | 完整模型 |
| 压缩比 | 6.9× | 1728→250 Gaussians |

### 关键论点（论文用）

1. **方法有效性**: 模型比简单方法高17 dB，证明latent学习有效
2. **性能瓶颈**: 数据空间不连续性(ρ=0.275)限制性能
3. **接近上限**: 当前28.5 dB接近估计上限32-35 dB (80-95%)
4. **诊断工具**: K-NN基线可区分数据问题vs模型问题

### 论文章节对应

- **第3章 方法**: 使用模型架构部分
- **第4章 实验**: 使用实验1-3的数据和表格
- **第5章 讨论**: 使用K-NN分析和空间平滑性分析
- **第6章 结论**: 引用主要发现和建议

---

**索引最后更新**: 2025-12-08
**完整性**: ✅ 所有实验已记录
**可用性**: ✅ 所有脚本和数据可复现
