# CLAUDE.md改进建议

基于对实验流程的完整梳理,以下是对现有CLAUDE.md的改进建议。

---

## 核心问题

当前CLAUDE.md (2026-01-04检查) 存在以下问题:

1. **混淆目标与现状**: 将任务书的"目标架构"与"当前实现"混在一起描述
2. **实验方法未区分**: 没有明确标注哪些是失败方法、哪些是当前最优
3. **缺少实验演进**: 没有说明TPE→MLP→TemporalMLP→PG-GCPL的演进逻辑
4. **代码路径错误**: 指向了不存在的`models/`目录,实际在`PG-GCPL/src/models/`和`TemporalMLP/src/models/`
5. **状态标注不清**: 没有明确说明PG-GCPL是当前最优,TemporalMLP已废弃

---

## 建议的新结构

### 第1部分: 项目概述 (保留+修改)

**修改要点**:
- 明确说明"任务书描述的是目标,而非当前实现"
- 添加"当前最优方法: PG-GCPL"的显著标注
- 压缩目标指标到表格 (当前太冗长)

**建议文本**:

```markdown
## 项目概述

**毕业论文题目**: 面向多时刻光照的层次化神经压缩方法

**IMPORTANT**: 本项目包含多个实验方法和演进路径:
- `docs/thesis/多时刻光照压缩任务书.md` 描述的是**目标架构** (4层时空级联,24时刻,<0.5ms解压)
- **当前实现**: PG-GCPL方法 (16.59× compression, SSIM 0.951 ✅) - 已验证成功
- **已废弃方法**: TPE, MLP Hybrid, TemporalMLP - 保留作baseline或失败案例分析

### 当前最优方法: PG-GCPL

**状态**: ✅ 验证成功 (2026-01-03)
**路径**: `multi_time_compression/PG-GCPL/`
**核心创新**: 空间Gaussian混合 + 时间物理引导低秩分解

**关键指标** (K30_r8配置):
| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| 压缩比 | 14-18× | 16.59× | ✅ |
| SH重建MAE | < 0.05 | 0.0318 | ✅ |
| 场景渲染SSIM | > 0.95 | 0.951 | ✅ |
| 查询延迟P95 | < 0.5ms | 0.145ms | ✅ |
| 内插退化 | < 1.5× | 1.28× | ✅ |
| 外推退化 | < 3.0× | 1.10× | ✅ |

**方法公式**:
```
SH(probe_pos, light_params) = Σ_j G_j(probe_pos) × [U_j @ (coeffs_j @ Φ_physics(light_params))]
```

**详细文档**: `PG-GCPL/README.md`, `PG-GCPL/docs/reports/05_Final_Validation.md`
```

---

### 第2部分: 仓库结构 (重写)

**问题**: 当前描述的`models/`不存在,需要明确不同方法的目录

**建议文本**:

```markdown
## 仓库结构

```
毕设/
├── data_generation/           # Mitsuba 3-based dataset generation
│   ├── generate_dataset.py    # Main generation script
│   ├── core/                  # Core rendering logic
│   ├── utils/                 # SH fitting, sun position calculation
│   └── output/                # Generated datasets
├── multi_time_compression/    # Main experimental directory
│   ├── PG-GCPL/               # ✅ CURRENT BEST METHOD
│   │   ├── src/
│   │   │   ├── models/
│   │   │   │   ├── gaussian_physics_5D.py       # ✅ Week 6 optimal (K30_r8)
│   │   │   │   ├── physics_low_rank.py          # Week 1-2 validation
│   │   │   │   └── gaussian_physics_compression.py
│   │   │   ├── data/
│   │   │   │   └── transfer_tensor_dataset.py   # 5D parametric data loader
│   │   │   ├── training/      # Training loop, losses, metrics
│   │   │   └── scripts/
│   │   │       ├── training/
│   │   │       ├── evaluation/
│   │   │       └── visualization/
│   │   ├── experiments/       # Experimental results (4 phases)
│   │   │   ├── 01_physics_low_rank_validation/
│   │   │   ├── 02_query_validation/
│   │   │   ├── 03_gaussian_physics_5D_training/
│   │   │   └── 04_ablation_and_visualization/
│   │   └── docs/reports/      # 5 detailed reports
│   ├── TemporalMLP/           # ⚠️ DEPRECATED (Task spec baseline)
│   │   ├── src/models/
│   │   │   ├── temporal_mlp.py
│   │   │   ├── decoder_mlp.py
│   │   │   └── full_model.py
│   │   ├── configs/           # baseline*.yaml
│   │   └── experiments/       # Training outputs
│   ├── shared/                # Shared utilities (data, training, losses)
│   ├── archive/               # ❌ FAILED METHODS
│   │   ├── TPE_Method_Archived/         # Phase 0: TPE failed (84.9% error)
│   │   └── Week3-4_MLP_Hybrid_Failed/   # Weeks 3-4: MLP hybrid failed
│   └── docs/
│       ├── EXPERIMENT_INDEX.md          # Experiment timeline and index
│       ├── experiment_reports/          # TPE phase reports (archived)
│       └── PG-GCPL_vs_TaskSpec_Gap_Analysis.md
├── docs/
│   ├── thesis/多时刻光照压缩任务书.md  # **AUTHORITATIVE TARGET SPEC**
│   └── research_notes/
├── mitsuba-docs/              # Local Mitsuba documentation
└── EXPERIMENTAL_WORKFLOW.md   # 实验演进时间线 (2026-01-04)
```

**关键目录说明**:

- **PG-GCPL/**: 当前最优方法,已验证成功 ✅
  - 使用此方法进行后续开发和实验
  - 最优配置: K30_r8 (16.59× compression, SSIM 0.951)

- **TemporalMLP/**: 任务书基线方法,已废弃 ⚠️
  - 保留作性能对比baseline
  - 不推荐用于新实验

- **archive/**: 失败方法存档 ❌
  - TPE方法: 假设不成立,84.9%误差贡献
  - MLP Hybrid: 训练不稳定
  - 保留作负面案例分析

- **shared/**: 跨方法共享工具库
  - 数据加载器 (`data/dataset.py`)
  - 训练工具 (`training/trainer.py`)
  - 损失函数 (`training/losses.py`)
```

---

### 第3部分: 开发命令 (新增)

当前缺少实际可用的开发命令,需要新增此节。

**建议文本**:

```markdown
## 常用开发命令

**Always activate virtual environment first**:
```bash
source venv/bin/activate
```

### PG-GCPL方法 (当前最优 ✅)

#### 训练

```bash
cd multi_time_compression

# Train optimal configuration (K30_r8)
python PG-GCPL/src/scripts/training/train_gaussian_physics_5D.py

# Train physics low-rank (single probe validation)
python PG-GCPL/src/scripts/training/train_physics_low_rank_proper.py
```

#### 评估

```bash
# Ablation study (6 configurations)
python PG-GCPL/src/scripts/evaluation/ablation_gaussian_physics_5D.py

# Query validation (interpolation, extrapolation, latency)
python PG-GCPL/src/scripts/evaluation/test_interpolation_query_physics.py
python PG-GCPL/src/scripts/evaluation/test_extrapolation_query_physics.py
python PG-GCPL/src/scripts/evaluation/test_latency_physics.py
```

#### 可视化

```bash
# SH coefficient reconstruction validation (10 samples)
python PG-GCPL/src/scripts/visualization/visual_validation_K30_r8.py

# Scene rendering comparison (GT vs Pred)
python PG-GCPL/src/scripts/visualization/render_scene_comparison_K30_r8.py
```

### TemporalMLP方法 (废弃,仅参考 ⚠️)

```bash
# Training (deprecated - use PG-GCPL instead)
# python TemporalMLP/src/scripts/train.py --config TemporalMLP/configs/baseline_2k.yaml

# KNN baseline (diagnostic tool)
python TemporalMLP/src/scripts/knn_baseline.py --data_dir ../data_generation/output/dataset_2k
```

### 数据生成

```bash
cd data_generation

# Generate 5D parametric dataset (for PG-GCPL)
python generate_dataset.py

# Validate dataset quality
python validate_dataset.py --data_dir output/5D_parametric_validation

# Analyze SH reconstruction quality
python analyze_sh_quality.py --data_dir output/5D_parametric_validation
```

### 测试

```bash
cd multi_time_compression

# Test PG-GCPL models
pytest PG-GCPL/tests/ -v

# Test shared utilities
pytest shared/tests/ -v
```
```

---

### 第4部分: 实验方法演进 (新增)

当前CLAUDE.md缺少对实验演进逻辑的说明,需要新增此节。

**建议文本**:

```markdown
## 实验方法演进

本项目经历了4个主要阶段,每个阶段都有关键发现和决策点:

### Phase 0: TPE方法 (2025-12月初) ❌ FAILED

**核心思想**: 时间→空间映射, `F(p,t) = F_base(p + δ(t))`

**实验**:
- Cornell Box (室内): TPE贡献率84.9% → 严重失效
- House (室外): TPE贡献率96.5%, 但E_total极小 → 问题太简单

**关键发现**:
- ✓ Grid插值 (Spline) >> 隐式神经表示
- ✓ 简单方法在平滑数据上更有效
- ✗ TPE假设不成立: 时间无法简单映射到空间扰动

**决策**: 放弃TPE → 转向物理引导方法

**文档**: `archive/TPE_Method_Archived/docs/` (7篇详细分析)

---

### Weeks 3-4: MLP Hybrid方法 ❌ FAILED

**核心思想**: 纯数据驱动MLP (未详细记录)

**结果**: 训练不稳定, 泛化能力差

**决策**: 放弃纯MLP → 尝试任务书基线

**文档**: `archive/Week3-4_MLP_Hybrid_Failed/`

---

### TemporalMLP: 任务书基线 (2025-12月) ⚠️ DEPRECATED

**核心思想**: Gaussian Mixture + Temporal MLP + Decoder MLP (任务书3层架构)

**实验序列**:
1. 小数据集 (343 probes) → 过拟合
2. 数据扩充 (1,728 probes) → 过拟合解决
3. 正则化调优 → Val PSNR 28.53 dB
4. 性能瓶颈分析 → 接近数据质量上限

**关键发现**:
- ✓ 方法有效: 模型PSNR 28.53 dB vs K-NN 11.09 dB (+17.44 dB)
- ✗ 压缩比低: 仅6.9× (远低于17×目标)
- ✗ 瓶颈在数据空间不连续性 (ρ=0.275)

**决策**: 保留作baseline → 转向物理引导压缩

**文档**: `TemporalMLP/README.md`, `docs/EXPERIMENT_INDEX.md`

---

### PG-GCPL: 物理引导Gaussian压缩 (2025-12月末-2026-01月) ✅ SUCCESS

**核心思想**: 空间Gaussian混合 + 时间物理引导低秩分解

**实验阶段**:

**Week 1-2: 物理低秩验证** (单探针)
- Cornell Box: 2.34× 压缩 + 33.5% 精度提升 (vs Spline)
- 证明: 物理基函数 > 学习嵌入

**Week 4: 查询验证**
- 内插退化 1.28× < 1.5× ✅
- 外推退化 1.10× < 3.0× ✅ (excellent)
- 查询延迟 0.145ms < 0.5ms ✅

**Week 5: 多探针架构 + SurveyGo改进**
- 扩展至125探针 × 41光源配置
- 集成: Charbonnier loss, L1 temporal正则化, K-Means+SVD初始化
- Baseline K20_r5: 39.31× (过压缩)

**Week 6: Ablation实验**
- 测试6个(K, rank)配置
- 找到最优: K30_r8 (16.59× compression, MAE 0.0343)

**Week 6.2: 可视化验证**
- SH重建: MAE 0.0318 < 0.05 ✅
- 场景渲染: SSIM 0.951 > 0.95 ✅

**最终决策**: ✅ GO - 方法验证成功, 所有核心指标达标

**文档**: `PG-GCPL/README.md`, `PG-GCPL/docs/reports/05_Final_Validation.md`

---

### 方法对比总结

| 方法 | 参数量 | 压缩比 | 质量指标 | 状态 |
|------|--------|--------|----------|------|
| TPE | - | - | E_total=0.359 | ❌ FAILED |
| MLP Hybrid | 17K+ | - | - | ❌ FAILED |
| TemporalMLP | 12.6K | 6.9× | PSNR 28.5 dB | ⚠️ DEPRECATED |
| **PG-GCPL K30_r8** | **8.34K** | **16.59×** | **SSIM 0.951** | **✅ SUCCESS** |

**核心教训**:
1. 物理先验 > 数据驱动 (TPE, MLP失败 vs PG-GCPL成功)
2. 低秩分解有效压缩时间维度 (16.59× vs 6.9×)
3. 场景几何补偿SH低频限制 (Envmap SSIM 0.699 → 场景SSIM 0.951)
```

---

### 第5部分: 配置系统 (更新)

**修改要点**: 明确PG-GCPL和TemporalMLP的配置路径

**建议文本**:

```markdown
## 配置系统

### PG-GCPL方法 (当前使用 ✅)

**配置位置**: `PG-GCPL/src/scripts/training/train_gaussian_physics_5D.py` (硬编码)

**最优配置 K30_r8**:
```python
num_gaussians = 30
rank = 8
learning_rate = 1e-3
num_epochs = 1000
batch_size = 512
lambda_temporal = 0.001  # L1 temporal regularization
use_charbonnier = True
charbonnier_epsilon = 1e-3
```

**数据集**: `data_generation/output/5D_parametric_validation/`
- 125探针 × 41光源配置
- 分割: Train 28 / Val 6 / Test 7

**初始化**: K-Means (probe positions) + SVD (per-Gaussian U matrix)

---

### TemporalMLP方法 (废弃 ⚠️)

**配置位置**: `TemporalMLP/configs/`

**推荐配置** (仅作baseline参考):
- `baseline_2k.yaml`: 稳定配置, Val PSNR 28.3 dB
- `baseline_2k_v2.yaml`: 降低正则化, Val PSNR 28.53 dB

**关键参数**:
```yaml
temporal_smooth: 0.05        # 时间平滑权重
num_gaussians: 250           # Gaussian数量
latent_dim_base: 6           # 基础潜在码维度
optimizer:
  gaussians: {lr: 0.008}
  mlp: {lr: 0.0008}
```

**数据集**: `data_generation/output/dataset_2k/`
- 1,728探针 × 6时刻
```

---

### 第6部分: 目标架构 (重命名+标注)

**修改要点**:
- 标题改为"目标架构 (From Task Specification)" → "任务书目标架构 (尚未完全实现)"
- 每个层次添加实现状态标注

**建议文本**:

```markdown
## 任务书目标架构 (部分实现)

**IMPORTANT**: 以下是任务书 (`docs/thesis/多时刻光照压缩任务书.md`) 描述的**目标架构**, 并非当前完整实现。

**当前实现状态**:
- ✅ 时间层: 已实现 (PG-GCPL物理基函数 ≈ 任务书时变潜在码)
- ✅ 空间层: 已实现 (PG-GCPL Gaussian混合 = 任务书Gaussian混合)
- ✅ 解码层: 已实现 (PG-GCPL低秩分解 ≈ 任务书Decoder MLP)
- ❌ 层次层: 未实现 (4×3级联体积, LRU缓存, CUDA融合)

---

### 1. 时间层: 时变潜在码 ✅ (已实现变体)

**任务书方案**:
```
F_j(t) = MLP_temporal([F_j^base, sun_dir(t)])
```

**PG-GCPL实现** (等价功能):
```
SH(t) = U_j @ (coeffs_j @ Φ_physics(sun_zenith, sun_azimuth, intensity, color_temp, cloud))
```

**关键区别**:
- 任务书: 学习的MLP映射
- PG-GCPL: 显式物理基函数 (7D: cos/sin太阳角 + 强度 + 色温 + 云量)
- **优势**: PG-GCPL外推能力excellent (1.10× vs 3.0×目标)

**压缩比分析** (时间维度):
- 朴素: K × T × D = 750 × 24 × 9 = 162,000 params
- 任务书: K × D_base + 2K ≈ 750 × 8 = 6,000 params (27×)
- PG-GCPL: K × (27×rank + rank×7) = 30 × (27×8 + 8×7) = 6,480+1,680 = 8,160 params (19.8×)

**实现**: `PG-GCPL/src/models/gaussian_physics_5D.py:32` (ExtendedPhysicsBasis5D)

---

### 2. 空间层: Gaussian混合表示 ✅ (已实现)

**任务书 = PG-GCPL** (完全一致):
```
F(p,t) = Σ_{j: p ∈ R(G_j)} F_j(t) · G_j(p)
```

**高斯参数**:
- 位置 μ_j ∈ ℝ³
- 尺度 s_j ∈ ℝ³
- 旋转 q_j ∈ ℝ⁴ (四元数) - PG-GCPL简化为各向同性,仅存储3D尺度
- 基础潜在码 F_j^base → PG-GCPL中为低秩矩阵U_j

**初始化**:
- K-Means聚类探针位置 → μ_j
- 最近邻距离 → s_j
- SVD(cluster_sh) → U_j (PG-GCPL特有)

**实现**: `PG-GCPL/src/models/gaussian_mixture.py`

---

### 3. 解码层: SH系数生成 ✅ (已实现变体)

**任务书方案**:
```
SH_coeff(p,t) = MLP_decoder([F(p,t), p, sun_dir(t)])
```

**PG-GCPL实现** (低秩分解):
```
SH_coeff(p,t) = Σ_j G_j(p) × [U_j @ (coeffs_j @ Φ_physics(t))]
```

**关键区别**:
- 任务书: 2层×64 MLP
- PG-GCPL: 低秩矩阵分解 (U: [27, rank], coeffs: [rank, 7])
- **优势**: PG-GCPL参数更少 (8,340 vs 任务书估计~12K)

**实现**: `PG-GCPL/src/models/gaussian_physics_5D.py:35` (GaussianPhysicsCompression5D.forward)

---

### 4. 层次层: 4×3级联体积 ❌ (未实现)

**任务书目标**:

**空间4级**:
- Level 0: 1m块 (近场, <5m)
- Level 1: 4m块 (中场, 5-20m)
- Level 2: 16m块 (远场, 20-80m)
- Level 3: 64m块 (极远, >80m)

**时间3级**:
- Level 0: 5分钟粒度 (快速变化, 如云)
- Level 1: 15分钟粒度 (太阳轨迹)
- Level 2: 1小时粒度 (慢速变化)

**选择策略**:
- 空间: 基于距相机距离
- 时间: 基于变化率 ||L(p,t) - L(p,t-Δt)||₂ / Δt

**总计**: 12个级联体积 (4 × 3)

**当前状态**: ❌ 未实现 - 这是与任务书最大差距

**实时优化** (未实现):
- Temporal LRU Cache: 预测性解压, 70%缓存命中率
- 融合CUDA kernel: 时间插值+高斯查找+MLP推理单kernel
- 目标: <0.5ms解压 (保守), <0.15ms (冲刺,Cache命中)

**PG-GCPL当前性能**: 0.145ms (已达<0.5ms目标, 无cache)

---

### 任务书 vs PG-GCPL对比

| 维度 | 任务书目标 | PG-GCPL实现 | 状态 | Gap |
|------|-----------|------------|------|-----|
| **时间编码** | 学习MLP | 物理基函数 | ✅ 变体 | 更好外推 |
| **空间编码** | Gaussian混合 | Gaussian混合 | ✅ 一致 | - |
| **解码** | Decoder MLP | 低秩分解 | ✅ 变体 | 更少参数 |
| **层次结构** | 4×3级联 | 单层 | ❌ 缺失 | 主要Gap |
| **时刻数** | 24 | 7-41 | ⚠️ 部分 | 可扩展 |
| **压缩比** | 1:17 | 16.59× | ✅ 达标 | - |
| **解压速度** | <0.5ms | 0.145ms | ✅ 超过 | - |
| **PSNR** | >38 dB | SSIM 0.951 | ✅ 变体 | 指标不同 |
| **LRU Cache** | 是 | 否 | ❌ 缺失 | 未来工作 |
| **CUDA融合** | 是 | 否 | ❌ 缺失 | 未来工作 |
| **光照解耦** | 直接+间接 | 否 | ❌ 缺失 | 未来工作 |

**结论**: PG-GCPL在核心压缩和质量指标上达标或超过任务书,但缺少层次化和实时优化部分。
```

---

### 第7部分: 当前实现状态 (重写)

**修改要点**: 移除误导性的"What Exists",改为基于实际代码的清晰描述

**建议文本**:

```markdown
## 当前实现状态

### PG-GCPL方法 (2026-01-03) ✅

**完整实现**:
1. ✅ **Gaussian混合层** (`models/gaussian_mixture.py`)
   - K-Means初始化 (聚类探针位置)
   - 3D高斯函数 (位置, 尺度, 基础潜在码)
   - Top-k加速查询 (3个最近高斯)

2. ✅ **物理基函数层** (`models/gaussian_physics_5D.py:32`)
   - 7D扩展物理基: [cos(θ), sin(θ), cos(φ), sin(φ), I, T_norm, exp(-cloud)]
   - 支持5D参数空间: [zenith, azimuth, intensity, color_temp, cloud_cover]
   - Excellent外推能力 (1.10× degradation)

3. ✅ **低秩时空分解** (`models/gaussian_physics_5D.py:35`)
   - Per-Gaussian低秩矩阵 U_j: [27, rank]
   - 时间混合系数 coeffs_j: [rank, 7]
   - SVD初始化 (92-97% energy capture)

4. ✅ **SurveyGo训练增强** (`models/gaussian_physics_5D.py:242`)
   - Charbonnier loss (鲁棒损失)
   - L1 temporal正则化 (稀疏化62% coefficients)
   - K-Means + SVD数据驱动初始化

5. ✅ **完整训练/评估Pipeline**
   - 训练脚本: `scripts/training/train_gaussian_physics_5D.py`
   - Ablation: `scripts/evaluation/ablation_gaussian_physics_5D.py`
   - 查询验证: `scripts/evaluation/test_{interpolation,extrapolation,latency}_physics.py`
   - 可视化: `scripts/visualization/{visual_validation,render_scene_comparison}_K30_r8.py`

**验证结果** (K30_r8, 2026-01-03):
- 压缩比: 16.59× (138,375 → 8,340 params)
- SH重建: MAE 0.0318 < 0.05 ✅
- 场景渲染: SSIM 0.951 > 0.95 ✅
- 查询延迟: P95 0.145ms < 0.5ms ✅
- 泛化能力: 内插1.28×, 外推1.10× ✅

**实验数据** (4个阶段):
- Phase 01: 物理低秩验证 (单探针, 33.5%精度提升 vs Spline)
- Phase 02: 查询验证 (内插/外推/延迟全通过)
- Phase 03: 多探针5D训练 (K20_r5 baseline)
- Phase 04: Ablation + 可视化 (K30_r8最优)

**Checkpoints**:
- 最优模型: `experiments/04_ablation_and_visualization/ablation/K30_r8/checkpoints/best_model.pt`
- 包含: Gaussian参数, U矩阵, time_coeffs, 优化器状态

---

### TemporalMLP方法 (2025-12月) ⚠️

**完整实现** (已废弃,保留作baseline):
1. ✓ Gaussian混合 (`TemporalMLP/src/models/gaussian_mixture.py` - 从shared复制)
2. ✓ Temporal MLP (`TemporalMLP/src/models/temporal_mlp.py`)
3. ✓ Decoder MLP (`TemporalMLP/src/models/decoder_mlp.py`)
4. ✓ 完整训练pipeline (`TemporalMLP/src/scripts/train.py`)

**最佳结果** (baseline_2k_v2):
- Val PSNR: 28.53 dB
- 压缩比: 6.9× (1,728 → 250 Gaussians)
- 参数量: 12,612

**诊断工具**:
- K-NN baseline: `scripts/knn_baseline.py` (验证数据质量上限)
- 空间平滑性分析: `scripts/analyze_data_smoothness.py`

**结论**: 方法有效但压缩比低 → 被PG-GCPL替代

---

### 与任务书的Gap ❌

以下是任务书目标但尚未实现的部分:

1. **4×3层次化级联体积** (Cascaded Lighting Volume)
   - 空间4级 (1m, 4m, 16m, 64m)
   - 时间3级 (5min, 15min, 1hr)
   - 自适应选择策略

2. **实时解压优化**
   - Temporal LRU Cache (时间相干性缓存, 70%命中率)
   - 融合CUDA kernel (单kernel完成插值+查询+推理)
   - 目标: <0.15ms解压 (Cache命中)
   - 当前: 0.145ms (无Cache) ✓ 已达<0.5ms目标

3. **10-bit量化**
   - 两阶段训练 (fine-tuning + decoder adaptation)
   - 目标存储: <100MB (24时刻)

4. **能量守恒损失** (物理软约束)
   ```
   ℒ_energy = max(0, ∫L_pred - ∫L_gt - ε)²
   ```
   - 当前仅有: ℒ_recon + ℒ_temporal

5. **光照解耦**
   - 直接光 + 间接光分离
   - 支持实时编辑 (太阳方向, 天气, 局部调整)

6. **24时刻扩展**
   - 当前: 3-7时刻 (实验)
   - 目标: 24时刻 (每小时)

**优先级排序** (基于Week 6报告):
1. 跨场景泛化测试 (短期, 1-2周)
2. 8D参数空间扩展 (中期, 1-2月)
3. 层次化Gaussian结构 (中期, 1-2月)
4. CUDA融合kernel (中期, 1-2月)
5. 24时刻扩展 + 量化 (长期, 3+月)
```

---

### 第8部分: 重要实现细节 (更新)

**修改要点**: 更新为PG-GCPL的实际实现细节

**建议文本**:

```markdown
## 重要实现细节

### 坐标系统

- **World coordinates**: 右手系, Y-up (Mitsuba惯例)
- **Probe positions**: 归一化至[-1, 1]³ (dataset loader)
- **Sun directions**: 单位向量, world坐标 (天文算法)
- **Light parameters**: 5D [zenith∈[0,π/2], azimuth∈[0,2π], intensity∈[0.5,1.5], color_temp∈[3000,7000]K, cloud∈[0,0.8]]
- **SH coefficients**: 实球谐, 2阶 (9 bases), 存储为[N, 27] (9 bases × 3 RGB)

### 数据流 (PG-GCPL)

1. **数据集生成** (data_generation/):
   - Mitsuba 3渲染: 41个光源配置 × 125探针位置
   - 分层采样: 几何12 (太阳角) + 强度20 + 大气9 (色温+云量)
   - SH拟合: 2阶球谐 (27系数), 64 samples/probe
   - 输出: `5D_parametric_validation/probes.npz`, `config_XX/sh_coeffs.npz`

2. **训练** (PG-GCPL/):
   - 加载: 125探针 × 41配置 = 5,125样本
   - 初始化: K-Means(probe_pos) → μ_j, SVD(cluster_sh) → U_j
   - Forward:
     ```python
     light_params → Φ_physics(7D) → coeffs @ Φ → [rank, 1]
     U @ (coeffs @ Φ) → SH_pred[27]
     Gaussian(probe_pos) → weights → Σ_j w_j × SH_j
     ```
   - Loss: Charbonnier(SH_pred, SH_gt) + λ_temp × ||coeffs||₁
   - Backprop: 更新μ, s, U, coeffs

3. **推理** (查询probe_pos + light_params):
   - 找top-k=3最近Gaussians (KDTree或暴力搜索)
   - 计算Φ_physics(light_params)
   - 每个Gaussian: U_j @ (coeffs_j @ Φ) → SH_j
   - 加权求和: Σ G_j(probe_pos) × SH_j → SH_final
   - Latency: 0.145ms (P95, GPU)

### 损失函数 (PG-GCPL)

```python
# 1. Charbonnier重建损失 (鲁棒MSE)
ℒ_recon = mean(√((SH_pred - SH_gt)² + ε²)), ε=1e-3
# 优势: 对蒙特卡洛噪声更鲁棒

# 2. L1 Temporal正则化 (稀疏约束)
ℒ_temporal = λ_temp × ||time_coeffs||₁, λ_temp=0.001
# 效果: 稀疏化62% coefficients, +3.8% Val性能

# 总损失
ℒ_total = ℒ_recon + ℒ_temporal
```

**对比TemporalMLP** (已废弃):
```python
# 1. Charbonnier重建 (相同)
ℒ_recon = mean(√((SH_pred - SH_gt)² + ε²))

# 2. 时间平滑 (二阶差分)
ℒ_temporal_smooth = mean(||SH(t-1) - 2·SH(t) + SH(t+1)||₁)
# 鼓励平滑时间过渡

# 问题: temporal_smooth权重难以调优 (21.5% → 0.02)
```

### Gaussian初始化 (PG-GCPL)

**两阶段初始化**:

**1. K-Means聚类** (探针空间分布):
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=K, random_state=42)
kmeans.fit(probe_positions)  # [N, 3]
μ_j = kmeans.cluster_centers_  # [K, 3]
```

**2. SVD初始化** (每个Gaussian的U矩阵):
```python
for j in range(K):
    # 找到属于cluster j的探针
    mask = (kmeans.labels_ == j)
    cluster_sh = all_sh[mask]  # [M, 27]

    # SVD分解
    U_full, S, Vt = np.linalg.svd(cluster_sh)
    U_j = Vt[:rank].T  # [27, rank] - 取前rank个右奇异向量

    # 验证能量
    energy = S[:rank].sum() / S.sum()  # 通常92-97%
```

**3. 尺度初始化**:
```python
# 最近邻距离 (各向同性)
dist = min_{k≠j} ||μ_j - μ_k||
s_j = [dist, dist, dist]
```

**4. 系数初始化**:
```python
time_coeffs = torch.randn(K, rank, 7) * 0.01
```

**Critical**: K ≤ N_probes (避免过拟合)

---

### TemporalMLP初始化 (已废弃, 仅参考)

**K-Means聚类** (相同):
```python
μ_j, s_j = KMeans_init(probe_positions, K)
```

**基础潜在码** (随机):
```python
F_j^base ~ N(0, 0.1²), shape=[K, D_base]
```

**MLP** (Xavier):
```python
TemporalMLP: Xavier_init
DecoderMLP: Xavier_init
```

**问题**: 没有数据驱动初始化 (如SVD), 收敛较慢

---

### 训练稳定性

**PG-GCPL** (✅ 稳定):
- ✓ 早期收敛 (80-100% epochs)
- ✓ Loss平滑下降
- ✓ 无震荡
- ✓ Charbonnier loss鲁棒性

**TemporalMLP** (⚠️ 需注意):
- ⚠️ NaN losses (不稳定Gaussian尺度)
  - Solution: Gradient clipping (grad_clip: 1.0)
- ⚠️ 过拟合 (小数据集)
  - Solution: 数据扩充 (343 → 1,728 probes)
- ⚠️ temporal_smooth权重难调
  - Solution: 降至0.02-0.05

**Common issues**:
1. **Gaussian覆盖不足** (< 95% probes within influence)
   - Solution: 增加K或增大初始尺度

2. **时间抖动** (temporal jitter)
   - Solution: 增加temporal正则化权重

3. **Poor interpolation** (时间泛化差)
   - Solution: 更多训练时刻, 增加数据多样性
```

---

### 第9部分: 快速参考 (新增)

新增快速查找表,方便用户快速定位关键信息。

**建议文本**:

```markdown
## 快速参考

### 我应该使用哪个方法?

| 你的目标 | 推荐方法 | 路径 | 状态 |
|---------|----------|------|------|
| **新实验/开发** | **PG-GCPL** | `PG-GCPL/` | ✅ CURRENT |
| 性能baseline对比 | TemporalMLP | `TemporalMLP/` | ⚠️ DEPRECATED |
| 失败案例分析 | TPE / MLP Hybrid | `archive/` | ❌ ARCHIVED |
| 理解目标架构 | 任务书 | `docs/thesis/多时刻光照压缩任务书.md` | 📖 SPEC |

### 关键文档路径

| 文档 | 路径 | 用途 |
|------|------|------|
| **实验演进时间线** | `/EXPERIMENTAL_WORKFLOW.md` | 完整实验流程梳理 |
| **PG-GCPL方法说明** | `PG-GCPL/README.md` | 当前最优方法文档 |
| **最终验证报告** | `PG-GCPL/docs/reports/05_Final_Validation.md` | Week 6验证结果 |
| **任务书 (目标)** | `docs/thesis/多时刻光照压缩任务书.md` | 权威目标架构 |
| **实验索引** | `multi_time_compression/docs/EXPERIMENT_INDEX.md` | TemporalMLP阶段实验 |
| **TPE失败分析** | `archive/TPE_Method_Archived/docs/06_Key_Findings_and_Insights.md` | Grid > INR教训 |

### 关键数字 (PG-GCPL K30_r8)

| 指标 | 值 | 说明 |
|-----|---|------|
| 压缩比 | 16.59× | 138,375 → 8,340 params |
| SH重建MAE | 0.0318 | < 0.05目标 ✅ |
| 场景SSIM | 0.951 | > 0.95目标 ✅ |
| 查询延迟P95 | 0.145ms | < 0.5ms目标 ✅ |
| 内插退化 | 1.28× | < 1.5×目标 ✅ |
| 外推退化 | 1.10× | < 3.0×目标 ✅ |
| 参数量 | 8,340 | K30 × (10+30r8) |
| 训练时长 | ~1.1h | 1000 epochs, GPU |
| 最优Epoch | 906 | 早期收敛 (91%) |

### 常见问题

**Q: 我应该从哪个文件开始阅读?**
A:
1. 先读 `EXPERIMENTAL_WORKFLOW.md` (理解演进)
2. 再读 `PG-GCPL/README.md` (理解当前方法)
3. 最后读 `docs/thesis/多时刻光照压缩任务书.md` (理解目标)

**Q: 为什么有这么多方法 (TPE, TemporalMLP, PG-GCPL)?**
A: 这是实验演进过程:
- TPE → 失败 (假设不成立)
- MLP Hybrid → 失败 (训练不稳定)
- TemporalMLP → 次优 (压缩比低)
- **PG-GCPL → 成功** ✅

**Q: PG-GCPL和任务书方法有什么区别?**
A:
- **相似**: 都用Gaussian混合编码空间, 都用物理先验 (太阳方向)
- **不同**: PG-GCPL用显式物理基+低秩分解, 任务书用学习MLP
- **优势**: PG-GCPL外推能力更好 (1.10× vs 估计3-5×)
- **缺失**: PG-GCPL没有层次化结构 (4×3级联), 没有LRU缓存

**Q: 我要训练新模型,应该用哪个配置?**
A: 使用PG-GCPL K30_r8:
```bash
python PG-GCPL/src/scripts/training/train_gaussian_physics_5D.py
```
(配置硬编码在脚本中: K=30, rank=8)

**Q: 如何评估模型质量?**
A: 运行完整验证pipeline:
```bash
# 1. Ablation (找最优配置)
python PG-GCPL/src/scripts/evaluation/ablation_gaussian_physics_5D.py

# 2. 查询验证 (泛化能力)
python PG-GCPL/src/scripts/evaluation/test_interpolation_query_physics.py
python PG-GCPL/src/scripts/evaluation/test_extrapolation_query_physics.py

# 3. 可视化 (SH + 场景渲染)
python PG-GCPL/src/scripts/visualization/visual_validation_K30_r8.py
python PG-GCPL/src/scripts/visualization/render_scene_comparison_K30_r8.py
```

**Q: 任务书中的4×3级联在哪里?**
A: ❌ 尚未实现。这是当前方法与任务书的主要gap。优先级: 中期工作 (1-2月)

**Q: 为什么envmap SSIM只有0.699但场景SSIM有0.951?**
A:
- Envmap直接展示SH重建 → 受SH低频限制
- 场景渲染包含几何细节 → 几何补偿SH限制
- **结论**: 场景SSIM才是真实应用质量指标 ✅

---

### 论文写作资源

**Methods章节**:
- PG-GCPL方法: `PG-GCPL/README.md` (公式+架构)
- 物理基函数: `PG-GCPL/docs/reports/01_Physics_Low_Rank_Results.md`
- SurveyGo改进: `PG-GCPL/docs/reports/05_Final_Validation.md` (第5.1-5.2节)

**Results章节**:
- Ablation表格: `PG-GCPL/docs/reports/05_Final_Validation.md` (第6.2节)
- 可视化结果: `PG-GCPL/experiments/04_ablation_and_visualization/visualization/`
- 压缩-质量Trade-off: `PG-GCPL/docs/reports/05_Final_Validation.md` (压缩比vs质量权衡分析)

**Discussion章节**:
- 物理 vs 数据驱动: `PG-GCPL/docs/reports/05_Final_Validation.md` (物理引导vs纯数据驱动对比)
- TPE失败教训: `archive/TPE_Method_Archived/docs/06_Key_Findings_and_Insights.md`
- 方法演进: `EXPERIMENTAL_WORKFLOW.md`

**Related Work**:
- Gaussian Compression (SIGGRAPH 2025): 空间压缩baseline
- K-Planes (CVPR 2023): 时间平滑性启发
- PBR-NeRF (2024): 能量守恒约束 (未实现)
```

---

## 建议的最终CLAUDE.md结构

```markdown
# CLAUDE.md

## 项目概述
- 明确当前最优方法 (PG-GCPL ✅)
- 区分任务书目标 vs 当前实现
- 压缩目标指标到表格

## 仓库结构
- PG-GCPL/ (✅ CURRENT BEST)
- TemporalMLP/ (⚠️ DEPRECATED)
- archive/ (❌ FAILED METHODS)
- 明确文件路径和状态标注

## 常用开发命令
- PG-GCPL训练/评估/可视化
- TemporalMLP (仅参考)
- 数据生成
- 测试

## 实验方法演进 (NEW)
- Phase 0: TPE ❌
- Weeks 3-4: MLP Hybrid ❌
- TemporalMLP ⚠️
- PG-GCPL ✅
- 方法对比总结

## 配置系统
- PG-GCPL配置 (当前使用)
- TemporalMLP配置 (废弃)

## 任务书目标架构 (部分实现)
- 明确标注实现状态
- 时间层 ✅, 空间层 ✅, 解码层 ✅, 层次层 ❌
- 任务书 vs PG-GCPL对比表

## 当前实现状态
- PG-GCPL完整实现 ✅
- TemporalMLP实现 ⚠️
- 与任务书的Gap ❌

## 重要实现细节
- 坐标系统
- 数据流 (PG-GCPL)
- 损失函数 (PG-GCPL vs TemporalMLP)
- Gaussian初始化 (两阶段)
- 训练稳定性

## 快速参考 (NEW)
- 方法选择指南
- 关键文档路径
- 关键数字
- 常见问题
- 论文写作资源

## 开发工作流 (保留现有)
## Physics and Mathematical Background (保留现有)
## 压缩比分析 (保留现有)
## Thesis Roadmap (更新状态)
## Tips for Working with this Codebase (更新)
## References (添加PG-GCPL相关文献)
```

---

## 实施步骤

1. **备份当前CLAUDE.md**:
   ```bash
   cp CLAUDE.md CLAUDE.md.backup.20260104
   ```

2. **逐节更新**:
   - 第1部分: 项目概述 (重点修改)
   - 第2部分: 仓库结构 (重写)
   - 第3部分: 开发命令 (新增)
   - 第4部分: 实验演进 (新增)
   - 第5-9部分: 按建议修改

3. **验证更新**:
   - 检查所有文件路径是否存在
   - 验证所有命令是否可执行
   - 确认所有数字与实验报告一致

4. **最终检查**:
   - 移除所有误导性描述
   - 确保状态标注清晰 (✅ ⚠️ ❌)
   - 验证方法名称一致性

---

## 关键改进点总结

1. **明确方法状态**: PG-GCPL ✅ / TemporalMLP ⚠️ / TPE ❌
2. **区分目标与现状**: 任务书 (目标) vs PG-GCPL (实现)
3. **修正文件路径**: `PG-GCPL/src/models/` 而非 `models/`
4. **新增实验演进**: 完整演进逻辑和教训
5. **新增开发命令**: 实际可用的训练/评估命令
6. **新增快速参考**: 方法选择/文档路径/常见问题

---

**生成时间**: 2026-01-04
**基于**: 完整代码库分析 + 实验报告梳理 + 文件时间戳验证
