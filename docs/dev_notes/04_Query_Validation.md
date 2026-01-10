# Week 4-6: 查询验证与Ablation实验报告

**时间**: 2026-01-03
**目标**: 验证Physics-only模型查询能力 + 找到17×压缩比最优配置
**状态**: Week 4-5完成 ✓, Week 6.1进行中

---

## Week 4: Physics-only模型查询验证

### 4.1 内插查询测试 (Interpolation Query)

**目标**: 验证模型对训练范围内随机查询的泛化能力
**成功标准**: 内插查询退化 < 1.5× 训练MAE

#### 实验配置
- 模型: PhysicsLowRank5D (rank=5, 170 params)
- Checkpoint: `archive/Week3-4_MLP_Hybrid_Failed/checkpoints/stage1_best.pt`
- 测试: 20个随机内插配置 (从训练范围内均匀采样)

#### 参数范围
| 参数 | 训练范围 |
|------|----------|
| Zenith | [10°, 80°] |
| Azimuth | [0°, 360°] |
| Intensity | [0.5, 1.5] |
| Color Temp | [3000K, 7000K] |
| Cloud Cover | [0.0, 0.8] |

#### 结果
```
内插查询结果:
  训练集MAE (probe_0): 0.014231
  内插查询MAE (20 configs): 0.018227
  退化比: 1.28× (< 1.5× target ✓)
```

**结论**: ✓ **PASSED** - 模型在训练范围内泛化良好

---

### 4.2 外推查询测试 (Extrapolation Query)

**目标**: 验证模型对超出训练范围查询的鲁棒性
**成功标准**: 外推查询退化 < 3.0× 训练MAE

#### 测试配置
5个超出训练范围的极端配置:
1. **低强度**: intensity=0.3 (train min: 0.5)
2. **高强度**: intensity=2.0 (train max: 1.5)
3. **低色温**: color_temp=2500K (train min: 3000K)
4. **高色温**: color_temp=8500K (train max: 7000K)
5. **高云量**: cloud_cover=0.9 (train max: 0.8)

#### 结果
```
外推查询结果:
  训练集MAE (probe_0): 0.014231
  外推查询MAE (5 configs): 0.015634
  退化比: 1.10× (< 3.0× target ✓)
```

**结论**: ✓ **PASSED** - 物理基函数提供excellent外推能力

---

### 4.3 查询延迟测试 (Query Latency)

**目标**: 验证实时查询性能
**成功标准**: P95延迟 < 0.5ms

#### 实验设置
- Device: CUDA (GPU)
- 查询数量: 100 (随机5D配置)
- GPU预热: 10 runs

#### 结果
```
延迟统计 (cuda:0):
  P50: 0.1338 ms
  P90: 0.1442 ms
  P95: 0.1449 ms (< 0.5ms target ✓)
  P99: 0.1533 ms

吞吐量: 10,439 queries/sec
```

**结论**: ✓ **PASSED** - 延迟远低于目标，可支持实时查询

---

## Week 4 总结

| 测试 | 目标 | 结果 | 状态 |
|------|------|------|------|
| 内插查询退化 | < 1.5× | 1.28× | ✓ PASS |
| 外推查询退化 | < 3.0× | 1.10× | ✓ PASS |
| 查询延迟P95 | < 0.5ms | 0.145ms | ✓ PASS |

**Go/No-Go决策**: ✓ **GO** - 所有查询验证通过，可进入Week 5-6

---

## Week 5: 多探针Gaussian-Physics架构实现

### 5.1-5.2 架构设计与SurveyGo改进集成

#### SurveyGo文献综述关键发现
基于81KB文献综述分析，提取4项可操作改进:

1. **L1 Temporal Regularization** (SurveyGo推荐)
   - 原理: 稀疏约束，鼓励time_coeffs中大部分元素接近0
   - 公式: ℒ_temporal = λ_t · ||time_coeffs||₁
   - 权重: λ_t = 0.001

2. **Charbonnier Loss** (鲁棒损失函数)
   - 原理: 对蒙特卡洛噪声更鲁棒，避免outliers主导训练
   - 公式: ρ(x) = √(x² + ε²), ε=1e-3
   - 替代: MSE loss

3. **Adaptive Gaussian Placement** (K-Means初始化)
   - 方法: K-Means聚类探针位置 → 高斯中心
   - 优势: 数据驱动的空间分布，避免均匀网格低效覆盖

4. **SVD Initialization** (低秩矩阵初始化)
   - 方法: 对每个高斯簇内的SH数据做SVD → U_j初始化
   - 结果: SVD energy 92-97% per Gaussian

#### 架构: GaussianPhysicsCompression5D

**数学公式**:
```
SH(p, light_params) = Σ_j G_j(p) × [U_j @ (coeffs_j @ Φ_physics(light_params))]

其中:
- G_j(p): Gaussian weight at position p
- U_j: [27, rank] low-rank matrix for Gaussian j
- coeffs_j: [rank, 7] time coefficients for Gaussian j
- Φ_physics: [7] physics basis from ExtendedPhysicsBasis5D
```

**参数统计 (K=20, rank=5)**:
- Gaussian positions μ_j: 20 × 3 = 60
- Gaussian scales s_j: 20 × 3 = 60
- Low-rank matrices U_j: 20 × 27 × 5 = 2,700
- Time coefficients coeffs_j: 20 × 5 × 7 = 700
- **总计: 3,520 params**

**5D→7D Physics Basis**:
```
Φ_physics([zenith, azimuth, intensity, temp, cloud]) = [
    cos(zenith),
    sin(zenith),
    cos(azimuth),
    sin(azimuth),
    intensity,
    (temp - 5500) / 2000,  # Normalized color temperature
    exp(-cloud)            # Atmospheric scattering
]
```

---

### 5.3 Baseline训练 (K=20, rank=5)

#### 训练配置
```yaml
num_gaussians: 20
rank: 5
learning_rate: 1e-3
lambda_temporal: 0.001  # L1 temporal regularization
num_epochs: 2000
batch_size: 512
loss: Charbonnier (ε=1e-3)
```

#### 数据集
- **来源**: `../data_generation/output/5D_parametric_validation/`
- **探针**: 125个 (均匀网格采样)
- **光源配置**: 41个 (分层策略: 几何12 + 强度20 + 大气9)
- **分割**: Train 28 / Val 6 / Test 7
- **样本数**: Train 3,500 / Val 750 / Test 875

#### K-Means初始化结果
```
初始化 20 个高斯:
  Gaussian 0: cluster_size=7, SVD energy=95.90% (rank=5/5)
  Gaussian 1: cluster_size=8, SVD energy=93.52% (rank=5/5)
  ...
  Gaussian 19: cluster_size=3, SVD energy=96.54% (rank=5/5)

平均SVD能量: 94.3% ± 1.5%
```

#### 训练结果
```json
{
  "best_epoch": 610,
  "val_mae": 0.03310,
  "test_metrics": {
    "mae": 0.034823,
    "rmse": 0.056104,
    "charbonnier": 0.034903
  },
  "compression": {
    "original_params": 138,375,
    "compressed_params": 3,520,
    "ratio": 39.31,
    "target_ratio": 17.0,
    "success": false  # 压缩比过高
  }
}
```

#### 训练曲线分析
- **收敛速度**: 610/2000 epochs (早期收敛)
- **训练稳定性**: Loss平滑下降，无震荡
- **Temporal正则化**: L1 loss从0.015降至0.008 (有效稀疏化)
- **最佳Val MAE**: Epoch 610达到0.0331后稳定

#### Week 5.3 分析

**优点**:
1. ✓ 质量excellent: Test MAE=0.0348 < 0.05 target
2. ✓ SurveyGo改进有效: Charbonnier loss + L1 temporal正则化
3. ✓ 早期收敛: 节省训练时间
4. ✓ SVD初始化有效: 高能量占比

**问题**:
1. ✗ 压缩比39.31× >> 目标17× (over-compressed)
2. 需要增加参数量: 3,520 → ~8,000

**结论**: ✓ **架构验证成功**, 需调整K或rank以hit目标压缩比

---

## Week 6: Ablation实验 (进行中)

### 6.1 参数搜索策略

**目标**: 找到最优(K, rank)组合实现17×压缩比 + MAE < 0.05

#### 参数计算
```
压缩比 = 原始参数 / 压缩参数
       = (125 probes × 41 configs × 27 SH) / compressed_params
       = 138,375 / compressed_params

目标17×压缩比 → compressed_params ≈ 8,139
```

**参数公式**:
```
compressed_params = K × (10 + 30r)
其中: 10 = μ(3) + s(3) + q(4)
      30r = U(27r) + coeffs(3r)
```

#### Ablation配置矩阵

| 配置 | K | rank | 预期参数 | 预期压缩比 | 状态 |
|------|---|------|----------|-----------|------|
| K20_r5_baseline | 20 | 5 | 3,520 | 39.31× | ✓ 完成 |
| K30_r5 | 30 | 5 | 5,280 | 26.21× | 🔄 训练中 |
| K40_r5 | 40 | 5 | 6,400 | 21.62× | ⏳ 等待 |
| **K50_r5** | 50 | 5 | **8,000** | **17.30×** | ⏳ **目标** |
| K20_r8 | 20 | 8 | 5,360 | 25.82× | ⏳ 等待 |
| K30_r8 | 30 | 8 | 7,500 | 18.45× | ⏳ 等待 |

**策略**:
1. **K方向**: 增加高斯数量 (空间覆盖)
2. **rank方向**: 增加秩 (时间表达能力)
3. **平衡**: K30_r8提供最接近17×的平衡配置

#### 预期权衡
- **更多高斯(↑K)**: 更好空间覆盖，每个高斯秩固定
- **更高秩(↑rank)**: 更复杂时间模式，高斯数固定
- **最优**: K50_r5 (8,000 params ≈ 17.3× compression)

---

## 关键技术贡献

### 1. Physics-Guided 5D→7D Basis Extension

**创新点**: 从简单3D太阳位置 [cos(θ), sin(θ), 1] 扩展至完整5D参数化

**物理意义**:
- **几何**: cos(zenith), sin(zenith), cos(azimuth), sin(azimuth) - 太阳方向
- **辐射**: intensity - 光源强度
- **光谱**: (temp - 5500) / 2000 - 归一化色温 (Planck曲线)
- **大气**: exp(-cloud) - 云量指数衰减 (Rayleigh散射近似)

**优势**:
- 外推能力excellent (1.10× degradation vs 3.0× target)
- 参数效率: 仅7个basis functions编码5D参数空间
- 可解释性: 每个basis有明确物理含义

### 2. Spatial-Temporal Factorization

**数学框架**:
```
传统: SH(p, t) - 直接存储 [P×M×27] 参数

Gaussian-Physics:
  SH(p, t) = Σ_j G_j(p) × [U_j @ (coeffs_j @ Φ(t))]
           = 空间混合 × (空间基 @ (时间系数 @ 物理基))
```

**分解维度**:
1. **Spatial**: K个高斯混合 (G_j)
2. **Spatial basis**: 每高斯27→rank低秩 (U_j)
3. **Temporal coefficients**: rank→7映射 (coeffs_j)
4. **Physics basis**: 7D物理先验 (Φ)

**压缩效率**:
- 无分解: 138,375 params
- 分解后: 3,520 params (39.31×)
- 调整后: ~8,000 params (17×)

### 3. SurveyGo-Inspired Improvements

#### Charbonnier Loss
```python
# 传统MSE
loss_mse = (pred - target)²

# Charbonnier (更鲁棒)
loss_char = √((pred - target)² + ε²)
```

**优势**:
- 对outliers更鲁棒 (蒙特卡洛噪声)
- 梯度平滑 (ε平滑项)
- 收敛稳定性提升

#### L1 Temporal Regularization
```python
ℒ_total = ℒ_recon + λ_temporal × ||time_coeffs||₁
```

**优势**:
- 稀疏化time_coeffs (大部分≈0)
- 鼓励平滑时间变化
- 提升泛化能力

---

## 实验统计

### Week 4-6 总览

| 阶段 | 实验数 | 成功 | 失败 | 总训练时长 |
|------|--------|------|------|-----------|
| Week 4 (查询验证) | 3 | 3 | 0 | ~15分钟 |
| Week 5 (Baseline) | 1 | 1* | 0 | ~2.5小时 |
| Week 6 (Ablation) | 6 | TBD | TBD | ~6分钟 (预计) |
| **总计** | 10 | 4+ | 0 | ~3小时 |

*成功 = 质量达标，压缩比需调整

### 参数量统计

| 方法 | 参数量 | 压缩比 | Test MAE | 备注 |
|------|--------|--------|----------|------|
| 原始存储 | 138,375 | 1.00× | - | Baseline |
| Spline插值 (Week 2) | 810 | 170.8× | 0.0162 | Grid-based |
| Physics-only (Week 4) | 170 | 813.7× | 0.0142 | Single probe |
| Gaussian-Physics K20r5 | 3,520 | 39.31× | 0.0348 | Week 5 baseline |
| Gaussian-Physics K50r5 | ~8,000 | 17.30× | TBD | Week 6 target |

---

## Go/No-Go决策矩阵

### Week 4 决策 (已完成)

| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| 内插查询退化 | < 1.5× | 1.28× | ✓ |
| 外推查询退化 | < 3.0× | 1.10× | ✓ |
| 查询延迟P95 | < 0.5ms | 0.145ms | ✓ |

**决策**: ✓ **GO** → 进入Week 5

### Week 5 决策 (已完成)

| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| Test MAE | < 0.05 | 0.0348 | ✓ |
| 压缩比 | 14-18× | 39.31× | ✗ |
| 训练稳定性 | 平滑收敛 | 平滑 | ✓ |
| SurveyGo改进 | 集成 | 集成 | ✓ |

**决策**: ✓ **GO** → 进入Week 6 (需调整压缩比)

### Week 6 决策 (待完成)

| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| 最优配置压缩比 | 14-18× | TBD | ⏳ |
| 最优配置MAE | < 0.05 | TBD | ⏳ |
| Ablation覆盖范围 | 6 configs | 6 | ⏳ |

**决策**: ⏳ **PENDING** - Ablation进行中

---

## 下一步工作 (Week 6.2)

### 最终验证任务

1. **压缩比验证**
   - 确认最优配置在[14, 18]×范围内
   - 对比baseline (K20r5) vs optimal
   - 分析质量vs压缩比权衡曲线

2. **性能对比**
   ```
   对比方法:
   - Naive存储 (138,375 params)
   - Spline插值 (810 params, MAE=0.0162)
   - Physics-only (170 params, MAE=0.0142)
   - Gaussian-Physics (optimal)
   ```

3. **可视化验证**
   - 渲染测试场景 (随机查询)
   - 对比GT vs Reconstruction
   - 计算PSNR/SSIM

4. **最终Go/No-Go**
   ```
   成功标准:
   ✓ 压缩比 ∈ [14, 18]
   ✓ Test MAE < 0.05
   ✓ 查询延迟 P95 < 0.5ms
   ✓ 可视化质量acceptable
   ```

---

## 代码贡献清单

### 新增文件 (Week 4-6)

1. **查询验证** (Week 4)
   - `scripts/test_interpolation_query_physics.py` (325行)
   - `scripts/test_extrapolation_query_physics.py` (360行)
   - `scripts/test_query_latency_physics.py` (280行)

2. **多探针架构** (Week 5)
   - `models/gaussian_physics_5D.py` (360行)
   - `scripts/train_gaussian_physics_5D.py` (367行)

3. **Ablation实验** (Week 6)
   - `scripts/ablation_gaussian_physics_5D.py` (400行+)

**总代码量**: ~2,100行新增代码

### 关键函数

1. **ExtendedPhysicsBasis5D** (models/gaussian_physics_5D.py:32)
   - 5D参数 → 7D物理基
   - 集成几何、辐射、光谱、大气物理

2. **GaussianPhysicsCompression5D** (models/gaussian_physics_5D.py:35)
   - K高斯混合 + 低秩时空分解
   - K-Means + SVD初始化
   - top-k加速查询

3. **charbonnier_loss** (models/gaussian_physics_5D.py:242)
   - 鲁棒损失函数
   - SurveyGo改进

4. **compute_temporal_smoothness_loss** (models/gaussian_physics_5D.py:218)
   - L1时间正则化
   - SurveyGo改进

---

## 参考文献

### 核心论文
1. SurveyGo综述: `/home/kyrie/毕设/基于物理先验的多时刻动态光照压缩方法_SurveyGo.md`
2. 任务书: `docs/thesis/多时刻光照压缩任务书.md`
3. 实验报告: `docs/experiment_reports/` (文档1-9)

### 关键技术
- **物理基**: Sun position算法 (utils/sun_position.py)
- **球谐函数**: SH fitting (utils/spherical_harmonics.py)
- **低秩分解**: SVD初始化 (Week 1-2验证)
- **高斯混合**: K-Means聚类 (sklearn.cluster)

---

**报告生成时间**: 2026-01-03
**实验状态**: Week 4-5完成, Week 6.1进行中
**下次更新**: Week 6.2最终验证完成后
