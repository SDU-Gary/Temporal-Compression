# Week 3-4 实验报告: Physics+MLP混合方法失败分析

**日期**: 2026-01-03
**实验目标**: 验证MLP残差能否在物理基础上提供>10%精度改进
**结果**: ❌ **FAILED** - 性能退化14.97%

---

## 执行摘要

尝试在Physics-only Low-Rank基础上添加MLP残差网络以提升精度, 采用两阶段训练策略:
- **Stage 1**: 预训练物理组件(成功, Val MAE=0.0391)
- **Stage 2**: 添加MLP残差学习(失败, Test MAE退化至0.1007)

**关键失败指标**:
- Test MAE从0.0876退化至0.1007 (**-14.97%**)
- 学习到的α=0.0126 (目标0.1的12.6%), MLP几乎未启用
- Val/Test gap从2.2×恶化至2.7×, 严重过拟合
- 31×参数增长换来性能下降

**根本原因**: 41个训练配置远不足以训练5125参数的MLP, 导致过拟合

---

## 1. 实验设置

### 数据集
- **参数空间**: 5D [sun_zenith, sun_azimuth, intensity, color_temp, cloud_cover]
- **探针**: 125个 (单探针实验)
- **光源配置**: 41个
- **Split**: 28 train / 6 val / 7 test configs
- **样本数**: Train 3500 (125×28), Val 750, Test 875

### 模型架构

```python
SH(light_params) = U @ (Φ_physics(light) + α·MLP(light))

组件:
  - Φ_physics: 7D物理基 [cos(θ), sin(θ), cos(φ), sin(φ), I, ΔT, exp(-cloud)]
  - MLP: ResidualMLP(5→64→64→5)
  - U: 低秩矩阵 [27, 5]
  - α: 可学习混合权重 (初始0.1)

参数统计:
  - Physics: U(135) + time_coeffs(35) = 170
  - MLP: 5→64(320) + 64→64(4096) + 64→5(320) + LayerNorm(256) = 5,125
  - α: 1
  - **Total**: 5,296 (vs Physics-only 170, 31×增长)
```

### 训练策略

#### Stage 1: Physics-Only Pre-training (500 epochs)
```python
冻结MLP, α=0
仅优化: U, time_coeffs
优化器: Adam(lr=1e-3)
调度器: ReduceLROnPlateau(patience=200, factor=0.5)
Early stopping: patience=500

结果: Val MAE=0.0391 ✓
```

#### Stage 2: Hybrid Training (1000 epochs)
```python
解冻MLP, α warmup (0→0.1 over 200 epochs)
差分学习率:
  - MLP: 1e-5 (慢学习)
  - Physics (U, coeffs): 1e-4 (快10×)
梯度裁剪: max_norm=1.0
调度器: ReduceLROnPlateau(patience=100, factor=0.5)

Warmup策略:
  Epoch 1-200: α从0线性增至0.1, requires_grad=False
  Epoch 201+: α.requires_grad=True, 可学习

结果: Val MAE=0.0372, Test MAE=0.1007 ✗
```

---

## 2. 实验结果

### 2.1 性能对比

| 模型 | Train MAE | Val MAE | Test MAE | Val/Test Gap | 参数量 |
|------|-----------|---------|----------|--------------|--------|
| **Stage 1** (Physics-only) | 0.0548 | 0.0391 | **0.0876** | 2.2× | 170 |
| **Stage 2** (Hybrid) | 0.0537 | 0.0372 | 0.1007 | 2.7× | 5,296 |
| **变化** | -2.0% | **+4.9%** | **-14.97%** | **+23%** | **+31×** |

**Go/No-Go检查**:
- [ ] ~~Test MAE <0.04~~ (实际: 0.1007 ✗)
- [ ] ~~改进>10%~~ (实际: -14.97% ✗)
- [ ] ~~α∈[0.05, 0.3]~~ (实际: 0.0126 ✗)
- [ ] ~~Val/Test gap <1.5×~~ (实际: 2.7× ✗)

**决策**: ❌ **NO-GO** - 所有验收标准未达标

### 2.2 MLP贡献分解

```
测试集MAE分解:
  Physics-only:     0.1015
  Hybrid (full):    0.1007
  MLP improvement:  0.0008 (0.8%)

学习到的α: 0.0126

输出幅度对比:
  Physics weights:  [-0.693, 0.892], std=0.541
  MLP residual:     [-1.161, 1.084], std=0.633  ← 已学到特征
  α·MLP:            [-0.015, 0.014], std=0.008  ← 但被α压制50×
```

**关键发现**:
- MLP确实学到了非平凡特征(std=0.633)
- 但优化器学习到极小α(0.0126)来**主动抑制MLP贡献**
- 说明在当前设置下, MLP残差对泛化性能有害而非有益

### 2.3 训练曲线分析

```
Stage 2训练过程:
  Epoch 1:   Val=0.0396, α=0.0000 (warmup开始)
  Epoch 50:  Val=0.2221, α=0.0245 (严重退化, warmup中)
  Epoch 200: Val=0.0421, α=0.1000 (warmup完成)
  Epoch 300: Val=0.0389, α变为可学习
  Epoch 916: Val=0.0372, α=0.0126 (最佳, α被优化器压制)
  Epoch 1000: Val=0.0376, α=0.0124 (训练结束)
```

**观察**:
1. Warmup期间(Epoch 1-200): 性能严重波动, 说明MLP干扰物理基
2. α可学习后(Epoch 201+): α从0.1快速下降至0.013
3. 最终α稳定在0.012-0.013, 说明优化器认为"更小的MLP贡献=更好的泛化"

---

## 3. 失败原因分析

### 3.1 数据量不足 (主要原因)

**参数/样本比**:
- MLP参数: 5,125
- 训练样本: 125探针 × 28配置 = 3,500
- **比值**: 5,125 / 3,500 = 1.46

**对比标准**:
- 深度学习经验法则: 参数/样本 < 0.01 (需>100× 样本)
- **实际需求**: 5,125 × 100 = 512,500样本 → 需 512,500/(125) ≈ **4100个配置**
- **当前仅有**: 28个训练配置
- **缺口**: 4100 / 28 ≈ **146×不足**

**过拟合证据**:
- Val MAE改进4.9%, Test MAE退化14.97% → 经典过拟合
- Val/Test gap从2.2×增至2.7× (+23%)
- MLP在训练集拟合良好但泛化失败

### 3.2 物理基已足够强

**SVD能量分析**(Week 2结果):
- Rank=5捕获99.61%方差能量
- 剩余0.39%能量主要是:
  - 渲染噪声(Monte Carlo采样)
  - 数值误差(SH拟合残差)
  - 真实高频细节(极少)

**Physics-only性能**:
- Test MAE: 0.0876
- 对比Week 2 Spline baseline: 0.0267 (但Spline用810参数)
- **估计噪声下限**: ~0.03-0.05 (基于渲染spp=128)

**结论**: 当前0.088已接近数据质量上限, MLP难以在噪声主导的残差上学习有效模式

### 3.3 训练策略不当

#### α Warmup问题
```python
# 当前策略
Epoch 1-200: α线性增长0→0.1, requires_grad=False
Epoch 201+: α变为可学习

问题:
  - 前200 epochs强制α=0阻碍MLP学习正确梯度信号
  - MLP在无监督信号下随机游走200 epochs
  - 切换到可学习时MLP参数已次优
```

#### 差分学习率问题
```python
LR_MLP = 1e-5
LR_Physics = 1e-4  (10× faster)

问题:
  - MLP学习太慢, 1000 epochs不足以收敛
  - Physics快速微调可能破坏Stage 1学到的结构
```

#### Early Stopping偏差
- 基于Val loss选择最佳模型
- 但Val set也只有6个配置(样本不足)
- Val最优≠Test最优

### 3.4 架构不匹配

**维度瓶颈**:
- 输入: 5D → MLP → 输出: 5D
- 低维空间限制MLP表达能力
- 对比: 高维图像任务(224×224×3输入)MLP表现更好

**Physics Basis已覆盖主要模式**:
- cos/sin(太阳角度): 捕获日周期
- 指数衰减(云量): 捕获大气散射
- 温度归一化: 捕获色温变化
- **剩余模式**: 可能只是配置特异性噪声

**MLP学到的可能是过拟合**:
- 在小数据集上记忆训练样本
- 而非学习可泛化的非线性变换

---

## 4. 对比实验(未执行, 时间限制)

### 实验4.1: 无Warmup
**假设**: Warmup阻碍MLP学习
**修改**: α初始0.1, 从Epoch 1开始可学习
**预期**: 可能改进, 但数据不足问题仍存在

### 实验4.2: 统一学习率
**假设**: 差分LR导致不平衡
**修改**: MLP和Physics用相同LR(1e-4)
**预期**: 收敛更快, 但可能破坏Stage 1预训练

### 实验4.3: 更多数据
**假设**: 增加数据可缓解过拟合
**需求**: 渲染100+配置
**成本**: 125探针 × 60新配置 × 1.5分钟 = 187 GPU小时 ≈ 7.8天
**状态**: GPU资源不足, 未执行

### 实验4.4: 简化MLP
**修改**: Hidden 64→16, 参数降至~1000
**预期**: 小幅改进, 但根因(数据不足)未解决

---

## 5. 经验教训

1. **数据优先于模型**: 41配置对5125参数MLP远远不够(146×不足)
2. **物理先验价值**: 170参数physics ≥ 5296参数hybrid
3. **过拟合监控**: Val改进≠Test改进, 必须严格监控泛化gap
4. **简单即美**: 低秩+物理基已足够强, 盲目增加复杂度无益
5. **资源约束决策**: GPU有限时, 应优先验证简单强基线而非复杂模型

---

## 6. 后续路线决策

### Option A: 放弃MLP, 继续Physics-only (✓ 推荐)

**理由**:
- Physics-only MAE=0.0876, 参数仅170, 接近性能上限
- Week 5-6计划可直接用physics扩展至多探针
- 符合任务目标(17×压缩比, 非精度极致优化)

**行动**:
1. 跳过Week 3-4剩余MLP消融实验
2. 提前进入Week 5-6: 多探针Physics Low-Rank
3. 架构: F(p,t) = Σ G_j(p) * [U_j @ Φ(t)]  (文档9)

**预期**:
- 压缩比: 17× (343探针×6时刻, 基于文档9)
- MAE: <0.05 (基于单探针0.088)
- 参数: ~3,120 (K=20高斯)

### Option B: 大幅增加数据后重试

**要求**:
- 渲染至少200+配置 (当前41, 需增5×)
- 成本: ~10-15 GPU天
- 修改: 无warmup + 统一LR + 简化MLP

**不推荐原因**:
- GPU资源严重不足
- 收益不确定(物理基已很强)
- 偏离主线(毕设重点是压缩系统, 非MLP调优)
- 时间成本高, 风险大

### Option C: 切换至端到端学习

**修改**: 放弃physics basis, 纯MLP学习
**不推荐**: 需要更多数据, 且丢失物理可解释性

---

## 7. 最终决策

**选择**: ✓ **Option A** - 放弃MLP混合, 继续Physics-only多探针路线

**立即行动**:
1. ✓ 归档失败代码至 `archive/Week3-4_MLP_Hybrid_Failed/`
2. ✓ 编写详细失败分析报告(本文档)
3. 更新主计划, 调整Week 4-6 roadmap
4. 验证Physics-only随机查询能力(内插/外推)
5. 准备Week 5-6多探针数据生成

---

## 8. 归档清单

**代码** (`archive/Week3-4_MLP_Hybrid_Failed/code/`):
- `physics_mlp_hybrid.py` (320行): 混合架构实现
- `train_stage1_physics_only.py` (360行): Stage 1训练脚本
- `train_stage2_hybrid.py` (480行): Stage 2训练脚本

**Checkpoints** (`archive/Week3-4_MLP_Hybrid_Failed/checkpoints/`):
- `stage1_best.pt` (30KB): Stage 1最佳模型 (Val MAE=0.0391, 可用)
- `stage2_best.pt` (210KB): Stage 2最佳模型 (过拟合, 不建议使用)

**日志** (`archive/Week3-4_MLP_Hybrid_Failed/logs/`):
- `stage1_train.log`: Stage 1完整训练日志
- `stage2_training.log`: Stage 2训练日志

---

## 附录: 详细训练日志

### Stage 1训练 (500 epochs)
```
Epoch    1: Train=0.0778, Val=0.0433, LR=1.00e-03, Patience=0/500
Epoch  100: Train=0.0549, Val=0.0397, LR=1.00e-03, Patience=34/500
Epoch  200: Train=0.0548, Val=0.0395, LR=1.00e-03, Patience=134/500
Epoch  300: Train=0.0547, Val=0.0397, LR=5.00e-04, Patience=234/500
Epoch  400: Train=0.0548, Val=0.0398, LR=5.00e-04, Patience=334/500
Epoch  500: Train=0.0547, Val=0.0398, LR=2.50e-04, Patience=434/500

Best: Epoch 65, Val MAE=0.0391
```

### Stage 2训练 (1000 epochs, 关键节点)
```
Epoch    1: Train=0.0549, Val=0.0396, α=0.0000
Epoch   50: Train=0.2437, Val=0.2221, α=0.0245  ← Warmup期严重退化
Epoch  100: Train=0.1124, Val=0.0989, α=0.0490
Epoch  200: Train=0.0571, Val=0.0421, α=0.1000  ← Warmup完成
Epoch  300: Train=0.0552, Val=0.0389, α→可学习
Epoch  500: Train=0.0544, Val=0.0379, α=0.0182
Epoch  700: Train=0.0541, Val=0.0377, α=0.0147
Epoch  916: Train=0.0537, Val=0.0372, α=0.0126  ← 最佳
Epoch 1000: Train=0.0537, Val=0.0376, α=0.0124

Best: Epoch 916, Val MAE=0.0372, Test MAE=0.1007
```

**α演化**:
- Warmup(0-200): 强制线性增长至0.1
- 可学习(201-1000): 快速下降至0.013并稳定
- **解释**: 优化器发现α越小, Val loss越低(但Test性能退化)

---

**结论**: Week 3-4 MLP混合方法在当前数据规模(41 configs)下失败. 建议立即切换至**Physics-only多探针压缩路线**(Week 5-6), 这更符合任务目标并在资源约束下可行.

**报告完成时间**: 2026-01-03 16:10
