# TPE研究关键空白分析

**深度思考分析报告**

**日期**: 2025-12-11
**状态**: 基于Cornell Box Level 2成功验证后的系统性反思
**目的**: 识别当前TPE研究的理论、实验、工程空白

---

## 执行摘要

经过20轮深度思考分析，识别出TPE研究的**18个关键问题**，分为3个优先级：

| 优先级 | 数量 | 核心关切 | 风险等级 |
|--------|------|---------|---------|
| **P0 - 紧急** | 4个 | **验证完整性** | 🔴 高 - 可能推翻现有结论 |
| **P1 - 重要** | 4个 | **可靠性和泛化性** | 🟡 中 - 影响毕设完整度 |
| **P2 - 有益** | 4个 | **理论完善** | 🟢 低 - 提升学术价值 |

**最关键发现**：
> 当前TPE验证缺少**Ground Truth检验**（实验F）。我们只验证了"能否用F_static(x+ε)拟合数据"，从未验证"F_static(x+ε)是否真的等于真实光场F_gt(x, t)"。这是validation的致命缺陷。

---

## 目录

1. [P0 - 紧急问题](#p0---紧急问题)
2. [P1 - 重要问题](#p1---重要问题)
3. [P2 - 有益问题](#p2---有益问题)
4. [数据集状态与可用性](#数据集状态与可用性)
5. [实验优先级与行动计划](#实验优先级与行动计划)
6. [附录：详细分析](#附录详细分析)

---

## P0 - 紧急问题

### 问题1：缺少Ground Truth验证（实验F）⚠️⚠️⚠️

**严重性**: 🔴🔴🔴 CRITICAL
**影响**: 可能推翻整个TPE假设

**当前状态**：
- ✅ 验证了：能用F_static(x + ε(t))拟合观测SH(x, t)
- ❌ 未验证：F_static(x + ε(t))是否真的近似真实F_gt(x, t)

**问题描述**：

当前验证只做了"data fitting"，没有做"physical validation"。可能的情况：

```
情况A（期望）：TPE物理正确
F_static(x + ε(t)) ≈ F_gt(x, t)
→ 扰动有物理意义

情况B（危险）：TPE只是黑箱拟合
F_static是overfitted函数，ε(t)是为了fit数据的数学构造
→ 没有真实物理意义
```

**如何区分**：实验F - 真实光场扰动响应

```python
# 关键测试
for t in test_hours:
    # 真实渲染：扰动位置，参考光照
    F_real_perturbed = render(position=x_probe + ε(t),
                               sun=sun_noon)

    # 真实渲染：原始位置，时刻t光照
    F_real_time_t = render(position=x_probe,
                           sun=sun_t)

    # TPE预测
    F_tpe = F_static(x_probe + ε(t))

    # 检查1：静态场泛化能力
    error_generalization = ||F_real_perturbed - F_tpe||

    # 检查2：TPE假设有效性
    error_tpe_assumption = ||F_real_time_t - F_real_perturbed||

    # 检查3：残差大小
    residual = F_real_time_t - F_tpe
```

**预期结果判定**：

| error_tpe_assumption | error_generalization | 结论 |
|---------------------|---------------------|------|
| < 0.01 | < 0.01 | ✅ TPE完全正确 |
| < 0.01 | 0.01-0.05 | ⚠️ TPE概念正确，但F_static表达力不足 |
| 0.01-0.05 | < 0.01 | ⚠️ TPE近似合理，需要残差修正 |
| > 0.05 | any | ❌ TPE假设失败，需要multi-anchor或hybrid |

**行动**：
- [ ] 选择4个代表性时刻（6, 9, 15, 18点）
- [ ] 渲染x + ε(t)在noon光照下的SH（4 × 5分钟 = 20分钟）
- [ ] 计算3个error指标
- [ ] 根据结果决定是否需要残差网络

---

### 问题2：残差模型的必要性未量化

**严重性**: 🔴🔴 HIGH
**影响**: 决定TPE的"天花板"

**问题描述**：

TPE假设 `F(x, t) = F_static(x + ε(t))` 是强假设。更现实的应该是：

```
F(x, t) = F_static(x + ε(t)) + R(x, t, ε)
```

其中R是残差，来源于：
- 间接光照路径变化
- 多次反射的非线性
- 材质BRDF与入射角的耦合
- 遮挡变化

**当前缺失**：我们完全不知道||R||的大小。

**测试方法**：

```python
# 方法1：通过实验F估计
residual_真实 = F_gt(x, t) - F_gt(x + ε(t), t_ref)

# 方法2：训练显式残差网络
residual_net = ResidualMLP(input_dim=18, output_dim=27)
for step in range(500):
    sh_tpe = F_static(x + ε(t))
    residual = residual_net(x, sun_t, ε(t), time_encoding(t))
    sh_pred = sh_tpe + residual
    loss = mse(sh_pred, sh_gt)

# 如果residual_net显著降低误差 → 需要hybrid model
improvement = loss_without_residual - loss_with_residual
```

**判定标准**：

| ||R|| (RMS) | 占比 (vs ||F||) | 结论 |
|------------|----------------|------|
| < 0.001 | < 1% | ✅ Pure TPE sufficient |
| 0.001-0.005 | 1%-5% | ⚠️ Residual helpful but not必需 |
| 0.005-0.02 | 5%-20% | ⚠️ Hybrid model recommended |
| > 0.02 | > 20% | ❌ TPE inadequate, need rethink |

**行动**：
- [ ] 实现ResidualMLP (3小时)
- [ ] 训练并评估residual大小
- [ ] 如果||R||大，将hybrid model纳入Level 3架构

---

### 问题3：空间泛化性完全未知

**严重性**: 🔴 MEDIUM-HIGH
**影响**: 单点成功不代表全场景成功

**问题描述**：

当前所有验证都在Cornell Box的**单个探针位置** `[0, 1, 0]` 进行。这个位置是场景几何中心，可能是"最简单"的情况。

**未回答的问题**：

A. 其他位置的TPE行为：
   - 靠近墙壁的点（如[0.9, 0.5, 0.5]）→ 不对称，可能需要多方向
   - 角落的点（如[0.9, 0.1, 0.9]）→ 强遮挡，可能扰动很大
   - 靠近顶部（如[0, 1.9, 0]）→ 直接光主导，可能扰动很小

B. 扰动方向的空间变化：
   - 中心点：Y轴主导
   - 其他点：可能需要不同方向d(x)
   - 如何参数化d(x)？

C. Level 3多探针的架构选择：
   - 选项A：共享ε(t) → 过度简化
   - 选项B：独立ε_i(t) → 参数爆炸
   - 选项C：插值ε(x, t) → 如何插值？
   - 选项D：网络ε_net(x, t) → 回到神经网络baseline

**测试方法**：

利用现有dataset_2k（1728探针）：

```python
# 选择10个代表性探针位置
test_probes = [
    [0, 1, 0],      # 中心（已验证）
    [0.5, 0.5, 0.5], # 偏心
    [0.9, 1.0, 0],   # 靠墙
    [0.9, 0.1, 0.9], # 角落
    ... # 6个更多位置
]

results = []
for probe_pos in test_probes:
    # 运行Level 2 TPE验证
    success, metrics = validate_tpe_single_probe(
        probe_pos, cornell_box_data
    )
    results.append({
        'position': probe_pos,
        'max_norm': metrics['max_norm'],
        'direction': metrics['principal_direction'],
        'success': success
    })

# 分析空间模式
analyze_spatial_pattern(results)
```

**预期发现**：

| 位置类型 | 预期||ε||_max | 预期主方向 | 成功率 |
|---------|-------------|-----------|--------|
| 中心 | 0.77m | Y轴 | ✅ 100% |
| 偏心 | 0.5-1.0m | Y轴或倾斜 | ⚠️ 70%? |
| 靠墙 | 1.0-1.5m | 法向或Y轴 | ⚠️ 50%? |
| 角落 | > 1.5m | 多方向 | ❌ 30%? |

**行动**：
- [ ] 在dataset_2k选择10个探针
- [ ] 并行运行10个Level 2验证（3小时）
- [ ] 分析空间模式，决定Level 3架构

---

### 问题4：时间插值能力未测试

**严重性**: 🔴 MEDIUM
**影响**: 决定TPE的实用价值

**问题描述**：

TPE的承诺：可以泛化到未见时刻。但我们完全没有验证！

**当前状态**：
- 训练：6,7,8,9,10,11,12,13,14,15,16,17,18点（13个小时）
- 测试：同样的13个小时
- 插值：❌ 未测试

**风险**：

如果不能插值，TPE只是个过拟合的lookup table，没有泛化价值。

**测试方法**：

```python
# 方案A：时间留出交叉验证
train_hours = [6, 8, 10, 12, 14, 16, 18]  # 奇数小时
test_hours = [7, 9, 11, 13, 15, 17]       # 偶数小时

# 训练TPE（只用train_hours）
F_static, alphas_train = train_tpe(sh_data[train_hours])

# 插值到test_hours
for t_test in test_hours:
    # 插值α(t)
    alpha_interp = cubic_spline_interpolate(
        train_hours, alphas_train, t_test
    )

    # 插值sun_dir
    sun_interp = interpolate_sun_direction(t_test)

    # 预测
    sh_pred = F_static(x_probe + alpha_interp * direction, sun_interp)
    sh_gt = sh_data[t_test]

    error_interp = mse(sh_pred, sh_gt)

# 对比：直接训练 vs 插值
error_direct = mse(F_tpe_full_training, sh_gt)
degradation = (error_interp - error_direct) / error_direct
```

**判定标准**：

| degradation | 结论 |
|-------------|------|
| < 10% | ✅ 插值能力excellent |
| 10%-30% | ⚠️ 插值能力acceptable |
| 30%-50% | ⚠️ 插值能力weak |
| > 50% | ❌ 过拟合，无泛化能力 |

**行动**：
- [ ] 实现时间交叉验证（1小时）
- [ ] 测试cubic spline vs linear插值
- [ ] 如果失败，增加时间平滑正则化

---

## P1 - 重要问题

### 问题5：压缩率计算错误

**严重性**: 🟡 MEDIUM
**影响**: 毕设声称的优势可能不存在

**问题描述**：

文档声称TPE有压缩优势，但单探针情况下**完全不是这样**！

**正确计算**（Level 2单探针）：

| 组件 | 参数量 | 计算 |
|------|--------|------|
| GaussianMixture(K=1) | 19 | means(3) + scales(3) + rotations(4) + latent(9) |
| DecoderMLP | 6784 | (15→64: 960) + (64→64: 4096) + (64→27: 1728) |
| Alphas(T=13) | 13 | 13个标量 |
| **TPE Total** | **6816** | |
| **Direct Storage** | **351** | 13时刻 × 27 SH系数 |

**结论**：TPE比直接存储大**19.4倍**！

**TPE何时有优势**：Level 3多探针

| 场景 | TPE参数 | 直接存储 | 压缩率 |
|------|---------|---------|--------|
| Level 2 (N=1, T=13) | 6816 | 351 | ❌ 0.05x (更大) |
| Level 3-small (N=100, T=13) | 6816 | 35100 | ✅ 5.1x |
| Level 3-medium (N=1000, T=100) | ~15000 | 2.7M | ✅ 180x |
| Level 3-large (N=10000, T=100) | ~15000 | 27M | ✅ 1800x |

**关键洞察**：

TPE的价值不在于单点时间压缩，而在于：
1. **空间共享**：F_static(高斯+decoder)被所有探针共享
2. **时间复杂度**：ε(t)的参数量是O(T)，而非O(NT)

**行动**：
- [ ] 修正文档中的压缩率声称
- [ ] 添加参数量随N, T变化的理论分析
- [ ] 实现Level 3-small (N=100)验证实际压缩率

---

### 问题6：缺少Baseline对比实验

**严重性**: 🟡 MEDIUM
**影响**: 毕设完整性

**问题描述**：

当前只有TPE方法，没有任何baseline对比。无法证明TPE的相对优势。

**应对比的方法**：

**1. 时间插值（Temporal Interpolation）**

```python
# 用B-spline拟合每个SH通道
from scipy.interpolate import splrep, splev

sh_interp_model = []
for ch in range(27):  # 27 SH channels
    # 对第ch个通道，拟合时间曲线
    tck = splrep(hours, sh_gt[:, ch], k=3)  # cubic spline
    sh_interp_model.append(tck)

# 参数量：~5 knots/channel × 27 = 135参数
# vs TPE: 6816参数
```

**2. TDML（时间度量学习）- 毕设已有**

```python
from models.time_encoder import TimeEncoder

time_encoder = TimeEncoder(dim=16)  # 16-dim time embedding
decoder_tdml = DecoderMLP(input_dim=3+16, output_dim=27)

for step in range(1000):
    time_emb = time_encoder(hours_tensor)  # [T, 16]
    sh_pred = decoder_tdml(pos, time_emb)
    loss = mse(sh_pred, sh_gt)
```

**3. 直接MLP**

```python
# 最简单baseline：直接用MLP
mlp_baseline = MLP(input_dim=3+3+1, output_dim=27)
# input: position + sun_dir + hour

for step in range(1000):
    sh_pred = mlp_baseline(position, sun_dir, hour_normalized)
    loss = mse(sh_pred, sh_gt)
```

**对比维度**：

| 方法 | 参数量 | 训练时间 | 推理时间 | 重建误差 | 插值能力 |
|------|--------|---------|---------|---------|---------|
| Direct Storage | 351 | 0s | 0s | 0 | Poor (linear) |
| Spline Interp | 135 | 1s | 0.1ms | ? | Good |
| TDML | ~5000 | 30s | 1ms | ? | Good |
| Direct MLP | ~3000 | 20s | 0.5ms | ? | Medium |
| **TPE (ours)** | 6816 | 60s | 2ms | 0.0039 | ? |

**行动**：
- [ ] 实现Spline baseline (30分钟)
- [ ] 实现TDML baseline (1小时，已有模型)
- [ ] 实现Direct MLP baseline (30分钟)
- [ ] 在相同数据上训练并对比

---

### 问题7：方向约束的物理解释薄弱

**严重性**: 🟡 MEDIUM
**影响**: 理论贡献的说服力

**问题描述**：

当前声称"Cornell Box垂直结构 → Y轴扰动"，但这是**事后解释**（post-hoc rationalization）。

**矛盾证据**：

方向分析结果：
- PCA explained variance = 95.3%（幅度主要在Y方向）
- Cosine similarity = 0.190（方向一致性很低！）

这说明什么？baseline的扰动方向并不一致，但幅度的variance沿Y轴。

**两种解释**：

解释A（物理）：
- Y轴是真实的物理方向
- Baseline的低cos sim是优化噪声
- 方向约束强制了正确的物理先验

解释B（正则化）：
- 方向约束只是降维正则化trick
- 成功是因为reduced overfitting，不是因为物理正确
- Y轴解释是coincidence

**如何区分**：

测试1 - 使用错误方向：
```python
# 用X轴约束（应该是"错误"的）
direction_x = [1, 0, 0]
success_x, metrics_x = train_directional_tpe(direction=direction_x)

# 如果X轴也成功 → 解释B正确（只是正则化效应）
# 如果X轴失败 → 解释A正确（Y轴有物理意义）
```

测试2 - 随机方向：
```python
# 用10个随机方向
for i in range(10):
    random_dir = normalize(np.random.randn(3))
    success, metrics = train_directional_tpe(direction=random_dir)

# 如果所有方向都差不多 → 解释B
# 如果只有Y轴附近好 → 解释A
```

**行动**：
- [ ] 测试X轴、Z轴、随机方向约束 (2小时)
- [ ] 如果都成功，承认是正则化效应
- [ ] 如果只有Y轴成功，探索如何从场景几何预测方向

---

### 问题8：实验可重复性未验证

**严重性**: 🟡 MEDIUM
**影响**: 结果的鲁棒性

**问题描述**：

当前所有实验都是**单次运行**，没有：
- 重复实验（不同random seed）
- Error bars
- 统计显著性检验

**风险**：

我们观察到的"成功"可能是lucky seed。可能：
- 90%的seed失败
- 只有10%的seed侥幸成功
- 我们恰好碰到了lucky case

**测试方法**：

```python
# 多seed实验
results = []
for seed in range(10):
    torch.manual_seed(seed)
    np.random.seed(seed)

    success, metrics = validate_directional_tpe(
        dataset_path,
        direction_mode='pca'
    )

    results.append({
        'seed': seed,
        'success': success,
        'max_norm': metrics['max_perturbation_norm'],
        'correlation': metrics['sun_correlation'],
        'final_loss': metrics['final_recon_loss']
    })

# 统计分析
success_rate = sum(r['success'] for r in results) / len(results)
max_norm_mean = np.mean([r['max_norm'] for r in results])
max_norm_std = np.std([r['max_norm'] for r in results])

print(f"Success rate: {success_rate:.0%}")
print(f"Max norm: {max_norm_mean:.3f} ± {max_norm_std:.3f}m")
```

**判定标准**：

| success_rate | 结论 |
|--------------|------|
| > 80% | ✅ Robust |
| 50%-80% | ⚠️ Moderately robust |
| 20%-50% | ⚠️ Unstable |
| < 20% | ❌ Lucky case, not reliable |

**行动**：
- [ ] 10个seed的重复实验 (4小时)
- [ ] 报告mean ± std
- [ ] 如果不稳定，改进初始化或增加训练步数

---

## P2 - 有益问题

### 问题9：多探针架构未决策

**影响**: Level 3实现的可行性

**4种选项对比**：

**选项A：共享扰动** `ε(t)` for all probes

```python
# Pros: 最简单，T个参数
# Cons: 过度简化，不同位置不应同步移动
```

**选项B：独立扰动** `ε_i(t)` per probe

```python
# Pros: 最灵活
# Cons: N×T×3参数，完全失去压缩优势
```

**选项C：插值扰动** `ε(x, t)` 连续场

```python
# 用K个control points（K << N）
control_points = [ε_1(t), ε_2(t), ..., ε_K(t)]

for probe in probes:
    # RBF interpolation
    weights = rbf_weights(probe, control_point_positions)
    ε_probe = sum(w_k * ε_k(t) for k in range(K))
    sh_pred = F_static(probe + ε_probe)

# Pros: K×T×3参数（K~10），可接受
# Cons: 如何选control points？如何插值？
```

**选项D：扰动网络** `ε_net(x, t)`

```python
epsilon_net = MLP(input=position+time_enc, output=3)
for probe in probes:
    ε_probe = epsilon_net(probe, t)
    sh_pred = F_static(probe + ε_probe)

# Pros: 灵活且参数可控
# Cons: 回到神经网络，失去TPE的简洁性
```

**推荐**：

先测试C（插值），如果失败考虑D。

**行动** (P2低优先级)：
- [ ] 实现选项C的prototype
- [ ] 在dataset_2k的100探针上测试
- [ ] 对比参数量和重建质量

---

### 问题10：方向推断自动化

**影响**: 工程便利性

**当前流程**：
1. 运行baseline TPE
2. PCA分析
3. 用主方向重新训练

**理想流程**：

```python
direction = infer_direction_from_scene(
    scene_geometry,
    probe_position,
    sun_trajectory
)

# One-shot训练
train_directional_tpe(direction=direction)
```

**方向推断方法**：

```python
def infer_direction_from_scene(scene_voxels, probe_pos, sun_traj):
    # 1. 场景对称轴
    symmetry_axes = analyze_pca_symmetry(scene_voxels)

    # 2. 太阳运动主方向
    sun_motion = pca(sun_traj)[0]

    # 3. 探针到最近表面
    nearest_surface = raycast_to_nearest_surface(probe_pos)
    surface_normal = nearest_surface.normal

    # 4. 组合决策（需调权重）
    direction = (
        0.5 * symmetry_axes[0] +
        0.3 * sun_motion +
        0.2 * surface_normal
    )

    return normalize(direction)
```

**行动** (P2低优先级)：
- [ ] 实现场景几何分析工具
- [ ] 在多个场景测试推断准确度
- [ ] 如果准确度> 80%，替换PCA方法

---

### 问题11：残差网络架构

**影响**: hybrid model的性能

如果实验F发现残差大（||R|| > 0.005），需要hybrid model：

```python
class HybridTPE(nn.Module):
    def __init__(self):
        self.gaussian = GaussianMixture(K, latent_dim)
        self.decoder = DecoderMLP()
        self.residual_net = ResidualMLP(
            input_dim=3+3+3+16,  # pos + sun + epsilon + time_enc
            hidden_dim=32,
            output_dim=27
        )

    def forward(self, x, t, epsilon):
        # TPE prediction
        sh_tpe = self.decoder(
            self.gaussian(x + epsilon),
            x + epsilon,
            sun_ref
        )

        # Residual correction
        time_emb = time_encoder(t)
        residual = self.residual_net(x, sun_t, epsilon, time_emb)

        return sh_tpe + residual
```

**参数量对比**：

| 组件 | 参数 |
|------|------|
| TPE (F_static + alphas) | 6816 |
| Residual net | ~2000 |
| **Hybrid total** | ~8816 |
| **vs TDML** | ~5000 |

**行动** (取决于实验F结果)：
- [ ] 如果||R|| > 0.005，实现hybrid model
- [ ] 评估是否值得额外的2000参数

---

### 问题12：理论形式化

**影响**: 学术贡献的深度

**当前状态**：定性描述

"强对称场景 → 小扰动"

**应该有**：定量理论

**理论框架草案**：

定义场景对称度：
```
S(scene, x) = max eigenvalue of Hessian(F_ref(x))
```

定理（待证明）：
```
在Lipschitz常数为L的场景中，TPE扰动满足：
||ε(t)||_max ≤ C · ||ΔF||_max / L

其中：
- ||ΔF||_max = max_t ||F(x, t) - F(x, t_ref)||
- C是与场景对称度S相关的常数
- L = Lipschitz constant of F_ref
```

**行动** (P2低优先级)：
- [ ] 数学推导上界公式
- [ ] 实验验证多个场景
- [ ] 写理论章节for论文

---

## 数据集状态与可用性

### 现有数据集概览

| 数据集 | 探针数 | 时刻数 | 场景 | 状态 | 可用性 |
|--------|--------|--------|------|------|--------|
| **dataset_2k** | 1728 | 6 (6,9,12,15,18,21) | ? | ✅ 完整 | **可用于多探针实验** |
| **dataset_15k** | 13824 | 6 | ? | ✅ 完整 | 可用但数据量大 |
| **method_test_v1** | ? | 6 | ? | ⚠️ 未知 | 需检查 |
| **level1_tpe/shadow_plane** | 25 | 11 | shadow_plane | ❌ 缺陷 | 不可用（常量光） |
| **level2_tpe/cornell-box** | 1 | 13 | cornell-box | ✅ 完整 | 当前验证用 |

### 数据集缺口

**缺少的关键数据**：

1. **其他场景的Level 2数据**：
   - classroom（已有场景文件，未生成数据）
   - house（已有场景文件，未生成数据）
   - 无法测试场景泛化性

2. **更大时间跨度**：
   - 当前：6-21点（白天）
   - 需要：0-24点（包括夜晚）
   - 测试间接光主导情况

3. **不同材质**：
   - 当前：纯diffuse
   - 需要：glossy, specular测试

### 数据集行动项

**立即可做**（利用现有数据）：
- [ ] 检查dataset_2k的场景和质量
- [ ] 如果是Cornell Box，立即用于多位置验证
- [ ] 如果不是，了解是什么场景

**未来生成**（需要渲染时间）：
- [ ] Classroom Level 2数据（~1小时渲染）
- [ ] House Level 2数据（~2小时渲染）
- [ ] Cornell Box夜晚数据（0-6, 18-24点，~30分钟）

---

## 实验优先级与行动计划

### 立即执行（今天）- P0

**实验F - Ground Truth验证** ⚡
**预期时间**: 2-3小时
**成本**: 4个扰动位置渲染（~20分钟）+ 分析代码（~1小时）

```bash
# Step 1: 实现渲染脚本
python scripts/render_perturbed_positions.py \
    --scene cornell-box \
    --base-position 0,1,0 \
    --perturbations validation_results/tpe_directional_validation_results.npz \
    --hours 6,9,15,18 \
    --output output/experiment_f_ground_truth

# Step 2: 分析
python scripts/analyze_ground_truth_tpe.py \
    --experiment-f-data output/experiment_f_ground_truth \
    --validation-results validation_results/
```

**判定标准**：
- ✅ Pass: error_tpe_assumption < 0.01
- ⚠️ Caution: 0.01 < error < 0.05
- ❌ Fail: error > 0.05

**如果Fail**: 立即转向residual model或multi-anchor

---

### 本周执行 - P0+P1

**多位置验证** (P0)
**预期时间**: 3-4小时

```bash
# 检查dataset_2k
python scripts/inspect_dataset.py --dataset dataset_2k

# 选择10个探针，并行验证
python scripts/batch_validate_tpe.py \
    --dataset dataset_2k \
    --num-probes 10 \
    --probe-selection-strategy spatial_sampling \
    --parallel 4
```

**时间插值测试** (P0)
**预期时间**: 1-2小时

```bash
# 交叉验证
python scripts/validate_tpe_interpolation.py \
    --dataset level2_tpe/cornell-box_test \
    --cv-folds 2 \  # train on odd hours, test on even
    --interpolation-method cubic_spline
```

**Baseline对比** (P1)
**预期时间**: 3-4小时

```bash
# 实现并训练3个baseline
python scripts/train_baseline_spline.py
python scripts/train_baseline_tdml.py
python scripts/train_baseline_mlp.py

# 对比
python scripts/compare_methods.py \
    --methods tpe,spline,tdml,mlp \
    --metrics params,time,error,interpolation
```

**重复实验** (P1)
**预期时间**: 4小时（可后台运行）

```bash
# 10个seed
for seed in {0..9}; do
    python scripts/validate_tpe_level2_directional.py \
        --seed $seed \
        --output validation_results/seed_$seed &
done
wait

# 统计分析
python scripts/analyze_robustness.py \
    --results validation_results/seed_*
```

---

### 下周执行 - P2

**Level 3 prototype** (如果P0+P1都通过)
**预期时间**: 2天

```bash
# 小规模Level 3
python scripts/train_level3_prototype.py \
    --dataset dataset_2k \
    --num-probes 100 \
    --num-gaussians 10 \
    --perturbation-mode interpolated \  # 选项C
    --control-points 10
```

**方向推断自动化**
**预期时间**: 1天

```python
# 实现场景分析模块
python scripts/develop_direction_predictor.py \
    --train-scenes cornell-box,classroom,house \
    --test-on-new-scenes
```

---

## 附录：详细分析

### A. 参数量详细分解

**GaussianMixture(K=1, latent_dim=9)**:
```
means: [K, 3] = 3
scales: [K, 3] = 3
rotations: [K, 4] = 4 (quaternion)
latent_codes: [K, 9] = 9
-------------------------
Total: 19 parameters
```

**DecoderMLP(input=15, hidden=64, layers=2, output=27)**:
```
Layer 1: (15 → 64) = 15*64 + 64 = 960 + 64 = 1024
Layer 2: (64 → 64) = 64*64 + 64 = 4096 + 64 = 4160
Layer 3: (64 → 27) = 64*27 + 27 = 1728 + 27 = 1755
-------------------------
Total: 6939 parameters（含bias）
简化计算: ~6784（不含bias）
```

**Directional TPE (T=13)**:
```
Alphas: [T] = 13
-------------------------
Total TPE: 19 + 6784 + 13 = 6816 parameters
```

### B. Cornell Box几何分析

```
场景尺寸: 2m × 2m × 2m
材质: Lambertian diffuse (ρ ≈ 0.7)
光源: Directional sun (top window)
探针: [0, 1, 0] (geometric center)

对称性：
- X-Z平面对称（除了墙面颜色）
- Y轴是唯一的特殊轴（重力方向，光源入射）
- 红墙(X=+1) vs 绿墙(X=-1)打破X对称
- 但对光照分布，颜色影响小（diffuse混合）

物理推理：
太阳高度角变化 → 顶部vs侧面光照比例
探针Y坐标变化 → 感受到顶部/侧面相对强度变化
→ 等效于太阳高度角变化的效果
→ 因此ε(t)主要沿Y轴
```

### C. TPE理论基础回顾

**Taylor展开（空间）**：
```
F(x + ε, t) ≈ F(x, t) + ∇_x F(x, t) · ε + O(||ε||²)
```

**TPE假设**：
```
F(x, t) ≈ F(x + ε(t), t_ref)
```

**组合**：
```
F(x, t) ≈ F(x, t_ref) + ∇_x F(x, t_ref) · ε(t)
```

**有效性条件**：
1. ||ε(t)|| 足够小（Taylor一阶近似有效）
2. ∇_x F在t_ref附近相对稳定（时间变化主要是平移，不是形状变化）
3. F的Hessian不太大（二阶项可忽略）

**数学等价**：
```
∂F/∂t ≈ ∇_x F · (dε/dt)
```
这是**advection equation**的形式，说明场F沿着"流"ε(t)演化。

---

## 总结

### 核心发现

1. **致命缺陷**：缺少Ground Truth验证（实验F）
2. **压缩悖论**：单探针TPE反而比直接存储大19倍
3. **泛化盲区**：空间、时间、场景泛化性完全未知
4. **理论薄弱**：物理解释是post-hoc，缺少预测能力

### 最紧急行动

| 实验 | 时间 | 风险 | 优先级 |
|------|------|------|--------|
| 实验F (Ground Truth) | 2-3h | 可能推翻TPE假设 | 🔴🔴🔴 P0 |
| 多位置验证 | 3-4h | 可能发现单点偶然成功 | 🔴🔴 P0 |
| 时间插值测试 | 1-2h | 可能发现过拟合 | 🔴 P0 |
| Baseline对比 | 3-4h | 可能发现TPE不是最优 | 🟡 P1 |

### 如果实验F失败

**Plan B - Hybrid Model**:
```
F(x, t) = F_static(x + ε(t)) + R_net(x, t, ε)
```

**Plan C - Multi-Anchor TPE**:
```
F(x, t) = Σ w_i(t) · F_static_i(x + ε_i(t))
```

**Plan D - 回归TDML**:
```
F(x, t) = F_baseline(x, TimeEnc(t))
```

---

**文档版本**: v1.0
**作者**: Claude (深度思考模式)
**审核建议**: 与导师讨论实验F的urgency
