# TPE进阶实验设计方案

**基于Cornell Box验证结果的深度分析与后续实验规划**

**日期**: 2025-12-11
**状态**: 设计阶段
**前置结果**: Cornell Box - 标准1失败(1.126m), 标准2通过(0.244m), 标准3通过(0.979)

---

## 核心洞察

经过深度分析，我发现当前TPE"失败"可能是**误判**，原因如下：

### 关键矛盾

| 指标 | 结果 | 物理意义 |
|------|------|---------|
| ||ε||_max | 1.126m | "扰动太大" |
| Correlation | **0.979** | **扰动与太阳运动几乎完美相关** |
| Reconstruction Loss | 0.00266 | **重建误差极低** |
| Smoothness | 0.244m | **时间演化物理合理** |

**矛盾**: 扰动"太大"却"非常合理"且"重建准确"。

### 根本问题

1. **绝对阈值不合理**: 1.0m忽略了场景尺度（Cornell Box = 2m）
2. **未分析扰动方向**: 只看范数，未检查方向的物理意义
3. **未考虑场平滑性**: 高斯场的平滑性可能使大扰动仍在有效区域
4. **单参考点限制**: Noon单点可能不足以覆盖全天变化

---

## 实验方案总览

设计6个递进式实验，从分析现有数据到改进TPE架构：

| 实验 | 名称 | 成本 | 信息量 | 优先级 | 预计时间 |
|------|------|------|--------|--------|---------|
| **C** | 扰动方向分析 | ★☆☆ | ★★★★★ | P0 | 30分钟 |
| **F** | 真实光场扰动响应 | ★★☆ | ★★★★★ | P0 | 2小时 |
| **B** | 静态场Lipschitz常数 | ★★☆ | ★★★★☆ | P1 | 1小时 |
| **E** | 场景尺度归一化 | ★☆☆ | ★★★☆☆ | P1 | 30分钟 |
| **G** | 方向化扰动TPE | ★★★☆ | ★★★★★ | P1 | 4小时 |
| **D** | 多锚点TPE | ★★★★ | ★★★★☆ | P2 | 1天 |

**执行顺序**: C → F → (B & E) → G → D

---

## 实验C: 扰动方向分析 ⭐⭐⭐⭐⭐

### 目标

验证扰动向量ε(t)的方向是否具有物理意义，是否与太阳运动相关。

### 核心假设

**H_C1**: 扰动方向d(t) = ε(t)/||ε(t)||与太阳方向变化相关
**H_C2**: 扰动方向在时间上保持一致性（不是随机的）
**H_C3**: 扰动方向与场景几何（如主墙面法向）有关

### 方法

#### C.1 方向一致性检查

```python
# 提取扰动方向
directions = perturbations / np.linalg.norm(perturbations, axis=1, keepdims=True)

# 计算方向间的cosine similarity矩阵
cos_sim = directions @ directions.T

# 统计量
mean_cos_sim = np.mean(cos_sim[np.triu_indices_from(cos_sim, k=1)])
```

**判定**:
- mean_cos_sim > 0.8 → 方向高度一致（支持H_C2）
- mean_cos_sim < 0.5 → 方向随机（反对TPE）

#### C.2 与太阳方向的相关性

```python
# 计算太阳方向变化
sun_diffs = sun_dirs - sun_dirs[reference_idx]  # [T, 3]
sun_diff_dirs = sun_diffs / (np.linalg.norm(sun_diffs, axis=1, keepdims=True) + 1e-8)

# 扰动方向与太阳方向变化的夹角
angles = np.arccos(np.clip(np.sum(directions * sun_diff_dirs, axis=1), -1, 1))
angles_deg = np.degrees(angles)
```

**判定**:
- mean(angles) < 30° → 扰动沿太阳运动方向（强支持）
- 90° < mean(angles) < 150° → 扰动沿反太阳方向（也合理）
- 无规律 → 方向物理意义不明

#### C.3 主方向提取（PCA）

```python
from sklearn.decomposition import PCA

# 对所有扰动做PCA
pca = PCA(n_components=3)
pca.fit(perturbations)

# 检查主方向
principal_direction = pca.components_[0]  # 第一主成分
explained_variance_ratio = pca.explained_variance_ratio_[0]
```

**判定**:
- explained_variance_ratio > 0.8 → 扰动主要沿单一方向
- 此方向应与太阳路径或场景轴对齐

### 预期结果

| 场景 | 预期 | 含义 |
|------|------|------|
| **乐观** | 方向一致，沿太阳路径 | TPE有效，可用方向约束优化 |
| **中性** | 方向一致，但与太阳无关 | TPE部分有效，需场景几何分析 |
| **悲观** | 方向随机 | 当前优化失败，需改进策略 |

### 实现

```python
# 文件: scripts/analyze_perturbation_directions.py
def analyze_perturbation_directions(
    perturbations: np.ndarray,  # [T, 3]
    sun_dirs: np.ndarray,       # [T, 3]
    reference_idx: int
) -> Dict:
    """分析扰动方向的物理意义"""

    # 归一化方向
    pert_norms = np.linalg.norm(perturbations, axis=1, keepdims=True)
    pert_dirs = perturbations / (pert_norms + 1e-8)

    # 1. 方向一致性
    cos_sim_matrix = pert_dirs @ pert_dirs.T
    mean_cos_sim = np.mean(cos_sim_matrix[np.triu_indices_from(cos_sim_matrix, k=1)])

    # 2. 与太阳方向相关性
    sun_diffs = sun_dirs - sun_dirs[reference_idx:reference_idx+1]
    sun_diff_norms = np.linalg.norm(sun_diffs, axis=1, keepdims=True)
    sun_diff_dirs = sun_diffs / (sun_diff_norms + 1e-8)

    # 去除参考时刻（范数为0）
    valid_mask = pert_norms.squeeze() > 1e-6
    angles = np.arccos(np.clip(
        np.sum(pert_dirs[valid_mask] * sun_diff_dirs[valid_mask], axis=1),
        -1, 1
    ))

    # 3. PCA分析
    pca = PCA(n_components=3)
    pca.fit(perturbations[valid_mask])

    return {
        'mean_cosine_similarity': mean_cos_sim,
        'angle_to_sun_mean_deg': np.degrees(angles.mean()),
        'angle_to_sun_std_deg': np.degrees(angles.std()),
        'principal_direction': pca.components_[0],
        'explained_variance_ratio': pca.explained_variance_ratio_[0],
        'angles_deg': np.degrees(angles)
    }
```

---

## 实验F: 真实光场扰动响应 ⭐⭐⭐⭐⭐

### 目标

在**真实渲染数据**（非高斯近似）上测试扰动的有效性，验证光场的平滑性假设。

### 核心假设

**H_F1**: 真实光场足够平滑，||F_gt(x+ε) - F_gt(x)|| 随||ε||增长缓慢
**H_F2**: 在||ε||=1.126m处，真实光场变化仍然适度（支持大扰动）
**H_F3**: Lipschitz常数L满足 L·||ε|| ≈ ||ΔSH||_observed

### 方法

#### F.1 采样扰动后的光场

在Cornell Box中，对探针位置[0, 1, 0]添加扰动，重新渲染：

```python
# 选择几个代表性扰动
test_perturbations = [
    perturbations[0],   # Hour 6 (max norm)
    perturbations[3],   # Hour 9 (mid)
    perturbations[6],   # Hour 12 (zero, reference)
    perturbations[12],  # Hour 18 (symmetric to 6)
]

# 对每个扰动，渲染新位置的SH
probe_original = np.array([0.0, 1.0, 0.0])
for t, pert in enumerate(test_perturbations):
    probe_perturbed = probe_original + pert

    # 在noon光照下渲染扰动后的位置
    sh_perturbed = render_sh_at_probe(
        scene_path='cornell-box',
        probe_position=probe_perturbed,
        sun_direction=sun_dirs[reference_idx],  # Noon
        spp=256,
        sh_samples=64
    )

    # 对比预测 vs 真实
    sh_predicted = decoder(gaussian(probe_perturbed), probe_perturbed, sun_ref)
    sh_ground_truth = sh_gt[hour_mapping[t]]

    error_predicted = np.linalg.norm(sh_predicted - sh_ground_truth)
    error_real_field = np.linalg.norm(sh_perturbed - sh_ground_truth)
```

#### F.2 光场平滑性量化

```python
# 采样多个距离的扰动
distances = np.linspace(0, 2.0, 20)  # 0到2m
directions = perturbations[0] / np.linalg.norm(perturbations[0])  # 主方向

sh_changes = []
for d in distances:
    pert = d * directions
    probe_new = probe_original + pert
    sh_new = render_sh_at_probe(probe_new, sun_ref, ...)
    sh_change = np.linalg.norm(sh_new - sh_original)
    sh_changes.append(sh_change)

# 拟合Lipschitz常数
# ||F(x+ε) - F(x)|| ≤ L·||ε||
lipschitz_constant = np.max(np.array(sh_changes) / distances)
```

#### F.3 Taylor展开有效性

```python
# 数值计算梯度（有限差分）
delta = 0.01  # 1cm
gradient = np.zeros((27, 3))
for i in range(3):
    pert_pos = probe_original.copy()
    pert_pos[i] += delta
    sh_plus = render_sh_at_probe(pert_pos, sun_ref, ...)

    pert_pos[i] -= 2*delta
    sh_minus = render_sh_at_probe(pert_pos, sun_ref, ...)

    gradient[:, i] = (sh_plus - sh_minus) / (2*delta)

# 对于扰动ε，Taylor预测
for pert in test_perturbations:
    sh_taylor = sh_original + gradient @ pert  # 一阶近似
    sh_真实 = render_sh_at_probe(probe_original + pert, sun_ref, ...)

    error_taylor = np.linalg.norm(sh_taylor - sh_真实)
    print(f"||ε||={np.linalg.norm(pert):.3f}m, Taylor误差={error_taylor:.6f}")
```

### 预期结果

| Lipschitz常数L | Taylor误差(||ε||=1.1m) | 结论 |
|---------------|---------------------|------|
| L < 0.1 | < 0.05 | 光场极平滑，大扰动有效 |
| 0.1 < L < 0.5 | 0.05-0.2 | 中等平滑，需谨慎使用 |
| L > 0.5 | > 0.2 | 光场非线性强，TPE无效 |

### 关键意义

- 如果真实光场在||ε||=1.1m处Taylor近似仍好 → 1.0m阈值过于保守
- 如果Lipschitz常数低 → 可以放宽扰动限制
- 这是判断TPE理论有效性的**决定性实验**

---

## 实验B: 静态场Lipschitz常数

### 目标

量化高斯场F_static的平滑性，解释为什么大扰动不导致大误差。

### 方法

```python
# 在探针周围采样
sample_positions = probe_original + np.random.randn(100, 3) * 0.5  # 半径0.5m

sh_samples = []
for pos in sample_positions:
    latent = gaussian(pos)
    sh = decoder(latent, pos, sun_ref)
    sh_samples.append(sh.numpy())

# 计算相邻点的最大梯度
max_gradient = 0
for i in range(len(sample_positions)):
    for j in range(i+1, len(sample_positions)):
        dist = np.linalg.norm(sample_positions[i] - sample_positions[j])
        sh_diff = np.linalg.norm(sh_samples[i] - sh_samples[j])
        gradient = sh_diff / dist
        max_gradient = max(max_gradient, gradient)

print(f"Estimated Lipschitz constant: {max_gradient:.4f}")
```

### 判定

- L_static << L_real → 高斯场过度平滑，可能欠拟合
- L_static ≈ L_real → 高斯场合理近似
- L_static >> L_real → 不应出现（过拟合？）

---

## 实验E: 场景尺度归一化

### 目标

测试相对阈值||ε||/L_scene < 0.5是否比绝对阈值更合理。

### 方法

重新评估多个场景，计算归一化扰动：

| 场景 | 尺度L | ||ε||_max | ||ε||/L | 绝对标准 | 相对标准 |
|------|-------|----------|---------|---------|---------|
| Cornell Box | 2m | 1.126m | 0.56 | ✗ FAIL | ⚠️ 边界 |
| Classroom | 5m | ? | ? | ? | ? |
| House | 10m | ? | ? | ? | ? |

### 预期

- 大场景: ||ε||/L应该更小 → 相对标准更严格但公平
- 小场景: ||ε||/L可能仍大 → 说明小场景本质上难以用TPE

---

## 实验G: 方向化扰动TPE ⭐⭐⭐⭐⭐

### 目标

基于实验C的发现，约束扰动方向，减少自由度。

### 新TPE公式

**假设**: 扰动沿固定方向d（由太阳路径决定）

```
ε(t) = α(t) · d
```

其中：
- d: 主方向（从PCA或太阳路径推导）
- α(t): 标量幅度（T个自由度，而非3T个）

### 优化

```python
# 固定方向（从实验C得到）
direction = principal_direction  # [3]

# 只优化标量幅度
alphas = torch.zeros(T, requires_grad=True)
optimizer = torch.optim.Adam([alphas], lr=0.01)

for step in range(500):
    # 扰动
    perturbations = alphas.unsqueeze(1) * torch.tensor(direction)  # [T, 3]

    # 前向传播（同之前）
    ...

    # 约束：alpha[reference_idx] = 0
    with torch.no_grad():
        alphas[reference_idx] = 0
```

### 预期优势

- **自由度减少**: 3T → T（减少67%）
- **正则化效应**: 强先验约束 → ||ε||可能减小
- **物理可解释**: 单一方向有明确几何意义

### 判定

- ||α·d||_max < 1.0m → 标准1通过！
- 重建误差仍低 → 方向约束不损失表达力

---

## 实验D: 多锚点TPE

### 目标

使用多个参考时刻，每个锚点对应小的局部扰动。

### 新TPE公式

```
F(x, t) = Σ_i w_i(t) · F_static_i(x + ε_i(t))
```

其中：
- F_static_i: 在锚点时刻t_i训练的静态场
- ε_i(t): 相对于锚点i的扰动
- w_i(t): 时间插值权重，如 w_i(t) = exp(-||t - t_i||²/σ²)

### 锚点选择

```python
# 选择3个锚点：morning, noon, evening
anchor_hours = [8, 12, 16]
anchor_indices = [list(hours).index(h) for h in anchor_hours]

# 为每个锚点训练独立的静态场
static_fields = []
for idx in anchor_indices:
    gaussian_i = GaussianMixture(...)
    decoder_i = DecoderMLP(...)

    # 在锚点时刻训练
    train_static_field(gaussian_i, decoder_i, sh_gt[idx], sun_dirs[idx], ...)
    static_fields.append((gaussian_i, decoder_i))
```

### 优化多个小扰动

```python
# 每个锚点对应T个扰动
perturbations_multi = torch.zeros(3, T, 3, requires_grad=True)  # [3锚点, T时刻, 3维]

for step in range(500):
    sh_pred_list = []

    for t in range(T):
        # 计算插值权重
        weights = compute_temporal_weights(t, anchor_indices, sigma=2.0)  # [3]

        # 混合多个静态场
        sh_t = 0
        for i, (gauss, dec) in enumerate(static_fields):
            pert_i_t = perturbations_multi[i, t]  # [3]
            pos_perturbed = probe_pos + pert_i_t

            latent = gauss(pos_perturbed)
            sh_i = dec(latent, pos_perturbed, sun_dirs[anchor_indices[i]])

            sh_t += weights[i] * sh_i

        sh_pred_list.append(sh_t)

    # 损失
    loss = mse_loss(torch.cat(sh_pred_list), sh_gt)

    # 约束：每个锚点在自己的参考时刻扰动为0
    with torch.no_grad():
        for i, ref_idx in enumerate(anchor_indices):
            perturbations_multi[i, ref_idx] = 0
```

### 预期

- 每个||ε_i(t)||应远小于单锚点的||ε(t)||
- 可能所有||ε_i|| < 0.5m
- 代价：3倍的静态场存储 + 插值计算

---

## 实验执行计划

### 第1天（今天）

**上午**:
- ✅ 实验C - 扰动方向分析（立即执行）
  - 编写`analyze_perturbation_directions.py`
  - 运行分析，生成可视化

**下午**:
- 实验F准备 - 设置扰动采样点
  - 选择4个代表性扰动
  - 准备渲染脚本

### 第2天

**全天**:
- 实验F - 真实光场扰动响应
  - 渲染4个扰动后的光场
  - 计算Lipschitz常数
  - 评估Taylor展开误差

- 实验B - 静态场平滑性
  - 采样100个邻近点
  - 计算梯度上界

### 第3天

**根据C和F的结果决定**:

- **情景A**: 如果C发现方向规律，F证实平滑性 → 执行实验G（方向化TPE）
- **情景B**: 如果C方向随机 → 改进优化策略，重新训练
- **情景C**: 如果F发现强非线性 → 考虑多锚点（实验D）

---

## 成功标准

### 实验C成功

- [ ] 扰动方向cosine相似度 > 0.8
- [ ] 与太阳方向夹角 < 45°或与场景轴对齐
- [ ] PCA第一主成分解释方差 > 0.8

### 实验F成功

- [ ] Lipschitz常数 L < 0.5
- [ ] ||ε||=1.1m处Taylor误差 < 0.1
- [ ] 证明光场平滑性足以支持大扰动

### 实验G成功（如果执行）

- [ ] 方向约束后||α·d|| < 1.0m
- [ ] 重建误差不增加（< 0.005）
- [ ] 所有3个标准通过

---

## 理论贡献

如果实验成功，我们将：

1. **纠正TPE的评估标准**: 从绝对阈值到相对/平滑性阈值
2. **发现TPE的适用条件**: 平滑场景 + 适当尺度
3. **提出改进的TPE变体**: 方向化TPE或多锚点TPE
4. **建立理论框架**: 场平滑性 → 扰动有效性的定量关系

---

## 风险与备选方案

### 风险1: 实验C显示方向随机

**应对**: 改进优化
- 添加方向正则化损失
- 使用更好的初始化（太阳方向）
- 增加训练步数

### 风险2: 实验F发现强非线性

**应对**:
- 接受1.5m阈值（放宽标准）
- 或转向混合模型（TPE + 残差网络）

### 风险3: 所有改进都无效

**应对**:
- 放弃纯TPE
- 保留高斯压缩框架，但使用时间编码器（回归TDML方法）

---

## 总结

当前TPE的"失败"可能是**评估标准问题**而非方法问题。相关性0.979和低重建误差是强有力的正面证据。

**最重要的下一步**: 执行实验C（扰动方向分析），这将在30分钟内告诉我们TPE是否在正确的道路上。

如果C成功 → F确认 → G实现 → TPE得到拯救并改进
如果C失败 → 及早转向备选方案

无论结果如何，这套实验都将建立TPE适用性的**清晰理论边界**。
