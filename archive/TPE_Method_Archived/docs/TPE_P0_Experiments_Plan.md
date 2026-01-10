# TPE P0级紧急实验详细规划

**目的**: 系统验证TPE假设的物理正确性、泛化能力和相对优势
**优先级**: P0 - 可能推翻现有结论
**预计总时间**: 12-15小时
**执行顺序**: F → (2+3并行) → 4

---

## 实验依赖关系图

```
实验F (Ground Truth验证)
   ↓ 提供真实数据
   ├→ 实验2 (残差分析) ← 使用实验F的渲染数据
   └→ 实验3 (空间泛化) ← 独立，可并行
        ↓
   实验4 (时间插值对比) ← 需要baseline实现
```

---

## 实验1: Ground Truth验证（实验F）⚡⚡⚡

### 核心问题

**当前盲点**: 我们只验证了"F_static(x+ε)能拟合数据"，从未验证"F_static(x+ε)是否物理正确"。

**需要回答**:
1. TPE预测 F_static(x + ε(t)) 是否真的近似 真实光场 F_gt(x, t)？
2. 静态场 F_static 在扰动位置的泛化能力如何？
3. 残差 R = F_gt(x, t) - F_static(x + ε(t)) 的大小？

### 实验设计

#### 阶段1: 数据收集（渲染）

选择4个代表性时刻进行真实渲染：

| 时刻 | 太阳高度 | ε范数 | 理由 |
|------|---------|-------|------|
| 6:00 | 低 | 0.77m (max) | 最大扰动 |
| 9:00 | 中 | 0.24m | 中等扰动 |
| 15:00 | 中-高 | 0.31m | 对称测试点 |
| 18:00 | 低 | 0.73m | 晚间最大扰动 |

每个时刻渲染2个位置：

```python
for hour in [6, 9, 15, 18]:
    epsilon = optimized_perturbations[hour_to_idx(hour)]

    # 位置A: 扰动后的位置，参考光照（noon）
    probe_A = original_position + epsilon  # e.g., [0, 1.77, 0]
    sun_A = sun_direction_noon
    SH_A = render(probe_A, sun_A, spp=256)

    # 位置B: 原始位置，时刻hour的光照
    probe_B = original_position  # [0, 1, 0]
    sun_B = sun_direction_hour
    SH_B = render(probe_B, sun_B, spp=256)
```

**渲染成本**: 8个probe-sun组合 × 5分钟 = **40分钟**

#### 阶段2: 误差分解

计算3种误差：

```python
# 加载已优化的TPE模型
F_static = load_trained_static_field()  # gaussian + decoder
epsilon = load_optimized_perturbations()

for hour in [6, 9, 15, 18]:
    # 1. TPE预测
    SH_tpe = F_static(original_position + epsilon[hour])

    # 2. 真实渲染：扰动位置，参考光照
    SH_gt_perturbed = load_rendered_SH(f"perturbed_{hour}_noon.npz")

    # 3. 真实渲染：原始位置，时刻hour光照
    SH_gt_time_t = load_rendered_SH(f"original_{hour}.npz")

    # ===== 误差分解 =====

    # Error 1: 静态场泛化误差（F_static在扰动位置的准确度）
    E_generalization = ||SH_gt_perturbed - SH_tpe||

    # Error 2: TPE假设误差（物理假设是否成立）
    E_tpe_assumption = ||SH_gt_time_t - SH_gt_perturbed||

    # Error 3: 总重建误差（当前优化目标）
    E_total = ||SH_gt_time_t - SH_tpe||

    # ===== 理论关系验证 =====
    # E_total ≈ E_generalization + E_tpe_assumption (如果独立)

    print(f"\nHour {hour}:")
    print(f"  E_generalization: {E_generalization:.6f}")
    print(f"  E_tpe_assumption: {E_tpe_assumption:.6f}")
    print(f"  E_total:          {E_total:.6f}")
    print(f"  Sum check:        {E_generalization + E_tpe_assumption:.6f}")
```

### 成功标准

| 指标 | 优秀 | 可接受 | 失败 | 含义 |
|------|------|--------|------|------|
| E_tpe_assumption | < 0.005 | 0.005-0.02 | > 0.02 | TPE物理假设有效性 |
| E_generalization | < 0.005 | 0.005-0.01 | > 0.01 | F_static泛化能力 |
| E_total | < 0.010 | 0.010-0.03 | > 0.03 | 整体重建质量 |

**决策树**:

```
if E_tpe_assumption < 0.005 and E_generalization < 0.005:
    ✅ TPE完全成功，继续Level 3
elif E_tpe_assumption < 0.02:
    if E_generalization > 0.01:
        ⚠️ F_static表达力不足，需要更大网络或更多高斯
    else:
        ⚠️ TPE近似合理，考虑添加小残差网络
elif E_tpe_assumption > 0.02:
    ❌ TPE假设不成立，需要multi-anchor或转向其他方法
```

### 实现清单

```bash
# 文件1: scripts/experiment_f_render_ground_truth.py
# 功能：渲染8个probe-sun组合
python experiment_f_render_ground_truth.py \
    --scene cornell-box \
    --base-position 0,1,0 \
    --perturbations-file ../data_generation/output/level2_tpe/cornell-box_test/validation_results/tpe_directional_validation_results.npz \
    --test-hours 6,9,15,18 \
    --spp 256 \
    --num-sh-samples 64 \
    --output experiment_f_data

# 文件2: scripts/experiment_f_analyze_errors.py
# 功能：误差分解和分析
python experiment_f_analyze_errors.py \
    --experiment-f-data experiment_f_data/ \
    --trained-model validation_results/ \
    --output experiment_f_results.npz
```

### 预计时间

- 实现渲染脚本: 30分钟
- 渲染执行: 40分钟
- 实现分析脚本: 30分钟
- 分析和可视化: 20分钟
- **总计: 2小时**

---

## 实验2: 残差模型必要性量化

**依赖**: 实验F的渲染数据

### 核心问题

TPE是否足够，还是需要残差修正？

```
Pure TPE:    F(x, t) = F_static(x + ε(t))
Hybrid TPE:  F(x, t) = F_static(x + ε(t)) + R(x, t, ε)
```

### 实验设计

#### 方法1: 直接估计残差（使用实验F数据）

```python
# 从实验F获取真实残差
residuals = []
for hour in [6, 9, 15, 18]:
    # 真实残差
    R_true = SH_gt_time_t[hour] - SH_gt_perturbed[hour]
    residuals.append(R_true)

# 统计分析
R_mean = np.mean([np.linalg.norm(r) for r in residuals])
R_max = np.max([np.linalg.norm(r) for r in residuals])
R_std = np.std([np.linalg.norm(r) for r in residuals])

# 残差模式分析
R_matrix = np.array(residuals)  # [4, 27]
pca = PCA(n_components=3)
pca.fit(R_matrix)

print(f"Residual statistics:")
print(f"  Mean ||R||: {R_mean:.6f}")
print(f"  Max ||R||:  {R_max:.6f}")
print(f"  Std ||R||:  {R_std:.6f}")
print(f"  PCA explained variance: {pca.explained_variance_ratio_}")
```

#### 方法2: 训练显式残差网络（如果R大）

```python
class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # input: position(3) + sun_dir(3) + epsilon(3) + time_features(9) = 18
        self.net = nn.Sequential(
            nn.Linear(18, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 27)  # SH coefficients
        )

    def forward(self, x, sun_t, epsilon, time_features):
        # time_features可以是hour的sin/cos编码
        inp = torch.cat([x, sun_t, epsilon, time_features], dim=-1)
        return self.net(inp)

# 训练
residual_net = ResidualMLP()
optimizer = torch.optim.Adam(residual_net.parameters(), lr=0.001)

for step in range(500):
    sh_tpe = F_static(x + epsilon(t))  # 固定
    residual_pred = residual_net(x, sun_t, epsilon(t), time_enc(t))
    sh_final = sh_tpe + residual_pred

    loss = F.mse_loss(sh_final, sh_gt)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 对比
loss_pure_tpe = mse(F_static(x + epsilon), sh_gt)
loss_hybrid = mse(sh_tpe + residual_pred, sh_gt)
improvement = (loss_pure_tpe - loss_hybrid) / loss_pure_tpe * 100

print(f"Pure TPE loss:  {loss_pure_tpe:.6f}")
print(f"Hybrid loss:    {loss_hybrid:.6f}")
print(f"Improvement:    {improvement:.1f}%")
```

### 成功标准

| ||R||_mean | Improvement with R_net | 决策 |
|-----------|----------------------|------|
| < 0.002 | - | ✅ Pure TPE足够 |
| 0.002-0.005 | < 20% | ⚠️ 残差小但可选 |
| 0.005-0.01 | 20%-50% | ⚠️ 推荐hybrid |
| > 0.01 | > 50% | ❌ 必须hybrid或改方法 |

### 实现清单

```bash
# scripts/experiment_2_residual_analysis.py
python experiment_2_residual_analysis.py \
    --experiment-f-data experiment_f_data/ \
    --trained-tpe validation_results/ \
    --train-residual-net \  # 可选flag
    --output residual_analysis_results.npz
```

### 预计时间

- 直接残差分析: 30分钟
- 残差网络实现和训练: 1小时
- **总计: 1.5小时**

---

## 实验3: 空间泛化性验证

### 核心问题

Cornell Box其他位置的TPE表现如何？扰动方向和大小的空间模式？

### 实验设计

#### 阶段1: 检查dataset_2k

```bash
python scripts/inspect_dataset.py --dataset ../../data_generation/output/dataset_2k
```

预期输出：
- 场景名称
- 探针数量和分布
- 时刻覆盖
- 数据质量

#### 阶段2: 选择测试探针

**策略**: 空间分层采样

```python
def select_test_probes_stratified(all_probes, n=10):
    """选择空间上有代表性的探针"""
    # Cornell Box: [-1, 1] × [0, 2] × [-1, 1]

    regions = {
        'center': (0, 1, 0),       # 已验证
        'high_center': (0, 1.7, 0),  # 靠近顶部
        'low_center': (0, 0.3, 0),   # 靠近地面
        'near_red_wall': (0.8, 1, 0),  # 靠近X+墙
        'near_green_wall': (-0.8, 1, 0),  # 靠近X-墙
        'near_back_wall': (0, 1, 0.8),   # 靠近Z+墙
        'corner_floor': (0.8, 0.2, 0.8),  # 角落底部
        'corner_ceiling': (0.8, 1.8, 0.8),  # 角落顶部
        'mid_space_1': (0.4, 0.7, 0.4),  # 偏心位置1
        'mid_space_2': (-0.4, 1.3, -0.4),  # 偏心位置2
    }

    # 找到最接近每个region的真实探针
    selected = []
    for region_name, target_pos in regions.items():
        closest_idx = find_closest_probe(all_probes, target_pos)
        selected.append({
            'index': closest_idx,
            'position': all_probes[closest_idx],
            'region': region_name
        })

    return selected
```

#### 阶段3: 并行验证

```python
# 对每个探针运行完整的Level 2 TPE验证
results = []

for probe_info in selected_probes:
    print(f"\n{'='*70}")
    print(f"Testing probe: {probe_info['region']}")
    print(f"Position: {probe_info['position']}")

    # 运行方向约束TPE
    success, metrics = validate_directional_tpe(
        dataset_path=dataset_2k_path,
        probe_index=probe_info['index'],
        reference_hour=12
    )

    results.append({
        'region': probe_info['region'],
        'position': probe_info['position'],
        'success': success,
        'max_norm': metrics['max_perturbation_norm'],
        'correlation': metrics['sun_correlation'],
        'smoothness': metrics['avg_temporal_smoothness'],
        'principal_direction': metrics.get('principal_direction', None),
        'direction_explained_variance': metrics.get('explained_variance', None)
    })
```

#### 阶段4: 空间模式分析

```python
# 分析空间变化模式
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. 成功率 vs 位置
positions = np.array([r['position'] for r in results])
success_flags = np.array([r['success'] for r in results])

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
colors = ['green' if s else 'red' for s in success_flags]
ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
            c=colors, s=100, alpha=0.6)
ax1.set_title('TPE Success by Position')

# 2. 扰动方向的空间分布
directions = np.array([r['principal_direction'] for r in results if r['principal_direction'] is not None])
ax2 = fig.add_subplot(122, projection='3d')
for i, (pos, dir_vec) in enumerate(zip(positions, directions)):
    ax2.quiver(pos[0], pos[1], pos[2],
               dir_vec[0], dir_vec[1], dir_vec[2],
               length=0.3, color='blue', alpha=0.6)
ax2.set_title('Principal Perturbation Directions')

# 3. 扰动大小 vs 位置特征
max_norms = np.array([r['max_norm'] for r in results])
heights = positions[:, 1]  # Y坐标

plt.figure(figsize=(8, 5))
plt.scatter(heights, max_norms)
plt.xlabel('Probe Height (m)')
plt.ylabel('Max Perturbation Norm (m)')
plt.title('Perturbation Magnitude vs Height')
plt.axhline(y=1.0, color='r', linestyle='--', label='Threshold')
plt.legend()

# 4. 方向一致性
if len(directions) > 1:
    # 计算所有方向对的余弦相似度
    cos_sims = []
    for i in range(len(directions)):
        for j in range(i+1, len(directions)):
            cos_sim = np.dot(directions[i], directions[j])
            cos_sims.append(cos_sim)

    print(f"\nDirection consistency:")
    print(f"  Mean cos similarity: {np.mean(cos_sims):.3f}")
    print(f"  Std: {np.std(cos_sims):.3f}")
```

### 成功标准

| 指标 | 优秀 | 可接受 | 需关注 |
|------|------|--------|--------|
| 成功率 | > 80% | 60%-80% | < 60% |
| 方向一致性 | cos > 0.8 | 0.6-0.8 | < 0.6 |
| 最大扰动均值 | < 0.8m | 0.8-1.2m | > 1.2m |

**空间模式预期**:

| 位置类型 | 预期max_norm | 预期方向 | 成功率预期 |
|---------|-------------|---------|-----------|
| 中心 | 0.77m | Y轴 | ✅ 100% |
| 高/低中心 | 0.5-1.0m | Y轴 | ✅ 90% |
| 靠墙 | 1.0-1.5m | Y轴或倾斜 | ⚠️ 70% |
| 角落 | > 1.5m | 复杂 | ⚠️ 40% |

### Level 3架构决策

根据空间模式选择：

```python
if direction_consistency > 0.8:
    # 所有位置方向接近 → 选项A: 共享扰动
    architecture = "shared_perturbation"
    params_count = T  # 最少
elif direction_consistency > 0.6:
    # 方向中等一致 → 选项C: 插值扰动场
    architecture = "interpolated_perturbation_field"
    params_count = K * T  # K~10个control points
else:
    # 方向不一致 → 选项D: 扰动网络
    architecture = "perturbation_network"
    params_count = ~5000  # MLP parameters
```

### 实现清单

```bash
# 文件1: scripts/experiment_3_inspect_dataset_2k.py
python experiment_3_inspect_dataset_2k.py \
    --dataset ../../data_generation/output/dataset_2k \
    --output dataset_2k_info.json

# 文件2: scripts/experiment_3_spatial_validation.py
python experiment_3_spatial_validation.py \
    --dataset ../../data_generation/output/dataset_2k \
    --num-test-probes 10 \
    --strategy stratified \
    --parallel 4 \  # 并行4个进程
    --output spatial_validation_results/
```

### 预计时间

- 数据集检查: 15分钟
- 实现批量验证脚本: 45分钟
- 运行10个探针验证: 2-3小时（可并行）
- 分析和可视化: 30分钟
- **总计: 4小时**（如果4核并行）

---

## 实验4: 时间插值能力对比 ⭐

### 核心问题

**你说得对**：必须与baseline对比才能判断TPE的时间泛化能力！

需要回答：
1. TPE的时间插值能力如何？
2. 相比简单方法（Spline, MLP）有优势吗？
3. 相比TDML（时间度量学习）呢？

### 实验设计

#### 对比方法设计

**Method 1: TPE (Ours)**

```python
# 训练：只用奇数小时
train_hours = [7, 9, 11, 13, 15, 17]
F_static, alphas_train = train_directional_tpe(sh_data[train_hours])

# 测试：插值到偶数小时
test_hours = [6, 8, 10, 12, 14, 16, 18]
for t_test in test_hours:
    # Cubic spline插值α(t)
    alpha_interp = interp1d(train_hours, alphas_train, kind='cubic')(t_test)

    # 预测
    sun_interp = compute_sun_direction(t_test)
    sh_pred = F_static(x_probe + alpha_interp * direction, sun_interp)
```

**Method 2: Spline Interpolation (Simple Baseline)**

```python
from scipy.interpolate import interp1d

# 训练：对每个SH通道拟合spline
spline_models = []
for ch in range(27):
    spline = interp1d(train_hours, sh_train[:, ch], kind='cubic')
    spline_models.append(spline)

# 测试：直接插值
for t_test in test_hours:
    sh_pred = np.array([spline(t_test) for spline in spline_models])
```

**Method 3: Direct MLP (Time as Input)**

```python
class DirectMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: position(3) + sun_dir(3) + time_features(6) = 12
        self.net = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 27)  # SH coefficients
        )

    def forward(self, x, sun_dir, t_normalized, t_sin_cos):
        # t_sin_cos: [sin(2π*t/24), cos(2π*t/24), sin(4π*t/24), cos(4π*t/24), ...]
        time_features = torch.cat([
            t_normalized.unsqueeze(-1),  # [0, 1]
            t_sin_cos  # [6]
        ], dim=-1)
        inp = torch.cat([x, sun_dir, time_features], dim=-1)
        return self.net(inp)

# 训练
mlp = DirectMLP()
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

for step in range(1000):
    # 只用train_hours
    sh_pred = mlp(x_probe, sun_dirs[train_indices], hours_norm[train_indices], sin_cos[train_indices])
    loss = F.mse_loss(sh_pred, sh_gt[train_indices])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 测试：直接推理
sh_pred_test = mlp(x_probe, sun_dirs[test_indices], hours_norm[test_indices], sin_cos[test_indices])
```

**Method 4: TDML (Time Distance Metric Learning)**

```python
from models.time_encoder import TimeEncoder

# 训练
time_encoder = TimeEncoder(dim=16)
decoder = DecoderMLP(input_dim=3+16, output_dim=27)

optimizer = torch.optim.Adam(
    list(time_encoder.parameters()) + list(decoder.parameters()),
    lr=0.001
)

for step in range(1000):
    # 只用train_hours
    time_emb = time_encoder(hours_tensor[train_indices])  # [N_train, 16]
    sh_pred = decoder(x_probe, time_emb)
    loss = F.mse_loss(sh_pred, sh_gt[train_indices])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 测试
time_emb_test = time_encoder(hours_tensor[test_indices])
sh_pred_test = decoder(x_probe, time_emb_test)
```

#### 交叉验证策略

**方案A: 奇偶交叉验证**

```python
# Split 1: 奇数训练，偶数测试
train_hours_1 = [7, 9, 11, 13, 15, 17]
test_hours_1 = [6, 8, 10, 12, 14, 16, 18]

# Split 2: 偶数训练，奇数测试
train_hours_2 = [6, 8, 10, 12, 14, 16, 18]
test_hours_2 = [7, 9, 11, 13, 15, 17]
```

**方案B: K-fold (K=3)**

```python
# Fold 1: 训练[6,9,12,15,18], 测试[7,8,10,11,13,14,16,17]
# Fold 2: 训练[7,8,10,11,13,14,16,17], 测试[6,9,12,15,18]
# Fold 3: 训练混合, 测试剩余
```

推荐**方案A**（更简单且足够）

#### 评估指标

```python
def evaluate_interpolation(method_name, sh_pred_test, sh_gt_test):
    """评估插值质量"""

    # 1. 整体MSE
    mse_total = np.mean((sh_pred_test - sh_gt_test)**2)

    # 2. Per-hour MSE
    mse_per_hour = []
    for i, hour in enumerate(test_hours):
        mse_h = np.mean((sh_pred_test[i] - sh_gt_test[i])**2)
        mse_per_hour.append(mse_h)

    # 3. 最坏情况
    mse_max = np.max(mse_per_hour)

    # 4. 与训练误差的对比（泛化gap）
    mse_train = compute_train_error(method_name)
    generalization_gap = (mse_total - mse_train) / mse_train * 100

    return {
        'method': method_name,
        'mse_test': mse_total,
        'mse_train': mse_train,
        'mse_max': mse_max,
        'generalization_gap': generalization_gap,
        'mse_per_hour': mse_per_hour
    }
```

### 对比维度

| 方法 | 参数量 | 训练时间 | 推理时间 | Test MSE | Gap% | 优势 |
|------|--------|---------|---------|----------|------|------|
| Spline | 135 | 0.1s | 0.01ms | ? | ? | 最简单 |
| Direct MLP | 3856 | 20s | 0.5ms | ? | ? | 灵活 |
| TDML | ~6000 | 30s | 1ms | ? | ? | 时间度量 |
| **TPE (Ours)** | 6816 | 60s | 2ms | ? | ? | 物理约束 |

**关键问题**：TPE的Test MSE是否显著低于其他方法？

### 统计显著性检验

```python
from scipy import stats

# 收集每个方法的per-hour errors
errors_spline = [mse_per_hour for method='spline']
errors_mlp = [...]
errors_tdml = [...]
errors_tpe = [...]

# Paired t-test (因为在同样的test hours上比较)
t_stat, p_value = stats.ttest_rel(errors_tpe, errors_spline)

if p_value < 0.05 and np.mean(errors_tpe) < np.mean(errors_spline):
    print(f"✅ TPE显著优于Spline (p={p_value:.4f})")
else:
    print(f"⚠️ TPE无显著优势 (p={p_value:.4f})")
```

### 成功标准

**相对优势判定**：

| 条件 | 结论 |
|------|------|
| TPE < Spline且p<0.05 | ✅ TPE有价值（超越简单方法） |
| TPE < MLP且p<0.05 | ✅ TPE的物理约束有效 |
| TPE ≈ TDML | ⚠️ TPE与TDML相当，但更可解释 |
| TPE > Spline | ❌ TPE过拟合，无泛化优势 |

**绝对性能判定**：

| Generalization Gap | 结论 |
|-------------------|------|
| < 20% | ✅ 插值能力excellent |
| 20%-50% | ⚠️ 插值能力acceptable |
| > 50% | ❌ 过拟合严重 |

### 实现清单

```bash
# 文件1: scripts/experiment_4_baseline_spline.py
python experiment_4_baseline_spline.py \
    --dataset level2_tpe/cornell-box_test \
    --cv-split odd_even \
    --output baselines/spline_results.npz

# 文件2: scripts/experiment_4_baseline_mlp.py
python experiment_4_baseline_mlp.py \
    --dataset level2_tpe/cornell-box_test \
    --cv-split odd_even \
    --hidden-dim 64 \
    --output baselines/mlp_results.npz

# 文件3: scripts/experiment_4_baseline_tdml.py
python experiment_4_baseline_tdml.py \
    --dataset level2_tpe/cornell-box_test \
    --cv-split odd_even \
    --time-dim 16 \
    --output baselines/tdml_results.npz

# 文件4: scripts/experiment_4_tpe_interpolation.py
python experiment_4_tpe_interpolation.py \
    --dataset level2_tpe/cornell-box_test \
    --cv-split odd_even \
    --interpolation cubic \
    --output baselines/tpe_results.npz

# 文件5: scripts/experiment_4_compare_all.py
python experiment_4_compare_all.py \
    --results baselines/*_results.npz \
    --output interpolation_comparison.pdf
```

### 预计时间

- Spline baseline实现: 30分钟
- Direct MLP实现和训练: 1小时
- TDML实现和训练: 1小时
- TPE交叉验证: 1小时
- 对比分析和统计检验: 30分钟
- **总计: 4小时**

---

## 总体执行计划

### 时间线（3天）

**Day 1 - 实验F（最关键）**
```
09:00 - 09:30  实现渲染脚本
09:30 - 10:10  执行渲染（40分钟，可后台）
10:10 - 10:40  实现分析脚本
10:40 - 11:00  误差分解和可视化
11:00 - 12:00  根据结果决定是否需要实验2
```

**Day 1下午 / Day 2上午 - 实验2+3并行**
```
# 如果实验F发现残差大
13:00 - 14:30  实验2: 残差分析和残差网络训练

# 并行执行（或Day 2）
10:00 - 10:15  实验3: 检查dataset_2k
10:15 - 11:00  实验3: 实现批量验证
11:00 - 14:00  实验3: 并行运行10个探针（后台）
14:00 - 14:30  实验3: 分析空间模式
```

**Day 2下午 / Day 3 - 实验4**
```
14:00 - 14:30  实现Spline baseline
14:30 - 15:30  实现和训练Direct MLP
15:30 - 16:30  实现和训练TDML
16:30 - 17:30  TPE交叉验证
17:30 - 18:00  对比分析和统计检验
```

### 并行化策略

```bash
# 可以同时运行的任务
# Terminal 1: 实验F渲染（后台）
nohup python experiment_f_render.py > render.log 2>&1 &

# Terminal 2: 实现实验2和3的代码
vim experiment_2_residual_analysis.py
vim experiment_3_spatial_validation.py

# Terminal 3: 实验3并行验证（后台）
python experiment_3_spatial_validation.py --parallel 4 &

# Terminal 4: 实验4 baselines训练
python experiment_4_baseline_mlp.py  # 可以在等待其他任务时训练
```

### 资源需求

- **计算**: 4核CPU（用于并行），16GB RAM
- **渲染**: ~40分钟（实验F的8个probe-sun组合）
- **训练**: ~3小时总GPU时间（各种模型训练）
- **存储**: ~2GB（渲染数据 + 模型权重）

---

## 风险与应对

### 风险1: 实验F显示E_tpe > 0.02

**应对**:
- 立即停止Level 3开发
- 转向Multi-Anchor TPE（选择3个锚点：morning, noon, evening）
- 或转向Hybrid Model

### 风险2: 实验3显示空间泛化性差（成功率<60%）

**应对**:
- 分析失败模式（哪些位置失败？为什么？）
- 考虑位置相关的方向场 d(x)
- 或者Level 3使用选项D（perturbation network）

### 风险3: 实验4显示TPE插值不如Spline

**应对**:
- 检查是否过拟合（增加正则化）
- 添加时间平滑性约束
- 或承认TPE不适合插值，只适合已知时刻

### 风险4: dataset_2k不可用或质量差

**应对**:
- 使用level2数据，手动选择Cornell Box内的其他位置
- 快速生成小规模多探针数据（5个探针，1小时渲染）

---

## 成果交付

### 每个实验的输出

**实验F**:
```
experiment_f_results/
├── rendered_data/
│   ├── hour_06_perturbed_noon.npz
│   ├── hour_06_original_t.npz
│   └── ...
├── error_decomposition.npz
├── error_analysis_report.md
└── visualization_errors.pdf
```

**实验2**:
```
residual_analysis/
├── residual_statistics.json
├── residual_patterns_pca.npz
├── residual_net_model.pth (如果训练)
└── residual_analysis_report.md
```

**实验3**:
```
spatial_validation/
├── dataset_2k_info.json
├── probe_selection.json
├── results_probe_*.npz (10个文件)
├── spatial_patterns_analysis.npz
├── spatial_visualization.pdf
└── level3_architecture_recommendation.md
```

**实验4**:
```
interpolation_comparison/
├── baselines/
│   ├── spline_results.npz
│   ├── mlp_results.npz
│   ├── tdml_results.npz
│   └── tpe_results.npz
├── comparison_table.csv
├── statistical_tests.json
├── interpolation_curves.pdf
└── interpolation_comparison_report.md
```

### 总结报告

最终生成一份 **P0_Experiments_Summary_Report.md** 包含：

1. 实验F结论：TPE物理假设是否成立？
2. 实验2结论：是否需要残差模型？
3. 实验3结论：空间泛化性如何？Level 3架构建议？
4. 实验4结论：时间插值能力如何？相对优势？
5. **总体决策**：是否继续Level 3？用什么架构？

---

## 检查清单

**实验开始前**:
- [ ] 确认Cornell Box Level 2数据完整
- [ ] 确认已训练的TPE模型可加载
- [ ] 检查dataset_2k状态
- [ ] 预留足够计算资源（4核，16GB RAM）

**实验F**:
- [ ] 渲染脚本实现并测试
- [ ] 8个probe-sun组合渲染完成
- [ ] 误差分解正确计算
- [ ] 3个error指标都有
- [ ] 可视化图表清晰

**实验2**:
- [ ] 使用实验F的真实数据
- [ ] 残差统计计算
- [ ] PCA分析残差模式
- [ ] 如果||R||大，训练残差网络

**实验3**:
- [ ] dataset_2k检查完成
- [ ] 10个探针合理选择
- [ ] 所有探针验证完成
- [ ] 空间模式分析
- [ ] Level 3架构建议明确

**实验4**:
- [ ] 4种方法都实现
- [ ] 交叉验证正确执行
- [ ] 统计显著性检验
- [ ] 对比表格和图表
- [ ] 相对优势结论明确

**报告**:
- [ ] 每个实验有独立报告
- [ ] P0_Experiments_Summary_Report.md完成
- [ ] 决策清晰（继续/调整/放弃）
- [ ] 下一步行动明确

---

**文档版本**: v1.0
**预计总耗时**: 12-15小时（3天）
**关键路径**: 实验F → 决策点 → 实验2/3 → 实验4
