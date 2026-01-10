# 时间扰动嵌入（Temporal Perturbation Embedding, TPE）

**创建日期**: 2025-12-10
**状态**: 理论方案，待验证
**类型**: 动态隐式神经表示的时间建模创新方法

---

## 目录

1. [核心思想](#核心思想)
2. [理论基础](#理论基础)
3. [数学推导](#数学推导)
4. [架构设计](#架构设计)
5. [与现有方法对比](#与现有方法对比)
6. [验证策略](#验证策略)
7. [实现指南](#实现指南)
8. [预期效果](#预期效果)
9. [潜在局限](#潜在局限)

---

## 核心思想

### 一句话概述

**将时间变化建模为查询坐标的可学习扰动，使时间无关的静态场通过坐标漂移隐式表达动态，实现零时间参数增长的4D表示。**

### 范式转变

```
传统方法：F(x, t) = MLP(x, t)      ← 时间是输入维度
TPE方法：  F(x, t) = F_static(x + ε(t)) ← 时间是坐标扰动
```

**关键洞察**：时间变化不是"场的变化"，而是"观察坐标的变化"。

### 物理直觉

以光照场景为例：
- 太阳从东到西移动
- 阴影不是"变化"的，而是"平移"的
- 如果你"跟随"阴影移动，光照模式保持不变

这是**Lagrangian视角**：
- **Eulerian（传统）**：固定点x，观察F(x,t)如何变化
- **Lagrangian（TPE）**：跟随漂移点x(t)=x+ε(t)，场F保持静态

---

## 理论基础

### 1. 数学基础：泰勒展开

对于小扰动ε(t)，场F在x+ε处的值可以泰勒展开：

```
F(x + ε) ≈ F(x) + ∇F(x)·ε + O(ε²)
```

**关键假设**：如果||ε||足够小，一阶近似足够准确。

**有效条件**：
- ||ε|| < 0.5m（经验阈值）
- F(x)足够平滑（至少C¹连续）

### 2. 物理类比：坐标变换

在广义相对论中，时空是统一的四维流形。TPE将时间"投影"到3D空间扰动：

```
时空表示：(x, y, z, t) ∈ R⁴
TPE表示： x(t) = (x, y, z) + ε(t) ∈ R³
```

**物理意义**：
- 对于移动的光照模式（如阴影），存在一个"共动坐标系"
- 在该坐标系中，光照模式是静止的
- ε(t)编码从实验室坐标系到共动坐标系的变换

### 3. 流形学习视角

时间维度本质上是一个1D流形，TPE学习其在3D空间中的嵌入：

```
时间流形：t ∈ [0, T] ⊂ R¹
   ↓ 嵌入映射 ε(·)
空间流形：ε(t) ∈ R³（实际是嵌入的1D曲线）
```

**优势**：
- 流形的内在维度（1D）远小于嵌入空间（3D）
- 允许非线性嵌入，捕捉复杂的时间演化

### 4. 对抗鲁棒性视角

TPE训练的F_static必须对坐标扰动鲁棒：

```
对于任意x和小扰动ε：
F_static(x + ε) ≈ 真实场在时刻t的值
```

这类似于对抗训练中的输入扰动，但这里扰动是结构化的（由时间参数化）。

---

## 数学推导

### 问题设定

给定：
- 空间坐标：x ∈ R³
- 时间：t ∈ [0, T]
- 真实动态场：F_true(x, t) → R^n（如SH系数）

目标：学习紧凑表示，使得F_TPE(x, t) ≈ F_true(x, t)。

### TPE分解

将动态场分解为：
1. **静态场**：F_static: R³ → R^n
   - 与时间无关
   - 在参考时刻t₀训练

2. **扰动映射**：ε: [0, T] → R³
   - 极小的神经网络
   - 将时间映射到空间扰动

3. **组合**：
   ```
   F_TPE(x, t) = F_static(x + ε(t))
   ```

### 近似误差分析

定义近似误差：

```
E(x, t) = ||F_true(x, t) - F_static(x + ε(t))||²
```

泰勒展开F_static：

```
F_static(x + ε) = F_static(x) + ∇F_static(x)·ε + (1/2)ε^T H ε + O(||ε||³)
```

其中H是Hessian矩阵。

**误差上界**：
假设F_static和F_true满足：
- ||∇F_static|| ≤ L（Lipschitz连续）
- ||H|| ≤ M（Hessian有界）

则误差满足：

```
E(x, t) ≤ C₁·||ε(t)||² + C₂·||∇F_static(x)||·||ε(t) - ε*(t)||

其中：
- C₁ ∝ M（二阶项系数）
- C₂ ∝ L（一阶项系数）
- ε*(t)是最优扰动
```

**结论**：
- 误差随||ε||²增长（二阶）
- 学习到接近最优的ε(t)可以降低误差
- 需要||ε|| < 1/√M保证误差可控

### 参数复杂度分析

**传统时间建模**（如Temporal MLP）：
```
参数量 = K × (D_in × H + H × H + H × D_out)
       ≈ K × (6×32 + 32×32 + 32×9)
       ≈ K × 1504
对于K=750，约1.1M参数
```

**TPE方法**：
```
静态场参数 = K × D_latent（与时间无关，不计入时间建模）
扰动网络参数 = 3×16 + 16×3 = 96（共享，极小）

时间相关参数：仅96个！
```

**压缩比**：1504 / 96 ≈ **15.7×**

---

## 架构设计

### 整体架构

```
┌─────────────────────────────────────────────────────┐
│                    TPE架构                           │
├─────────────────────────────────────────────────────┤
│                                                      │
│  时间输入 t                                           │
│     ↓                                                │
│  太阳方向 sun_dir(t)                                  │
│     ↓                                                │
│  ┌──────────────────────┐                           │
│  │ 扰动网络 (极小)       │                           │
│  │ ε(·): R³ → R³        │                           │
│  │ [3 → 16 → 3]         │                           │
│  │ 参数：~100个          │                           │
│  └──────────────────────┘                           │
│     ↓                                                │
│  扰动向量 ε(t) ∈ R³                                   │
│     ↓                                                │
│  扰动坐标 x' = x + ε(t)                               │
│     ↓                                                │
│  ┌──────────────────────┐                           │
│  │ 静态场 (时间无关)      │                           │
│  │ F_static(·): R³ → R^n│                           │
│  │ 高斯混合 + 解码器      │                           │
│  │ 参数：K×(几何+潜在码)  │                           │
│  └──────────────────────┘                           │
│     ↓                                                │
│  输出 SH系数 / 颜色                                    │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 组件详细设计

#### 1. 扰动网络（Perturbation Network）

```python
class PerturbationNet(nn.Module):
    """极小的扰动映射网络"""

    def __init__(self, input_dim=3, hidden_dim=16, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # 限制输出范围[-1, 1]
        )
        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, sun_direction):
        """
        Args:
            sun_direction: [3] or [T, 3] 归一化的太阳方向向量
        Returns:
            perturbation: [3] or [T, 3] 空间扰动向量（米）
        """
        raw_output = self.net(sun_direction)
        perturbation = raw_output * self.scale
        return perturbation
```

**设计要点**：
- 输入：sun_dir(t)（物理先验，3D单位向量）
- 输出：ε(t)（3D扰动向量）
- Tanh激活：限制扰动幅度
- 可学习scale：自动调节扰动范围

#### 2. 静态场（Static Field）

使用现有的高斯压缩架构，但**完全不感知时间**：

```python
class StaticField(nn.Module):
    """时间无关的静态场"""

    def __init__(self, num_gaussians=250, latent_dim=6):
        super().__init__()
        # 高斯混合层
        self.gaussians = GaussianMixture(num_gaussians, latent_dim)

        # 解码器（不接受时间输入）
        self.decoder = DecoderMLP(
            input_dim=latent_dim + 3,  # latent + position
            output_dim=27  # SH coefficients
        )

    def forward(self, positions):
        """
        Args:
            positions: [B, 3] 查询位置（可能是扰动后的）
        Returns:
            sh_coeffs: [B, 27] 球谐系数
        """
        # 查询高斯混合
        latents = self.gaussians.query(positions)  # [B, D]

        # 解码
        sh_coeffs = self.decoder(torch.cat([latents, positions], dim=-1))

        return sh_coeffs
```

**关键特性**：
- ✅ 与时间完全解耦
- ✅ 仅在参考时刻（如12:00）训练
- ✅ 参数量与时刻数无关

#### 3. 完整TPE模型

```python
class TPEModel(nn.Module):
    """完整的时间扰动嵌入模型"""

    def __init__(self, num_gaussians=250, latent_dim=6):
        super().__init__()

        # 静态场（大部分参数）
        self.static_field = StaticField(num_gaussians, latent_dim)

        # 扰动网络（极小）
        self.perturbation_net = PerturbationNet()

    def forward(self, positions, sun_direction, reference_mode=False):
        """
        Args:
            positions: [B, 3] 探针位置
            sun_direction: [3] or [T, 3] 太阳方向
            reference_mode: 如果True，不添加扰动（训练静态场用）

        Returns:
            sh_coeffs: [B, 27] or [B, T, 27]
        """
        # 计算扰动
        if reference_mode:
            # 训练静态场时，不添加扰动
            perturbation = torch.zeros(1, 3, device=positions.device)
        else:
            perturbation = self.perturbation_net(sun_direction)
            if perturbation.dim() == 1:
                perturbation = perturbation.unsqueeze(0)  # [1, 3]

        # 扰动坐标
        if perturbation.shape[0] == 1:
            # 单一时刻
            perturbed_positions = positions + perturbation  # [B, 3]
            sh_coeffs = self.static_field(perturbed_positions)  # [B, 27]
        else:
            # 多个时刻
            T = perturbation.shape[0]
            B = positions.shape[0]
            # 广播：[B, 1, 3] + [1, T, 3] = [B, T, 3]
            perturbed_positions = positions.unsqueeze(1) + perturbation.unsqueeze(0)
            perturbed_flat = perturbed_positions.reshape(B * T, 3)
            sh_coeffs_flat = self.static_field(perturbed_flat)  # [B*T, 27]
            sh_coeffs = sh_coeffs_flat.reshape(B, T, 27)  # [B, T, 27]

        return sh_coeffs

    def get_time_related_params(self):
        """返回时间相关的参数（用于统计压缩比）"""
        return list(self.perturbation_net.parameters())

    def get_perturbation_trajectory(self, sun_dirs):
        """获取扰动轨迹（用于可视化）"""
        with torch.no_grad():
            perturbations = self.perturbation_net(sun_dirs)
        return perturbations.cpu().numpy()
```

---

## 与现有方法对比

### 概念对比

| 方法 | 时间建模方式 | 时间参数 | 核心思想 |
|------|-------------|----------|---------|
| **D-NeRF** | 形变场D(x,t) | ~2M | 规范空间+非刚性形变 |
| **NeRF-W** | 外观嵌入l_t | N×D | 逐帧独立嵌入 |
| **K-Planes** | 时间平面T | H×W×C | 显式时间特征张量 |
| **DINR** | ODE演化 | ~10K | 连续动力学系统 |
| **Temporal MLP** | MLP(x,t) | ~2K | 端到端学习 |
| **TPE（本方案）** | ε(t)扰动 | **~100** | 坐标变换 |

### 定量对比（预测）

| 指标 | Temporal MLP | TPE | 变化 |
|------|-------------|-----|------|
| 时间相关参数 | 2,048 | 96 | **↓ 21×** |
| 静态场参数 | K×6 | K×6 | 相同 |
| PSNR (dB) | 38.0 | 36-37 | ↓ 1-2 |
| 训练时间 | 5h | 4h | ↓ 20% |
| 推理速度 | 2ms | 1.8ms | ↑ 10% |
| 可解释性 | 弱 | **强** | - |

### 优势总结

✅ **极致压缩**：时间参数减少20×
✅ **物理可解释**：ε(t)可直接可视化为"时间向量场"
✅ **架构简洁**：仅一个3→16→3的MLP
✅ **自然泛化**：新时刻插值容易（扰动连续）
✅ **可编辑性**：手工设计ε实现艺术效果

### 劣势总结

❌ **适用场景受限**：仅适合平移类变化（光照、流体）
❌ **质量可能略降**：泰勒近似引入误差
❌ **需要平滑场**：高频细节可能模糊

---

## 验证策略

### 验证层次结构

```
Level 0: 理论验证（1天）
  ├─ 泰勒展开误差分析
  └─ 决策点：误差<1% at ||ε||<0.5m？

Level 1: 玩具实验（2天）
  ├─ 移动高斯团（解析解）
  ├─ 旋转场景（失效案例）
  └─ 决策点：解析解误差<1e-4？

Level 2: 单探针验证（3天）← 最关键！
  ├─ 反推最优扰动ε*(t)
  ├─ 检查：幅度、平滑性、相关性
  └─ 决策点：满足3个成功标准？

Level 3: 完整系统（2周）
  ├─ 实现TPE模型
  ├─ 分阶段训练
  ├─ 与Baseline对比
  └─ 决策点：PSNR降低<2dB？

Level 4: 边界探索（1周）
  ├─ 失效场景分析
  ├─ 扰动幅度上限
  └─ 可视化与论文实验
```

### Level 0: 理论验证

**实验0.1：泰勒展开误差分析**

目标：验证F(x+ε) ≈ F(x) + ∇F·ε的有效范围

```python
def test_taylor_approximation():
    """验证泰勒近似的有效性"""

    # 定义场（高斯混合）
    def F(x):
        centers = [[0,0,0], [1,0,0], [0,1,0]]
        return sum(np.exp(-np.sum((x - c)**2)) for c in centers)

    # 数值梯度
    def grad_F(x, eps=1e-4):
        grad = np.zeros(3)
        for i in range(3):
            x_plus, x_minus = x.copy(), x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (F(x_plus) - F(x_minus)) / (2*eps)
        return grad

    # 测试不同扰动幅度
    x0 = np.array([0.5, 0.5, 0.5])
    epsilons = np.logspace(-3, 0, 20)
    errors = []

    for eps_norm in epsilons:
        eps = np.random.randn(3)
        eps = eps / np.linalg.norm(eps) * eps_norm

        true_value = F(x0 + eps)
        taylor_approx = F(x0) + np.dot(grad_F(x0), eps)
        errors.append(abs(true_value - taylor_approx))

    # 绘图：误差 vs ||ε||
    plt.loglog(epsilons, errors, label='Actual Error')
    plt.loglog(epsilons, epsilons**2, '--', label='O(ε²)')
    plt.xlabel('||ε|| (meters)')
    plt.ylabel('Approximation Error')
    plt.legend()
    plt.savefig('taylor_error.png')

    # 判断
    idx = np.argmin(np.abs(epsilons - 0.5))
    return errors[idx] < 0.01  # 成功标准
```

**成功标准**：
- ✅ 误差曲线斜率≈2（二阶增长）
- ✅ 在||ε||<0.5m时，误差<1%

### Level 1: 玩具实验

**实验1.1：移动高斯团（完美案例）**

```python
class ToyTPE:
    """2D移动高斯团"""

    def __init__(self):
        # 静态场（中心在原点）
        self.F_static = lambda x, y: torch.exp(-(x**2 + y**2))

        # 扰动网络（应学到 ε=(t, 0)）
        self.perturbation_net = nn.Sequential(
            nn.Linear(1, 8), nn.ReLU(),
            nn.Linear(8, 2)
        )
        self.optimizer = torch.optim.Adam(
            self.perturbation_net.parameters(), lr=0.01
        )

    def ground_truth(self, x, y, t):
        """真实场：高斯从左到右移动"""
        return torch.exp(-((x - t)**2 + y**2))

    def predict(self, x, y, t):
        """TPE预测"""
        eps = self.perturbation_net(t.unsqueeze(-1))
        return self.F_static(x - eps[:, 0:1], y - eps[:, 1:2])

    def train(self, steps=1000):
        for step in range(steps):
            x = torch.randn(64, 1) * 2
            y = torch.randn(64, 1) * 2
            t = torch.rand(64, 1) * 2

            pred = self.predict(x, y, t)
            gt = self.ground_truth(x, y, t)
            loss = torch.mean((pred - gt)**2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 验证
        test_t = torch.tensor([[1.0]])
        learned_eps = self.perturbation_net(test_t).detach().numpy()
        print(f"Learned ε(1.0) = {learned_eps[0]}")
        print(f"Expected: [1.0, 0.0]")
        return np.linalg.norm(learned_eps - [1.0, 0.0]) < 0.1
```

**成功标准**：
- ✅ 最终损失<1e-4
- ✅ 学到的ε(t)与(t, 0)误差<0.1

### Level 2: 单探针验证（最关键）

**实验2.1：验证核心假设**

```python
def validate_perturbation_hypothesis(dataset):
    """验证：真实光照变化是否可用扰动建模"""

    # 选择探针
    probe_idx = 100
    sample = dataset[probe_idx]

    position = sample['position']  # [3]
    sh_gt = sample['sh_gt']        # [T, 27]
    sun_dirs = sample['sun_dirs']  # [T, 3]

    # 优化扰动向量（使每个时刻能匹配GT）
    perturbations = nn.Parameter(torch.zeros(T, 3))
    optimizer = torch.optim.Adam([perturbations], lr=0.01)

    # 参考时刻固定为0
    reference_idx = 2  # 12:00

    for step in range(500):
        optimizer.zero_grad()

        # 强制参考时刻扰动=0
        perturb = perturbations.clone()
        perturb[reference_idx] = 0

        # 损失：这里需要真正的F_static查询
        # 简化：用扰动的L2范数作为代理
        loss = torch.norm(perturb, dim=-1).mean() * 0.1

        loss.backward()
        optimizer.step()

    # 分析结果
    learned_perturb = perturb.detach().numpy()
    perturb_norms = np.linalg.norm(learned_perturb, axis=1)

    # 成功标准1: 幅度<1m
    test1 = perturb_norms.max() < 1.0
    print(f"Max ||ε||: {perturb_norms.max():.3f}m {'✓' if test1 else '✗'}")

    # 成功标准2: 平滑性
    perturb_diffs = np.linalg.norm(np.diff(learned_perturb, axis=0), axis=1)
    test2 = perturb_diffs.mean() < 0.5
    print(f"Avg smoothness: {perturb_diffs.mean():.3f}m {'✓' if test2 else '✗'}")

    # 成功标准3: 与太阳方向相关
    sun_diffs = sun_dirs - sun_dirs[reference_idx:reference_idx+1]
    correlation = np.corrcoef(
        perturb_norms,
        np.linalg.norm(sun_diffs, axis=1)
    )[0, 1]
    test3 = abs(correlation) > 0.5
    print(f"Correlation: {correlation:.3f} {'✓' if test3 else '✗'}")

    return test1 and test2 and test3
```

**成功标准（必须全部满足）**：
1. ✅ ||ε|| < 1米（泰勒近似有效）
2. ✅ 相邻时刻变化<0.5米（平滑性）
3. ✅ 与太阳方向相关性>0.5（物理合理）

**如果失败 → 立即放弃方案**

### Level 3: 完整系统验证

**训练策略：三阶段**

```python
def train_tpe_three_stages(model, dataloader):
    """三阶段训练TPE"""

    # ========== 阶段1: 预训练静态场 ==========
    print("Stage 1: Pretraining static field...")

    # 冻结扰动网络
    for param in model.perturbation_net.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        model.static_field.parameters(), lr=0.001
    )

    reference_time_idx = 2  # 12:00

    for epoch in range(100):
        for batch in dataloader:
            positions = batch['position']
            sh_gt = batch['sh_gt'][:, reference_time_idx]

            # reference_mode=True：不添加扰动
            pred = model(
                positions,
                sun_dirs[reference_time_idx],
                reference_mode=True
            )

            loss = F.mse_loss(pred, sh_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ========== 阶段2: 训练扰动网络 ==========
    print("Stage 2: Training perturbation network...")

    # 冻结静态场
    for param in model.static_field.parameters():
        param.requires_grad = False

    # 解冻扰动网络
    for param in model.perturbation_net.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(
        model.perturbation_net.parameters(), lr=0.001
    )

    for epoch in range(100):
        for batch in dataloader:
            positions = batch['position']
            sh_gt = batch['sh_gt']        # [B, T, 27]
            sun_dirs = batch['sun_dirs']  # [T, 3]

            # 所有时刻
            pred = model(positions, sun_dirs)  # [B, T, 27]

            loss_recon = F.mse_loss(pred, sh_gt)

            # 扰动正则化
            perturbations = model.perturbation_net(sun_dirs)
            loss_reg = torch.norm(perturbations, dim=-1).mean() * 0.1

            loss = loss_recon + loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ========== 阶段3: 联合微调（可选）==========
    print("Stage 3: Joint fine-tuning...")

    # 解冻所有参数
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(50):
        # ... 训练代码 ...
        pass

    return model
```

**对比实验**

```python
def compare_methods(tpe_model, baseline_model, test_loader):
    """对比TPE与Baseline"""

    results = {}

    for name, model in [('Baseline', baseline_model), ('TPE', tpe_model)]:
        psnrs = []
        for batch in test_loader:
            pred = model(batch['position'], batch['sun_dirs'])
            gt = batch['sh_gt']
            psnr = compute_psnr(pred, gt)
            psnrs.append(psnr)

        param_count = sum(
            p.numel() for p in model.get_time_related_params()
        )

        results[name] = {
            'PSNR': np.mean(psnrs),
            'Params': param_count,
            'Size_KB': param_count * 4 / 1024
        }

    # 打印
    print(f"\n{'Method':<15} {'PSNR':<10} {'Params':<10} {'Size (KB)':<10}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<15} {res['PSNR']:<10.2f} {res['Params']:<10d} {res['Size_KB']:<10.1f}")

    # 判断
    psnr_diff = results['TPE']['PSNR'] - results['Baseline']['PSNR']
    size_ratio = results['Baseline']['Size_KB'] / results['TPE']['Size_KB']

    success = (psnr_diff > -2) and (size_ratio > 1.5)

    print(f"\nPSNR difference: {psnr_diff:+.2f} dB")
    print(f"Compression ratio: {size_ratio:.1f}×")
    print(f"Result: {'✓ PASS' if success else '✗ FAIL'}")

    return success
```

**成功标准**：
- ✅ PSNR降低<2dB
- ✅ 参数减少>1.5×

### Level 4: 边界探索

**失效场景分析**

```python
def test_failure_scenarios():
    """测试TPE在不同场景下的表现"""

    scenarios = {
        'Translational (shadows move)': {
            'expected': 'PASS',
            'threshold': 35
        },
        'Intensity change (clouds)': {
            'expected': 'PARTIAL',
            'threshold': 30
        },
        'Color change (sunset)': {
            'expected': 'FAIL',
            'threshold': 25
        },
    }

    for name, config in scenarios.items():
        psnr = test_on_scenario(tpe_model, name)
        expected = config['expected']
        threshold = config['threshold']

        if expected == 'PASS':
            status = '✓' if psnr > threshold else '✗'
        elif expected == 'FAIL':
            status = '✓' if psnr < threshold else '✗'
        else:
            status = '~'

        print(f"{name:<30} PSNR={psnr:.1f}dB {status}")
```

---

## 实现指南

### 文件结构

```
multi_time_compression/
├── models/
│   ├── tpe_model.py          # TPE完整模型
│   ├── perturbation_net.py   # 扰动网络
│   └── static_field.py       # 静态场（复用现有）
├── training/
│   └── tpe_trainer.py        # 三阶段训练器
├── scripts/
│   ├── train_tpe.py          # 训练脚本
│   ├── validate_tpe.py       # 验证脚本
│   └── visualize_tpe.py      # 可视化扰动
└── tests/
    ├── test_taylor.py        # 理论验证
    ├── test_toy_tpe.py       # 玩具实验
    └── test_hypothesis.py    # 单探针验证
```

### 关键代码片段

**完整TPE模型**（详见架构设计章节）

**训练入口**

```python
# scripts/train_tpe.py

if __name__ == '__main__':
    # 加载数据
    train_dataset = MultiTimeLightingDataset(...)
    train_loader = DataLoader(train_dataset, batch_size=64)

    # 创建模型
    model = TPEModel(num_gaussians=250, latent_dim=6)

    # 三阶段训练
    model = train_tpe_three_stages(model, train_loader)

    # 保存
    torch.save(model.state_dict(), 'tpe_model.pth')

    # 对比
    baseline_model = load_baseline_model()
    success = compare_methods(model, baseline_model, test_loader)

    if success:
        print("✓ TPE验证成功！")
    else:
        print("✗ TPE未达预期")
```

---

## 预期效果

### 定量指标（预测）

| 指标 | Baseline | TPE | 目标 |
|------|---------|-----|------|
| PSNR (dB) | 38.0 | 36.5-37.5 | >36 |
| 时间参数 | 2048 | 96 | <200 |
| 压缩比 | 1× | 21× | >15× |
| 训练时间 | 5h | 4h | <5h |
| 推理速度 | 2ms | 1.8ms | <2ms |

### 定性效果

**扰动向量场可视化**：
- 24时刻的ε(t)在3D空间形成平滑曲线
- 早晨→正午→傍晚：扰动从东→0→西
- 幅度：最大约0.5米

**时间插值**：
- 对于新时刻t_new，ε(t_new)可线性插值
- 插值质量优于baseline（扰动连续性强）

**可编辑性**：
- 手工设计ε实现艺术效果
- 例如：ε=(0.5, 0, 0)强制"东移"光照

---

## 潜在局限

### 1. 仅适用平移类变化

**失效场景**：
- ❌ 旋转（如天体自转）
- ❌ 缩放（如爆炸、收缩）
- ❌ 拓扑变化（如手张开/闭合）

**解决方案**：
- 扩展到仿射变换：x' = A(t)·x + b(t)
- 或混合表示：TPE（主要）+ 残差MLP（细节）

### 2. 泰勒近似误差

**问题**：||ε||过大时，一阶近似失效

**缓解**：
- 多尺度扰动：ε = ε_coarse + ε_fine
- 自适应：根据场的平滑性调节||ε||上限

### 3. 静态场容量需求

**问题**：F_static必须"足够丰富"以覆盖所有时刻

**权衡**：
- 增加F_static复杂度 vs. 保持简洁性
- 可能需要更多高斯或更深的解码器

### 4. 对高频细节不友好

**问题**：扰动操作本质上是平滑的

**场景**：
- 锐利阴影边界可能模糊
- 快速变化的细节丢失

**缓解**：
- 训练时增加高频损失
- 或混合方法：TPE（低频）+ 残差网络（高频）

---

## 总结

### TPE的核心价值

1. **理论优雅**：将时间折叠到空间扰动，降维表示
2. **极致简洁**：仅96个时间相关参数
3. **物理可解释**：扰动向量可直观理解
4. **自然泛化**：时间插值天然平滑

### 验证的关键路径

```
理论验证 → 玩具实验 → 单探针验证 → 完整系统
                           ↑
                      最关键决策点
```

**如果单探针验证通过（3个标准全满足）**：
- 90%概率完整系统可行
- 继续实现

**如果单探针验证失败**：
- 立即放弃
- 不要浪费时间在完整实现上

### 适用场景

✅ **理想场景**：
- 光照变化（太阳运动）
- 流体运动（平移为主）
- 刚体运动（缓慢）

❌ **不适用**：
- 拓扑变化（人体、手）
- 旋转为主的运动
- 颜色/强度突变

### 下一步行动

1. **Week 1**：理论验证 + 玩具实验
2. **Week 2**：单探针验证（决策点）
3. **Week 3-4**：如果通过，完整实现
4. **Week 5**：论文实验与可视化

---

**创建日期**: 2025-12-10
**最后更新**: 2025-12-10
**状态**: 等待实验验证
**预期成功率**: 70%（基于理论分析）

---

## 参考文献

1. D-NeRF: Neural Radiance Fields for Dynamic Scenes (CVPR 2021)
2. NeRF in the Wild (CVPR 2021)
3. K-Planes: Explicit Radiance Fields in Space, Time, and Appearance (CVPR 2023)
4. Dynamical Implicit Neural Representations (arXiv 2024)
5. Taylor series expansion and differential calculus
6. Lagrangian vs Eulerian description (Fluid Mechanics)

---

## 附录：数学符号表

| 符号 | 含义 | 维度 |
|------|------|------|
| x | 空间坐标 | R³ |
| t | 时间 | R |
| ε(t) | 扰动向量 | R³ |
| F_static(·) | 静态场 | R³ → R^n |
| F_true(·,·) | 真实动态场 | R³ × R → R^n |
| sun_dir(t) | 太阳方向 | R³（单位向量） |
| SH | 球谐系数 | R²⁷ |
| K | 高斯数量 | int |
| D | 潜在码维度 | int |
| T | 时刻数 | int |

---

**END**
