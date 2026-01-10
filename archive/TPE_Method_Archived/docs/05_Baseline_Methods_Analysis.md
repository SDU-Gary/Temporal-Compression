# Baseline方法详细分析

对比三种时间插值方法：Spline, DirectMLP, TDML

---

## 方法概览

| 方法 | 类型 | 参数量 | 训练 |
|------|------|--------|------|
| **Spline** | 确定性插值 | 0 | 无需训练 |
| **DirectMLP** | 神经网络 | ~20k | 1000 epochs |
| **TDML** | 度量学习 | ~30k | 1000 epochs |

---

## 1. Spline（三次样条插值）

### 实现
```python
from scipy.interpolate import CubicSpline

class SplineBaseline:
    def train(self, hours, sh_gt):
        self.splines = []
        for i in range(27):  # 每个SH系数独立
            spline = CubicSpline(hours, sh_gt[:, i])
            self.splines.append(spline)

    def predict(self, hours_query):
        predictions = [spline(hours_query) for spline in self.splines]
        return np.stack(predictions, axis=-1)
```

### 理论保证
- **C²连续**: 二阶导数连续
- **插值性**: 通过所有训练点
- **最优性**: 在样条空间中最小化曲率

### 性能

| 场景 | MAE | RMSE |
|------|-----|------|
| Cornell Box | **0.0312** | 0.0524 |
| House | **0.0002** | 0.0002 |

### 为什么这么好？

1. **匹配问题特性**: SH系数时间变化平滑
2. **无需学习**: 数学保证的平滑性
3. **无参数**: 不会过拟合

---

## 2. DirectMLP（直接神经网络）

### 架构
```python
Input: hour (1D)
   ↓
Linear(1 → 128) + ReLU
   ↓
Linear(128 → 128) + ReLU
   ↓
Linear(128 → 27)
   ↓
Output: 27 SH系数
```

### 训练
```
Optimizer: Adam (lr=0.001)
Epochs: 1000
Loss: MSE(sh_pred, sh_gt)
```

### 性能

| 场景 | MAE | RMSE | vs Spline |
|------|-----|------|-----------|
| Cornell Box | 0.0458 | 0.0591 | +47% |
| House | 0.0003 | 0.0006 | +50% |

### 为什么比Spline差？

1. **需要学习平滑性**: ReLU需要学习C²性质
2. **小样本过拟合**: 6个训练点 vs 20k参数
3. **缺乏归纳偏置**: 没有编码时间的特殊性

---

## 3. TDML（时间距离度量学习）

### 架构
```python
Stage 1: Time Encoding
  hour → Fourier Features (16D)
        → TimeEncoder MLP → 16D embedding

Stage 2: Decoding
  16D embedding → Linear → 27 SH系数
```

### Fourier特征
```python
freqs = [f1, f2, ..., f8]  # 可学习
features = [sin(2πf1·t), cos(2πf1·t), ..., sin(2πf8·t), cos(2πf8·t)]
```

### 性能

| 场景 | MAE | RMSE | vs Spline |
|------|-----|------|-----------|
| Cornell Box | 0.0725 | 0.0935 | +132% |
| House | 0.0158 | 0.0195 | **+7800%** |

### 为什么崩溃了？

**训练现象**:
```
Epoch 200: Loss = 0.000000
Epoch 400: Loss = 0.000000
...
训练集完全拟合，测试集巨大误差
```

**原因分析**:
1. **过拟合严重**: 30k参数 vs 6个训练点
2. **错误的归纳偏置**:
   - Fourier适合周期信号
   - SH变化不是周期的（6-18点是单调趋势）
3. **高频噪声**: 8个频率学到了噪声模式

---

## 性能总结表

### Cornell Box

| 方法 | MAE | RMSE | 训练时间 | 推理时间 |
|------|-----|------|----------|----------|
| **Spline** | **0.0312** | **0.0524** | 0s | <1ms |
| DirectMLP | 0.0458 | 0.0591 | ~30s | <1ms |
| TDML | 0.0725 | 0.0935 | ~40s | <1ms |

### House

| 方法 | MAE | RMSE | 训练时间 | 推理时间 |
|------|-----|------|----------|----------|
| **Spline** | **0.0002** | **0.0002** | 0s | <1ms |
| DirectMLP | 0.0003 | 0.0006 | ~30s | <1ms |
| TDML | 0.0158 | 0.0195 | ~40s | <1ms |

---

## 核心教训

### 1. Occam's Razor胜利

```
最简单的方法（Spline）效果最好
```

### 2. 归纳偏置至关重要

| 方法 | 归纳偏置 | 匹配度 |
|------|----------|--------|
| Spline | C²连续性 | ✓ 完美匹配平滑信号 |
| DirectMLP | ReLU非线性 | △ 需要学习平滑性 |
| TDML | Fourier周期性 | ✗ 不匹配单调趋势 |

### 3. 数据量决定方法复杂度

```
6个训练点:
  Spline    → 刚好（唯一确定三次曲线）
  DirectMLP → 容易欠拟合
  TDML      → 严重过拟合
```

### 4. 不要为了深度学习而深度学习

对于平滑插值问题，传统方法更好。

---

[← 返回索引](00_README.md)
