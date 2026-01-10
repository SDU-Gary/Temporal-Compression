# P0实验设计

**目标**: 系统性验证TPE假设的有效性

---

## 实验体系结构

P0阶段包含4个独立但相互关联的实验：

```
实验F (Ground Truth Validation)
   ↓ 提供数据
实验2 (Residual Analysis) ──────→ 判断是否需要混合模型

实验3 (Spatial Generalization) ──→ 测试空间泛化能力

实验4 (Baseline Comparison) ────→ 对比传统方法
```

---

## 实验F：真值验证

### 目的
通过渲染验证TPE的核心假设是否成立。

### 方法
对于每个测试小时t，渲染两个配置：

**Config A**: `F_static(x + ε(t))` - TPE预测的配置
- 位置：x + ε(t)（扰动后）
- 太阳方向：参考时刻

**Config B**: `F_gt(x, t)` - 真实配置
- 位置：x（原始）
- 太阳方向：当前小时

### 误差分解

```
E_generalization = ||F_static(x+ε) - F_tpe_pred(x+ε)||
                   ↑ 模型拟合误差

E_tpe_assumption = ||F_gt(x,t) - F_static(x+ε)||
                   ↑ TPE物理假设误差

E_total = ||F_gt(x,t) - F_tpe_pred(x+ε)||
          ↑ 总预测误差
```

### 判断标准
```
Success: E_tpe_assumption / E_total < 50%
```

TPE假设误差应小于总误差的一半。

---

## 实验2：残差分析

### 目的
量化TPE假设的系统性偏差，决定是否需要混合模型。

### 方法
```python
residual(t) = F_gt(x, t) - F_static(x + ε(t))
```

### 统计指标
- Mean |residual|
- Max |residual|
- Std(residual)
- 每小时残差范数

### 判断标准
```
if TPE_contribution > 5%:
    recommend hybrid model
```

---

## 实验3：空间泛化

### 目的
测试TPE是否能泛化到未训练的空间位置。

### 方法
在10个空间分布的探针上独立运行TPE验证，检查：
1. 扰动范数 < 1.0m
2. 时间平滑性 < 0.5m
3. 太阳相关性 > 0.5

### 判断标准
```
Success: pass_rate >= 70%
```

---

## 实验4：基线对比

### 目的
对比TPE与传统时间插值方法的性能。

### Baseline方法

1. **Spline**: 三次样条插值（每个SH系数独立）
2. **DirectMLP**: hour → 27 SH系数
3. **TDML**: TimeEncoder + Decoder

### 数据分割
```
Train: [7, 9, 11, 13, 15, 17]  # 奇数小时
Test:  [6, 8, 10, 12, 14, 16, 18]  # 偶数小时
```

### 评价指标
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)

---

## 实验配置

### 场景选择
- **Cornell Box**: 室内封闭，复杂间接光照
- **House**: 室外开阔，直射光为主

### 渲染质量
- SPP: 256
- SH samples: 64
- SH阶数: 3阶（27系数）

### 时间采样
- 范围: 6:00 - 18:00
- 间隔: 1小时
- 总计: 13个时刻

### 验证小时
选择4个代表性时刻：[6, 9, 15, 18]
- 覆盖早中晚
- 包含极端太阳角度

---

[→ 下一章：Cornell Box结果](03_Cornell_Box_Results.md)
