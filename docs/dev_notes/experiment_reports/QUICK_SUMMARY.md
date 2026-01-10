# 实验快速摘要

**一句话总结**: TPE假设在两个测试场景都不成立，但简单的Spline插值表现最优。

---

## 核心数据

### Cornell Box（室内封闭）
```
TPE贡献率: 84.9% (阈值<50%) → FAIL
E_total: 0.3590
Spline MAE: 0.0312 (最优)
```

### House（室外开阔）
```
TPE贡献率: 96.5% → 看似更差
E_total: 0.0019 → 但误差极小（189倍差距）
Spline MAE: 0.0002 (几乎完美)
```

---

## 关键发现

### 1. TPE失效但方式不同

- **Cornell Box**: 真失败（复杂光照无法简化）
- **House**: 假失败（问题本身太简单）

### 2. Spline完胜

```
Spline > DirectMLP > TDML
```

两个场景都是Spline最优。

### 3. Grid > INR

学术界真相：
- 顶会方法都在用Grid+Interpolation
- 包装成"神经网络"
- 2024年研究证实：Grid常优于INR

---

## 三个教训

1. **不要被百分比迷惑** - 看绝对误差
2. **简单方法常常更好** - Occam's Razor
3. **归纳偏置很重要** - TDML的Fourier特征不匹配

---

## 下一步

**短期**: 测试Staircase/Classroom（中等复杂度）
**中期**: Hybrid TPE（TPE+残差网络）
**长期**: 场景分类框架

---

## 毕设建议

**Option A**: 诚实对比研究（推荐）
- 标题："When Do We Need Neural Methods?"
- 贡献：系统对比，定义边界

**核心价值**: 负面结果也是贡献！

---

**完整文档**: 见 `00_README.md`

**文档位置**: `/home/kyrie/毕设/multi_time_compression/docs/experiment_reports/`
