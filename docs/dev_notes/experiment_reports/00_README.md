# 多时刻光照压缩实验报告

**日期**: 2025-12-12
**实验周期**: P0阶段
**主要场景**: Cornell Box, House

---

## 文档结构

本报告分为以下7个部分，请按顺序阅读：

### 理论与设计

1. [**TPE理论与动机**](01_TPE_Theory_and_Motivation.md)
   - TPE假设的提出背景
   - 核心思想：时间→空间的映射
   - 理论优势与预期效果
   - 数学表达与物理直觉

2. [**P0实验设计**](02_P0_Experiment_Design.md)
   - 四个验证实验的设计思路
   - 实验F：真值验证
   - 实验2：残差分析
   - 实验3：空间泛化
   - 实验4：基线对比

### 实验结果

3. [**Cornell Box场景结果**](03_Cornell_Box_Results.md)
   - 室内封闭场景的完整实验
   - TPE假设失效的详细分析
   - 误差分解与每小时数据
   - 关键发现：TPE贡献率84.9%

4. [**House场景对比实验**](04_House_Scene_Comparison.md)
   - 室外开阔场景的验证
   - 与Cornell Box的对比分析
   - 惊人的悖论：误差小但贡献率高
   - 扰动范数的数量级差异

### 深度分析

5. [**Baseline方法详细分析**](05_Baseline_Methods_Analysis.md)
   - Spline：三次样条插值
   - DirectMLP：直接神经网络
   - TDML：时间距离度量学习
   - 性能对比与方法评价

6. [**核心发现与启示**](06_Key_Findings_and_Insights.md)
   - TPE适用性的重新审视
   - Grid+Interpolation vs INR的学术真相
   - 为什么简单方法表现更好
   - 对论文发表的启示

### 展望与新方法

7. [**未来工作方向**](07_Future_Work_and_Recommendations.md)
   - 短期验证实验
   - 中期方法改进
   - 长期研究方向
   - 毕设建议

8. [**物理引导低秩压缩**](08_Physics_Low_Rank_Results.md)
   - 低秩假设验证（SVD分析）
   - 物理基函数 vs 神经基函数
   - Cornell Box: 2.34x压缩 + 33.5%精度提升
   - House: 2.92x压缩

9. [**高斯-物理混合压缩**](09_Gaussian_Physics_Hybrid.md)
   - 空间高斯混合 + 时间物理基
   - 多探针场景（343 probes）
   - 17.8x压缩比
   - 与任务书方法对比

---

## 快速导航

### 如果你想了解...

- **TPE是什么** → 从文档1开始
- **实验怎么设计的** → 直接看文档2
- **Cornell Box为什么失败** → 文档3
- **House场景的矛盾** → 文档4
- **为什么Spline这么好** → 文档5
- **最重要的发现** → 文档6（必读）
- **接下来做什么** → 文档7
- **低秩压缩方法** → 文档8（新方法）
- **空间-时间混合压缩** → 文档9（毕设方向）

### 关键数据速查

| 场景 | E_total | TPE贡献率 | Spline MAE | 结论 |
|------|---------|-----------|------------|------|
| Cornell Box | 0.3590 | 84.9% | 0.0312 | TPE严重失效 |
| House | 0.0019 | 96.5% | 0.0002 | 问题太简单 |

---

## 实验环境

- **渲染器**: Mitsuba 3 (CUDA variant)
- **数据集**: Level 2 TPE (单探针, 13小时)
- **质量**: SPP=256, SH_samples=64
- **SH阶数**: 3阶 (27系数)
- **时间范围**: 6:00-18:00 (每小时)

---

## 文件位置

- **实验代码**: `/home/kyrie/毕设/multi_time_compression/scripts/`
- **数据集**: `/home/kyrie/毕设/data_generation/output/level2_tpe/`
- **结果输出**: `/home/kyrie/毕设/multi_time_compression/output/`
- **本文档**: `/home/kyrie/毕设/multi_time_compression/docs/experiment_reports/`

---

## 版本历史

- **v1.2** (2025-12-12): 新增混合压缩方法
  - 新增：高斯-物理混合压缩（文档9）
  - 多探针场景实验（343 probes）
  - 17.8x压缩比验证

- **v1.1** (2025-12-12): 新增低秩压缩方法
  - 新增：物理引导低秩压缩（文档8）
  - 单探针2-3x压缩 + 精度提升

- **v1.0** (2025-12-12): 初始版本，完成P0阶段所有实验
  - Cornell Box: 4个实验全部完成
  - House: 4个实验全部完成
  - Baseline对比: 3种方法完整测试

---

**开始阅读**: [01 - TPE理论与动机 →](01_TPE_Theory_and_Motivation.md)
