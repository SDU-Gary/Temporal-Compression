# Neural Precomputed Radiance Transfer

## 基本信息
- **作者**: Gilles Rainer, Adrien Bousseau, Tobias Ritschel, George Drettakis
- **机构**: Inria (Université Côte d'Azur), University College London
- **发表**: EUROGRAPHICS 2022 (Volume 41, Number 2)
- **类型**: 神经渲染 + 传统PRT结合

## 解决的核心问题

### 主要问题
1. **实时全局光照挑战**: 需要昂贵的高端光追GPU
2. **神经渲染设计空间探索**: 如何设计高质量的场景特定神经渲染器
3. **传统PRT限制**: 球谐基函数表达高频细节能力有限
4. **神经方法缺乏物理先验**: 现有神经渲染方法未充分利用传统渲染知识

### 研究配置
- **静态场景**: 几何和材质固定
- **动态光照**: 环境贴图光照可变
- **固定神经预算**: 训练时间和网络大小固定
- **对比传统PRT**: 预计算 → 运行时快速渲染

## 技术方法与Pipeline

### 核心思想: PRT启发的神经架构

受PRT原理启发，设计4种递进式神经架构（图2）:

#### **架构1: Baseline Transport Operator**
```
输入: [光照编码eˆ, G-buffer(x,ωo,ωr,n,kd,ks,α)]
  ↓
单个MLP (ReLU激活)
  ↓
输出: L (辐射度)
```

**公式**: L = Ŝ(eˆ, x, ωo, ωr, n, kd, ks, α)

**问题**:
- 64维光照 + 19维场景属性直接拼接
- 难以正确学习全局光照
- 倾向"烘焙"直接光照和高光

#### **架构2: PRT-Inspired Transport Operator**

**灵感来源**: PRT的矩阵-向量乘积 L = Se (Equation 9)

```
场景属性(x,ωo,ωr,n,kd,ks,α)
  ↓
FT (MLP with SIREN激活)
  ↓
传输描述符 sˆ (64维)

环境贴图 E
  ↓
FL (CNN编码器)
  ↓
光照描述符 eˆ (64维)

sˆ + eˆ
  ↓
Φ (MLP with ReLU激活)
  ↓
输出: L
```

**公式**:
- sˆ = FT(x, ωo, ωr, n, kd, ks, α)
- L = Φ(sˆ, eˆ)

**优势**:
- 光照和传输等维度 (64维 vs. 64维)
- FT: SIREN激活 → 高频传输
- Φ: ReLU激活 → 高维到RGB映射
- 更难"烘焙"直接光照

**PRT类比**:
- FT ↔ 传输矩阵 S
- FL ↔ 光照投影到基
- Φ ↔ 矩阵-向量乘积

#### **架构3: Albedo Factorization**

**灵感来源**: PRT分离反射算子 L = RSE (Equation 10)

```
sˆ + eˆ
  ↓
Φ (输出2×RGB)
  ↓
(LD, LS)

最终输出: L = kdLD + ksLS
```

**优势**:
- 网络不再学习空间变化的反照率
- 对复杂纹理场景特别有效
- 更连贯的辐射度分布

**PRT类比**: 反射算子分解 R = RD + RS

#### **架构4: Diffuse-Specular Separation**

**灵感来源**: PRT的漫反射-镜面分离 L = SDE + SSE (Equation 12)

```
漫反射分支:
  (x, n, kd) → FT^D (SIREN) → sˆD
  sˆD + eˆ → ΦD → LD
  输出: kdΦD(sˆD, eˆ)

镜面分支:
  (x, n, ks, ωo, ωr, α) → FT^S (SIREN) → sˆS
  sˆS + eˆ → ΦS → LS
  输出: ksΦS(sˆS, eˆ)

最终: L = kdΦD(sˆD, eˆ) + ksΦS(sˆS, eˆ)
```

**优势**:
- 注入物理归纳偏置
- 漫反射: 专注位置输入
- 镜面: 专注方向输入
- 每个分支神经元减少但总体质量提升

### 环境贴图编码 (Section 5.1)

**CNN编码器 FL**:
- 5个卷积块 (Conv + ReLU + MaxPool + Skip)
- 32通道卷积层
- 最终全连接层输出64维 eˆ

**训练策略**: 与传输算子联合训练，学习场景特定紧凑表示

### 传统PRT回顾 (Section 4)

#### 积分形式
```
Lo(x,ωo) = ∫Ω fr(ωi,ωo,kd,ks,α) Li(x,ωi) (ωi·n) dωi
```

#### 漫反射-镜面分离
```
Lo = kdLoD + ksLoS

LoD = ∫Ω (ωi·n)/π Li(x,ωi) dωi

LoS = ∫Ω [D(α)FG / (4(ωi·n)(ωo·n))] Li(x,ωi) (ωi·n) dωi
```

#### 矩阵形式 (Lehtinen 2007)
```
L = SE (solution operator)
L = RSE (reflection分解)
L = SDE + SSE (漫反射-镜面分离)
```

**Transfer vs. Transport**:
- Transfer: 环境光 → 局部入射光 (S)
- Transport: Transfer + 与BRDF卷积 (RS)

## 实验设置与结果

### 数据生成
- **渲染器**: Falcor实时路径追踪器
- **训练数据**: 每场景3000张 (256×256)
- **测试数据**: 每场景200张 (400×400)
- **光照**: Laval HDR数据库 (室内/室外环境贴图)
- **相机放置**: 主动体积内随机 + 金字塔视角

### 场景
1. **ATELIER**: 533K多边形 (自建)
2. **BEDROOM**: 1,499K多边形
3. **KITCHEN**: 1,443K多边形
4. **SANMIGUEL**: 5,608K多边形

### 训练细节
- **优化器**: Adam (lr=10^-4)
- **Epoch**: 500
- **损失函数**: L1 (log(1+x) tonemap后)
- **训练时间**: 16-18小时 (NVIDIA RTX6000)

### 性能对比 (Table 6, RMSE×10³)

| 方法 | ATELIER | BEDROOM |
|------|---------|---------|
| RTPT (5spp) | 230.02 | 511.64 |
| RTPT+Denoise | 453.46 | 1526.00 |
| NRC | 137.64 | 164.90 |
| NRC-cache | 72.22 | 77.44 |
| Deep Shading | 39.11 | 47.19 |
| PRT-9 (SH order 2) | 72.53 | 47.64 |
| PRT-25 (SH order 4) | 29.99 | 19.62 |
| **Ours (D/S)** | **22.02** | **21.76** |

### 推理速度 (512×512, RTX 3090)
| 组件 | 时间 |
|------|------|
| Encoder | 0.34 ms |
| Baseline | 9.73 ms |
| PRT-Inspired | 10.57 ms |
| Albedo Fact. | 10.60 ms |
| D/S Separation | 16.02 ms |

**对比**: Falcor RTPT (5spp) = 19-36 ms (场景相关)

### 架构性能递进 (Table 5, RMSE×10³)

**ATELIER场景**:
1. Baseline: 57.6
2. PRT-Inspired: 49.4 ↓14%
3. Albedo Fact.: 47.9 ↓3%
4. **D/S Separation**: **46.8** ↓2%

**BEDROOM场景**:
1. Baseline: 23.5
2. PRT-Inspired: 21.3 ↓9%
3. Albedo Fact.: 20.5 ↓4%
4. **D/S Separation**: **20.0** ↓2%

## 重要前置工作 (引用分析)

### Neural Rendering基础
1. **Deep Shading** [Nalbach+ EGSR'17]: 首个完整神经渲染器
2. **NeRF** [Mildenhall+ ECCV'20]: MLP表示场景
3. **Radiance Regression** [Ren+ TOG'13]: 神经网络表示间接光照

### PRT经典工作
4. **Sloan+ SKS02** [SIGGRAPH'02]: PRT开创性工作
5. **Lehtinen'07**: PRT算子理论框架
6. **Ng+ NRH04**: Haar小波基
7. **Tsai&Shih TS06**: 球径向基函数

### Neural + PRT结合
8. **Ren+ RDL'15**: 神经网络学习传输矩阵
9. **Xu+ XSHR18**: CNN预测PRT-SH系数
10. **Plenoctrees** [Yu+ ICCV'21]: SH编码NeRF方向信息

### 材质神经表示
11. **Neural BTF** [Rainer+ EG'19,'20]: 神经BTF压缩和插值
12. **Deep Appearance Maps** [Maximov+ ICCV'19]

### 重光照
13. **NeRF-W** [Martin-Brualla+ CVPR'21]: 野外场景重光照
14. **NeRD, NeRFactor** [Boss+, Zhang+ ICCV'21]: 显式BRDF学习

## 技术突破与贡献

### 1. 方法论创新
**将传统渲染理论注入神经架构设计**:
- 证明PRT原理可指导神经网络架构选择
- 系统探索设计空间 (4种递进架构)
- 物理归纳偏置提升质量而不增加成本

### 2. 架构设计原则 (来自PRT)
- ✓ **分离光照和传输**: 等维度描述符
- ✓ **因式分解反照率**: 学习辐射度分布
- ✓ **漫反射-镜面分离**: 物理正确的解耦
- ✓ **传输vs.运输**: 学习transfer而非transport

### 3. 技术优势
- **无需高端GPU**: 不依赖RT Cores
- **紧凑表示**: 2MB网络 vs. 传统PRT的巨大存储
- **连续插值**: 神经网络自带插值能力
- **场景特定优化**: 训练集中注意力在重要区域

### 4. 性能优势
- 超越传统PRT (SH-25): RMSE降低 ~10-40%
- 超越实时路径追踪 (5spp): RMSE降低 ~90%
- 超越Deep Shading: RMSE降低 ~40-50%
- 接近或超越Neural Radiance Cache

## 局限性

### 1. 场景特定训练
- 每个场景需要16-18小时训练
- 无法泛化到新场景
- 需要大量渲染数据 (3000张训练图)

### 2. 观测依赖
- 未见区域质量下降 (Fig. 14)
- 高频效果需要充分观测
- 相机放置策略影响质量

### 3. 网络容量限制
- 小MLP难以完全拟合大场景所有细节
- 完美镜面材质在未见区域有伪影
- 需要自适应场景划分 (future work)

### 4. 抗锯齿缺失
- 中心采样，无内置抗锯齿
- 需要后处理或多样本平均

## 与研究主题的关联

**"面向多时刻光照的层次化神经压缩方法"相关性分析:**

✓ **高度相关点:**
- **神经-PRT结合**: 提供了神经方法与传统PRT结合的完整范例
- **动态光照**: 环境贴图光照变化 → 扩展到时变光照
- **层次化思想**: 漫反射-镜面分离 → 多尺度光照分解
- **神经压缩**: 2MB网络表示完整光传输

✓ **可借鉴设计**:
- **编码器-传输-应用**三阶段架构
- **SIREN用于高频传输**
- **物理先验注入**: 漫反射/镜面分离
- **联合训练策略**: 光照编码器 + 传输网络

✗ **缺失方向 (研究机会)**:
- 未处理时变光照 (仅空间变化环境光)
- 未显式分解光照分量
- 静态场景假设
- 无法编辑光照

**定位**: Neural PRT展示了如何用PRT理论指导神经渲染设计，是神经方法与传统PRT结合的里程碑，可作为多时刻光照神经建模的方法论基础

## Milestone属性评估: ★★★★ (重要方法论)

**理由:**
1. **方法论贡献**: 首次系统展示PRT理论指导神经架构设计
2. **设计空间探索**: 4种架构的递进式分析
3. **性能验证**: 超越传统PRT和多种神经基线
4. **理论深度**: 深入连接传统渲染理论与神经方法

## 未来方向 (论文提出)

1. **动态场景**: 扩展到变形/动态几何
2. **缓存策略**: 静态光照/视角时缓存中间结果
3. **大场景**: 自适应场景划分 (类似KiloNeRF)
4. **环境贴图编码器**: 鲁棒球面畸变和时间闪烁
5. **抗锯齿**: 多样本或后处理滤波

## 技术细节补充

### SIREN vs. ReLU选择 (Table 2)
| 架构 | ATELIER RMSE | BEDROOM RMSE |
|------|--------------|--------------|
| Baseline (ReLU) | 0.058 | 0.023 |
| Baseline (SIREN) | 0.068 ❌ | 0.056 ❌ |
| PRT-Inspired (ReLU) | 0.056 | 0.022 |
| **PRT-Inspired (SIREN+ReLU)** | **0.049** ✓ | **0.021** ✓ |

**结论**: SIREN在FT中编码高频，ReLU在Φ中聚合

### 参数量对比 (Table 1)
| 架构 | 光照编码器 | 传输网络 | 总参数 | 存储 |
|------|-----------|---------|--------|------|
| Baseline | 290,848 | 219,651 | 510,499 | 2.0 MB |
| PRT-Insp. | 290,848 | 186,947 | 477,795 | 1.8 MB |
| Albedo Fact. | 290,848 | 187,718 | 478,566 | 1.8 MB |
| D/S Separ. | 290,848 | 194,226 | 482,074 | 1.9 MB |

**结论**: 几乎等参数量，但质量递进提升
