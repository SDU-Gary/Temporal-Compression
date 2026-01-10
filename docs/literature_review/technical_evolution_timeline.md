# 神经渲染与光照压缩技术演变时间轴

## 总览

本文档整合所有已分析论文，构建从经典渲染到现代神经方法的完整技术演变脉络，聚焦**光照表示、神经压缩、实时渲染**三大主题。

**时间跨度**: 1984-2025 (41年)
**分析论文**: 12篇核心 + 29篇里程碑
**研究主题**: 面向多时刻光照的层次化神经压缩方法

---

## Phase 1: 经典渲染基础 (1984-2002)

### 核心突破: 体积渲染与预计算光照

#### 1984 | Volume Rendering ★★★★★
**论文**: Volume Rendering [Kajiya & Von Herzen, SIGGRAPH'84]
**贡献**:
- 体积密度光线追踪理论
- 渲染方程: C = ∫ T(t)σ(t)c(t)dt
- 透射率: T(t) = exp(-∫σ(s)ds)

**影响**: 所有神经体渲染的物理基础

---

#### 1996 | Light Field Rendering ★★★★★
**论文**: The Light Field [Levoy & Hanrahan, SIGGRAPH'96]
**贡献**:
- 7D全光函数 → 5D光场 (假设无遮挡)
- 双平面参数化: L(u,v,s,t)
- 四维插值渲染

**影响**: 基于图像的渲染经典，NeRF的概念前身

---

#### 2002 | Precomputed Radiance Transfer (PRT) ★★★★★
**论文**: [Sloan, Kautz, Snyder, SIGGRAPH'02]
**贡献**:
- 球谐基函数: y = ∫ L(ω)Y_i(ω)dω
- 传输矩阵: E = TL (光传输解耦)
- 实时低频全局光照

**影响**: 预计算光照的开创性工作，Neural PRT的直接前身

**技术要点**:
- 漫反射PRT: ρ/π ∫ V(ω)max(n·ω,0)dω
- 球谐投影: L_i = ∫ L(ω)Y_i(ω)dω
- 实时渲染: E = Σ c_i T_i y_i

---

## Phase 2: 神经网络初探 (2013-2019)

### 核心突破: 神经隐式表示

#### 2013 | VoxelHashing ★★★★
**论文**: Real-time 3D Reconstruction [Nießner et al., SIGGRAPH'13]
**贡献**:
- 哈希表用于3D场景表示
- 稀疏体素存储

**影响**: Instant NGP多分辨率哈希编码的灵感来源

---

#### 2017 | Transformer ★★★★
**论文**: Attention is All You Need [Vaswani et al., NeurIPS'17]
**贡献**:
- 位置编码: PE(pos, 2i) = sin(pos/10000^(2i/d))
- 虽然目的不同，但启发了频率编码概念

---

#### 2018 | Spectral Bias ★★★★★
**论文**: On the Spectral Bias of Neural Networks [Rahaman et al., ICML'18]
**贡献**:
- 揭示MLP倾向学习低频函数
- 高频信号需要更深网络或特殊编码

**影响**: 启发NeRF位置编码设计的关键理论

---

#### 2019 | Scene Representation Networks (SRN) ★★★★
**论文**: [Sitzmann et al., NeurIPS'19]
**贡献**:
- 神经隐式场景表示
- 单点表面 (vs. NeRF的体积)

---

#### 2019 | Local Light Field Fusion (LLFF) ★★★★
**论文**: [Mildenhall et al., SIGGRAPH'19]
**作者**: Ben Mildenhall (NeRF第一作者)
**贡献**:
- 多平面图像(MPI)表示
- 局部光场融合

**影响**: NeRF的直接前身

---

## Phase 3: NeRF范式 (2020)

### 核心突破: 5D神经辐射场

#### 2020 | Neural Radiance Fields (NeRF) ★★★★★
**论文**: [Mildenhall et al., ECCV'20]
**贡献**:
- **5D表示**: F_Θ: (x,y,z,θ,φ) → (r,g,b,σ)
- **位置编码**: γ(p) = [sin(2^0πp), cos(2^0πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp)]
- **体积渲染**: C(r) = Σ T_i α_i c_i, T_i = exp(-Σ_{j<i} σ_j δ_j)
- **分层采样**: 粗网络 + 细网络

**性能**:
- PSNR 31.01 dB (Lego场景)
- 训练时间: 1-2天
- 渲染速度: 30秒/帧

**影响**: 开启神经辐射场时代

**已分析**: `/home/kyrie/毕设/analysis/nerf_analysis.md`

---

#### 2020 | SIREN ★★★★
**论文**: Implicit Neural Representations with Periodic Activation Functions [Sitzmann et al., NeurIPS'20]
**贡献**:
- **周期激活**: φ(x) = sin(ω_0 x)
- **初始化**: 均匀分布 U(-√(6/n)/ω_0, √(6/n)/ω_0)
- **高频细节**: 无需位置编码即可学习高频

**应用**:
- 图像: PSNR 35-40 dB
- SDF: Chamfer距离 0.001
- 音频: 高保真重建

**影响**: 周期激活函数成为神经隐式表示的重要选择

**已分析**: `/home/kyrie/毕设/analysis/siren_analysis.md`

---

## Phase 4: NeRF扩展与应用 (2021)

### 方向1: 质量提升

#### 2021 | mip-NeRF ★★★★
**论文**: [Barron et al., ICCV'21]
**贡献**:
- 集成位置编码(IPE): 用圆锥代替光线
- 抗锯齿，多尺度一致

**性能**: PSNR提升, 比NeRF快7%

---

### 方向2: 逆渲染与光照分解

#### 2021 | NeRFactor ★★★★
**论文**: Neural Factorization of Shape and Reflectance Under an Unknown Illumination [Zhang et al., SIGGRAPH'21]
**贡献**:
- **NeRF几何蒸馏**: 表面点 x_surf = o + ∫ T(t)σ(r(t))t dt · d
- **BRDF分解**: Lambertian + 学习镜面反射
- **可见性建模**: MLP预测阴影
- **数据驱动BRDF先验**: 从MERL数据集学习

**架构**:
```
NeRF预训练 → 表面提取
  ↓
法线MLP + 可见性MLP + 反照率MLP + BRDF Identity MLP
  ↓
联合优化: 形状 + 反射 + 光照
```

**性能**:
- 重光照PSNR 25-30 dB
- 支持自由视角 + 任意光照
- 材质编辑

**影响**: 逆渲染框架，可扩展到多时刻光照

**已分析**: `/home/kyrie/毕设/analysis/nerfactor_analysis.md`

---

#### 2021 | Neural Radiance Caching (NRC) ★★★★★
**论文**: Real-time Neural Radiance Caching for Path Tracing [Müller et al., SIGGRAPH'21]
**贡献**:
- **自训练机制**: Q-learning for radiance transport
- **完全融合神经网络**: 9× faster than TensorFlow
- **在线学习**: 实时适应场景变化
- **路径终止启发式**: a(x_1...x_n) = (Σ√(d²/p|cosθ|))²

**架构**:
```
路径追踪 → 终止判断 → 缓存查询
  ↓                      ↓
训练路径              推理路径
  ↓                      ↓
自训练 (L_target)    MLP输出辐射度
```

**网络**: 7层×64神经元, 无偏置
**编码**: 频率编码 + One-blob编码
**优化**: Adam, 相对L2损失, EMA平滑

**性能**:
- 60+ FPS on all test scenes
- 13.6× average speedup
- MRSE降低1-2个数量级

**影响**: 在线学习范式可扩展到时变光照

**已分析**: `/home/kyrie/毕设/analysis/neural_radiance_caching_analysis.md`

---

### 方向3: 野外场景与泛化

#### 2021 | NeRF in the Wild (NeRF-W) ★★★★
**论文**: [Martin-Brualla et al., CVPR'21]
**贡献**:
- **外观嵌入**: 每张图像一个潜在码
- **瞬态嵌入**: 建模动态对象
- **不确定性估计**

**影响**: 光照变化建模的早期探索

---

## Phase 5: 效率革命 (2022)

### 核心突破: 从小时到秒

#### 2022 | Instant Neural Graphics Primitives (Instant NGP) ★★★★★
**论文**: [Müller et al., SIGGRAPH'22]
**贡献**:
- **多分辨率哈希编码**: 16层几何级数分辨率
- **哈希函数**: h(x) = (⊕x_i π_i) mod T
- **隐式碰撞解决**: 神经网络学习消歧
- **完全融合CUDA内核**: 极致GPU优化

**架构**:
```
输入x → 哈希编码(16层×2维) → 32维
  ↓
MLP (1层×64) → 16维 (首维=log密度)
  ↓
拼接 球谐编码(方向)
  ↓
MLP (2层×64) → RGB
```

**性能**:
- **训练速度**: 5秒达NeRF质量, 15秒超越, 1分钟接近mip-NeRF
- **vs. NeRF**: 1000×+ 加速
- **渲染**: 60 FPS @ 1920×1080
- **压缩**: 5 MB模型

**影响**: NeRF的效率革命，实时神经渲染新时代

**已分析**: `/home/kyrie/毕设/analysis/instant_ngp_analysis.md`

---

#### 2022 | Neural Precomputed Radiance Transfer (Neural PRT) ★★★★
**论文**: [Rainer et al., SIGGRAPH'22]
**贡献**:
- **神经传输算子**: T_Θ: L_env → E_scene
- **算子因式分解**: T = SE (Slicing + Environment mapping)
- **架构**: Image Encoder + ViT + Image Decoder
- **快速烘焙**: 2-10分钟 (vs. 传统数小时)

**性能**:
- PSNR 36-44 dB
- 推理: 5-30 ms
- 支持动态光照

**影响**: PRT与神经网络融合的典范

**已分析**: `/home/kyrie/毕设/analysis/neural_prt_analysis.md`

---

## Phase 6: 动态场景与时空表示 (2022-2023)

### 核心突破: 时空解耦与4D表示

#### 2022 | NeRV ★★★★
**论文**: NeRV: Neural Representations for Videos [Chen et al., NeurIPS'21]
**贡献**:
- **帧索引输入**: 直接用帧号作为输入
- **上采样架构**: NeRV Block = Conv + PixelShuffle
- **紧凑表示**: 网络权重即视频

**性能**:
- 压缩比: 10-100×
- 解码速度: 10-30× faster than H.264

**影响**: 隐式神经表示用于视频压缩

**已分析**: `/home/kyrie/毕设/analysis/nerv_analysis.md`

---

#### 2023 | K-Planes ★★★★
**论文**: K-Planes: Explicit Radiance Fields in Space, Time, and Appearance [Fridovich-Keil et al., CVPR'23]
**贡献**:
- **6平面分解**: (XY, XZ, YZ) × (静态, 动态)
- **时空解耦**: 时间平面独立建模
- **外观平面**: 光照变化建模

**架构**:
```
(x,y,z,t) → 投影到6个平面
  ↓
双线性采样 → 6个特征向量
  ↓
逐元素相乘 → 聚合特征
  ↓
小MLP → (σ, RGB)
```

**性能**:
- 动态场景PSNR 30-35 dB
- 训练时间: 分钟级
- 支持时空插值

**影响**: 显式4D表示，时空解耦思想

**已分析**: `/home/kyrie/毕设/analysis/k_planes_analysis.md`

---

#### 2023 | Neural Light Transport ★★★★
**论文**: Learning Neural Light Transport [Lin et al., SIGGRAPH'23 - 推测]
**贡献**:
- **可微光传输**: 神经网络建模L = Θ(I_direct)
- **多次反弹**: 递归光传输
- **重光照**: 支持光源编辑

**架构**: Encoder-Decoder with attention

**影响**: 神经光传输建模

**已分析**: `/home/kyrie/毕设/analysis/neural_light_transport_analysis.md`

---

#### 2023 | Dynamic Light Fields ★★★
**论文**: [作者未知, 2023推测]
**贡献**:
- 时变光场表示
- 动态光源建模

**影响**: 光场的时间扩展

**已分析**: `/home/kyrie/毕设/analysis/dynamic_light_fields_analysis.md`

---

### 混合表示: 显式+隐式

#### 2021 | Light Field Networks ★★★★
**论文**: Light Field Networks: Neural Scene Representations with Single-Evaluation Rendering [Sitzmann et al., NeurIPS'21]
**贡献**:
- **直接光场预测**: Φ:(r,d) → (RGB, α)
- **无体积渲染**: 单次评估
- **SIREN架构**: sin激活函数

**架构**:
```
光线(r,d) → SIREN网络 → (RGB, α)
  ↓
直接输出，无需积分
```

**性能**:
- 推理速度: 100× faster than NeRF
- 质量: 接近NeRF

**影响**: 直接光场预测范式

**已分析**: `/home/kyrie/毕设/analysis/light_field_networks_analysis.md`

---

## Phase 7: 压缩与实用化 (2024-2025)

### 核心突破: 神经压缩

#### 2025 | Gaussian Compression for Precomputed Indirect Illumination ★★★★
**论文**: [Zhou et al., SIGGRAPH'25]
**贡献**:
- **高斯探针压缩(GPC)**: 高斯函数 + MLP解码
- **级联光照体积(CLV)**: 动态更新，节省运行时内存
- **自适应量化**: 10-bit量化 + 精调
- **自定义CUDA**: 高效前向/反向传播

**架构**:
```
K个3D高斯 G_j (位置μ, 尺度s, 旋转q, 潜在码F)
  ↓
探针p → F(p) = Σ F_j G_j(p) (加权求和)
  ↓
MLP解码: [F(p), p] → SH系数
```

**CLV运行时**:
```
4级级联体积 (2m, 4m, 8m, 32m块)
  ↓
每帧更新≤64块 (帧分割)
  ↓
八叉树查找探针 → 实时解压
  ↓
LRU缓存 + 滚动索引
```

**性能**:
- **压缩比**: 1:50
- **质量**: PSNR 42.91 dB (低压缩), 44.82 dB (高压缩)
- **vs. CPCA**: PSNR +3.8 dB, MSE -78.6%
- **压缩时间**: 48-126秒 (取决于高斯数量)
- **解压缩**: 0.30-1.37 ms/frame
- **内存**: 27.95-160.87 KB

**影响**:
- 光照的神经压缩前沿
- 明确提出Time of Day为未来方向
- 与研究主题高度契合

**已分析**: `/home/kyrie/毕设/analysis/gaussian_compression_analysis.md`

---

## 技术演变主线

### 主线1: 表示方式演变

```
离散采样 (Light Field'96)
  ↓
预计算传输矩阵 (PRT'02)
  ↓
神经隐式表面 (DeepSDF'19, SRN'19)
  ↓
神经体积 (NeRF'20)
  ↓
混合表示 (Instant NGP'22: 哈希+MLP)
  ↓
显式平面 (K-Planes'23: 6平面分解)
  ↓
高斯混合 (Gaussian Compression'25: 高斯+MLP)
```

**趋势**: 离散 → 连续 → 混合(显式空间结构+隐式神经解码)

---

### 主线2: 编码策略演变

```
无编码 (MLP'19)
  ↓
位置编码 (NeRF'20: sin/cos频率编码)
  ↓
周期激活 (SIREN'20: sin(ω₀x))
  ↓
哈希编码 (Instant NGP'22: 多分辨率哈希)
  ↓
平面投影 (K-Planes'23: 6平面双线性采样)
  ↓
高斯基函数 (Gaussian Compression'25: 加权高斯)
```

**趋势**: 克服频谱偏差 → 自适应空间编码

---

### 主线3: 训练速度演变

```
NeRF'20: 1-2天
  ↓
NSVF'20: 数小时 (稀疏体素)
  ↓
Instant NGP'22: 5秒-1分钟 (1000×加速)
  ↓
K-Planes'23: 分钟级
  ↓
Gaussian Compression'25: 48-126秒 (压缩时间)
```

**关键突破**: Instant NGP的完全融合CUDA内核

---

### 主线4: 光照建模演变

```
隐式编码 (NeRF'20: 烘焙在RGB中)
  ↓
外观嵌入 (NeRF-W'21: 潜在码建模变化)
  ↓
显式分解 (NeRFactor'21: BRDF + 光照 + 几何)
  ↓
传输算子 (Neural PRT'22: T_Θ: L_env → E_scene)
  ↓
实时缓存 (NRC'21: 在线学习辐射度)
  ↓
时空解耦 (K-Planes'23: 外观平面)
  ↓
探针压缩 (Gaussian Compression'25: SH系数压缩)
```

**未来方向**: 多时刻光照 (Time of Day)

---

### 主线5: 实时性演变

```
NeRF'20: 30秒/帧
  ↓
Plenoctrees'21: 30 FPS
  ↓
Instant NGP'22: 60 FPS @ 1080p
  ↓
NRC'21: 60+ FPS (路径追踪)
  ↓
Light Field Networks'21: 100× faster
  ↓
Gaussian Compression'25: 实时解压(0.48ms) + 60 FPS渲染
```

**关键**: GPU优化 + 混合表示

---

## 关键技术演变对比表

| 维度 | NeRF'20 | Instant NGP'22 | K-Planes'23 | Gaussian Comp.'25 |
|------|---------|----------------|-------------|-------------------|
| **表示** | 纯MLP | 哈希+小MLP | 6平面+小MLP | 高斯+小MLP |
| **编码** | 位置编码(60维) | 多分辨率哈希(32维) | 平面投影 | 高斯加权和(9维) |
| **训练** | 1-2天 | 5秒-1分钟 | 分钟级 | 48-126秒(压缩) |
| **推理** | 30秒/帧 | 60 FPS | 实时 | 0.48ms解压 |
| **参数** | ~5M | ~5M | 平面参数 | 27-161KB |
| **场景** | 静态 | 静态 | 动态(时空) | 静态(多时刻潜力) |
| **光照** | 隐式烘焙 | 隐式烘焙 | 外观平面解耦 | SH系数压缩 |
| **质量** | PSNR~31 | PSNR~35(15s) | PSNR~30-35 | PSNR~43-45 |

---

## 面向多时刻光照的技术演变

### 相关技术线索

#### 1. 光照分解技术
```
PRT'02 (球谐分解)
  ↓
NeRFactor'21 (BRDF分解)
  ↓
Neural PRT'22 (传输算子)
  ↓
Gaussian Compression'25 (SH系数)
  ↓
? Time of Day (时间维度扩展)
```

#### 2. 时空建模技术
```
NeRF-W'21 (外观嵌入)
  ↓
K-Planes'23 (时空6平面)
  ↓
? 4D高斯 (空间3D + 时间1D)
```

#### 3. 压缩技术
```
NeRV'22 (视频压缩)
  ↓
Gaussian Compression'25 (光照压缩)
  ↓
? 时变光照联合压缩
```

#### 4. 层次化技术
```
Instant NGP'22 (16层分辨率)
  ↓
NRC'21 (在线自适应)
  ↓
Gaussian Compression'25 (4级级联体积)
  ↓
? 时空层次(空间级联 × 时间级联)
```

---

## 研究主题定位: 技术栈整合

### 可借鉴的核心技术

#### 从Gaussian Compression'25:
- ✅ 高斯+MLP混合表示 → 扩展到4D高斯
- ✅ 级联体积(CLV) → 时空级联
- ✅ 自定义CUDA优化 → 时空网络加速
- ✅ 自适应量化 → 时间维度量化
- ✅ LRU缓存 → 时空相干性

#### 从K-Planes'23:
- ✅ 时空解耦思想 → 空间3平面 + 时间平面
- ✅ 外观平面 → 建模光照变化
- ✅ 显式分解 → 可解释性

#### 从Neural PRT'22:
- ✅ 传输算子 → 光传输物理约束
- ✅ ViT架构 → 时空注意力
- ✅ 快速烘焙 → 多时刻预计算

#### 从NRC'21:
- ✅ 在线学习 → 时变场景自适应
- ✅ 完全融合网络 → GPU优化范式
- ✅ 自训练机制 → 时间相干性训练

#### 从Instant NGP'22:
- ✅ 多分辨率哈希 → 时空哈希编码
- ✅ 隐式碰撞解决 → 时间碰撞处理
- ✅ 极致GPU优化 → 实时解压缩

#### 从NeRFactor'21:
- ✅ 光照分解 → 直接光/间接光分离
- ✅ 可见性建模 → 时变阴影
- ✅ 数据驱动先验 → 光照先验学习

---

## 可能的技术路线

### 路线1: 4D高斯 + 时空级联
```
输入: (x,y,z,t)
  ↓
4D高斯: G_j(x,y,z,t) = exp(-½(p-μ_j)ᵀΣ_j⁻¹(p-μ_j))
  - 位置: μ_j(t) (时变中心)
  - 或: 静态μ_j + 时间范围[t_start, t_end]
  ↓
时空加权: F(x,y,z,t) = Σ F_j(t)·G_j(x,y,z,t)
  ↓
时空MLP解码: Φ([F(x,y,z,t), x,y,z,t]) → (直接光, 间接光, 阴影)
  ↓
4级空间级联 × 多时刻
```

**优势**:
- 继承Gaussian Compression的高压缩比
- 时空连续表示
- 自适应高斯分布

---

### 路线2: 时空6+N平面 + 传输算子
```
空间: 3平面(XY,XZ,YZ)
时间: N个时刻平面
光照: M个光源平面
  ↓
投影采样 → 特征向量
  ↓
神经传输算子: T_Θ: (几何, 时间) → 光照
  ↓
分解输出: (直接光, 间接光, AO, 阴影)
```

**优势**:
- 显式时空解耦
- 可解释性强
- 支持光照编辑

---

### 路线3: 层次化时空哈希
```
空间: 16层哈希编码 (Instant NGP)
时间: L_t层时间哈希
  ↓
时空联合哈希: h(x,y,z,t) = h_spatial ⊕ h_temporal
  ↓
小MLP → 光照分量
  ↓
级联体积: 4级空间 × T个时刻
```

**优势**:
- 训练极快
- 紧凑表示
- 实时推理

---

## 技术挑战与对应解决方案

### 挑战1: 时间维度爆炸
**问题**: T个时刻 × N个探针 = TN倍数据
**解决**:
- Gaussian Compression: 联合压缩多时刻
- K-Planes: 时间平面共享空间特征
- NeRV: 时间索引输入

### 挑战2: 时间连续性
**问题**: 时刻间跳变，闪烁
**解决**:
- NRC: EMA平滑
- Gaussian Compression: 时间相干性量化
- 时间正则化损失

### 挑战3: 实时解压缩
**问题**: T个时刻同时解压
**解决**:
- Gaussian Compression: 毫秒级解压 + CLV动态更新
- Instant NGP: 完全融合CUDA内核
- 预测 + 插值混合

### 挑战4: 光照物理正确性
**问题**: 纯数据驱动丢失物理
**解决**:
- Neural PRT: 传输算子约束
- NeRFactor: BRDF物理模型
- 混合: 物理先验 + 神经残差

---

## 里程碑论文总结

### ★★★★★ 顶级里程碑 (7篇)
1. **Volume Rendering'84** - 体积渲染理论基石
2. **Light Field'96** - 光场概念
3. **PRT'02** - 预计算光传输开创
4. **Spectral Bias'18** - 频谱偏差理论
5. **NeRF'20** - 神经辐射场范式
6. **Instant NGP'22** - 1000×效率革命
7. **NRC'21** - 实时神经缓存

### ★★★★ 重要里程碑 (11篇)
8. **VoxelHashing'13** - 哈希表3D表示
9. **Transformer'17** - 位置编码启发
10. **SRN'19** - 神经隐式场景
11. **LLFF'19** - MPI表示
12. **SIREN'20** - 周期激活
13. **mip-NeRF'21** - 抗锯齿多尺度
14. **NeRF-W'21** - 野外场景
15. **NeRFactor'21** - 逆渲染分解
16. **Neural PRT'22** - 神经传输算子
17. **K-Planes'23** - 时空6平面
18. **Gaussian Compression'25** - 神经光照压缩

### ★★★ 补充里程碑 (7篇)
19. Light Field Networks'21
20. NeRV'22
21. Neural Light Transport'23
22. Dynamic Light Fields'23
23. Plenoctrees'21
24. NSVF'20
25. Fourier Features'20

---

## 研究主题技术栈建议

基于41年技术演变分析，面向**"多时刻光照的层次化神经压缩"**的推荐技术栈:

### 核心表示:
- **4D高斯** (空间3D + 时间1D) [借鉴Gaussian Compression'25]
- **时空级联体积** [借鉴CLV + K-Planes]
- **时变潜在码** [借鉴NeRF-W + Gaussian Compression]

### 编码策略:
- **时空哈希编码** [借鉴Instant NGP + 时间扩展]
- **或: 时空平面投影** [借鉴K-Planes + 扩展]

### 光照分解:
- **直接光 + 间接光** [借鉴NeRFactor]
- **传输算子约束** [借鉴Neural PRT]
- **SH系数表示** [借鉴PRT + Gaussian Compression]

### 压缩策略:
- **自适应量化** (10-bit) [Gaussian Compression]
- **高斯参数共享** (时间聚类)
- **时间冗余提取** [NeRV思想]

### 实时优化:
- **完全融合CUDA** [NRC + Instant NGP]
- **LRU时空缓存** [Gaussian Compression CLV]
- **预测 + 按需解压** [CLV帧分割]

### 训练策略:
- **端到端优化** [Gaussian Compression]
- **时间相干性损失**
- **物理约束正则化** [Neural PRT]

---

## 未来趋势预测 (2025-2027)

### 1. 时变光照神经压缩
- Time of Day全天光照压缩
- 动态光源编辑
- 季节/天气变化建模

### 2. 物理约束神经表示
- 光传输方程嵌入神经网络
- BRDF先验 + 神经残差
- 可验证的物理正确性

### 3. 极致实时性
- < 1ms解压缩
- 完全硬件加速(Tensor Core)
- 端到端可微渲染管线

### 4. 统一表示
- 静态 + 动态 + 时变光照统一框架
- 几何 + 材质 + 光照联合优化
- 多模态压缩(几何, 纹理, 光照)

---

## 参考文献索引

本时间轴基于以下已分析论文构建:

1. **NeRF** - `/home/kyrie/毕设/analysis/nerf_analysis.md`
2. **Instant NGP** - `/home/kyrie/毕设/analysis/instant_ngp_analysis.md`
3. **Neural PRT** - `/home/kyrie/毕设/analysis/neural_prt_analysis.md`
4. **K-Planes** - `/home/kyrie/毕设/analysis/k_planes_analysis.md`
5. **NeRV** - `/home/kyrie/毕设/analysis/nerv_analysis.md`
6. **Neural Light Transport** - `/home/kyrie/毕设/analysis/neural_light_transport_analysis.md`
7. **Dynamic Light Fields** - `/home/kyrie/毕设/analysis/dynamic_light_fields_analysis.md`
8. **Light Field Networks** - `/home/kyrie/毕设/analysis/light_field_networks_analysis.md`
9. **SIREN** - `/home/kyrie/毕设/analysis/siren_analysis.md`
10. **NeRFactor** - `/home/kyrie/毕设/analysis/nerfactor_analysis.md`
11. **Neural Radiance Caching** - `/home/kyrie/毕设/analysis/neural_radiance_caching_analysis.md`
12. **Gaussian Compression** - `/home/kyrie/毕设/analysis/gaussian_compression_analysis.md`

加上29篇里程碑论文识别文档:
- **Milestone Papers** - `/home/kyrie/毕设/analysis/milestone_papers_identification.md`
- **Citation Network** - `/home/kyrie/毕设/analysis/citation_network_layer2.md`
- **NeRF Forward Citations** - `/home/kyrie/毕设/analysis/nerf_forward_citations.md`

---

**文档版本**: v1.0
**创建日期**: 2025-12-06
**覆盖时间跨度**: 1984-2025 (41年)
**总论文数**: 41篇 (12核心 + 29里程碑)
