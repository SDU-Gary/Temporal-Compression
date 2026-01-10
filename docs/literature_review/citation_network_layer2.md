# Citation Network Analysis - Layer 2 (Detailed Milestone Papers)

## Web Search Results Summary

本文档基于web search对Layer 1识别的milestone papers进行深入分析，并追踪前向引用到2023-2024年。

---

## Layer 2A: 理论基础论文详解

### 📌 Milestone 15: Spectral Bias of Neural Networks ★★★★★

**完整引用**:
- **Title**: "On the Spectral Bias of Neural Networks"
- **Authors**: Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred A. Hamprecht, Yoshua Bengio, Aaron Courville
- **Published**: ICML 2019 (International Conference on Machine Learning), Long Beach, CA
- **ArXiv**: [1806.08734](https://arxiv.org/abs/1806.08734)
- **PMLR**: [Proceedings](https://proceedings.mlr.press/v97/rahaman19a/)

**核心发现**:
1. **频谱偏差现象**: 深度ReLU网络对低频函数有偏好，难以学习局部波动
2. **频率依赖的学习速度**: 高频分量学习速度显著慢于低频分量
3. **工具**: 使用傅里叶分析工具证明这一现象

**理论意义**:
- ReLU-based MLPs在坐标网络中表现出频谱偏差，优先低频信号
- 这些网络学习高频分量更慢
- 启发了NeRF的位置编码设计

**被引用者**:
- NeRF [Mildenhall+ ECCV'20]
- Instant NGP [Müller+ SIGGRAPH'22]
- Fourier Features [Tancik+ NeurIPS'20]

**重要性**: ★★★★★ (理论基础石)

**Sources**:
- [ArXiv Paper](https://arxiv.org/abs/1806.08734)
- [PMLR Proceedings](https://proceedings.mlr.press/v97/rahaman19a/rahaman19a.pdf)

---

### 📌 Milestone 17: Fourier Features ★★★★

**完整引用**:
- **Title**: "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
- **Authors**: Matthew Tancik, Pratul Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan Barron, Ren Ng
- **Published**: NeurIPS 2020
- **ArXiv**: [2006.10739](https://arxiv.org/abs/2006.10739)
- **GitHub**: [tancik/fourier-feature-networks](https://github.com/tancik/fourier-feature-networks)
- **Project**: [Fourier Feature Networks](https://bmild.github.io/fourfeat/)

**核心贡献**:
1. **Fourier Feature Mapping**: 将输入点通过简单傅里叶特征映射，使MLP能学习高频函数
2. **NTK分析**: 使用Neural Tangent Kernel理论证明标准MLP对高频信号收敛慢
3. **可调带宽**: 傅里叶特征映射将有效NTK转换为可调带宽的平稳核

**技术细节**:
```
γ(v) = [cos(2πBv), sin(2πBv)]^T
其中B是从高斯分布采样的频率矩阵
```

**与NeRF关系**:
- 与NeRF同期发表 (2020)
- 互相验证位置编码的有效性
- NeRF使用确定性频率，Fourier Features使用随机频率

**影响**:
- 计算机视觉和图形学中神经表示的基础技术
- 3D场景和物体神经表示的重要工具

**重要性**: ★★★★

**Sources**:
- [NeurIPS 2020 Paper](https://proceedings.neurips.cc/paper/2020/file/55053683268957697aa39fba6f231c68-Paper.pdf)
- [ArXiv](https://arxiv.org/abs/2006.10739)
- [GitHub Repository](https://github.com/tancik/fourier-feature-networks)

---

## Layer 2B: PRT经典论文详解

### 📌 Milestone 4: Precomputed Radiance Transfer ★★★★★

**完整引用**:
- **Title**: "Precomputed Radiance Transfer for Real-Time Rendering in Dynamic, Low-Frequency Lighting Environments"
- **Authors**: Peter-Pike Sloan, Jan Kautz, John Snyder
- **Published**: SIGGRAPH 2002
- **Journal**: ACM Transactions on Graphics (Proc. SIGGRAPH 02), Vol. 21, No. 3, pp. 527-536
- **PDF**: [Direct Link](https://jankautz.com/publications/prtSIG02.pdf)

**技术方法**:
1. **实时技术**: 在低频光照环境下渲染漫反射和光泽物体
2. **效果**: 捕捉软阴影、相互反射、焦散
3. **球谐光照**: 在PRT系统中广泛使用，SH系数预计算并存储在物体顶点
4. **交互式组合**: SH光照系数与预计算传输系数交互式组合

**影响**:
- 实时图形学高度影响力工作
- 催生大量follow-up扩展工作：可变形物体、光泽材质等

**相关工作**:
- Companion paper: Kautz, J., Sloan, P., Snyder, J. "Fast, arbitrary BRDF shading for low-frequency lighting using spherical harmonics" (Eurographics Workshop on Rendering 2002, pp. 291-296)

**重要性**: ★★★★★ (开创性工作)

**Sources**:
- [SIGGRAPH History](https://history.siggraph.org/learning/precomputed-radiance-transfer-for-real-time-rendering-in-dynamic-low-frequency-lighting-environments/)
- [Tutorial PDF](https://www.inf.ufrgs.br/~oliveira/pubs_files/Slomp_Oliveira_Patricio-Tutorial-PRT.pdf)

---

## Layer 2C: NeRF前身与改进

### 📌 Milestone 9: Local Light Field Fusion (LLFF) ★★★★

**完整引用**:
- **Title**: "Local Light Field Fusion: Practical View Synthesis with Prescriptive Sampling Guidelines"
- **Authors**: Ben Mildenhall, Pratul Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, Abhishek Kar
- **Published**: SIGGRAPH 2019
- **ArXiv**: [1905.00889](https://arxiv.org/abs/1905.00889)
- **Project**: [https://bmild.github.io/llff/](https://bmild.github.io/llff/)
- **Code**: [GitHub - Fyusion/LLFF](https://github.com/Fyusion/LLFF)

**技术方法**:
1. **Multiplane Images (MPI)**: 将每个采样视图扩展为局部光场
2. **局部光场融合**: 渲染新视角时融合相邻局部光场
3. **Pipeline**:
   - 输入: 静态场景的图像集
   - 提升每张图像为局部分层表示 (MPI)
   - 从MPI渲染局部光场并融合

**性能**:
- 达到Nyquist采样率的感知质量
- 使用视图数量减少高达 **4000×**

**与NeRF关系**:
- NeRF第一作者 Ben Mildenhall 的前一年工作
- MPI表示 → NeRF的连续体积表示
- LLFF数据集成为NeRF的标准测试集

**重要性**: ★★★★ (NeRF直接前导)

**Sources**:
- [SIGGRAPH 2019 Paper](https://bmild.github.io/llff/index.html)
- [ArXiv](https://arxiv.org/abs/1905.00889)
- [ACM Digital Library](https://dl.acm.org/doi/10.1145/3306346.3322980)

---

### 📌 Milestone 25: mip-NeRF ★★★★

**完整引用**:
- **Title**: "Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields"
- **Authors**: Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, Pratul P. Srinivasan
- **Published**: ICCV 2021
- **ArXiv**: [2103.13415](https://arxiv.org/abs/2103.13415)
- **Project**: [https://jonbarron.info/mipnerf/](https://jonbarron.info/mipnerf/)

**核心创新**:
1. **Conical Frustums**: 用圆锥台替代光线进行渲染
2. **Integrated Positional Encoding (IPE)**:
   - 考虑高斯空间区域而非无穷小点
   - 自然输入"空间区域"作为查询
   - 允许网络推理采样和混叠
3. **3D Conical Frustum**: 每个像素投射完整3D圆锥，查询点关联3D圆锥台

**性能提升**:
- 比NeRF快 **7%**
- 模型大小减半
- NeRF数据集误差降低 **17%**
- 多尺度数据集误差降低 **60%**
- 比暴力超采样NeRF快 **22倍**，精度相当

**技术意义**:
- 高效渲染抗锯齿圆锥台
- 显著改善NeRF表示精细细节能力
- 减少混叠伪影

**重要性**: ★★★★ (质量重要改进)

**Sources**:
- [ICCV 2021 Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Barron_Mip-NeRF_A_Multiscale_Representation_for_Anti-Aliasing_Neural_Radiance_Fields_ICCV_2021_paper.pdf)
- [Project Page](https://jonbarron.info/mipnerf/)
- [ArXiv](https://arxiv.org/abs/2103.13415)

---

## Layer 2D: 神经渲染开创性工作

### 📌 Milestone 20: Deep Shading ★★★★

**完整引用**:
- **Title**: "Deep Shading: Convolutional Neural Networks for Screen Space Shading"
- **Authors**: Oliver Nalbach, Elena Arabadzhiyska, Dushyant Mehta, Hans-Peter Seidel, Tobias Ritschel
- **Published**: EGSR 2017 (28th Eurographics Symposium on Rendering)
- **Journal**: Computer Graphics Forum, Volume 36, Issue 4
- **DOI**: 10.1111/cgf.13225
- **ArXiv**: [1603.06078](https://arxiv.org/abs/1603.06078)
- **Project**: [http://deep-shading-datasets.mpi-inf.mpg.de/](http://deep-shading-datasets.mpi-inf.mpg.de/)
- **GitHub**: [marcelsan/DeepShading](https://github.com/marcelsan/DeepShading)

**核心贡献**:
1. **首个完整神经渲染器**: 使用CNN从像素属性合成外观
2. **Screen Space Effects**: 实时渲染屏幕空间效果
   - 环境光遮蔽
   - 间接光照
   - 散射
3. **从示例学习**: 从示例图像学习而非人工编程

**技术特点**:
- 竞争质量和速度
- 网络定义、预训练网络和数据集公开

**影响**:
- 神经渲染领域开创性工作
- 证明神经网络可学习复杂渲染效果

**重要性**: ★★★★ (开创性)

**Sources**:
- [Wiley Online Library](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.13225)
- [ArXiv](https://arxiv.org/abs/1603.06078)
- [Project Page](http://deep-shading-datasets.mpi-inf.mpg.de/)

---

## Layer 2E: Instant NGP前向引用 (2023-2024)

### 🔥 Follow-up 1: Compact Neural Graphics Primitives ★★★★

**完整引用**:
- **Title**: "Compact Neural Graphics Primitives with Learned Hash Probing"
- **Authors**: Towaki Takikawa, Josef Spjut, Xiaohui Zeng, et al.
- **Published**: SIGGRAPH Asia 2023
- **ArXiv**: [2312.17241](https://arxiv.org/abs/2312.17241)
- **Project**: [NVIDIA Toronto AI Lab](https://research.nvidia.com/labs/toronto-ai/compact-ngp/)

**核心改进**:
1. **Learned Hash Probing**: 学习哈希探测策略
2. **性能**:
   - 训练慢 1.2-2.6×
   - **推理比Instant NGP更快** (显著减小的文件大小更好利用缓存)
   - 全方位超越Instant NGP质量
3. **Size-Speed Trade-off**: 最优大小和速度组合

**技术突破**: 证明哈希表可以通过学习探测策略进一步优化

**重要性**: ★★★★

**Sources**:
- [SIGGRAPH Asia 2023](https://dl.acm.org/doi/10.1145/3610548.3618167)
- [ArXiv](https://arxiv.org/abs/2312.17241)
- [NVIDIA Research](https://research.nvidia.com/labs/toronto-ai/compact-ngp/)

---

### 🔥 Follow-up 2: CNC - Context-based NeRF Compression ★★★

**完整引用**:
- **Title**: "How Far Can We Compress Instant-NGP-Based NeRF?"
- **Published**: CVPR 2024
- **Authors**: Yihang Chen, Qianyi Wu
- **ArXiv**: [2406.04101](https://arxiv.org/abs/2406.04101)

**核心成果**:
1. **压缩比**:
   - Synthetic-NeRF: **100× 大小减少** (质量提升)
   - Tanks and Temples: **70× 大小减少** (质量提升)
2. **上下文建模**: 利用Instant-NGP结构进行上下文压缩

**意义**: 证明哈希编码具有极大压缩潜力

**重要性**: ★★★

**Sources**:
- [CVPR 2024 Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_How_Far_Can_We_Compress_Instant-NGP-Based_NeRF_CVPR_2024_paper.pdf)
- [ArXiv](https://arxiv.org/abs/2406.04101)

---

### 🔥 Follow-up 3: Grid4D - 4D Hash Encoding ★★★★

**完整引用**:
- **Title**: "Grid4D: 4D Decomposed Hash Encoding for High-Fidelity Dynamic Gaussian Splatting"
- **Authors**: Jiawei Xu, et al.
- **Published**: NeurIPS 2024
- **ArXiv**: [2410.20815](https://arxiv.org/html/2410.20815)
- **GitHub**: [JiaweiXu8/Grid4D](https://github.com/JiaweiXu8/Grid4D)

**核心创新**:
1. **4D分解**:
   - 1个空间3D哈希编码
   - 3个时间3D哈希编码
2. **动态场景**: 扩展哈希编码到时变场景
3. **vs. Plane-based**: 避免不适合的低秩假设

**技术意义**:
- 哈希编码从静态扩展到动态
- 与用户研究主题"多时刻光照"高度相关

**重要性**: ★★★★ (动态扩展)

**Sources**:
- [ArXiv](https://arxiv.org/html/2410.20815)
- [NeurIPS 2024](https://neurips.cc/virtual/2024/poster/94235)
- [GitHub](https://github.com/JiaweiXu8/Grid4D)

---

### 🔥 Follow-up 4: HAC - Hash-grid for 3DGS Compression ★★★

**完整引用**:
- **Title**: "HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression"
- **Authors**: Yihang Chen, et al.
- **Published**: ECCV 2024
- **ArXiv**: [2403.14530](https://arxiv.org/abs/2403.14530)
- **Project**: [https://yihangchen-ee.github.io/project_hac/](https://yihangchen-ee.github.io/project_hac/)

**核心成果**:
1. **压缩比**:
   - vs. vanilla 3DGS: **75× 大小减少** (质量提升)
   - vs. Scaffold-GS (SOTA): **11× 大小减少**
2. **Binary Hash Grid**: 建立连续空间一致性
3. **上下文建模**: 利用无组织锚点与结构化哈希网格的关系

**应用**: 3D Gaussian Splatting压缩

**重要性**: ★★★

**Sources**:
- [ECCV 2024](https://link.springer.com/chapter/10.1007/978-3-031-72667-5_24)
- [ArXiv](https://arxiv.org/abs/2403.14530)
- [Project Page](https://yihangchen-ee.github.io/project_hac/)

---

### 🔥 Follow-up 5: NGP-RT - Real-Time Novel View Synthesis ★★★

**完整引用**:
- **Title**: "NGP-RT: Fusing Multi-Level Hash Features with Lightweight Attention for Real-Time Novel View Synthesis"
- **Published**: July 2024
- **ArXiv**: [2407.10482](https://arxiv.org/abs/2407.10482)

**核心创新**:
1. **显式存储**: 颜色和密度作为哈希特征显式存储
2. **轻量级注意力**: 消歧哈希碰撞
3. **vs. Instant NGP**: 避免计算密集的MLP

**目标**: 进一步加速实时渲染

**重要性**: ★★★

**Sources**:
- [ArXiv](https://arxiv.org/abs/2407.10482)

---

### 📊 Instant NGP影响力总结

**2025年理论分析**:
> "Instant-NGP has been the state-of-the-art architecture of neural fields in recent years, with its signal-fitting capabilities attributed to its multi-resolution hash grid structure and used and improved in numerous following works."

**应用扩展**:
- NeRF压缩
- 3D Gaussian Splatting
- 动态场景渲染
- 实时新视角合成

**技术演变方向**:
1. **压缩**: 100×减少存储 (CNC)
2. **速度**: 优化哈希探测 (Compact NGP)
3. **动态**: 4D分解编码 (Grid4D)
4. **应用迁移**: 3DGS压缩 (HAC)

---

## Layer 2F: Neural PRT前向引用 (2022-2025)

### 🔥 Follow-up 1: Neural Radiance Transfer Fields ★★★

**完整引用**:
- **Title**: "Neural Radiance Transfer Fields for Relightable Novel-View Synthesis with Global Illumination"
- **Authors**: Lyu et al.
- **Published**: ECCV 2022

**核心方法**:
- 学习神经预计算辐射传输函数
- 隐式处理全局光照效果
- 使用新环境贴图重光照

**关系**: 引用Neural PRT (Rainer+ 2022)

**重要性**: ★★★

**Sources**:
- [SpringerLink](https://link.springer.com/chapter/10.1007/978-3-031-19790-1_10)

---

### 🔥 Follow-up 2: PRTGS - PRT for Gaussian Splatting ★★★★

**完整引用**:
- **Title**: "PRTGS: Precomputed Radiance Transfer of Gaussian Splats for Real-Time High-Quality Relighting"
- **Published**: 2024
- **ArXiv**: [2408.03538](https://arxiv.org/html/2408.03538)

**解决问题**:
- 3D Gaussian Splatting虽然加速训练/渲染
- 但实时高质量间接光照计算仍有挑战
- 动态重光照和阴影估计困难

**方法**: PRT应用于3D Gaussian Splatting

**意义**: 传统PRT与最新3DGS表示结合

**重要性**: ★★★★

**Sources**:
- [ArXiv](https://arxiv.org/html/2408.03538)

---

### 🔥 Follow-up 3: Neural-GASh ★★★

**完整引用**:
- **Title**: "Neural-GASh: A CGA-based neural radiance prediction pipeline for real-time shading"
- **Published**: 2025
- **ArXiv**: [2507.13917](https://arxiv.org/html/2507.13917)

**核心方法**:
1. **神经场方法**: 替代传统PRT刚性预计算阶段
2. **灵活神经网络**: 预测辐射传输
3. **实时全局光照**: 静态场景 + 动态光照

**技术特点**: 神经网络与传统PRT集成

**重要性**: ★★★

**Sources**:
- [ArXiv](https://arxiv.org/html/2507.13917)

---

## Layer 2总结: 技术演变矩阵

### 理论基础线 (2018-2020)

```
Spectral Bias (2018) → Fourier Features (2020)
    ↓                         ↓
 NeRF位置编码 (2020)    随机傅里叶特征
    ↓
Instant NGP哈希编码 (2022)
```

### PRT演变线 (2002-2025)

```
PRT原始 (2002) → Neural PRT (2022)
                      ↓
    ┌─────────────────┼─────────────────┐
    ↓                 ↓                 ↓
Neural Radiance   PRTGS (2024)    Neural-GASh (2025)
Transfer (2022)   (3DGS+PRT)      (神经+传统PRT)
```

### NeRF演变线 (2019-2024)

```
LLFF (2019) → NeRF (2020) → mip-NeRF (2021)
                  ↓
         Instant NGP (2022)
                  ↓
    ┌─────┬───────┼────────┬────────┐
    ↓     ↓       ↓        ↓        ↓
Compact  CNC   Grid4D    HAC    NGP-RT
NGP(23) (24)   (24)     (24)    (24)
```

### 跨领域融合 (2024+)

```
Instant NGP哈希编码
    ↓
┌───┴────────────┐
↓                ↓
NeRF压缩      3DGS压缩 ← PRT理论
(CNC)         (HAC)     (PRTGS)
    ↓            ↓
  动态场景 (Grid4D)
```

---

## 与研究主题的关联度评估

### 高度相关 (★★★★★):
1. **Grid4D** - 4D时变哈希编码，直接对应"多时刻"
2. **Neural PRT系列** - 动态光照建模
3. **Instant NGP系列** - 层次化哈希编码 = "层次化神经压缩"

### 中度相关 (★★★):
4. **Spectral Bias** - 理论基础
5. **mip-NeRF** - 多尺度表示
6. **PRTGS** - 实时重光照

### 基础性 (★★):
7. **PRT原始** - 传统PRT理论
8. **LLFF** - MPI多层表示

---

## 下一步行动

### Layer 3扩展计划:

#### 必读论文:
1. **Grid4D详细分析** - 4D时变编码直接相关
2. **PRTGS详细分析** - PRT+最新表示
3. **Compact NGP** - 哈希编码优化

#### Web Search计划:
- [ ] Grid4D backward citations (基于什么前置工作?)
- [ ] 3D Gaussian Splatting基础论文
- [ ] 时变光照建模相关工作 (temporal light transport)
- [ ] Neural relighting综述

#### 补充阅读:
- [ ] Reference文件夹中剩余10篇论文
- [ ] 识别与"多时刻光照"最相关的论文

### 时间线构建准备:

当前已覆盖时间范围: **2002-2025**
- Layer 1: 29个milestone papers
- Layer 2: 8个详细分析 + 8个前向引用
- **总计**: 37篇论文构成citation网络

准备构建:
1. **技术发展时间轴** (按年份 + 技术范式)
2. **问题演变时间轴** (按时期 + 核心问题)
