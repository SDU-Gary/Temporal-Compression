# Milestone Papers Identification (Layer 1 - Backward Citations)

## Methodology

从已分析的3篇核心论文中提取backward citations，根据以下标准识别milestone papers:

### Milestone判断标准
1. **被多篇论文引用** (跨论文引用频次)
2. **基础性/开创性工作** (paradigm-shifting)
3. **顶级会议发表** (SIGGRAPH, ECCV, CVPR, NeurIPS, ICCV)
4. **时间跨度** (覆盖重要历史节点)

## 引用统计

### 被多次引用的论文 (High Cross-Citation)

#### ⭐⭐⭐⭐⭐ Tier 1: 被所有3篇论文引用

1. **NeRF** [Mildenhall+ ECCV'20]
   - 引用者: Instant NGP, Neural PRT
   - 地位: 神经辐射场开创性工作
   - 类型: **MILESTONE**

#### ⭐⭐⭐⭐ Tier 2: 被2篇论文引用

2. **Spectral Bias** [Rahaman+ ICML'18]
   - 引用者: NeRF, Instant NGP
   - 贡献: 解释神经网络频谱偏差，启发位置编码
   - 类型: **MILESTONE** (理论基础)

3. **Transformer** [Vaswani+ NeurIPS'17]
   - 引用者: NeRF, Instant NGP
   - 贡献: 位置编码概念来源
   - 类型: **MILESTONE** (概念启发)

4. **Volume Rendering Foundations** [Kajiya&Von Herzen SIGGRAPH'84]
   - 引用者: NeRF, Instant NGP
   - 贡献: 体积渲染开创性工作
   - 类型: **MILESTONE** (经典基础)

5. **Radiance Regression** [Ren+ TOG'13]
   - 引用者: NeRF, Neural PRT
   - 贡献: 神经网络表示间接光照
   - 类型: **MILESTONE** (神经+渲染结合)

6. **NeRF-W** [Martin-Brualla+ CVPR'21]
   - 引用者: Neural PRT, 前向引用分析
   - 贡献: 野外场景+外观嵌入
   - 类型: **MILESTONE** (重要扩展)

## 按研究方向分类的Milestone Papers

### A. 体积渲染基础 (1984-1995)

#### 📌 **Milestone 1: Volume Rendering**
- **论文**: "Volume Rendering" [Kajiya & Von Herzen SIGGRAPH'84]
- **贡献**: 体积密度光线追踪基础理论
- **影响**: 所有神经体渲染方法的理论基石
- **Milestone等级**: ★★★★★ (经典基础)

#### 📌 **Milestone 2: Optical Models for Direct Volume Rendering**
- **论文**: [Max IEEE TVCG'95]
- **贡献**: 直接体积渲染的光学模型
- **影响**: 体积渲染物理正确性
- **Milestone等级**: ★★★★

#### 📌 **Milestone 3: Efficient Ray Tracing**
- **论文**: [Levoy TOG'90]
- **贡献**: 高效体积数据光线追踪
- **Milestone等级**: ★★★

### B. 预计算辐射传输 (2002-2007)

#### 📌 **Milestone 4: Precomputed Radiance Transfer**
- **论文**: "Precomputed Radiance Transfer for Real-Time Rendering in Dynamic, Low-Frequency Lighting Environments" [Sloan, Kautz, Snyder SIGGRAPH'02]
- **贡献**: PRT开创性工作，球谐基函数传输
- **影响**: 实时全局光照的经典方法
- **Milestone等级**: ★★★★★ (开创性)

#### 📌 **Milestone 5: All-Frequency Shadows using Non-linear Wavelet Lighting Approximation**
- **论文**: [Ng+ SIGGRAPH'03]
- **贡献**: Haar小波基替代球谐，支持高频
- **Milestone等级**: ★★★★

#### 📌 **Milestone 6: Matrix Row-Column Sampling**
- **论文**: "Matrix Row-Column Sampling for the Many-Light Problem" [Lehtinen'07]
- **贡献**: PRT算子理论框架 (L=SE, L=RSE)
- **影响**: Neural PRT架构设计的理论基础
- **Milestone等级**: ★★★★

### C. 基于图像的渲染 (1996-2019)

#### 📌 **Milestone 7: Light Field Rendering**
- **论文**: [Levoy & Hanrahan SIGGRAPH'96]
- **贡献**: 光场采样与渲染
- **影响**: 新视角合成的经典方法
- **Milestone等级**: ★★★★★ (经典)

#### 📌 **Milestone 8: The Lumigraph**
- **论文**: [Gortler+ SIGGRAPH'96]
- **贡献**: 非结构化光场表示
- **Milestone等级**: ★★★★

#### 📌 **Milestone 9: Local Light Field Fusion (LLFF)**
- **论文**: [Mildenhall+ SIGGRAPH'19]
- **作者**: Ben Mildenhall (NeRF第一作者)
- **贡献**: 多平面图像(MPI)表示，NeRF的直接前身
- **Milestone等级**: ★★★★ (NeRF前导)

### D. 神经隐式表示 (2019-2020)

#### 📌 **Milestone 10: Scene Representation Networks (SRN)**
- **论文**: [Sitzmann+ NeurIPS'19]
- **贡献**: 神经场景表示，单点表面
- **影响**: 神经隐式表示早期工作
- **Milestone等级**: ★★★★

#### 📌 **Milestone 11: DeepSDF**
- **论文**: [Park+ CVPR'19]
- **贡献**: 连续SDF表示，需要3D ground truth
- **影响**: 神经隐式几何表示
- **Milestone等级**: ★★★★

#### 📌 **Milestone 12: Occupancy Networks**
- **论文**: [Mescheder+ CVPR'19]
- **贡献**: 占据场表示
- **Milestone等级**: ★★★

#### 📌 **Milestone 13: Neural Volumes**
- **论文**: [Lombardi+ SIGGRAPH'19]
- **贡献**: 可变形体素网格
- **Milestone等级**: ★★★

#### 📌 **Milestone 14: Differentiable Volumetric Rendering (DVR)**
- **论文**: [Niemeyer+ CVPR'20]
- **贡献**: 可微体积渲染
- **Milestone等级**: ★★★★

### E. 理论基础 (2017-2018)

#### 📌 **Milestone 15: Spectral Bias of Neural Networks**
- **论文**: "On the Spectral Bias of Neural Networks" [Rahaman+ ICML'18]
- **贡献**: 揭示MLP的频谱偏差 (倾向学习低频)
- **影响**: 启发NeRF的位置编码设计
- **Milestone等级**: ★★★★★ (理论突破)

#### 📌 **Milestone 16: Attention is All You Need (Transformer)**
- **论文**: [Vaswani+ NeurIPS'17]
- **贡献**: 位置编码概念 (虽然目的不同)
- **影响**: 启发频率编码
- **Milestone等级**: ★★★★ (概念启发)

#### 📌 **Milestone 17: Fourier Features Let Networks Learn High Frequency Functions**
- **论文**: [Tancik+ NeurIPS'20]
- **贡献**: 系统研究傅里叶特征映射
- **影响**: 与NeRF同期，互相验证
- **Milestone等级**: ★★★★

### F. 空间数据结构 (2003-2013)

#### 📌 **Milestone 18: Spatial Hashing**
- **论文**: [Teschner+ 2003]
- **贡献**: 空间哈希函数
- **影响**: Instant NGP哈希编码的基础
- **Milestone等级**: ★★★

#### 📌 **Milestone 19: VoxelHashing - Real-time 3D Reconstruction**
- **论文**: [Nießner+ SIGGRAPH'13]
- **贡献**: 哈希表用于3D重建
- **影响**: 哈希表在3D场景表示中的应用
- **Milestone等级**: ★★★★

### G. 神经渲染 (2017-2020)

#### 📌 **Milestone 20: Deep Shading**
- **论文**: "Deep Shading: Convolutional Neural Networks for Screen-Space Shading" [Nalbach+ EGSR'17]
- **贡献**: 首个完整神经渲染器
- **影响**: 神经渲染开创性工作
- **Milestone等级**: ★★★★

#### 📌 **Milestone 21: Neural Bidirectional Texture Function (Neural BTF)**
- **论文**: [Rainer+ EG'19, EG'20]
- **作者**: Gilles Rainer (Neural PRT第一作者)
- **贡献**: 神经BTF压缩和插值
- **Milestone等级**: ★★★

### H. 快速神经表示 (2020-2021)

#### 📌 **Milestone 22: Neural Sparse Voxel Fields (NSVF)**
- **论文**: [Liu+ NeurIPS'20]
- **贡献**: 稀疏体素特征加速NeRF
- **Milestone等级**: ★★★★

#### 📌 **Milestone 23: Neural Geometric Level of Detail (NGLOD)**
- **论文**: [Takikawa+ 2021]
- **贡献**: 八叉树特征向量
- **Milestone等级**: ★★★

#### 📌 **Milestone 24: ACORN (Adaptive Coordinate Networks)**
- **论文**: [Martel+ 2021]
- **贡献**: 自适应坐标编码器网络
- **影响**: 树结构 vs. Instant NGP哈希的对比
- **Milestone等级**: ★★★

### I. NeRF重要扩展 (2021)

#### 📌 **Milestone 25: mip-NeRF**
- **论文**: "Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields" [Barron+ ICCV'21]
- **贡献**: 多尺度抗锯齿，集成位置编码(IPE)
- **性能**: 比NeRF快7%，误差降低17%
- **Milestone等级**: ★★★★

#### 📌 **Milestone 26: NeRF in the Wild (NeRF-W)**
- **论文**: [Martin-Brualla+ CVPR'21]
- **贡献**: 外观嵌入+瞬态嵌入，处理野外场景
- **影响**: 光照建模的重要探索
- **Milestone等级**: ★★★★

#### 📌 **Milestone 27: pixelNeRF**
- **论文**: [Yu+ CVPR'21]
- **贡献**: Few-shot学习，跨场景泛化
- **Milestone等级**: ★★★★

#### 📌 **Milestone 28: Plenoctrees**
- **论文**: [Yu+ ICCV'21]
- **贡献**: 球谐编码NeRF方向信息，加速
- **Milestone等级**: ★★★

#### 📌 **Milestone 29: NeRD & NeRFactor**
- **论文**: [Boss+, Zhang+ ICCV'21]
- **贡献**: 显式BRDF学习，可重光照
- **Milestone等级**: ★★★★

## 时间线可视化 (Layer 1)

```
1984 ━━━ Volume Rendering [Kajiya] ★★★★★
1990 ━━━ Efficient Ray Tracing [Levoy]
1995 ━━━ Optical Models [Max] ★★★★
1996 ━━━ Light Field [Levoy&Hanrahan] ★★★★★
1996 ━━━ Lumigraph [Gortler+]
2002 ━━━ PRT [Sloan+] ★★★★★
2003 ━━━ Spatial Hashing [Teschner]
2007 ━━━ PRT Operator Theory [Lehtinen] ★★★★
2013 ━━━ VoxelHashing [Nießner+] ★★★★
2017 ━━━ Transformer [Vaswani+] ★★★★
2017 ━━━ Deep Shading [Nalbach+] ★★★★
2018 ━━━ Spectral Bias [Rahaman+] ★★★★★
2019 ━━━ DeepSDF [Park+] ★★★★
2019 ━━━ SRN [Sitzmann+] ★★★★
2019 ━━━ LLFF [Mildenhall+] ★★★★
2019 ━━━ Neural Volumes [Lombardi+]
2020 ━━━ NeRF [Mildenhall+] ★★★★★ ← 核心论文1
2020 ━━━ Fourier Features [Tancik+] ★★★★
2020 ━━━ NSVF [Liu+] ★★★★
2021 ━━━ mip-NeRF [Barron+] ★★★★
2021 ━━━ NeRF-W [Martin-Brualla+] ★★★★
2021 ━━━ pixelNeRF [Yu+] ★★★★
2021 ━━━ Plenoctrees [Yu+]
2021 ━━━ NeRD/NeRFactor [Boss+,Zhang+] ★★★★
2021 ━━━ ACORN [Martel+]
2022 ━━━ Instant NGP [Müller+] ★★★★★ ← 核心论文2
2022 ━━━ Neural PRT [Rainer+] ★★★★ ← 核心论文3
```

## 研究方向演变

### Phase 1: 经典渲染基础 (1984-2002)
**核心问题**: 快速逼真渲染
**代表工作**: 体积渲染, 光场, PRT
**技术特点**: 离散表示, 预计算

### Phase 2: 神经网络初探 (2013-2019)
**核心问题**: 紧凑表示, 视角合成
**代表工作**: Deep Shading, DeepSDF, SRN
**技术特点**: 神经隐式表示, 需要3D监督

### Phase 3: NeRF范式 (2020)
**核心问题**: 高质量新视角合成
**代表工作**: NeRF, Fourier Features
**技术突破**: 仅需2D监督, 连续5D表示
**理论基础**: Spectral Bias

### Phase 4: NeRF扩展 (2021)
**核心问题**: 质量/效率/鲁棒性/泛化
**代表工作**: mip-NeRF, NeRF-W, pixelNeRF
**研究方向**: 多尺度, 野外场景, Few-shot

### Phase 5: 效率革命 (2022)
**核心问题**: 实时训练+渲染
**代表工作**: Instant NGP
**技术突破**: 哈希编码, 1000×加速

### Phase 6: 神经+传统融合 (2022)
**核心问题**: 物理正确性, 可重光照
**代表工作**: Neural PRT
**技术特点**: PRT理论指导神经架构

## 下一步行动

### 需要深入分析的Top Milestone Papers (Layer 2)

#### 必读 (★★★★★):
1. **Spectral Bias** [Rahaman+ ICML'18] - 理论基础
2. **PRT** [Sloan+ SIGGRAPH'02] - 传统PRT基础
3. **Light Field Rendering** [Levoy&Hanrahan SIGGRAPH'96] - IBR经典

#### 重要 (★★★★):
4. **mip-NeRF** [Barron+ ICCV'21] - NeRF质量改进
5. **NeRF-W** [Martin-Brualla+ CVPR'21] - 光照建模
6. **LLFF** [Mildenhall+ SIGGRAPH'19] - NeRF前身
7. **DeepSDF** [Park+ CVPR'19] - 神经隐式基础
8. **Deep Shading** [Nalbach+ EGSR'17] - 神经渲染开创

#### 补充 (★★★):
9. **Fourier Features** [Tancik+ NeurIPS'20] - 位置编码理论
10. **NSVF** [Liu+ NeurIPS'20] - NeRF加速

### Web Search计划

搜索以下milestone papers的详细信息:
- [ ] Spectral Bias (Rahaman+ ICML'18)
- [ ] PRT原始论文 (Sloan+ SIGGRAPH'02)
- [ ] mip-NeRF技术细节
- [ ] NeRF-W光照分解方法
- [ ] Deep Shading架构

### 前向引用搜索计划

已完成: NeRF前向引用 (2021-2022)
待搜索:
- [ ] Instant NGP前向引用 (2022-2024)
- [ ] Neural PRT前向引用 (2022-2024)
- [ ] mip-NeRF前向引用
