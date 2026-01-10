# 多时刻光照压缩 - 毕业设计项目

**学生**：葛恺尧
**时间**：2025学年
**主题**：基于高斯混合和物理引导的多时刻光照神经压缩

---

## 📁 项目结构

```
毕设/
├── README.md                    # 本文件
├── CLAUDE.md                    # Claude Code 使用指南
│
├── docs/                        # 文档目录
│   ├── thesis/                  # 论文相关
│   │   └── 多时刻光照压缩任务书.md
│   ├── research_notes/          # 研究笔记
│   │   ├── 研究故事_多时刻光照压缩.md
│   │   ├── TPE_Temporal_Perturbation_Embedding.md
│   │   ├── G2_CS-LT_研究报告.md
│   │   └── TAD-GF.md
│   └── assets/                  # 资源文件
│       ├── sun_trajectory.png
│       └── sun_directions_24h.txt
│
├── data_generation/             # 数据生成模块 (757MB)
│   ├── core/                    # 核心渲染逻辑
│   ├── utils/                   # 工具函数
│   ├── scenes/                  # 场景定义
│   ├── output/                  # 生成的数据集
│   │   ├── level1_tpe/          # (188K)
│   │   ├── level2_tpe/          # (452K)
│   │   ├── method_test_v1/      # (63M)
│   │   ├── dataset_2k/          # (190M)
│   │   └── dataset_15k/         # (203M)
│   ├── generate_dataset.py      # 主生成脚本
│   └── README.md
│
├── multi_time_compression/      # 主压缩系统 (6.9MB)
│   ├── models/                  # 神经网络模型
│   │   ├── physics_low_rank.py           # 物理引导低秩模型
│   │   ├── gaussian_physics_compression.py  # 高斯-物理混合模型
│   │   ├── temporal_mlp.py
│   │   └── ...
│   ├── scripts/                 # 训练和评估脚本
│   │   ├── train_physics_low_rank_proper.py
│   │   ├── train_gaussian_physics.py
│   │   └── experiments/         # P0实验脚本
│   ├── docs/                    # 实验报告
│   │   └── experiment_reports/  # 9篇详细报告
│   │       ├── 00_README.md
│   │       ├── 01-07_*.md      # TPE实验
│   │       ├── 08_Physics_Low_Rank_Results.md
│   │       └── 09_Gaussian_Physics_Hybrid.md
│   ├── output/                  # 实验结果
│   │   ├── p0_results/
│   │   ├── physics_low_rank/
│   │   └── gaussian_physics/
│   ├── configs/                 # 配置文件
│   ├── data/                    # 数据加载器
│   ├── training/                # 训练循环
│   └── tests/                   # 单元测试
│
├── mitsuba-docs/                # Mitsuba 3 本地文档 (3.2MB)
├── reference/                   # 参考论文 (35MB)
├── analysis/                    # 论文分析 (300KB)
├── venv/                        # Python虚拟环境 (7.4GB)
└── 202200300003-葛恺尧-开题报告.docx

```

---

## 🎯 项目目标

### 核心任务

在保证渲染质量的前提下，压缩多时刻光照场景的存储空间，实现实时查询。

### 技术路线

1. **空间压缩**：高斯混合模型表示探针分布
2. **时间压缩**：物理引导的低秩因子分解
3. **混合方法**：空间高斯 + 时间物理基函数

### 目标指标

- 压缩比：> 10x
- 精度：PSNR > 35dB
- 查询速度：< 0.5ms

---

## 📊 主要成果

### 1. TPE方法验证（文档1-7）

- ✗ TPE假设在真实场景失效
- Cornell Box: 84.9% 误差来自TPE假设
- House: 虽然贡献率96.5%，但绝对误差极小

### 2. 物理引导低秩压缩（文档8）

- ✅ Cornell Box: 2.34x压缩 + **33.5%精度提升**
- ✅ House: 2.92x压缩
- 方法：SH(t) ≈ U @ (coeffs @ [cos(θ), sin(θ), 1])
- 参数：150个（rank=5）

### 3. 高斯-物理混合压缩（文档9）

- ✅ **17.8x压缩比**（343探针 × 6时刻）
- 参数：3,120个（K=20高斯）
- 方法：F(p,t) = Σ G_j(p) \* [U_j @ Φ(t)]

---

## 🚀 快速开始

### 环境配置

```bash
# 激活虚拟环境
source venv/bin/activate

# 验证安装
python -c "import mitsuba as mi; print(mi.variants())"
python -c "import torch; print(torch.__version__)"
```

### 数据生成

```bash
cd data_generation
python generate_dataset.py
```

### 训练模型

```bash
cd multi_time_compression

# 物理引导低秩模型（单探针）
python scripts/train_physics_low_rank_proper.py

# 高斯-物理混合模型（多探针）
python scripts/train_gaussian_physics.py
```

---

## 📖 文档导航

### 核心文档

- **项目指南**：[CLAUDE.md](CLAUDE.md) - Claude Code使用说明
- **任务书**：[docs/thesis/多时刻光照压缩任务书.md](docs/thesis/多时刻光照压缩任务书.md)
- **实验报告**：[multi_time_compression/docs/experiment_reports/](multi_time_compression/docs/experiment_reports/)

### 快速链接

- [数据生成指南](data_generation/README.md)
- [实验总结](multi_time_compression/docs/experiment_reports/00_README.md)
- [物理低秩方法](multi_time_compression/docs/experiment_reports/08_Physics_Low_Rank_Results.md)
- [混合压缩方法](multi_time_compression/docs/experiment_reports/09_Gaussian_Physics_Hybrid.md)

---

## 🔧 依赖环境

### 主要依赖

- Python 3.13
- PyTorch 2.9.1 (CUDA 12.8)
- Mitsuba 3.7.3 (cuda_ad_rgb variant)
- DrJit 0.4.6
- NumPy 1.26.4
- SciPy 1.16.3

### 硬件要求

- GPU: NVIDIA RTX 4090 (24GB VRAM)
- RAM: 32GB+
- 存储: ~10GB（含数据集）

---

## 📈 实验时间线

- **2025-12-07**: 项目启动，环境配置
- **2025-12-09**: TPE方法提出与初步实验
- **2025-12-10**: P0实验（Cornell Box）
- **2025-12-11**: House场景对比实验
- **2025-12-12**:
  - 物理引导低秩方法验证
  - 高斯-物理混合方法实现
  - 完整实验报告撰写

---

## 📝 待办事项

### 短期（1周）

- [ ] 增加高斯数量（K=20→50）
- [ ] 使用完整训练数据（3→6时刻）
- [ ] 更长训练（1000→5000 epochs）

### 中期（1月）

- [ ] 加入方位角（3维→5维基函数）
- [ ] 扩展到24小时实验
- [ ] 与任务书MLP方法对比
- [ ] 实现实时解压缩（CUDA kernel）

### 长期（3月+）

- [ ] 多场景泛化测试
- [ ] 层次化高斯结构
- [ ] 端到端渲染集成
- [ ] 论文撰写与投稿

---

## 🤝 贡献者

- **葛恺尧** - 主要开发者
- **指导老师** - [待补充]

---

## 📄 许可证

本项目仅用于学术研究和毕业设计，未经许可不得用于商业用途。

---

**最后更新**: 2025-12-12
