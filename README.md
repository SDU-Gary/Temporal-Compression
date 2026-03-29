# PG‑GCPL：多时刻光照压缩（毕业设计仓库）

本仓库是一个图形学向的**多时刻/多光源全局光照压缩**项目，核心方向为 **PG‑GCPL (Physics‑Guided, GMM‑constrained Probe Learning)**：

- **空间**：用少量 3D Gaussians 做 probe 空间软分区/聚合。
- **时间/光照状态**：用物理可解释的“线性调制”思想（以及 FiLM 动态基调制）驱动低秩基函数权重变化。
- **目标**：高压缩比（~1:17）、可控误差、实时解压（<0.5ms）。

如果你要快速理解“主线/历史线/候选清理项”，建议先看：

- `docs/PGGCPL_AUDIT.md`：全仓库审计与主线定位
- `docs/PGGCPL_TIME_MODELS.md`：时间建模路线对比与演进
- `docs/dev_notes/DEVELOPMENT_TRUTH_2026-03-08.md`：当前代码真相（按日期维护）

## 📁 当前目录结构（以 numbered stages 为主）

```
Temporal-Compression/
├── 1_data_generation/          # Stage 1: 数据生成（Mitsuba/Falcor）
├── 2_src/                      # Stage 2: 核心算法（models/training/data/utils）
├── 3_experiments/              # Stage 3: 训练/评估/可视化（configs/scripts/results）
├── 4_thesis/                   # Stage 4: 论文材料
├── archive/                    # 历史/弃用路线（TemporalMLP/TPE等）
├── docs/                       # 文档与开发笔记
├── metadata/                   # 项目数据库与 schema
├── pipelines/                  # workflow pipeline 定义
└── tools/                      # 工具（实验记录、manifest等）
```

## ✅ 当前主线（PG‑GCPL / FiLM Unified）

主线模型以“**光照描述符 → pooled embedding Z → 低秩权重 + FiLM 动态基**”为核心：

- `2_src/models/gaussian_physics_unified.py`：`GaussianPhysicsCompressionUnified` + `LightSetEncoder`
- `2_src/utils/light_descriptor.py`：把 1D/5D 参数构建为统一 descriptor（默认 12D）
- `2_src/training/gaussian_physics_trainer.py`：统一 trainer（recon/temporal/linearity/spatial 等 loss）
- `3_experiments/scripts/train.py`：统一训练入口（当前主线 `variant: unified_set`）
- `metadata/schemas/train.schema.json`：训练配置 schema（YAML 可做校验）

## 🗃️ 历史路线（已归档但保留参考价值）

- `archive/TemporalMLP/`：TemporalMLP（三层结构基线，已弃用）
- `archive/TPE_Method_Archived/`：TPE 失败路线存档
- `2_src/models/physics_low_rank.py` 等：硬编码物理基函数的低秩模型（legacy_experiment）

## 🚀 快速开始（主线示例）

```bash
# 1) 激活环境
source venv/bin/activate

# 2) 生成数据（示例）
cd 1_data_generation
python generate_dataset.py

# 3) 训练主线模型（示例配置）
cd ../3_experiments
python scripts/train.py --config configs/bistro_clean_train.yaml
```

## 🔧 依赖环境（摘要）

- Python 3.13
- PyTorch 2.9.1
- Mitsuba 3.7.3
