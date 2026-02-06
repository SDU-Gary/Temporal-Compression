# PG‑GCPL 主线总览（Mainline Overview）

本文档给出仓库的“**当前主线** / **历史路线** / **候选清理项**”的全局视图，帮助后续做系统清理与温和重构。

## 1. 当前主线：PG‑GCPL（Gaussian‑Physics / FiLM Unified）

主线目标：用**空间软分区（Gaussians）** + **低秩基函数** + **光照/时间驱动的线性调制**来压缩并重建 probe SH。

核心实现链路（从数据到训练）：

1) 数据生成
- `1_data_generation/`：Mitsuba/Falcor 生成 probe → SH 数据集

2) 数据加载
- `2_src/data/`：训练/评估用 dataset + dataloader

3) 光照描述符（Schema-ish）
- `2_src/utils/light_descriptor.py`：把不同来源的光照参数（1D/5D/光照集合）统一成 descriptor（默认 12D）

4) 模型（主线）
- `2_src/models/gaussian_physics_unified.py`：
  - `LightSetEncoder`：可变数量光源集合 → pooled embedding `Z`
  - `GaussianPhysicsCompressionUnified`：Gaussians 软分区 + 低秩权重 `coeffs @ Z` +（可选）FiLM 动态基调制

5) 训练入口
- `3_experiments/scripts/train.py`：统一训练入口（推荐从这里跑实验）

6) 配置与校验
- `3_experiments/configs/*.yaml`：实验配置
- `metadata/schemas/train.schema.json`：训练配置 schema（`scripts/train.py` 会做 schema validate）

推荐主线配置：
- `3_experiments/configs/bistro_clean_train.yaml`

## 2. 历史路线（保留参考价值）

历史路线主要用于：论文写作对比、失败分析、以及复现实验结论。

- `archive/TemporalMLP/`：TemporalMLP 三层 MLP 基线（已弃用）
- `archive/TPE_Method_Archived/`：TPE 失败路线存档
- `3_experiments/scripts/legacy_training/`：早期训练脚本（硬编码物理基函数、或旧数据格式）
- `2_src/models/physics_low_rank.py`、`2_src/models/pg_rglt.py` 等：不同阶段的物理低秩尝试

## 3. 现在最容易“混淆”的点（建议优先处理）

1) 文档与真实目录结构不一致
- `docs/dev_notes/CLAUDE.md` 历史上引用过 `data_generation/`、`multi_time_compression/`，目前仓库以 `1_data_generation/2_src/3_experiments/` 为准。

2) 两套 config 语义共存
- TemporalMLP 的 config（`experiment.name / data.dataset_path / training.num_steps`）
- PG‑GCPL 的 config（`experiment.variant / data.data_root / training.epochs`）

3) 旧诊断脚本仍在引用 TemporalMLP 代码
- 例如 `3_experiments/scripts/analysis/diagnose_training.py` 仍在 import `models.temporal_mlp`（这是历史基线，不是当前主线）。

## 4. “候选清理项”建议（先标记，不直接删除）

清理建议遵循：先 **标记** → 再 **断引用** → 最后 **归档/删除**。

候选项（需要你确认）：

- `3_experiments/scripts/analysis/diagnose_training.py`：当前与 TemporalMLP 强耦合，容易误导为主线工具
- `docs/dev_notes/EXPERIMENTAL_WORKFLOW.md` 中一些历史路径引用（可做一次路径修正/加注释）
- `3_experiments/scripts/legacy_training/`：保留但建议改名为 `archive_training/` 或加统一 banner

