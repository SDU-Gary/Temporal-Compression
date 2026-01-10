# 项目重构计划 (Project Refactoring Plan)

**版本**: 1.0
**日期**: 2026-01-10
**目标**: 为了更好地服务于毕业设计工作，将当前项目结构进行重构，使其更清晰、更有逻辑，并与科研工作流保持一致。

---

## 1. 核心问题与重构目标

当前项目结构存在顶层目录混乱、核心代码与实验代码混合、文档分散等问题。

本次重构旨在建立一个以**“数据 -> 源码 -> 实验 -> 论文”**为核心工作流的清晰目录结构，实现职责分离，提升项目可维护性。

**重构原则**:
- 核心代码、实验产物、历史归档清晰分离。
- 不对 `Falcor` 和 `venv` 目录进行任何修改。
- 过程清晰可控，优先采用“移动”而非“删除”操作。

---

## 2. 规划的最终目录结构

重构完成后，项目将采用以下结构：

```
.
├── 1_data_generation/       # 1️⃣ 数据 - 数据生成脚本
├── 2_src/                   # 2️⃣ 源码 - 项目核心源代码 (PG-GCPL)
│   ├── data/                #    - 数据加载器
│   ├── models/              #    - 神经网络模型
│   ├── tests/               #    - 单元测试
│   ├── training/            #    - 训练逻辑、损失函数
│   └── utils/               #    - 共享工具函数
├── 3_experiments/           # 3️⃣ 实验 - 实验配置、脚本和结果
│   ├── configs/             #    - 所有实验的 YAML 配置文件
│   ├── results/             #    - 保存所有实验的输出 (日志, checkpoints, 图像)
│   └── scripts/             #    - 运行训练、评估的入口脚本
├── 4_thesis/                # 4️⃣ 论文 - 存放所有与毕业论文直接相关的材料
│   ├── figures/             #    - 论文中使用的图表、架构图等
│   ├── presentation/        #    - 答辩PPT
│   └── report/              #    - 开题报告、中期报告、最终论文的 .docx 或 .tex 文件
├── Falcor/                  # [不变] - 开源渲染器
├── venv/                    # [不变] - Python 虚拟环境
├── tools/                   # [不变] - 项目级工具 (如 logexp.py)
├── reference/               # [不变] - 参考的论文PDF
│
├── archive/                 # 🗄️ 归档 - 已废弃或仅供参考的历史代码
├── docs/                    # 📚 文档 - 所有参考和过程文档
│   ├── dev_notes/           #    - CLAUDE.md, EXPERIMENTAL_WORKFLOW.md 等开发笔记
│   ├── literature_review/   #    - (原 analysis/ 目录) 论文分析
│   └── mitsuba/             #    - (原 mitsuba-docs/ 目录) Mitsuba文档
├── metadata/                # 🗃️ 元数据 - 项目数据库及相关文件
│   ├── project.db
│   └── schema.sql
│
├── .gitignore
└── README.md                # [待更新] - 指向新结构的项目总览
```

---

## 3. 详细重构流程

以下是分三阶段执行的详细步骤，以 `bash` 命令的形式呈现，确保操作的精确性。

### **第一阶段：创建新的、干净的目录结构**

此阶段的目标是建立起目标框架，为后续的文件迁移做准备。

```bash
# 1. 创建四大核心工作流目录
mkdir -p 2_src/{models,training,data,utils,tests}
mkdir -p 3_experiments/{configs,scripts,results}
mkdir -p 4_thesis/{report,figures,presentation}

# 2. 创建一个临时的、用于整合所有文档的新目录
mkdir docs_new
mkdir docs_new/literature_review
mkdir docs_new/dev_notes
mkdir docs_new/mitsuba

# 3. 创建归档和元数据目录
mkdir archive_new
mkdir metadata
```

### **第二阶段：迁移文件与目录**

此阶段是重构的核心，我们将一步步地将现有文件和目录迁移到新的位置。

```bash
# === 步骤 1: 整理根目录下的文件 ===

# 迁移毕设论文相关文件
mv "202200300003-葛恺尧-开题报告.docx" 4_thesis/report/
mv "面向多时刻光照的层次化神经压缩方法.pdf" 4_thesis/report/

# 迁移数据库和其结构定义
mv project.db schema.sql metadata/

# 迁移各类开发笔记和说明文档
mv CLAUDE.md CLAUDE_MD_IMPROVEMENTS.md DATABASE_VERIFICATION_REPORT.md EXPERIMENTAL_WORKFLOW.md EXPERIMENT_DATA_MAPPING.md "基于物理先验的多时刻动态光照压缩方法_SurveyGo.md" project_state.md docs_new/dev_notes/


# === 步骤 2: 重组核心目录 ===

# 重命名数据生成目录，符合新工作流编号
mv data_generation 1_data_generation

# 迁移 `analysis` 和 `mitsuba-docs` 的内容
mv analysis/* docs_new/literature_review/
mv mitsuba-docs/* docs_new/mitsuba/


# === 步骤 3: 拆解并重组 `multi_time_compression` (最关键的一步) ===

# a) 提取最终成功的 PG-GCPL 项目的核心代码 -> `2_src`
mv multi_time_compression/PG-GCPL/src/models/* 2_src/models/
mv multi_time_compression/PG-GCPL/src/data/* 2_src/data/
mv multi_time_compression/PG-GCPL/src/training/* 2_src/training/
mv multi_time_compression/shared/utils/* 2_src/utils/  # 合并共享的 utils
mv multi_time_compression/tests/* 2_src/tests/

# b) 提取实验相关部分 -> `3_experiments`
mv multi_time_compression/PG-GCPL/src/scripts/* 3_experiments/scripts/ # 这是运行实验的入口
mv multi_time_compression/TemporalMLP/configs/* 3_experiments/configs/ # 保留旧配置作为参考
mv multi_time_compression/PG-GCPL/experiments/* 3_experiments/results/ # 迁移实验结果

# c) 归档历史代码和文档 -> `archive_new` 和 `docs_new`
mv multi_time_compression/archive/* archive_new/
mv multi_time_compression/TemporalMLP archive_new/ # 归档整个旧项目
mv multi_time_compression/docs/* docs_new/dev_notes/
mv multi_time_compression/PG-GCPL/docs/reports/* docs_new/dev_notes/

# === 步骤 4: 整合和重命名 `docs` 目录 ===
# 将您现有的 `docs` 目录中的重要内容也迁移过来
mv docs/research_notes/* docs_new/dev_notes/
mv docs/thesis/* 4_thesis/report/
# 注意：现有的 `docs/assets` 目录可根据需要迁移。

# 最后，用我们新建的 docs_new 替换旧的
mv docs docs_to_delete_later
mv docs_new docs
mv archive_new archive
```

### **第三阶段：清理旧目录**

在确认所有文件都已正确迁移后，此步骤将删除那些已经被清空的旧目录。

```bash
# 警告：这是一个删除操作，执行前需要确认文件已完全迁移
rm -r analysis
rm -r mitsuba-docs
rm -r multi_time_compression
rm -r docs_to_delete_later
```

---
此计划文档创建完毕。
