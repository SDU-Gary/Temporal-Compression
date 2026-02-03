# 项目重构文件迁移对照表

**版本**: 1.0
**日期**: 2026-01-10
**目的**: 为 `REFACTORING_PLAN.md` 中描述的重构过程提供一份清晰、详细的“前后路径”对照清单，方便追踪和验证。

---

## 1. 顶层目录迁移对照表

此表展示了根目录下主要文件夹的迁移路径。

| 原始路径 (Old Path) | 新路径 (New Path)         | 备注 (Notes)                       |
| :------------------ | :------------------------ | :--------------------------------- |
| `data_generation/`  | `1_data_generation/`      | 重命名以体现工作流顺序。           |
| `analysis/`         | `docs/literature_review/` | 归入新的中央文档结构。             |
| `mitsuba-docs/`     | `docs/mitsuba/`           | 作为第三方库文档，归入新文档结构。 |

---

## 2. `docs` 目录内部调整

旧 `docs` 目录中的内容被重新分类。

| 原始路径 (Old Path)    | 新路径 (New Path)  | 备注 (Notes)                 |
| :--------------------- | :----------------- | :--------------------------- |
| `docs/research_notes/` | `docs/dev_notes/`  | 与其他开发笔记合并。         |
| `docs/thesis/`         | `4_thesis/report/` | 归入新的论文专属目录。       |
| `docs/assets/`         | `docs/assets/`     | 静态资源，保留在文档目录内。 |

---

## 3. `multi_time_compression` 目录拆解对照表

`multi_time_compression` 是本次重构的核心，它被完全拆解，其内容根据职责被分发到新的 `2_src`、`3_experiments`、`archive` 和 `docs` 目录中。

| `multi_time_compression` 内的原始路径 | 新的目标路径 (New Path)  | 新职责 (Responsibility)       |
| :------------------------------------ | :----------------------- | :---------------------------- |
| **(核心代码)**                        |                          |                               |
| `PG-GCPL/src/models/`                 | `2_src/models/`          | 核心算法模型                  |
| `PG-GCPL/src/data/`                   | `2_src/data/`            | 数据加载器                    |
| `PG-GCPL/src/training/`               | `2_src/training/`        | 通用训练逻辑 (如Trainer类)    |
| `shared/utils/`                       | `2_src/utils/`           | 项目级共享工具函数            |
| `tests/`                              | `2_src/tests/`           | 单元与集成测试                |
| **(实验相关)**                        |                          |                               |
| `PG-GCPL/src/scripts/`                | `3_experiments/scripts/` | 运行各项实验的入口脚本        |
| `TemporalMLP/configs/`                | `3_experiments/configs/` | 实验配置文件 (保留作为参考)   |
| `PG-GCPL/experiments/`                | `3_experiments/results/` | 所有实验的输出结果            |
| **(文档与归档)**                      |                          |                               |
| `docs/`                               | `docs/dev_notes/`        | 开发过程中的文档和报告        |
| `PG-GCPL/docs/reports/`               | `docs/dev_notes/`        | 针对PG-GCPL方法的详细报告     |
| `archive/`                            | `archive/`               | 归档旧的、失败的实验代码      |
| `TemporalMLP/`                        | `archive/TemporalMLP/`   | 归档整个 `TemporalMLP` 旧项目 |

---

## 4. 根目录主要文件迁移对照表

此表展示了直接放在根目录下的重要文件的新家。

| 原始文件路径 (Old File Path)                         | 新的目标路径 (New Path) |
| :--------------------------------------------------- | :---------------------- |
| `./202200300003-葛恺尧-开题报告.docx`                | `4_thesis/report/`      |
| `./面向多时刻光照的层次化神经压缩方法.pdf`           | `4_thesis/report/`      |
| `./project.db`                                       | `metadata/`             |
| `./schema.sql`                                       | `metadata/`             |
| `./CLAUDE.md`                                        | `docs/dev_notes/`       |
| `./CLAUDE_MD_IMPROVEMENTS.md`                        | `docs/dev_notes/`       |
| `./DATABASE_VERIFICATION_REPORT.md`                  | `docs/dev_notes/`       |
| `./EXPERIMENTAL_WORKFLOW.md`                         | `docs/dev_notes/`       |
| `./EXPERIMENT_DATA_MAPPING.md`                       | `docs/dev_notes/`       |
| `./基于物理先验的多时刻动态光照压缩方法_SurveyGo.md` | `docs/dev_notes/`       |
| `./project_state.md`                                 | `docs/dev_notes/`       |
