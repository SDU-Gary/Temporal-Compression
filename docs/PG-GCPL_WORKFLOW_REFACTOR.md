# PG-GCPL 工作流深度解析与分阶段重构方案

> 角色：资深图形学科研工程师 + 系统架构师
> 目标：系统梳理并重构 PG‑GCPL 项目工作流，聚焦数据生成（Falcor）、训练、评估、实验管理与可视化。

---

# 《现有工作流深度盘点》

本节目标：把当前 workflow “画出来”，同时明确隐形步骤、决策点与痛点来源。以下内容基于仓库现状 + 必要假设。

## 0) 关键假设（如不成立可替换）

- 假设：Falcor 侧为主要数据生成引擎（`1_data_generation/falcor/*`），Mitsuba 侧以原型/历史脚本为主。
- 假设：主要在单机 Kubuntu + NVIDIA GPU 上完成生成/训练/评估。
- 假设：脚本中仍存在绝对路径与环境变量依赖（如 `/home/kyrie/毕设`），导致迁移与复现难度增加。

## 1) 数据生成链路（Falcor 侧）

### 1.1 场景配置
- 主要用 `.pyscene` 场景（`1_data_generation/falcor/scenes`、`1_data_generation/scenes`）。
- Bistro 管线依赖环境变量（如 `BISTRO_FBX/BISTRO_NUM_FRAMES/BISTRO_FPS`）和脚本内部同步参数。
- 场景预览依赖单独脚本（如 `tools/render_bistro_preview.py`）。

### 1.2 Probe 定义与布局
- Grid 模式：`--grid-resolution + --margin`。
- Adaptive 模式：`probe_uniform_ratio` + AABB 表面采样 `probe_surface_offset_*`。
- 外部文件输入：`--probe-file`（由 `tools/sample_surface_probes.py` 深度引导采样生成）。

### 1.3 Falcor 渲染参数管理
- 参数来源混杂：CLI + env + 脚本硬编码。
- 关键参数：`spp / accum-frames / num-sh-samples / cube-res / fixed-seed / radiance-clamp / frame-step`。
- Falcor PathTracer 每帧 SPP 上限 16，实际靠 `accum-frames` 累积实现高 SPP。

### 1.4 输出数据结构/命名规则
- Falcor 新管线：输出 `parametric_tensor.npz` + `metadata.json`。
- 旧管线：输出 `probes.npz + moment_*/sh_coeffs.npz + images/*`。
- 目录分散：`1_data_generation/output/*` 下按任务/实验自定义命名。

### 1.5 手工步骤与易错点
- 启动 Falcor 需大量手动 env 配置（VK/LD_LIBRARY_PATH/FALCOR_PYTHON_PATH）。
- 脚本与 README/配置仍存在旧路径（`data_generation/`）残留。
- 真实 probe 数与 config 声明不一致的历史记录已存在（见 `tools/import_existing_data.py`）。

---

## 2) 训练与评估链路

### 2.1 训练脚本如何定位数据
- 主入口 `3_experiments/scripts/train.py` 支持 CLI 参数（如 `--data-root`）。
- 仍有大量专项脚本硬编码数据路径。
- YAML 配置存在于 `3_experiments/configs/*`，但未被统一入口消费。

### 2.2 模型配置管理
- 多数训练脚本直接写死超参（K/r/lr/epochs）。
- `2_src/utils/config.py` 已具备 YAML 读取，但尚未成为默认入口。

### 2.3 评估指标与保存方式
- 评估脚本散落在 `3_experiments/scripts/evaluation/*`。
- 输出形式多样：`results.json`、`training_log.csv`、`training_curve.png` 等。
- 结果目录分散在 `3_experiments/output/` 与 `3_experiments/results/`。

### 2.4 实验管理工具现状
- 已有 SQLite 方案（`metadata/project.db` + `tools/logexp.py` + `metadata/schema.sql`）。
- 训练/评估脚本默认不写入 DB；且 schema 路径默认指向根目录，可能初始化失败。

---

## 3) 实验组织与版本管理

### 3.1 实验最小单位
- 实际上是“数据集 + 训练 + 评估 + 可视化”，但缺少强绑定与标准目录规范。

### 3.2 参数记录方式
- 依赖 README/手写 JSON/命名习惯，缺少统一机器可读记录。

### 3.3 Git 与目录组织
- 已进行一次重构（见 `REFACTORING_FILE_MAPPING.md`），但旧路径残留造成混乱与复现成本。

---

## 4) 交互与可视化

### 4.1 Probe/场景调试方式
- 预览脚本、单帧渲染、probe 采样等均存在，但各自为政。
- 缺乏“低成本、快速预览、可交互修改”闭环。

### 4.2 快速预览机制
- 许多脚本支持 `--max-probes/--max-configs/--spp`，但缺统一 smoke preset。

### 4.3 DCC/引擎使用情况
- 暂无 Blender/Godot 接入；Rerun 已有 logging 代码但未成为默认路径。

---

## 5) 资源与约束

- GPU 被 Falcor 与训练共享，容易串行阻塞。
- 生成侧和训练侧脚本风格不统一，改造成本较高。
- 需要“低成本增量改造”，不能影响论文主线。

---

## 6) 当前工作流流程图（文字版）

1. 设计场景/光源/探针规则
2. 选择 Falcor 脚本 + 手动拼环境变量
3. 运行数据生成 → 输出 `parametric_tensor.npz` 或旧格式
4. （可选）手动验证/分析数据
5. 手动配置训练脚本（或 YAML）并运行训练
6. 手动运行评估脚本
7. 手动整理结果/曲线/README
8. （可选）手动记录实验（DB 或 README）

---

## 7) 核心痛点清单

- **配置与路径漂移**：CLI/脚本/环境变量/YAML 混杂。
- **数据可追溯性不足**：配置声明与实际产出不一致。
- **实验产物分散**：output/results/log/curve 多路径多格式。
- **缺少快速预览闭环**：probe/场景调试需全量生成。
- **串行阻塞严重**：生成-训练-评估缺乏小样迭代机制。
- **实验管理未默认化**：DB 体系存在但非默认执行路径。

---

# 《工作流重构方案（分阶段可落地）》

## 第 1 阶段：最低成本、立竿见影的改造

### 目标
- 把“改配置 → 跑一条命令”变成默认方式。
- 让每个数据集自动携带可追溯 metadata。
- 初步打通“生成 → 训练”的轻量 pipeline，缩短反馈回路。

### 需要修改/新增的模块
- **统一配置入口**：新增 `1_data_generation/configs/*.yaml`（Falcor 参数 + probe + 输出路径 + seed）。
- **生成 orchestrator**：新增 `tools/run_dataset.py` 读取 YAML，调用现有 Falcor 脚本，不改渲染逻辑。
- **数据集 manifest**：生成后自动写 `manifest.json`（记录 config、脚本、git commit、真实 probe/config 数）。
- **数据验证**：自动调用 `1_data_generation/validate_dataset.py`，保存 `validation_report.json`。
- **实验记录默认化**：训练结束自动调用 `tools/logexp.py log` 写入 `metadata/project.db`。

### 实施顺序
1. 引入 `dataset_config.yaml` 与 `tools/run_dataset.py`（封装 Falcor 生成）。
2. 统一输出元数据（`manifest.json` + `metadata.json`）。
3. 训练入口加 `--preset smoke`，自动写 DB。

### 风险与权衡
- 不要重写 Falcor 核心逻辑，只做外层 orchestration。
- 允许 CLI/YAML 并存，避免破坏已有脚本。

### 具体价值场景
- 你想测试新的 probe 分布：改 YAML → `python tools/run_dataset.py --config xxx.yaml` → 10 分钟生成小样 → `train.py --data-root ...`，无需改 3 个脚本、手动记路径。

---

## 第 2 阶段：引入前端 + 可视化 + 实验管理工具

### 目标
- Probe 与场景配置可视化，降低“盲跑”成本。
- 训练与评估结果自动可视化，避免手工整理。
- 实验对比和复现更快捷。

### 需要修改/新增的模块
- **数据前端（Blender 作为核心前端）**：以 Blender 作为交互编辑器，负责“发光球体动画、探针采样范围/约束、相机/视角参考”。导出 `scene.json` + `lights_trajectory.npy` +（可选）`probes.npy`。
- **Falcor 读取 JSON**：Falcor 生成脚本支持 `--scene-json`，读取光源轨迹与 probe 文件；场景材质/环境光仍以 Falcor `.pyscene` 为真值源，保证渲染一致性。
- **可视化默认化**：训练脚本默认支持 `--rerun`，输出 `.rrd`。
- **评估结果标准化**：评估脚本统一 `results.json` + `summary.json` 并写入 DB。

### 实施顺序
1. 训练/评估结果可视化默认化（Rerun/可选 Polyscope）。
2. 统一输出格式与 DB 写入。
3. 引入 Blender JSON → Falcor 读取（先做“光源轨迹 + probe 文件”，不做材质/灯光资产转换）。

### 风险与权衡
- Blender 插件成本较高，可先用 Rerun + 简易 JSON 编辑器替代。
- 不强推 WandB/MLflow，先稳住本地 DB。

---

## 第 3 阶段：完整的“前后端分离 + 统一配置”架构

### 目标
- 用统一 schema 描述场景/探针/光源/渲染/训练/评估。
- Falcor/Mitsuba 双后端协同，形成高质量真值对照。
- 支持批量实验与自动报告。

### 需要修改/新增的模块
- **统一 Schema**：`pgcpl.schema.json`（版本化）。
- **编译器/生成器**：`tools/compile_pipeline.py` 把 schema 编译为 Falcor configs、训练 configs、评估 configs、metadata。
- **Orchestrator**：简化版 Python DAG / Makefile，必要时可扩展为 Snakemake/Hydra。
- **多后端支持**：Falcor（快）+ Mitsuba（高质量）输出同一数据格式。

### 实施顺序
1. Schema 最小化版本（Falcor + 训练参数）。
2. 编译器 + 简单 orchestrator。
3. 扩展多后端与自动报告。

### 风险与权衡
- 第 3 阶段会提高复杂度，必须建立在前两阶段稳定成果上。
- 避免一次性重写脚本，应以“封装 + 迁移”为主。

---

## 建议优先技术栈（不强推）

- 配置管理：YAML/JSON + Python（复用 `2_src/utils/config.py`）。
- 数据前端：Blender（优先），Godot/ImGui 作为备选。
- 可视化：Rerun（已有代码基础）+ Polyscope（轻量 3D）。
- 实验 orchestrator：简单 Python/Bash 先行，必要时再引 Hydra/Snakemake。

---

## 最小落地优先级（若立刻动手）

1. 统一数据生成入口 + YAML configs
2. 自动写 manifest + dataset validation
3. 训练脚本默认 logexp + rerun

---

# 《补充：Blender 作为核心前端的链路逻辑（Level 2 定位）》

本节明确 Blender 在整体链路中的职责，确保前端与后端解耦、可交互、可追溯。

## 1) Blender 负责什么（核心前端职责）

- **光源交互设计**：放置/动画化自发光球体（位置、半径、颜色、强度、轨迹）。  
- **探针采样约束**：定义 probe 的采样区域、密度与排除区域（可通过空物体/包围盒/体积来表达）。  
- **相机/视角参考**：提供预览视角（仅用于交互与确认，不用于渲染资产替换）。  
- **导出中间数据**：输出 `scene.json` + `lights_trajectory.npy` +（可选）`probes.npy`。  

> 注意：Blender 不负责真实渲染。Falcor 仍以 `.pyscene` 为场景真值源，保证材质/环境光一致性。

## 2) Falcor 负责什么（后端渲染职责）

- 加载 Falcor 原生 `.pyscene`（Bistro 等官方场景）。  
- 读取 Blender 导出的 JSON/轨迹文件，驱动发光球体位置/强度。  
- 执行 SH / cubemap 渲染并生成数据集。  
- 输出 `parametric_tensor.npz` + `metadata.json` + `manifest.json`。  

## 3) 整体链路逻辑（端到端）

```
Blender 交互编辑
  └── 输出 scene.json + lights_trajectory.npy (+ probes.npy)
        └── Falcor 读取 JSON/轨迹 → 生成 SH 数据集
              └── 写入 manifest.json + validation_report.json
                    └── 训练/评估统一入口（写 DB + 可视化）
```

## 4) Blender 输出最小规范（示例）

```json
{
  "scene": "1_data_generation/falcor/scenes/bistro_exterior_emissive_spheres.pyscene",
  "frames": 600,
  "fps": 30,
  "emissive_objects": [
    {
      "name": "BallRed",
      "color": [20.0, 0.0, 0.0],
      "radius": 0.3,
      "trajectory": "red_traj.npy"
    }
  ],
  "probes": {
    "file": "probes.npy",
    "mode": "adaptive"
  },
  "render": {
    "sh_mode": "cubemap",
    "cube_res": 16,
    "spp": 32
  }
}
```

## 5) 关键好处（对 PG‑GCPL 直接收益）

- **交互性提升**：光源轨迹与探针范围可视化设计，减少“盲改”。  
- **迭代成本降低**：改 Blender → 导出 → Falcor 直用，避免手改脚本。  
- **可追溯**：所有变更落在 JSON + manifest，可复现实验。  


---

# 《补充：工作流契约与统一入口（深度补全）》

本节为“工程完备性补全”，目标是把方案从“建议”升级为“可执行规范”，同时确保简单与高效之间的平衡。

## A) Workflow Contract（最小可执行流程契约）

**核心原则**：所有主流程必须通过“统一入口 + 统一配置”启动，禁止脚本内硬编码路径作为默认行为。  
**最低要求**：只要提供 `pipeline.yaml`，即可从“数据 → 训练 → 评估 → 可视化”完成一次实验。

### A.1 统一入口设计草图

建议新增入口脚本（名称可调整）：  

```
tools/run_pipeline.py --config pipelines/pgcpl_baseline.yaml
```

`run_pipeline.py` 的职责：
1. 解析 pipeline.yaml（只做调度，不做渲染/训练逻辑）。
2. 调用现有 Falcor 生成脚本（或训练脚本）。
3. 自动写入 manifest/DB。
4. 若失败，保证输出可诊断的错误（日志 + step 报告）。

### A.2 pipeline.yaml 最小结构（示意）

```yaml
id: PGCPL_5D_smoke
dataset:
  generator: falcor_5d
  config: 1_data_generation/configs/5d_smoke.yaml
train:
  entry: 3_experiments/scripts/train.py
  args:
    variant: 5d
    data_root: ${dataset.output}
    preset: smoke
eval:
  entry: 3_experiments/scripts/evaluation/run_all.py
  args:
    data_root: ${dataset.output}
viz:
  rerun: true
  out_dir: 3_experiments/output/${id}
```

> 说明：这里仅是“最小约束”，不引入复杂 DSL。  

---

## B) Dataset Contract（数据契约）

**目标**：任何数据集都必须可追溯、可验证、可复现。  
**最低产出**：除 `parametric_tensor.npz` 外必须存在 `manifest.json`。

### B.1 manifest.json 必要字段（最小规范）

```json
{
  "dataset_id": "BISTRO_TEMPORAL_200P_SMOKE",
  "created_at": "2026-02-01T12:34:56",
  "generator": "1_data_generation/falcor/generate_bistro_temporal_spheres.py",
  "config_path": "1_data_generation/configs/bistro_smoke.yaml",
  "scene": "1_data_generation/falcor/scenes/bistro_exterior_emissive_spheres.pyscene",
  "output_dir": "1_data_generation/output/bistro_temporal/smoke",
  "tensor_format": "parametric_tensor.npz",
  "num_probes": 200,
  "num_configs": 30,
  "spp": 32,
  "git_commit": "abc1234",
  "env": {
    "falcor_python_path": "/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/python",
    "vk_icd": "/usr/share/vulkan/icd.d/nvidia_icd.json"
  }
}
```

### B.2 验证规则（必须自动执行）
- `num_probes` 与 `probe_positions.shape[0]` 必须一致。  
- `num_configs` 与 `tensor.shape[1]` 必须一致。  
- 若不一致，写入 `validation_report.json` 并标记 `status=failed`。  

---

## C) 预览/小样预设（缩短反馈回路）

**原则**：任何生成脚本必须支持 `--preset smoke|dev|prod`。  
**建议预设**：

| Preset | Probe | Configs | SPP | 用途 |
| ------ | ----- | ------- | --- | ---- |
| smoke  | 8-32  | 2-5     | 8-16 | 5-10 分钟内验证流程 |
| dev    | 128-512 | 10-30 | 32-64 | 功能调试与小规模训练 |
| prod   | 1k+   | 40+    | 256+ | 正式论文/报告 |

**入口示例**：  
```
python tools/run_dataset.py --config configs/bistro.yaml --preset smoke
```

---

## D) 资源调度与串行阻塞控制（最小工程策略）

**问题**：Falcor 与训练争用 GPU。  
**最小解决方案**：
1. 规定“渲染阶段 GPU 独占、训练阶段 GPU 独占”。  
2. 通过 `CUDA_VISIBLE_DEVICES` 或 `FALCOR_GPU` 进行手动隔离。  
3. 引入简单队列文件（如 `metadata/queue.json`），保证一次只运行一个重任务。  

> 不推荐第 1 阶段引入复杂调度系统，避免超出研究任务范围。\n
---

## E) 评估与可视化一致性（统一 runner）

**现状**：评估脚本过多，输出格式不一。  
**建议**：新增 `3_experiments/scripts/evaluation/run_all.py` 作为统一入口：\n
职责：  
1. 读取模型 checkpoint + dataset manifest  
2. 执行常规评估（插值、外推、延迟等）  
3. 输出统一 `results.json + summary.json + plots/`  
4. 自动写入 DB `results_json`  

---

## F) 阶段验收标准（确保工程可控）

### 阶段 1 完成标准
- [ ] 数据生成必须通过 `run_dataset.py` 入口完成。  
- [ ] 每个数据集生成后自动写 `manifest.json` 与 `validation_report.json`。  
- [ ] 训练结束自动写入 DB（`tools/logexp.py`）。  

### 阶段 2 完成标准
- [ ] 训练过程默认可视化（Rerun/Polyscope）。  
- [ ] 评估统一入口 `run_all.py` 产出标准结果。  
- [ ] Probe 分布可视化可在 5 分钟内定位异常。  

### 阶段 3 完成标准
- [ ] Schema 可描述场景/探针/渲染/训练/评估。  
- [ ] Falcor + Mitsuba 至少 1 个数据集可通过同一 schema 生成。  
- [ ] pipeline 自动生成报告（summary + plots）。  

---

## G) 简单与高效之间的平衡策略（工程底线）

- **简单**：所有新增工具必须是“薄封装”。不重写 Falcor/训练逻辑。  
- **高效**：强制统一入口与数据契约，避免“脚本野生化”。  
- **可维护**：不引入复杂依赖（Hydra/Snakemake 等放到阶段 3 之后）。  

---

## H) 如果要立即落地的最小补强路径

1. 为 `tools/run_dataset.py` 增加 `--preset` 和 `manifest.json` 自动写入。  
2. 修正 DB schema 路径，确保 logexp 可直接写入。  
3. 新增 `run_all.py` 评估入口，统一输出格式。  

## I) 已落地的入口与配置位置（参考）

- 数据生成入口：`tools/run_dataset.py`  
- Pipeline 入口：`tools/run_pipeline.py`  
- Blender 导出：`tools/blender/export_emissive_trajectories.py`  
- 场景/清单 schema：`metadata/schemas/*.json`  
- 数据集配置：`1_data_generation/configs/*.yaml`  
- Pipeline 示例：`pipelines/*.yaml`  
- 备注：pipeline 的 eval 若未显式给 output_dir，会优先使用 **train args 的 output_dir**，或 **train config 里的 output_dir**；两者都没有时需手动指定。  
- YAML 模板：`3_experiments/configs/template_unified_set.yaml`  
- YAML Schema：`metadata/schemas/train.schema.json`（训练/评估脚本可选校验）  

## J) Python 选择规则（避免环境混用）

- **dataset 生成**：优先使用 `dataset.config` 内的 `python` 字段，其次 `generator.python` 字段，再次环境变量 `FALCOR_PYTHON`。  
- **train/eval**：优先使用 pipeline 层 `python_train/python_eval` 或各自 `train.python / eval.python` 字段；若未指定，使用当前默认 Python。  
- **说明**：Falcor Python 与训练 Python 应严格隔离，避免依赖缺失导致崩溃。  

## K) 旧训练脚本的处理策略

- 历史 1D/5D 训练脚本已迁移到 `3_experiments/scripts/legacy_training/`  
- 统一入口为 `3_experiments/scripts/train.py`  
