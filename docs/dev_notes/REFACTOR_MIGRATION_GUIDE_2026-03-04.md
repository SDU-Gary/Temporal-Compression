# PG-GCPL 重构迁移说明（2026-03-04）

本文档用于说明 2026-03-04 这轮“三阶段重构”后的迁移要点，确保：

- 现有训练/评估命令不中断
- 新增模块职责清晰，便于后续扩展
- 性能开关可控，出现问题可快速回退

---

## 1. 重构范围总览

本次改造覆盖以下主线区域：

1. **训练入口稳定化与模块化**（`3_experiments/scripts/`）
2. **实验编排去重**（`3_experiments/` + `tools/`）
3. **数据加载与训练传输性能优化**（`2_src/data/` + `2_src/training/`）

不包含破坏性 CLI 重命名，不要求改历史实验命令。

---

## 2. 模块迁移映射

### 2.1 训练参数与配置

原先逻辑大多堆叠在 `3_experiments/scripts/train.py`，现已拆分为：

- `3_experiments/scripts/train_arg_utils.py`
  - `normalize_namespace(...)`
  - `apply_variant_defaults(...)`
- `3_experiments/scripts/train_config_utils.py`
  - `load_yaml(...)`
  - `apply_config(...)`

`train.py` 中原有 `_load_yaml`、`_apply_config`、`apply_variant_defaults` 仍保留调用入口（内部改为委托），因此**旧调用方无需修改**。

### 2.2 路径引导

脚本路径注入统一收口到：

- `3_experiments/scripts/_path_setup.py`
  - `ensure_repo_paths(current_file, root_levels=...)`

已接入：

- `3_experiments/scripts/train.py`
- `3_experiments/scripts/eval.py`
- `3_experiments/scripts/analysis/analyze_data_smoothness.py`

### 2.3 实验脚本公共逻辑

以下重复逻辑已抽出：

- `3_experiments/experiment_driver_common.py`
  - `set_nested_value(...)`
  - `modify_config(...)`
  - `run_command(...)`
  - `read_json_if_exists(...)`

并由以下脚本复用：

- `3_experiments/run_controlled_experiments.py`
- `3_experiments/phase1_optimization_experiments.py`

### 2.4 Tools CLI 参数拼接

`tools` 内参数拼接逻辑统一为：

- `tools/workflow_config.py` -> `extend_cli_args(cmd, args)`

已接入：

- `tools/workflow_config.py` (`build_generator_command`)
- `tools/run_pipeline.py`（train/eval 参数构建）

---

## 3. 性能相关迁移点

### 3.1 `LightSetDataset` 新增开关

`2_src/data/lightset_dataset.py` 新增：

- `use_npz_cache: bool = True`
  - 启用后 train/val/test 多实例共享同一 NPZ 载入结果（带文件变更失效）
- `use_torch_views: bool = True`
  - 预构建 torch 视图，降低 `__getitem__` 的重复转换开销

`create_dataloaders_lightset(...)` 同步支持以上参数透传。

### 3.2 训练批次传输

`2_src/training/gaussian_physics_trainer.py` 中 `BatchAdapter` 新增：

- `non_blocking_transfer: bool = True`

在 `pin_memory=True` 的 DataLoader 条件下，可减少 H2D 传输等待。

---

## 4. 兼容性说明（重要）

### 4.1 CLI 兼容

- 训练主命令保持不变：
  - `python 3_experiments/scripts/train.py --config ...`
- 评估主命令保持不变：
  - `python 3_experiments/scripts/eval.py --data-root ... --checkpoint ...`

### 4.2 训练参数兼容

`train.py` 已加入参数归一化流程，旧测试或旧脚本传入的 `SimpleNamespace` 缺少新增字段时，会自动补默认值，不再因字段缺失崩溃。

### 4.3 行为等价原则

本次以“结构重组 + 性能微优化”为主，不主动改变训练指标语义；若出现指标波动，优先检查：

1. 数据路径与 split 是否一致
2. 是否手动改了 `use_npz_cache/use_torch_views/non_blocking_transfer`
3. 是否混用了旧 checkpoint 与新配置

---

## 5. 推荐迁移步骤

### 步骤 A（必做）

1. 拉取最新代码后运行：
   - `pytest -q 2_src/tests`
2. 确认至少通过以下关键回归：
   - `test_train_script_coverage.py`
   - `test_train_config_utils.py`
   - `test_lightset_dataset_cache.py`

### 步骤 B（建议）

对已有实验脚本仅做“导入路径对齐”，不要复制旧的参数拼接代码；优先复用：

- `train_arg_utils.py`
- `train_config_utils.py`
- `experiment_driver_common.py`
- `workflow_config.extend_cli_args(...)`

### 步骤 C（性能调参）

如果要定位性能回归/波动，可用以下开关快速排查：

- 关闭 NPZ 缓存：`use_npz_cache=False`
- 关闭 torch 视图快路径：`use_torch_views=False`
- 关闭 non-blocking 传输：`BatchAdapter(non_blocking_transfer=False)`

---

## 6. 常见问题（FAQ）

### Q1：为什么不直接把 `train.py` 全拆成多个类？

当前采用“兼容优先”的渐进策略：先拆功能模块并保留原入口函数，降低对现有实验流水线的破坏风险。

### Q2：缓存会不会读到脏数据？

不会。NPZ 缓存使用文件 `mtime_ns + size` 做失效判定，文件有变化会自动重载。

### Q3：这次迁移是否要求更新历史实验配置？

不要求。旧配置在本轮兼容层下可继续运行。

---

## 7. 后续建议（下一轮）

1. 继续把 `train.py` 的 staged 训练与 heartbeat 逻辑下沉到独立 runtime 模块
2. 对 `gaussian_physics_trainer.py` 做更细粒度性能剖析（按 loss 分段计时）
3. 将 `tools` 与 `3_experiments` 的 command builder 进一步统一成一个公共子包

