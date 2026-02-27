# 项目大链路：数据生成 → 训练与验证 → 测试

本文档基于代码阅读与推理，明确写出**数据生成–训练与验证–测试**整条大链路，以及每个部分的子链路逻辑与推理依据。

---

## 一、整体大链路概览

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  1. 数据生成 (1_data_generation/ + Falcor/Mitsuba)                              │
│     → 输出: parametric_tensor.npz + metadata.json (+ 可选 manifest.json)          │
└───────────────────────────────────┬─────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  2. 训练与验证 (2_src/ + 3_experiments/)                                         │
│     → 输入: data_root/manifest → 输出: best_model.pt, last_model.pt, 可选 sh_scaler │
└───────────────────────────────────┬─────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  3. 测试 (2_src/tests/ 单元/集成) + 评估 (3_experiments/scripts/eval.py)         │
│     → 输入: checkpoint + data_root → 输出: eval.json (MAE/RMSE 等)               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**统一编排入口**（可选）：
- `tools/run_dataset.py`：仅跑数据生成步骤。
- `tools/run_pipeline.py`：按 YAML 串联 dataset → train → eval，并把 dataset 的 `output_dir` 注入 train 的 `data_root`，把 train 的 `output_dir` 用于解析 eval 的 checkpoint。

---

## 二、第一部分：数据生成（子链路）

### 2.1 目标与产物

- **目标**：得到「探针位置 × 光照配置 × 球谐系数」的离散表，供后续用低秩+FiLM 模型学习压缩。
- **核心产物**：`parametric_tensor.npz`，内含：
  - `tensor`: `[P, M, 27]`（P 探针，M 个时刻/配置，27 维 SH）
  - `probe_positions`: `[P, 3]`
  - `light_configs`: `[M, N, 12]`（每配置 N 个光源的 12 维描述子）
  - `light_mask`: `[M, N]`（可选，默认全 1）
  - （部分生成器还有 `metadata`、`valid_mask` 等）

### 2.2 两条主数据生成子链路

项目里存在**两条并列**的数据生成子链路，最终都写入上述格式的 `parametric_tensor.npz`。

#### 子链路 A：Mitsuba 3 路径

- **入口**：`1_data_generation/generate_dataset.py`（多时刻太阳轨迹）、`generate_5D_parametric.py`（5D 参数化）、`generate_transfer_tensor.py`（传输张量）等。
- **逻辑**：
  1. **场景与边界**：加载 Mitsuba 场景或内置简单场景，得到 `scene_bounds`（或手动指定）。
  2. **探针采样**：`core/probe_sampler.sample_probe_positions(num_probes, scene_bounds)` → 在边界内均匀网格或随机采样，得到 `[P, 3]`。
  3. **光照配置**：按脚本不同，生成多组光照（例如太阳方向、5D 参数、灯位等），共 M 组。
  4. **SH 烘焙**：对每组光照、每个探针，调用 `core/sh_baker.bake_sh_at_probe(scene, probe_pos, num_samples, spp)`：
     - 用 `utils/spherical_harmonics.fibonacci_sphere(num_samples)` 得到采样方向；
     - 每个方向用极小视野相机在探针处渲染得到 radiance；
     - 用 `fit_sh_coefficients` 拟合 2 阶 SH → 27 维。
  5. **组装与保存**：将 `[P, M, 27]`、`probe_positions`、`light_configs`（及可选 `light_mask`）写入 `parametric_tensor.npz`（如 `generate_5D_parametric.py` 第 393–409 行）。

**推理依据**：`probe_sampler.py` 提供网格/随机采样；`sh_baker.py` 明确「Fibonacci 球采样 → 每方向渲染 → fit_sh」；各 `generate_*.py` 中循环「场景/光照 → 每探针 bake_sh → 填 tensor → savez」。

#### 子链路 B：Falcor 路径（当前主推）

- **入口**：`1_data_generation/falcor/generate_bistro_temporal_spheres.py`（Bistro 场景+运动发光球）、`falcor/generate_multilight_falcor.py`（多灯）、`falcor/generate_5d_parametric_falcor.py`（5D Falcor）等。
- **逻辑**：
  1. **场景与边界**：`falcor_render.load_scene` / `get_scene_bounds` 加载 .pyscene 等，得到 `bounds`。
  2. **探针**：网格或混合（均匀+表面）采样，得到 `probes [P, 3]`（如 `generate_probe_grid`、`_sample_aabb_surface_positions`）。
  3. **光照配置与轨迹**：例如 Bistro 脚本里用 `_compute_trajectories` 生成多帧、多球位置，每帧对应一组光源描述子 `light_configs[out_idx, :, :]`（12 维/灯）。
  4. **SH 烘焙**：对每一帧、每个探针调用 Falcor 渲染：
     - **cubemap 模式**：`render_sh_cubemap(testbed, graph, scene, probe, cube_res=..., num_frames=..., radiance_clamp=...)`；
     - **dir 模式**：对预计算方向用 `render_single_pixel` 取 radiance，再 `fit_sh_coefficients(directions, radiances, max_order=2)` 得到 27 维。
  5. **组装与保存**：`tensor[probe_idx, out_idx, :] = sh`，最后 `np.savez_compressed(output_dir / "parametric_tensor.npz", tensor=..., probe_positions=..., light_configs=..., light_mask=..., metadata=...)`（如 `generate_bistro_temporal_spheres.py` 第 501–512 行）。

**推理依据**：Falcor 脚本中显式循环 `frame_indices` × `probes`，内层调用 `render_probe_sh(probe)`（内部要么 cubemap 要么 dir+fit_sh）；最终统一写入同一 npz 格式。

### 2.3 可选后处理子链路

- **验证**：`1_data_generation/validate_dataset.py`（`DatasetValidator`）检查目录下是否有 `parametric_tensor.npz`，做文件完整性、形状、数值范围、SH 统计等（对 parametric 走 `check_parametric_tensor` 等）。
- **Manifest**：`tools/run_dataset.py` 在 `--write-manifest` 时用 `manifest_utils.read_parametric_stats` 读 npz，再 `write_manifest(...)` 写 manifest，便于追溯和训练时解析 `data_root`。

### 2.4 数据生成小结

| 环节       | 子步骤                     | 关键代码/模块                                      |
|------------|----------------------------|----------------------------------------------------|
| 场景/边界  | 加载场景、取 AABB          | Mitsuba `load_file` / Falcor `load_scene`、bounds  |
| 探针       | 采样 P 个位置              | `probe_sampler` / Falcor `generate_probe_grid` 等  |
| 光照配置   | 生成 M 组光源描述 [M,N,12] | 各 generator 的 config/trajectory 逻辑             |
| SH 烘焙    | 每 (probe, config) → 27 维 | `sh_baker.bake_sh_at_probe` / Falcor render_sh_*   |
| 写入       | 保存 npz + metadata        | `np.savez_compressed(..., parametric_tensor.npz)`  |
| 可选       | 验证 + manifest            | `validate_dataset.py`、`run_dataset.py` + manifest  |

---

## 三、第二部分：训练与验证（子链路）

### 3.1 目标与产物

- **目标**：用「探针位置 + 光源描述子」预测 27 维 SH，学到可压缩的 Gaussian–低秩–FiLM 表示。
- **产物**：`best_model.pt` / `last_model.pt`，以及可选的 `sh_scaler.npz`、训练日志/ Rerun 等。

### 3.2 训练入口与配置

- **入口**：`3_experiments/scripts/train.py`。
- **配置**：YAML（如 `3_experiments/configs/bistro_clean_train.yaml`），通过 `merge_configs` / `validate_with_schema` 与 CLI 合并，映射到 `experiment` / `data` / `model` / `training`（见 `_apply_config`）。

### 3.3 数据加载子链路

1. **data_root 解析**：若未提供 `--data-root`，则从 `--manifest` 的 `output_dir` 解析（`resolve_data_root`）。
2. **Dataset**：`2_src/data/lightset_dataset.create_dataloaders_lightset(data_root, batch_size, train_ratio, val_ratio, ...)`：
   - 读 `data_root/parametric_tensor.npz`；
   - 取 `tensor [P,M,27]`、`probe_positions`、`light_configs`、`light_mask`（及可选 `valid_mask`），无效探针过滤；
   - 按 config 索引划分 train/val/test（如 0.7/0.15/0.15）；
   - 可选 `normalize_probes` 将探针坐标归一化到 [-1,1]。
3. **DataLoader**：对 train/val/test 各建 DataLoader，batch 中键包括 `probe_position`、`light_params`、`sh_coeffs`、`light_mask` 等。

**推理依据**：`lightset_dataset.py` 明确期望 npz 含 `tensor`/`probe_positions`/`light_configs`/`light_mask`，并按 config 划分；`train.py` 的 `build_variant("unified_set", ...)` 只使用 `create_dataloaders_lightset`。

### 3.4 模型与适配器

- **模型**：`2_src/models/gaussian_physics_unified.GaussianPhysicsCompressionUnified`（K 个 3D Gaussian、低秩 U、FiLM、系数矩阵等），输入为位置与 light set，输出 27 维 SH。
- **适配器**：`BatchAdapter(params_key="light_params", mask_key="light_mask")`，将 batch 解包为 `(positions, params, targets, mask)`，并支持 `target_transform`/`target_inverse`（如 SH scaler）。

### 3.5 初始化与可选 SH Scaler

- **初始化**：若非 `no_init`，调用 `init_fn(model, train_loader)`；对 `unified_set` 即 `_init_5d`：用 dataset 的 `tensor`、`probe_positions`、`light_configs_subset` 做 K-Means/SVD 等，`model.init_from_kmeans(...)`。
- **SH Scaler**：若 `enable_sh_scaler`，用 `AdaptiveSHScaler.fit_from_dataset(train_loader.dataset, ...)` 拟合，保存到 `output_dir/sh_scaler.npz`，并把 `scaler.transform`/`inverse` 挂到 adapter 的 `target_transform`/`target_inverse`。

### 3.6 单 epoch 训练子链路（train_epoch）

在 `2_src/training/gaussian_physics_trainer.py` 中：

1. **前向**：`adapter.unpack(batch)` → `(positions, params, targets, mask)`；`model(positions, params, top_k=..., light_mask=mask)` → `preds`。
2. **损失**：
   - **recon**：`_recon_loss(preds, targets)`（MSE/L1/Charbonnier，可选按 SH basis 加权）；
   - **image**：可选 `_compute_image_loss(preds_eval, targets_eval)`（在 64 方向等上重建 radiance 再 MSE/Charbonnier）；
   - **temporal**：`_compute_temporal_loss()`（模型内或外部 `temporal_loss_fn`）；
   - **linearity**：`_compute_linearity_loss` + `_compute_linearity_aug_loss`（线性叠加约束）；
   - **spatial**：`_compute_spatial_loss`（邻探针平滑）。
3. **总损失**：`loss_total = recon + λ_image*image + λ_temporal*temporal + λ_linearity*(linearity+linearity_aug) + λ_spatial*spatial`。
4. **反向与优化**：`loss_total.backward()`，可选 `clip_grad_norm_`，`optimizer.step()`。

### 3.7 验证子链路（validate_epoch）

- 与训练同一 adapter 与 model 前向；对 pred/target 做 `adapter.inverse_targets` 后计算 **MAE、RMSE、Charbonnier、superposition、img_mae/img_rmse/img_psnr** 等，不反传。

### 3.8 主循环（fit）

- 按 epoch 循环：`train_epoch(train_loader)` → `validate_epoch(val_loader)`；
- 学习率：可选 `CosineAnnealingLR` 或 `ReduceLROnPlateau`；
- 按 `best_metric`（默认 "mae"）保存 `best_model.pt`，每轮保存 `last_model.pt`；
- checkpoint 中写入 `model_state_dict` 与 `meta`（split、model 超参、sh_scaler 等），供 eval 复现；
- 可选 Rerun 可视化（高斯中心、探针误差、SH 对比等）。

**推理依据**：`gaussian_physics_trainer.py` 中 `train_epoch`/`validate_epoch`/`fit` 的代码路径与上述一致；`train.py` 中 `run_training` 构建 model/adapter/loaders → init → scaler → trainer.fit → 可选 test_metrics。

### 3.9 训练与验证小结

| 环节         | 子步骤                     | 关键代码/模块                                  |
|--------------|----------------------------|------------------------------------------------|
| 数据         | 读 npz、划分、归一化探针   | `lightset_dataset.create_dataloaders_lightset`  |
| 模型         | 构建 + 可选 K-Means/SVD 初始化 | `GaussianPhysicsCompressionUnified`、`_init_5d` |
| 目标变换     | 可选 SH scaler             | `AdaptiveSHScaler`、adapter transform/inverse   |
| 单步训练     | 前向、多损失、反传、优化   | `GaussianPhysicsTrainer.train_epoch`           |
| 验证         | 前向、MAE/RMSE/PSNR 等     | `GaussianPhysicsTrainer.validate_epoch`        |
| 循环与保存   | epoch、best/last、meta     | `GaussianPhysicsTrainer.fit`、`train.py`       |

---

## 四、第三部分：测试与评估（子链路）

### 4.1 两类「测试」

- **单元/集成测试**：`2_src/tests/` 下 pytest，验证数据、模型、训练脚本、管线工具等行为，不产出业务指标。
- **评估（Eval）**：`3_experiments/scripts/eval.py`，在指定 split 上对 checkpoint 算 MAE/RMSE 等并写 JSON，是「验证模型好坏」的正式步骤。

### 4.2 单元/集成测试子链路

- **数据**：如 `test_mainline_smoke.py` 构造最小 `parametric_tensor.npz`（随机 [P,M,27]、probe_positions、light_configs、light_mask），用 `LightSetDataset` 加载并取 sample，检查键与形状。
- **模型**：同一测试里用 `GaussianPhysicsCompressionUnified` 做一次前向，检查输出 shape `(1, 27)`。
- **训练脚本**：`test_train_script_coverage.py` 等覆盖 `train.py` 的参数解析、config 合并、dry-run 等。
- **管线/工具**：`test_run_dataset.py`、`test_manifest_utils.py`、`test_workflow_config.py` 等验证 run_dataset、manifest、workflow_config 行为。

**推理依据**：tests 目录下各 `test_*.py` 的 import 与断言直接对应上述模块。

### 4.3 评估脚本子链路（eval.py）

1. **输入**：`--data-root`（含 `parametric_tensor.npz`）、`--checkpoint`（best_model.pt / last_model.pt）、`--split`（train/val/test）。
2. **加载 checkpoint**：`torch.load`，取 `model_state_dict` 与 `meta`；从 state 推断 K、rank、embed_dim、sh_dim；从 meta 恢复 train_ratio、val_ratio、top_k、light_dim、scaler 等。
3. **模型**：构建与 checkpoint 同结构的 `GaussianPhysicsCompressionUnified`，`load_state_dict(state)`；若 meta 中有 sh_scaler，则构建 `AdaptiveSHScaler` 并设为 adapter 的 `target_transform`/`target_inverse`。
4. **数据**：`create_dataloaders_lightset(data_root, ..., train_ratio, val_ratio)`，按 `split` 选用对应 loader。
5. **评估循环**：`adapter.unpack(batch)` → `model(positions, params, top_k, light_mask)` → `adapter.inverse_targets(preds, targets)` → 累加 |diff| 与 diff²，得到 MAE、RMSE。
6. **输出**：将 split、checkpoint 路径、data_root、MAE、RMSE、numel、模型超参等写入 `eval.json`（默认在 checkpoint 同目录）。

**推理依据**：`eval.py` 的 `main()` 与 `run_eval` 逻辑与上述步骤一致；输出格式见 `payload` 与 `out_path.write_text(json.dumps(...))`。

### 4.4 测试与评估小结

| 类型       | 目的               | 入口/输入                     | 输出/断言                    |
|------------|--------------------|-------------------------------|------------------------------|
| 单元/集成  | 逻辑正确性         | pytest `2_src/tests/`         | 断言 shape、键、返回值       |
| 评估       | 指标（MAE/RMSE 等）| `eval.py --data-root --checkpoint --split` | `eval.json`                 |

---

## 五、推理过程简要总结

1. **数据生成**：通过搜索「parametric_tensor.npz 的写入位置」和「bake_sh / render_sh」的调用，区分出 Mitsuba（probe_sampler + sh_baker + 各 generate_*.py）与 Falcor（falcor_render + 各 falcor/generate_*.py）两条子链路；两者最终都写入同一 npz 约定。
2. **训练与验证**：从 `train.py` 的 `run_training` 和 `build_variant` 追到 `lightset_dataset` 的 npz 约定与划分方式，再追到 `GaussianPhysicsTrainer` 的 `train_epoch`/`validate_epoch`/`fit`，归纳出数据→初始化→scaler→多损失训练→验证→保存的完整子链路。
3. **测试**：区分「pytest 测试」与「eval 脚本」；从 `eval.py` 的 main 和 `run_eval` 归纳出 checkpoint 加载、模型重建、scaler 恢复、按 split 评估、写 JSON 的子链路。

以上即项目「数据生成–训练与验证–测试」大链路及各子链路的逻辑与推理过程。
