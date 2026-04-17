# 可微 Image Loss（GBuffer 监督）实现与审计

## 0. 文档元信息
- 更新时间：2026-04-13（可微化改造已合入）
- 分支：`lite/pg-gcpl-min`
- 基线提交：`aa36c02`
- 审计范围：当前仓库中 `gbuffer_image_loss` 的训练接入链路（配置解析、数据加载、训练执行、指标与 checkpoint 记录、测试覆盖）

## 0.1 本次修改目的（2026-04-13）
本次代码修改的核心目标是：**将 GBuffer image loss 分支收敛为“全程可微分”的训练路径**，避免离散路由和硬截断导致的不可微点干扰梯度。

已完成的关键改动：
1. **Full-K soft routing（无 top-k 截断）**  
   文件：`2_src/models/gaussian_physics_unified.py:400-429`  
   在 `training_soft_routing=True` 且 `routing_soft_topk<=0` 时，改为对全部 `K` 专家做 dense softmax 路由，不再做 top-k 截断。
2. **GBuffer loss 分支强制走软路由**  
   文件：`2_src/training/gaussian_physics_trainer.py:1507-1515`  
   `_compute_gbuffer_image_loss()` 中调用 `_forward(..., training_soft_routing=True, routing_soft_topk=0)`，明确启用 full-K soft 路由。
3. **渲染正值约束由硬 `clamp` 改为平滑 `softplus`**  
   文件：`2_src/training/gaussian_physics_trainer.py:1446-1450`  
   `irradiance` 不再使用 `clamp(min=0)`，改为 `softplus(beta=8.0)`（参数见 `:411`），保持正值同时保留平滑梯度。
4. **GBuffer loss 类型固定为 Charbonnier**  
   文件：`2_src/training/gaussian_physics_trainer.py:231-233, 374-377, 1428-1433`  
   训练端强制 `charbonnier`，不再允许 `l1/huber`。  
   配置层同步收敛：`3_experiments/scripts/train_runtime_config.py:369-373`。  
   CLI 同步收敛：`3_experiments/scripts/train.py:2943-2946`。
5. **测试同步更新并通过**  
   文件：`2_src/tests/test_train_runtime_config.py:159-165, 224-225, 339-363`  
   命令：`pytest -q tests/test_trainer_image_loss.py tests/test_train_runtime_config.py tests/test_train_script_coverage.py`  
   结果：`28 passed`。

---

## 1. 目标与边界

### 1.1 模块目标
在现有 `proxy_sh` 训练主路径上新增一个可插拔的 `gbuffer_image_loss` 辅助分支，用离线 GBuffer 样本做像素级监督，提升最终图像一致性。

### 1.2 当前边界（重要）
- 当前实现是**离线监督**：训练端读取 `.pt/.npz` 样本，不是 Falcor 在线零拷贝 autograd 图。
- Falcor 仅作为上游数据生产端（当前仓库尚未提供现成导出脚本，见风险章节）。
- 训练主目标（probe SH 重建）仍是核心；`gbuffer_image_loss` 是附加项。

---

## 2. 端到端实现链路（文件可对照）

## 2.1 入口与配置解析
1. `train.py` 接收 CLI 参数并合并 YAML。
2. `train_runtime_config.py` 解析 `training.gbuffer_image_loss`。
3. runtime override 回写到 `args`。

关键位置：
- `3_experiments/scripts/train.py:2932-2975`：CLI 参数定义（`--gbuffer-image-loss-*`）
- `3_experiments/scripts/train_runtime_config.py:349-456`：`resolve_gbuffer_image_loss_config`
- `3_experiments/scripts/train_runtime_config.py:885-985`：`apply_runtime_overrides` 回填 args
- `3_experiments/scripts/train_runtime_config.py:1085-1122`：`resolve_runtime_config_bundle` 把 gbuffer cfg 打包

## 2.2 数据加载与字段规约
`train.py` 在启用 gbuffer loss 时创建 dataloader，并把 probe 归一化边界传给 trainer。

关键位置：
- `3_experiments/scripts/train.py:1598-1636`：创建 `gbuffer_loader`，检查 `probe_min/probe_max` 与 `dataset_root`
- `2_src/data/gbuffer_supervision_dataset.py:38-221`：`GBufferSupervisionDataset`
- `2_src/data/gbuffer_supervision_dataset.py:224-264`：`create_gbuffer_supervision_dataloader`

## 2.3 Trainer 接入与训练执行
`train.py` 将 `gbuffer_image_loss_cfg + gbuffer_loader + probe bounds` 传入 trainer。

关键位置：
- `3_experiments/scripts/train.py:1648-1732`：trainer 构造参数
- `2_src/training/gaussian_physics_trainer.py:223-267`：trainer 中保存并校验 gbuffer 配置
- `2_src/training/gaussian_physics_trainer.py:1457-1514`：`_compute_gbuffer_image_loss` 核心逻辑
- `2_src/training/gaussian_physics_trainer.py:1769-1776`：把 gbuffer loss 加入 `loss_total`
- `2_src/training/gaussian_physics_trainer.py:2761-2765`：按 epoch 更新 gbuffer loss warmup 权重

## 2.4 Checkpoint / Metrics 口径落盘
- `3_experiments/scripts/train.py:1420`：`checkpoint_meta["training"]["gbuffer_image_loss"]`
- `3_experiments/scripts/train.py:2657-2676`：`result_metrics` 写入 gbuffer 参数快照
- `2_src/training/gaussian_physics_trainer.py:2068-2082`：train totals 中含 `gbuffer` 与 `gbuffer_samples`
- `2_src/training/gaussian_physics_trainer.py:2189-2192`：输出 `gbuffer_image_loss_weight`

---

## 3. 模块计算逻辑（逐步）

每个训练 step（eager 路径）：

1. 主任务前向与常规 loss 先计算  
   位置：`2_src/training/gaussian_physics_trainer.py:1760-1768`

2. 调用 gbuffer 分支  
   位置：`2_src/training/gaussian_physics_trainer.py:1769`

3. gbuffer 分支内部流程（`_compute_gbuffer_image_loss`）  
   位置：`2_src/training/gaussian_physics_trainer.py:1457-1514`
   - 按 `every_steps` 触发（`1465-1466`）
   - 从 `gbuffer_loader` 取 batch（`1468-1470`）
   - 读取 `posW/normW/albedo/gt_linear/light_params/light_mask`（`1472-1477`）
   - 世界坐标归一化到 probe 空间（`1491`，函数定义 `1419-1425`）
   - 扩展 `light_params/light_mask` 到像素维（`1492-1509`）
   - 调用模型得到 `pred_sh`（软路由 full-K，`1507-1515`）
   - SH + 法线 -> irradiance -> RGB（`1512`，渲染函数 `1442-1455`）
   - 线性域损失（固定 Charbonnier）（`1517`，loss函数 `1428-1433`）

4. 乘以当前权重并并入总损失  
   位置：`2_src/training/gaussian_physics_trainer.py:1772-1776`

---

## 4. 数据契约（GBuffer 样本格式）

`GBufferSupervisionDataset` 最终输出固定 keys：
- `posW`: `[N,3]`
- `normW`: `[N,3]`
- `albedo`: `[N,3]`
- `gt_linear`: `[N,3]`
- `light_params`: `[S,F]`
- `light_mask`: `[S]`（缺省时自动补 1）
- `source_path`: `str`

实现要点：
- 支持输入样本为 `.pt` 或 `.npz`（`78-87`）
- `strict_keys=True` 时禁止 fallback key（`103-119`）
- `valid_mask` 会先过滤再采样像素（`165-189`）
- 支持 `gt_color_space=srgb` 自动 decode 到线性域（`24-29`, `153-154`）
- 当前 `domain` 仅支持 `linear`（配置层与 trainer 双重校验）

---

## 5. 配置口径与参数解释（当前实现）

来自 `training.gbuffer_image_loss`（YAML）或同名 CLI 参数：

- `enabled`: 是否启用分支
- `lambda`: 分支损失权重
- `warmup_epochs`: 权重 warmup 周期
- `every_steps`: 每多少 step 计算一次 gbuffer loss
- `loss_type`: 固定 `charbonnier`（配置/CLI 已限制）
- `huber_delta`: 兼容保留字段（不再参与 gbuffer loss 计算）
- `dataset_root`: 离线样本路径（启用时必填）
- `pixel_sample_count`: 每样本抽样像素数
- `domain`: 当前仅 `linear`
- `gt_color_space`: `linear | srgb`
- `strict_keys`: 是否强制主键，不允许 fallback
- `pos_key/normal_key/albedo_key/gt_linear_key/light_params_key/light_mask_key/valid_mask_key`

约束校验位置：
- `3_experiments/scripts/train_runtime_config.py:349-456`
- `2_src/training/gaussian_physics_trainer.py:373-385`

---

## 6. 路由口径一致性审计结论

## 6.1 已一致的部分
- gbuffer 分支在训练 epoch 内执行，`_train_routing_mode=True`。  
  位置：`2_src/training/gaussian_physics_trainer.py:2066`
- gbuffer 分支已显式传入 `training_soft_routing=True` 且 `routing_soft_topk=0`。  
  位置：`2_src/training/gaussian_physics_trainer.py:1507-1515`
- 路由器在 `routing_soft_topk<=0` 时使用 full-K dense soft routing。  
  位置：`2_src/models/gaussian_physics_unified.py:400-417`

结论：本次改造后，gbuffer image loss 主链路已切到 full-K soft 路由口径。

## 6.2 当前不一致风险（重点）
- 主训练分支：先 `_compute_routing(...)`，再 `_forward(..., routing=routing)`，从而可应用 `train_routing_param_mode`。  
  位置：`2_src/training/gaussian_physics_trainer.py:1683-1684`, `989-996`
- gbuffer 分支：`_forward(...)` 没传 `routing`，走 `model.forward(...)`。  
  位置：`2_src/training/gaussian_physics_trainer.py:1511`
- `model.forward(...)` 内部调用 `forward_with_routing(...)` 时没有传 `param_select_mode`，默认是 `gather`。  
  位置：`2_src/models/gaussian_physics_unified.py:456`, `523-540`

结论：若主训练设为 `train_routing_param_mode=dense_masked`，gbuffer 分支仍是 `gather` 路径，存在口径分叉风险。

---

## 7. 文件增删改清单（当前工作树）

以下为 `gbuffer_image_loss` 直接相关文件。

### 7.1 新增（A）
- `2_src/data/gbuffer_supervision_dataset.py`
- `3_experiments/scripts/train_runtime_config.py`
- `2_src/tests/test_train_runtime_config.py`

### 7.2 修改（M）
- `2_src/data/__init__.py`
- `2_src/training/gaussian_physics_trainer.py`
- `3_experiments/scripts/train.py`
- `2_src/tests/test_trainer_image_loss.py`
- `2_src/tests/test_train_script_coverage.py`

### 7.3 删除（D）
- 当前未发现与该模块直接相关的删除文件。

---

## 8. 已验证行为与测试覆盖

本次执行过的测试命令：

```bash
cd 2_src
pytest -q tests/test_trainer_image_loss.py tests/test_train_runtime_config.py tests/test_train_script_coverage.py
```

结果：
- `28 passed, 1 warning`
- warning 为当前环境 CUDA 初始化限制引起，不影响 CPU 单测覆盖结论。

关键覆盖点：
- gbuffer train step 能产出 `gbuffer` 指标：`test_trainer_image_loss.py:99-141`
- strict key 行为：`test_trainer_image_loss.py:144-165`
- srgb->linear 解码：`test_trainer_image_loss.py:167-197`
- runtime 配置解析与 override：`test_train_runtime_config.py:107-261`, `298-400`
- 脚本层 config resolve 覆盖：`test_train_script_coverage.py:235-311`

---

## 9. 当前潜在问题（按优先级）

## P0. `dense_masked` 与 gbuffer 分支路径不一致
现象：
- 主分支可走 `dense_masked`，gbuffer 分支固定 `gather`。

影响：
- 训练目标与辅助图像监督在算子路径上不一致，可能导致梯度方向偏差，尤其在你们当前重点优化路由/索引反向热点时会被放大。

证据：
- `2_src/training/gaussian_physics_trainer.py:1511`
- `2_src/training/gaussian_physics_trainer.py:989-996`
- `2_src/models/gaussian_physics_unified.py:456`, `523-540`

## P1. SH scaler 在 gbuffer 分支可能存在域不一致
现象：
- 主分支 proxy image loss 会对 `preds/targets` 应用 `adapter.inverse_targets`。  
  位置：`2_src/training/gaussian_physics_trainer.py:1688-1691`
- gbuffer 分支直接拿 `pred_sh` 渲染，不经过 `target_inverse`。  
  位置：`2_src/training/gaussian_physics_trainer.py:1511-1513`
- SH scaler 的 transform/inverse 在训练入口是可开启的。  
  位置：`3_experiments/scripts/train.py:1117-1137`

影响：
- 若启用 SH scaler，gbuffer loss 可能在“缩放域 SH”上渲染，导致亮度/色调偏差与 Falcor 口径不一致。

## P1. GT 色彩域标注错误会直接造成亮度/色偏
现象：
- `gt_color_space=srgb` 时会先 clamp 到 `[0,1]` 再做逆变换（`_srgb_to_linear`）。  
  位置：`2_src/data/gbuffer_supervision_dataset.py:24-29`

影响：
- 若 GT 实际是 linear HDR，却被当作 srgb 解码，会发生显著失真（常见为整体偏亮/偏色）。

## P2. GBuffer 离线数据生产脚本缺口
现象：
- 仓库当前仅有 dataset loader 与训练接入，未发现现成 `generate_gbuffer_dataset` 脚本。

影响：
- 数据集构建流程不可复现，字段口径容易漂移（特别是 `light_params/light_mask` 与主训练契约）。

## P2. 坐标归一化依赖主训练集边界
现象：
- gbuffer 像素世界坐标按主训练集 `probe_min/probe_max` 归一化。  
  位置：`3_experiments/scripts/train.py:1602-1610`, `2_src/training/gaussian_physics_trainer.py:1419-1425`

影响：
- 若 gbuffer 数据来自不同场景/坐标系，输入会超域，导致监督信号不稳定。

## P3. 开启 gbuffer loss 时 CUDA Graph 自动失效（性能侧）
现象：
- 只要 gbuffer loss 有效，就禁用 CUDA Graph。  
  位置：`2_src/training/gaussian_physics_trainer.py:683-686`

影响：
- 会牺牲一部分 launch-overhead 优化空间（这是当前实现的保守策略，不是功能错误）。

---

## 10. 建议的最小治理顺序（不改代码版）

1. 先固定训练口径：若使用 gbuffer loss，优先保持 `train_routing_param_mode=gather`，避免 P0 分叉。
2. 若启用 SH scaler，先做一次对照实验验证 gbuffer 分支亮度是否偏移（重点看第 0 帧 GT/Pred 一致性）。
3. 数据生产前锁定样本规范：`gt_linear` 必须线性域，且 `light_params/light_mask` 与主训练数据编码完全同构。
4. 在配置里显式写 `training.gbuffer_image_loss`，不要依赖 CLI 临时覆盖，避免复现实验时口径丢失。

---

## 11. 当前状态结论

- `gbuffer_image_loss` 已接入训练主循环，可工作，可记录指标，可被 checkpoint / result metrics 追踪。
- 模块的“是否能跑”已经成立；当前关键是“口径一致性”与“数据规范化流程”。
- 现阶段最重要风险是：
  - `dense_masked` 与 gbuffer 分支 `gather` 的路径分叉（P0）
  - SH scaler + gbuffer 渲染域不一致（P1）
