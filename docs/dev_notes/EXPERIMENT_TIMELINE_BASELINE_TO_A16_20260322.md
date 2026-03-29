# Baseline 到 A1.6 全流程实验时间线（技术实现细化版，2026-03-22）

## 0. 口径与证据说明（先读）

1. 结果来源优先级
   - `full-suite`（外部统一复算，最适合跨实验横向对比）
   - 训练内 `falcor_best_summary.json` / `global_best_falcor_summary.json`（A1.3+主线大量依赖）
2. 同名目录复用风险（关键）
   - `exp_A1_1_joint_lrs_ema` 在不同时期被复用，旧A1.1与后续“新A1.1（后命名A1.2）”产物混在同一路径。
   - 因此同一路径的 checkpoint 与后续 full-suite 复算结果可能对应不同实验阶段。
3. 本文规则
   - 每个实验都写：相对上一个实验改动、相对 baseline 累积变化、结果、来源。
   - 所有“术语”都映射到代码函数/配置键，不再用口头简称。

---

## A. 项目基础架构（给初次接手者）

### A.1 任务定义与最小心智模型

本项目要学习一个函数：

- 输入：`(probe_position p, light_set L_set)`
- 输出：该位置在该光照集合下的 `27 维 SH 系数`（RGB 各 9 维）

可把它理解为“空间条件 + 光照条件 -> SH 场值”的压缩重建器。  
训练和大部分验证在 SH 域完成，最终质量用 Falcor 渲染域 HDR PSNR 校验。

### A.2 数据与样本粒度（`2_src/data/lightset_dataset.py`）

`parametric_tensor.npz` 的关键字段：

- `tensor`: `[P, M, 27]`，P 个 probe、M 种光照配置下的 SH 真值
- `probe_positions`: `[P, 3]`
- `light_configs`: `[M, N, 12]`，每个配置包含 N 个灯，每灯 12 维描述符
- `light_mask`: `[M, N]`（可选）

训练样本粒度是笛卡尔展开的 `(probe_idx, config_idx)`，即一条样本对应一个 `(p, L_set, SH_gt)`。

### A.3 模型主干（`2_src/models/gaussian_physics_unified.py`）

统一模型 `GaussianPhysicsCompressionUnified` 可拆成 5 个部件：

1. 空间专家库（可训练参数）
- `mu, log_scale`：专家中心与尺度（路由）
- `U, U_l0`：专家低秩基（basis）
- `coeffs, coeffs_l0`：把光照潜码映射到 rank 权重（coeff）

2. 光照编码器
- 默认：`LightSetEncoder`（normal）
- 可替代：`bypass_fixed / bypass_linear / bypass_mlp_16_32`

3. 路由器（由位置决定）
- `compute_gaussian_routing(p)` 产生 top-k 专家与权重（hard/soft 两种）

4. rank 权重生成
- `time_weights = coeffs @ z`（按专家计算）

5. FiLM 动态基调制（可开关）
- `U <- U * (1 + gamma(z)) + beta(z)`，再与 `time_weights` 合成 SH

核心前向可写成：

- `SH(p,L) = Σ_j w_j(p) * [U_j(z(L)) @ time_weights_j(z(L))]`

### A.4 训练系统编排（`3_experiments/scripts/train.py` + `2_src/training/gaussian_physics_trainer.py`）

训练入口的职责分层：

- `train.py`
- 读 YAML -> 建 dataset/model/trainer
- 执行 staged 或 single-stage 训练
- 管理 resume/load_model/global_best/val_profiles/falcor periodic 等“流程级逻辑”
- `GaussianPhysicsTrainer`
- 实现 batch 前向、loss 计算、优化器 step、EMA、保存 best、val profile 评估

可训练参数分组（staged 的核心控制面）：

- `routing` / `basis` / `coeff` / `encoder` / `film` / `all`

因此时间线里所有“warmup/joint/refine/anchor”本质都是：

- 哪些分组在训（`trainable_groups`）
- 每组多大学习率（`group_lrs`）
- 以哪个指标写入当前 stage 的 `best_model.pt`（`best_metric`）

### A.5 损失栈与评估栈（必须区分）

损失栈（训练时优化）：

- 主重建损失：SH 域 `recon_loss`（MSE/Charbonnier）
- 可选：`proxy_image_loss`（SH -> 采样伪图像后的 linear/log/mix）
- 可选：`routing_balance_loss`（专家负载方差归一）
- 可选：linearity / temporal / spatial 等正则项

评估栈（用于选模和最终对比）：

1. `val` 默认口径（通常 hard3）
- 指标包含 `mae/rmse/charbonnier/img_psnr...`

2. `val_profiles` 多口径
- 例如 `hard3/soft8_t018/...`
- 每个 profile 可单独产出 `best_model_val_<profile>.pt`

3. `falcor_periodic_eval`
- 低频调用 Falcor 渲染，输出渲染域指标
- 可维护 `falcor_best_summary.json` 与 `global_best_falcor.pt`

4. `full-suite`
- 训练外统一复算脚本，最适合跨实验横向比较

### A.6 结果文件体系（看目录就知道实验跑到哪）

以 `3_experiments/results/bistro_clean_v2/<exp_name>/` 为例：

- `stageXX_*/best_model.pt`：stage 内主 best（按 stage best_metric）
- `best_model_val_<profile>.pt`：profile 内 best
- `global_best_val_<profile>.pt`：跨 stage profile best
- `global_best_falcor.pt`：跨 stage Falcor best
- `falcor_best_summary.json` / `global_best_falcor_summary.json`：Falcor 选模摘要
- `staged_training_summary.json`：每个 stage 的 trainable groups / group_lrs / 末轮指标

### A.7 时间线如何直接映射到架构改动

把 A1~A1.6 看成“对 A.3/A.4/A.5 各部件开关的连续操作”：

- Baseline：`normal encoder + hard routing + single-stage`，无 proxy/EMA/profile/falcor periodic
- A1：只改 A.4（引入 staged 与分组冻结）
- 旧A1.1：改 A.4（soft routing train + EMA + 分组LR）
- A1.2：改 A.5（proxy image loss）+ 评估链路（val_profiles + falcor periodic）
- A1.3：在 A1.2 上改 A.4（两阶段细化）+ A.5（routing balance 退火）
- A1.4：改 A.4 流程控制（strict resume + anchor stage + global_best）
- A1.5-proxyfix：改 A.5 数值细节（proxy clamp）+ 训练启动方式（load_model）
- A1.5-bypassA/B 与 A1.6-P2：直接改 A.3 光照编码分支（normal vs bypass 系列）
- A1.6-P1：只扫 A.5 中 proxy 的 `lambda`

这也是为什么同样“PSNR 上下波动”，其根因可能来自完全不同层次：

- 架构层（encoder/routing/basis）
- 优化层（staged/group_lrs/anneal）
- 选模层（best_metric/profile/falcor/global_best）

---

## 1. 术语到代码实现（统一定义）

### 1.1 `trainable_groups` 到参数映射（joint/refine 实际含义）

在 `3_experiments/scripts/train.py::_param_group_from_name` + `_set_trainable_groups` 中，分组是硬编码映射：

- `routing`: `mu`, `log_scale`
- `basis`: `U`, `U_l0`
- `coeff`: `coeffs`, `coeffs_l0`
- `encoder`: `light_encoder.*`, `bypass_proj.*`, `bypass_mlp.*`
- `film`: `gamma.*`, `beta.*`

术语落地：

- “joint” = `trainable_groups=['all']`，所有参数 `requires_grad=True`
- “refine 某部分” = 仅对应分组参与优化，其他参数冻结

### 1.2 `warmup_joint_fast` / `joint_refine` 到底是什么

它们只是 stage 名字，不是算法名称。真实行为由以下配置决定：

- `staged.stages[*].trainable_groups`
- `staged.stages[*].group_lrs`
- `staged.stages[*].best_metric`
- `staged.stages[*].epochs/warmup_epochs/lr_scheduler`

以 A1.3/A1.4 为例：

- `warmup_joint_fast`
- 可训练组：`coeff+encoder+film+routing`
- 目的：先把 `z->time_weights`、路由权重和 FiLM 调制对齐起来
- `joint_refine` / `stage2b_joint_refine_lowlr`
- 可训练组：`all`（包括 `basis`）
- 目的：在更低学习率下做全量联合微调
- `stage2a_anchor_refine`（A1.4）
- 可训练组：`coeff+encoder+film`
- 明确冻结 `routing+basis`，防止 stage2 直接破坏 stage1 的空间锚点

### 1.3 stage 切换时优化器状态怎么处理（常被误解）

实现点：`train.py` 在每个 stage 都重新 `_build_trainer(...)`，因此会新建 optimizer/scheduler。

- 正常跨 stage（不 resume）：
- optimizer/scheduler 重新初始化，不继承上一 stage 的动量状态
- “refine”主要通过更低 LR + 不同可训练组实现，不是靠沿用动量
- strict `resume` 同一 stage 继续训练：
- `trainer.fit(..., resume_state=...)` 会恢复 model/optimizer/scheduler/EMA/RNG/history
- `load_model`：
- 仅加载模型权重，不恢复 optimizer/scheduler/EMA/epoch/stage 进度

### 1.4 “训练路由”具体是训练哪部分

路由在 `2_src/models/gaussian_physics_unified.py::compute_gaussian_routing`，训练对象是 `mu` 与 `log_scale`。

公式（对每个样本位置 `p` 和专家 `j`）：

- `dist_j^2 = ||p - mu_j||^2`
- `exponent_j = -0.5 * ||(p - mu_j) / exp(log_scale_j)||^2`

Hard 路由（默认推理口径）：

- 先选最近 top-`k` 专家（离散索引）
- 再在选中专家内做 `exp(exponent)` 归一化

Soft 路由（训练可选）：

- 先选最近 top-`soft_topk`
- `weights = softmax(exponent / temperature)`

因此“训练路由”在工程上是：

- 开启 `routing` 参数组参与反向
- 并可选使用 `routing_soft_train=true` 让路由权重在 top-`soft_topk` 上可微混合

### 1.5 `proxy image loss` 的实现细节（谁和谁做差）

实现点：`2_src/training/gaussian_physics_trainer.py::_compute_image_loss_components`。

输入是预测 SH (`pred_sh`) 与真值 SH (`target_sh`)，不是 Falcor 渲染图像本身。过程：

1. 用固定球面采样方向（`image_samples`, 默认 64）构建 SH 基
2. `pred_sh/target_sh` 分别重建成“伪图像采样值” `pred_img/target_img`
3. 在采样图像上计算损失：
- `linear`: `loss(pred_img, target_img)`
- `log`: `loss(log1p(clamp(pred_img, min=0)), log1p(clamp(target_img, min=0)))`
- `linear_log_mix`: `(1-w)*linear + w*log`
4. 乘以 `lambda_image(epoch)`，其中 warmup 线性增长：
- `lambda_image(epoch)=target_lambda*min(1, epoch / warmup_epochs)`

线性分支口径：

- 旧实现：`clamp(0,1)`（上界截断）
- 当前实现：`clamp(min=0)`（只截负值）

### 1.6 `img_psnr` 的计算与选模链路（容易混淆）

`img_psnr` 由 `_compute_image_metrics` 计算，基于上面 proxy 伪图像而非 Falcor：

- `mse = mean((pred_img-target_img)^2)`
- `img_psnr = 10 * log10(1 / mse)`（动态范围固定按 1.0）

三套 best 并行存在：

1. Stage 主 best：`best_model.pt`
- 由 `trainer.fit(..., best_metric=stage.best_metric)` 决定
- 若 stage 设 `img_psnr`，是“默认验证口径”的 `val.img_psnr`（默认 hard3）
- 当 `ema.eval_on_ema=true` 时，该 `val` 指标来自 EMA 权重前向（raw 权重仅按 `raw_eval_every_epochs` 抽样记录）

2. Profile best：`best_model_val_<profile>.pt`
- 由 `training.val_profiles.best_metric` 决定
- 常见：`best_model_val_soft8_t018.pt`

3. Global best（跨 stage）：
- `global_best_val_soft8_t018.pt`（soft profile 指标）
- `global_best_falcor.pt`（Falcor 周期评估指标）

### 1.7 `routing balance` 与退火实现

实现点：`_compute_routing_balance_loss` + `_routing_balance_weight_for_epoch`。

计算步骤：

1. 把当前 batch 的 `(weights, indices)` 展平
2. 用 `scatter_add` 按“全局专家 ID”聚合 usage（不是按 top-k 槽位）
3. `usage = usage_sum / sum(usage_sum)`
4. `balance_loss = var(usage) / (mean(usage)^2 + eps)`

退火权重：

- `lambda_balance(epoch_global)=start + progress*(end-start)`
- `progress=(global_epoch-1)/anneal_epochs`，`global_epoch` 包含 stage offset，不会在 stage 切换时重置

### 1.8 `bypass_*` 各模式具体改了什么

实现点：`2_src/models/gaussian_physics_unified.py::_encode_light_set`。

共性：

- 从 `light_params` 中按 `bypass_feature_pairs` 抽取特征（当前实验是 `[0,5]` 和 `[1,6]`）
- 用配置里的 `bypass_feature_norm_mean/std` 做标准化
- 用该特征替代原 `LightSetEncoder` 的输入编码路径

差异：

- `normal`: 原编码器 `LightSetEncoder`（含 intensity decoupling + pooling）
- `bypass_fixed`: 标准化后 2 维特征直接 repeat 到 32 维，无新增可学习层
- `bypass_linear`: 新增 `Linear(2,32)` 作为可学习投影
- `bypass_mlp_16_32`: 新增 `Linear(2,16)->ReLU->Linear(16,32)`

### 1.9 proxyfix 到底做了什么

`A1.5-proxyfix` 不是“随便微调”，而是两件事同时发生：

- 代码层：proxy 线性分支从 `clamp(0,1)` 改成 `clamp(min=0)`
- 训练层：用 `training.load_model` 从 A1.4 权重初始化，跑单 stage `proxyfix_finetune`（`trainable_groups=['all']`，`best_metric=img_psnr`）

即它测试的是“新 proxy 数值口径 + 新训练日程”组合，不是 strict resume 的同轨续跑。

---

## 2. 时间线（按时间顺序，逐实验写清变更）

## 2.1 Baseline（`exp_A_K30_r8_baseline`，2026-02-27~02-28）

- 相对上一个：起点。
- 技术配置（核心）：
  - `light_encoder_mode=normal`
  - 单阶段 `epochs=2000`
  - 无 `staged`、无 `ema`、无 `proxy_image_loss`、无 `val_profiles`、无 `falcor_periodic_eval`
  - 路由为默认 hard top-3（未启用 `routing_soft_train`）
- 相对 baseline 累积变化：N/A
- 结果：
  - Hard HDR PSNR = **30.9159**
  - Soft t0.18 HDR PSNR = **30.8206**
- 来源：
  - `3_experiments/results/full_suite_best_last_baseline/full_suite_20260309_210325/suite_summary.json`
  - `3_experiments/results/full_suite_best_last_baseline_soft_t018/full_suite_20260309_221949/suite_summary.json`

## 2.2 A1（coeff-first，`exp_A1_coeff_first_20260302_202419`，2026-03-02）

- 相对 Baseline 改动：
  - 引入 `staged` 两阶段（`staged_training_summary.json` 可核对）：
  - stage1 `coeff`：
  - `epochs=1200`, `lr=1e-3`, `lr_scheduler=none`
  - `trainable_groups=['coeff','encoder']`
  - trainable 参数量 `22240 / 34884`
  - `best_metric=mae`
  - stage2 `joint`：
  - `epochs=800`, `lr=5e-4`, `lr_scheduler=none`
  - `trainable_groups=['all']`
  - trainable 参数量 `34884 / 34884`
  - `best_metric=mae`
  - 仍无 soft routing / proxy / EMA / val_profiles / falcor periodic
- 相对 baseline 累积变化：
  - 首次把 “只训 coeff+encoder -> 全量联合” 引入主线
- 结果：
  - Hard HDR PSNR = **25.2204**（vs baseline hard: -5.6956 dB）
  - Soft t0.18 HDR PSNR = **30.8200**（vs baseline soft: 基本持平）
- 来源：
  - `full_suite_20260309_210325/suite_summary.json`
  - `full_suite_20260309_221949/suite_summary.json`

## 2.3 旧A1.1（`exp_A1_1_joint_lrs_ema` 早期阶段）

- 相对 A1 改动：
  - staged 训练范式改为 `warmup_joint_fast -> joint_refine`：
  - stage1（warmup）只训 `coeff+encoder+film+routing`
  - stage2（joint_refine）放开 `all`
  - 该阶段可观测配置（同目录日志 + 同时期配置）：
  - stage1 group_lrs：`coeff/encoder/film=8e-4`, `routing=2e-4`, `all=1e-4`
  - stage2 group_lrs：`coeff/encoder/film=4e-4`, `basis=5e-5`, `routing=1e-5`, `all=1e-5`
  - 启用 soft 训练路由：
  - `routing_soft_train=true`
  - `routing_soft_topk=8`
  - `routing_temp: 1.0 -> 0.2`（`anneal_epochs=320`）
  - 启用 `lambda_routing_balance=0.02`
  - 启用 EMA：`decay=0.999`, `eval_on_ema=true`, `save_best_with_ema=true`
  - stage 主选模仍为 `best_metric=mae`
- 关键澄清（来自该目录 train.log 启动日志）：
  - 有 `[ema]`、`[routing]`、`[staged]`
  - 无 `[proxy-image-loss]`、`[val-profiles]`、`[falcor-periodic]`、`[global-best]`
  - 即旧A1.1高点阶段并未走后来的 proxy+多profile+falcor 周期选模链路
- 相对 baseline 累积变化：
  - soft 路由 + EMA + 分组 LR 的旧主线成型
- 结果（旧A1.1历史口径）：
  - Soft t0.18 HDR PSNR = **33.3438**
  - Hard HDR PSNR = **21.9068**
- 来源：
  - `full_suite_20260309_221949/suite_summary.json`
  - `full_suite_20260309_210325/suite_summary.json`

## 2.4 新A1.1（后更名 A1.2，同目录延续产物）

- 命名说明：
  - 你已确认这一段应叫 **A1.2**，不是“新A1.1”。
- 相对旧A1.1 改动（按你定义 + 目录证据）：
  - 打开 `training.proxy_image_loss`：
  - `lambda=0.02`, `warmup_epochs=120`
  - `mix_mode=linear_log_mix`, `log_mix_weight=0.3`
  - 打开 `training.val_profiles`：
  - profiles=`hard3 / soft8_t018 / soft8_t020 / soft8_t022`
  - profile best 指标=`img_psnr`
  - 打开 `training.falcor_periodic_eval`：
  - `every_n_epochs=50`, `profile=soft8_t018`
  - `best_metric=mean_real_render_hdr_psnr`, `maximize=true`
  - 产出 profile-best 与 falcor-best 链路 checkpoint（含 `best_model_val_soft8_t018_falcor.pt`）
- 相对 baseline 累积变化：
  - 目标函数、验证与选模从“SH误差主导”转到“proxy图像 + 多口径 + Falcor 周期评估”
- 结果（recompute 时点）：
  - Soft HDR PSNR = **22.9663**（另有 stage 内记录约 23.0712）
- 来源：
  - `3_experiments/results/full_suite_a11_resume_vs_baseline_soft018_recompute/full_suite_20260311_224321/suite_summary.json`
  - 同目录的 `falcor_best_summary.json` / profile best 文件

## 2.5 A1.3（`exp_A1_3_proxylog_balance_anneal`，2026-03-13~03-14）

- 相对 A1.2 改动：
  - 在新目录重跑，避免 A1.1 目录复用污染
  - 两阶段明确化（均为 cosine + warmup）：
  - stage1 `warmup_joint_fast`：
  - `epochs=800`, `trainable_groups=['coeff','encoder','film','routing']`
  - group_lrs：`coeff/encoder/film=8e-4`, `routing=2e-4`, `all=1e-4`
  - `best_metric=mae`
  - stage2 `joint_refine`：
  - `epochs=1200`, `trainable_groups=['all']`
  - group_lrs：`coeff/encoder/film=4e-4`, `basis=5e-5`, `routing=1e-5`, `all=1e-5`
  - `best_metric=mae`
  - `routing_balance_anneal`: `start=0.02 -> end=0.0`（到全局 60% epoch 截止）
  - `proxy_image_loss` 继续启用（linear/log mix, `lambda=0.02`）
  - 注意：stage 主 best 仍是 `mae`；profile best 与 falcor best 另存，不覆盖 stage 主 best 逻辑
- 相对 baseline 累积变化：
  - 完整进入“soft+EMA+proxy+val_profiles+falcor periodic”路线
- 结果（训练内 Falcor best）：
  - HDR PSNR = **27.8670**
- 来源：
  - `3_experiments/results/bistro_clean_v2/exp_A1_3_proxylog_balance_anneal/falcor_best_summary.json`

## 2.6 A1.4（`exp_A1_4_anchor_globalbest`，2026-03-15）

- 相对 A1.3 改动：
  - `training.resume` 严格续训自 A1.3 的 `best_model_val_soft8_t018_falcor.pt`
    - 续训恢复模型/优化器/scheduler/EMA/epoch-stage 状态
  - 三阶段重构：
  - stage1 `warmup_joint_fast`：
  - `epochs=800`, groups=`coeff+encoder+film+routing`
  - group_lrs：`coeff/encoder/film=8e-4`, `routing=2e-4`, `all=1e-4`
  - `best_metric=img_psnr`
  - stage2a `stage2a_anchor_refine`：
  - `epochs=600`, groups=`coeff+encoder+film`
  - group_lrs：`coeff/encoder/film=3e-4`, `all=1e-5`
  - `best_metric=img_psnr`
  - stage2b `stage2b_joint_refine_lowlr`：
  - `epochs=600`, groups=`all`
  - group_lrs：`coeff/encoder/film=2e-4`, `basis=2e-5`, `routing=5e-6`, `all=5e-6`
  - `best_metric=img_psnr`
  - 启用 `training.global_best`（跨 stage 保护 soft best 与 falcor best）
  - `global_best_val_soft8_t018.pt` 与 `global_best_falcor.pt` 从此跨 stage 维护
  - `routing_balance_anneal.end` 从 0 调到 **0.005**（保底，不归零）
- 相对 baseline 累积变化：
  - 在 A1.3 路线加入“锚点阶段 + 全局最优保护 + 保底均衡”
- 结果（训练内 Falcor best）：
  - HDR PSNR = **27.3836**（比 A1.3 低 0.4835 dB）
- 来源：
  - `exp_A1_4_anchor_globalbest/falcor_best_summary.json`
  - `exp_A1_4_anchor_globalbest/global_best_falcor_summary.json`

## 2.7 A1.5-proxyfix（`exp_A1_5_proxyfix`，2026-03-19~03-20）

- 相对 A1.4 改动：
  - 明确不是 strict resume；而是 `training.load_model` 从 A1.4 的 `global_best_val_soft8_t018.pt` 只载入权重后重新训练
  - 不恢复 A1.4 的 optimizer/scheduler/EMA/history/stage 进度
  - staged 改为单阶段：
  - `proxyfix_finetune`（`epochs=200`, groups=`all`, `best=img_psnr`）
  - group_lrs：`coeff/encoder/film=2e-4`, `basis=2e-5`, `routing=5e-6`, `all=1e-4`
  - proxy 数值路径代码变更（核心）：
  - 历史版本 linear 分支：`clamp(0,1)`
  - 当前版本 linear 分支：`clamp(min=0)`（去掉上限截断）
  - log 分支保持 `log1p(clamp(min=0))`
- 相对 baseline 累积变化：
  - 保留 A1.4 的大多数机制，仅做 proxy 数值与训练形态调整
- 结果（训练内 Falcor best）：
  - HDR PSNR = **21.7116**（大幅下降）
- 来源：
  - `exp_A1_5_proxyfix/falcor_best_summary.json`

## 2.8 A1.5-bypassA（`exp_A1_5_bypassA`）

- 相对 A1.5-proxyfix 改动：
  - `light_encoder_mode: normal -> bypass_fixed`
  - `bypass_feature_pairs=[[0,5],[1,6]]`，并使用固定标准化：
  - `mean=[-0.8408854604, 0.0039021266]`
  - `std=[0.0843038857, 0.4981141388]`
  - 单 stage：`stage1_bypass_fixed`, `epochs=400`, `best_metric=img_psnr`
  - `trainable_groups=['coeff','encoder','film','routing']`（冻结 basis）
  - group_lrs：`coeff/encoder/film=8e-4`, `routing=2e-4`, `all=1e-4`
- 相对 baseline 累积变化：
  - 进入“绕过原 light encoder 信息瓶颈”的实验线
- 结果：
  - HDR PSNR = **29.5016**
- 来源：
  - `exp_A1_5_bypassA/falcor_best_summary.json`

## 2.9 A1.5-bypassB（`exp_A1_5_bypassB`）

- 相对 A1.5-bypassA 改动：
  - `light_encoder_mode: bypass_fixed -> bypass_linear`
  - 即从“固定复制特征”改为“可学习线性投影 2->32（新增 `bypass_proj`）”
- 其余训练策略保持（same stage/group_lrs/proxy/soft routing/EMA/val_profiles/falcor periodic）
- 结果：
  - HDR PSNR = **34.3938**
- 来源：
  - `exp_A1_5_bypassB/falcor_best_summary.json`

## 2.10 A1.5-bypassB_noproxy（`exp_A1_5_bypassB_noproxy`）

- 相对 A1.5-bypassB 改动：
  - 唯一改动：`proxy_image_loss.enabled=false`
  - 注意：这只关掉 proxy 损失项，不会改变核心架构与路由策略
- 结果：
  - HDR PSNR = **31.8811**
- 来源：
  - `exp_A1_5_bypassB_noproxy/falcor_best_summary.json`

## 2.11 A1.6 P0（BypassB 稳定性复现）

- 相对 A1.5-bypassB 改动：
  - 仅改随机种子（7/19/73）
  - 同一模型与训练日程：`bypass_linear` + 单 stage 400 epoch + groups=`coeff+encoder+film+routing`
  - 同一 proxy/routing/EMA/val_profiles/falcor 配置
- 结果：
  - seed7 = **34.5215**
  - seed19 = **35.0076**
  - seed73 = **34.1041**
- 来源：
  - `exp_A1_6_P0_bypassB_seed7/falcor_best_summary.json`
  - `exp_A1_6_P0_bypassB_seed19/falcor_best_summary.json`
  - `exp_A1_6_P0_bypassB_seed73/falcor_best_summary.json`

## 2.12 A1.6 P1（proxy λ 扫描）

- 相对 P0 改动：
  - 固定 `bypass_linear` 与同一训练日程，仅扫描 `proxy_image_loss.lambda`
  - 扫描值：`0.00 / 0.01 / 0.02`
  - `mix_mode=linear_log_mix` 与 `log_mix_weight=0.3` 保持不变
- 结果：
  - λ=0.00: **33.6670**
  - λ=0.01: **34.4288**
  - λ=0.02: **34.8678**（当前 rerun）
  - 同目录历史峰值（global_best_falcor）: **35.2238**
- 来源：
  - `exp_A1_6_P1_bypassB_proxy_l000/falcor_best_summary.json`
  - `exp_A1_6_P1_bypassB_proxy_l001/falcor_best_summary.json`
  - `exp_A1_6_P1_bypassB_proxy_l002/falcor_best_summary.json`
  - `exp_A1_6_P1_bypassB_proxy_l002/global_best_falcor_summary.json`

## 2.13 A1.6 P2（encoder 模式对照）

- 相对 P1 改动：
  - 固定同一训练策略，只改 `light_encoder_mode`
  - 共同训练设置：
  - 单 stage 400 epoch（`stage1_enc_*`）
  - `trainable_groups=['coeff','encoder','film','routing']`（冻结 basis）
  - group_lrs：`coeff/encoder/film=8e-4`, `routing=2e-4`, `all=1e-4`
  - `best_metric=img_psnr`，soft 路由训练与 proxy/falcor 周期评估保持开启
  - 四个 encoder 变体：
  - `normal`（原 LightSetEncoder）
  - `bypass_fixed`（2维特征重复到32维）
  - `bypass_linear`（`Linear(2,32)`）
  - `bypass_mlp_16_32`（`2->16->32`）
  - 重要：A1.6 的 `normal` 不是 A1.4 复现
  - A1.6-normal 仍是“单 stage + groups=`coeff+encoder+film+routing`（冻结 basis）”
  - A1.4 是“三 stage + anchor + joint low-lr + strict resume”
- 结果：
  - normal = **33.7373**
  - bypass_fixed = **30.7549**
  - bypass_linear = **34.8626**
  - bypass_mlp_16_32 = **34.5837**
- 来源：
  - `exp_A1_6_P2_enc_normal/falcor_best_summary.json`
  - `exp_A1_6_P2_enc_bypass_fixed/falcor_best_summary.json`
  - `exp_A1_6_P2_enc_bypass_linear/falcor_best_summary.json`
  - `exp_A1_6_P2_enc_bypass_mlp1632/falcor_best_summary.json`

---

## 3. 同期关键代码改动（补充版）

1. 训练主流程与选模体系
   - `3_experiments/scripts/train.py`
   - 关键能力：`staged`、`val_profiles`、`global_best`、`falcor_periodic_eval`、strict `resume`、`load_model` 初始化加载
2. 训练器损失与路由统计
   - `2_src/training/gaussian_physics_trainer.py`
   - 关键能力：proxy image loss（linear/log mix）、routing balance scatter_add 统计、退火权重、EMA eval/save best
3. 模型编码分支
   - `2_src/models/gaussian_physics_unified.py`
   - 关键能力：`light_encoder_mode` 四分支、bypass 特征抽取与标准化、soft/hard routing 前向
4. 分析与复算脚本
   - `3_experiments/scripts/analysis/run_checkpoint_full_suite.py`
   - `3_experiments/scripts/analysis/run_expert_utilization_audit.py`
   - `3_experiments/scripts/analysis/run_semantic_drift_audit.py`
   - `3_experiments/scripts/analysis/run_suite_best_last_baseline.sh`
   - `3_experiments/scripts/analysis/wait_and_run_post_audits.py`

---

## 4. 当前结论（基于本时间线）

1. A1.3/A1.4 证明：复杂 staged 联合后期并不自动带来 HDR 提升；“有锚点 + 全局best保护”能防误覆盖，但不能保证超过阶段早期高点。
2. A1.5/A1.6 证明：瓶颈更集中在 light 编码链路；`bypass_linear` 在当前任务上稳定优于 `normal` 与 `bypass_fixed`。
3. proxy loss 在“编码链路有效”前提下是正收益（bypassB vs noproxy），但其数值实现（如 clamp 口径）会显著改变训练行为。
4. 目录复用（尤其 A1.1）会污染跨实验比较；最终结论应继续依赖“固定 checkpoint + full-suite 同口径复算”。

---

## 5. 核心证据索引

- Baseline/A1/旧A1.1（hard+soft）：
  - `3_experiments/results/full_suite_best_last_baseline/full_suite_20260309_210325/suite_summary.json`
  - `3_experiments/results/full_suite_best_last_baseline_soft_t018/full_suite_20260309_221949/suite_summary.json`
- 新A1.1(A1.2)复算：
  - `3_experiments/results/full_suite_a11_resume_vs_baseline_soft018_recompute/full_suite_20260311_224321/suite_summary.json`
- A1.3~A1.6 训练内 Falcor best：
  - `3_experiments/results/bistro_clean_v2/*/falcor_best_summary.json`
  - `3_experiments/results/bistro_clean_v2/*/global_best_falcor_summary.json`
- staged 配置快照：
  - `3_experiments/results/bistro_clean_v2/*/staged_training_summary.json`
- 旧A1.1日志证据（是否启用 proxy/val_profiles/falcor/global_best）：
  - `3_experiments/results/bistro_clean_v2/exp_A1_1_joint_lrs_ema/train.log`
