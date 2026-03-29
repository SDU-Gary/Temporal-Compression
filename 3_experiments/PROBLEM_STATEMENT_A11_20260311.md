# PG-GCPL A1.1/A1.2 综合问题陈述（截至 2026-03-11）

## 0. 命名更正（重要）

本文件采用以下命名口径（修正此前混用）：

- **旧 A1.1**：分阶段训练 + soft 口径 + EMA（你之前确认“描述没错”的版本）
- **新 A1.2**：在旧 A1.1 基础上，进一步引入  
  `proxy image loss + img_psnr 选模 + Falcor 低频周期评估 + soft_t018_falcor best`

因此，文中所有“新 A1.1”均应理解为 **A1.2**。  
另外，性能退化的归因应描述为“训练目标与选模策略改动后的新实验分支结果”，**不是**“断点续训本身导致退化”。

## 1. 面向外部接手者的最短背景

我们在做一个**可重光照 SH 场压缩**系统（PG-GCPL），核心目标是：

- 高质量：HDR 画质指标（尤其 `mean_hdr_psnr`）尽量高
- 高压缩：维持 `U * coeff` 低秩可解释结构
- 可部署：实时推理链路可运行

核心模型形式（单样本）：

1. 空间路由：`(weights, topk_indices) = routing(p)`
2. 光照编码：`z = LightSetEncoder(light_params, light_mask)`
3. 局部低秩重建：`time_weights = coeffs[topk] @ z`，`SH = sum_k weights_k * (U[topk] @ time_weights_k)`
4. FiLM：`U` 分支在 rank 维上受 `gamma(z), beta(z)` 调制

代码入口：

- 模型：[2_src/models/gaussian_physics_unified.py](/home/kyrie/毕设/2_src/models/gaussian_physics_unified.py)
- 训练器：[2_src/training/gaussian_physics_trainer.py](/home/kyrie/毕设/2_src/training/gaussian_physics_trainer.py)
- 训练主脚本：[3_experiments/scripts/train.py](/home/kyrie/毕设/3_experiments/scripts/train.py)

---

## 2. 我们做过哪些关键代码改动（不是“只调参数”）

### 2.1 训练/评估能力扩展

1. **分阶段训练 + 分组学习率 + EMA**
   - 支持 `staged.stages[*].trainable_groups + group_lrs`
   - 支持 `ema.eval_on_ema / save_best_with_ema`
   - 位置：
     - [3_experiments/configs/bistro_clean_train_A1_1.yaml](/home/kyrie/毕设/3_experiments/configs/bistro_clean_train_A1_1.yaml)
     - [2_src/training/gaussian_physics_trainer.py](/home/kyrie/毕设/2_src/training/gaussian_physics_trainer.py)

2. **多 profile 验证与 best 保存**
   - 每个 epoch 在 `hard3` + `soft8@{0.18,0.20,0.22}` 上同时验证，分别保存 `best_model_val_*.pt`
   - 位置：
     - [3_experiments/configs/bistro_clean_train_A1_1.yaml](/home/kyrie/毕设/3_experiments/configs/bistro_clean_train_A1_1.yaml)
     - [2_src/training/gaussian_physics_trainer.py](/home/kyrie/毕设/2_src/training/gaussian_physics_trainer.py)
     - [3_experiments/scripts/train.py](/home/kyrie/毕设/3_experiments/scripts/train.py)

3. **训练期周期 Falcor 评估（低频）**
   - 在训练中每 N epoch 跑真实渲染指标，支持 `stage_end_full` 和 best 记录
   - 位置：
     - [3_experiments/configs/bistro_clean_train_A1_1.yaml](/home/kyrie/毕设/3_experiments/configs/bistro_clean_train_A1_1.yaml)
     - [3_experiments/scripts/train.py](/home/kyrie/毕设/3_experiments/scripts/train.py)
     - [3_experiments/results/bistro_clean_v2/exp_A1_1_joint_lrs_ema/falcor_best_summary.json](/home/kyrie/毕设/3_experiments/results/bistro_clean_v2/exp_A1_1_joint_lrs_ema/falcor_best_summary.json)

4. **可微 proxy image loss（SH->采样方向图像）**
   - 在训练中引入 SH 渲染代理损失，目标是向图像域靠拢（不替代 Falcor）
   - 位置：
     - [2_src/training/gaussian_physics_trainer.py](/home/kyrie/毕设/2_src/training/gaussian_physics_trainer.py)
     - [3_experiments/configs/bistro_clean_train_A1_1.yaml](/home/kyrie/毕设/3_experiments/configs/bistro_clean_train_A1_1.yaml)

### 2.2 路由相关改动

1. **训练软路由支持**
   - `training_soft_routing`, `routing_soft_topk`, `routing_temperature`
   - 位置：[2_src/models/gaussian_physics_unified.py](/home/kyrie/毕设/2_src/models/gaussian_physics_unified.py)

2. **路由温度退火**
   - `routing_temp_start -> routing_temp_end`，按 epoch 线性退火
   - 位置：[2_src/training/gaussian_physics_trainer.py](/home/kyrie/毕设/2_src/training/gaussian_physics_trainer.py)

3. **路由负载均衡损失修正**
   - 从“槽位统计”修为“按 expert 索引聚合”，使用 `scatter_add` 到全局 expert
   - 位置：[2_src/training/gaussian_physics_trainer.py](/home/kyrie/毕设/2_src/training/gaussian_physics_trainer.py)

### 2.3 诊断链路改动

1. **Oracle 诊断 + coeff 可学习性套件**
   - Oracle 最优系数上界、per-band、rmsnorm、probe 回归（linear/MLP/structured）
   - 位置：
     - [3_experiments/scripts/analysis/run_sh_oracle_diagnostics.py](/home/kyrie/毕设/3_experiments/scripts/analysis/run_sh_oracle_diagnostics.py)
     - [3_experiments/scripts/analysis/run_coeff_learnability_suite.py](/home/kyrie/毕设/3_experiments/scripts/analysis/run_coeff_learnability_suite.py)

2. **全套并行评估脚本 full-suite**
   - benchmark + expert utilization + semantic drift 一次完成
   - 支持 benchmark 软路由参数（避免评估口径错配）
   - 位置：[3_experiments/scripts/analysis/run_checkpoint_full_suite.py](/home/kyrie/毕设/3_experiments/scripts/analysis/run_checkpoint_full_suite.py)

3. **新增两类独立审计**
   - 专家利用率：全局引用计数/质量分布/覆盖率
   - 语义漂移：参数漂移 + 功能漂移 + 补偿指数
   - 位置：
     - [3_experiments/scripts/analysis/run_expert_utilization_audit.py](/home/kyrie/毕设/3_experiments/scripts/analysis/run_expert_utilization_audit.py)
     - [3_experiments/scripts/analysis/run_semantic_drift_audit.py](/home/kyrie/毕设/3_experiments/scripts/analysis/run_semantic_drift_audit.py)

### 2.4 可靠性改动（训练卡死与续训）

1. 严格 resume：恢复 `optimizer/scheduler/EMA/history/stage进度/RNG`
2. heartbeat + history + watchdog
3. DataLoader timeout 可配置、Oracle/coeff suite timeout 可配置
4. 一键 tmux + watchdog 脚本

位置：

- [3_experiments/scripts/train.py](/home/kyrie/毕设/3_experiments/scripts/train.py)
- [tools/watchdog_train_heartbeat.py](/home/kyrie/毕设/tools/watchdog_train_heartbeat.py)
- [3_experiments/scripts/run_tmux_train_watchdog.sh](/home/kyrie/毕设/3_experiments/scripts/run_tmux_train_watchdog.sh)

---

## 3. 实验演化路径（时间线）

## 3.1 Baseline 阶段：先证明瓶颈在哪里（Oracle）

证据文件：

- [baseline Oracle summary](/home/kyrie/毕设/3_experiments/results/bistro_clean_v2/exp_A_K30_r8_baseline/oracle_diag_cpu_verify_fix/summary.json)

关键结论：

- baseline SH-PSNR(all27): `11.7869`
- Oracle（固定 U/路由后最优 time_weights）: `23.0692`
- 可提升空间：`+11.2823 dB`
- 诊断标签：`coeff-limited`

解释：在“U 和路由固定”的前提下，真实上界很高，主要瓶颈在 `z -> coeff/time_weights` 链路。

---

## 3.2 A1（Coeff-first）阶段：先训 coeff 再 joint

配置与结果：

- 配置：[bistro_clean_train_A1_coeff_first.yaml](/home/kyrie/毕设/3_experiments/configs/bistro_clean_train_A1_coeff_first.yaml)
- 报告：[a1_vs_baseline_report.json](/home/kyrie/毕设/3_experiments/results/bistro_clean_v2/exp_A1_coeff_first_20260302_202419/a1_vs_baseline_report.json)
- Oracle 诊断：[A1 oracle summary](/home/kyrie/毕设/3_experiments/results/bistro_clean_v2/exp_A1_coeff_first_20260302_202419/oracle_diag/summary.json)

观察：

- A1 自身 Oracle gap 仍大，且比 baseline 更大（all27 gap 从 `11.28` 到 `15.31` dB）
- 图像域也退化（HDR PSNR 约 `-5.70 dB` vs baseline）

结论：简单的 coeff-first 分段并没有自动带来“更接近 Oracle + 更好渲染”。

---

## 3.3 旧 A1.1（联合+分组LR+EMA+软路由）阶段

配置：

- [bistro_clean_train_A1_1.yaml](/home/kyrie/毕设/3_experiments/configs/bistro_clean_train_A1_1.yaml)

核心策略：

1. stage1: `coeff+encoder+film+routing` 高 LR 快速对齐  
2. stage2: 全量 joint，`basis/routing` 低 LR 精修  
3. soft routing 训练（top8 + temperature anneal）  
4. 多 profile 验证 + best_by_profile  
5. periodic Falcor eval 参与监控

## 3.4 A1.2（新分支：loss/选模策略增强）阶段

这是你后续明确指定的新实验分支（此前文档误写为“新A1.1”）：

1. 打开 `proxy_image_loss`（可微 SH->图像代理）
2. 以 `img_psnr` 作为 val profile 选模主指标
3. 开启 Falcor 低频周期评估，并记录/导出 Falcor best
4. 重点跟踪 `soft_t018` profile（包含 `best_model_val_soft8_t018` 及 Falcor best 线索）

说明：A1.2 与旧 A1.1 共享大量基础机制（staged/EMA/soft routing），但**优化目标与选模路径不同**，应视为独立分支。

---

## 4. 关键结果（最重要数字）

### 4.1 旧 A1.1（2026-03-09）在 soft 口径下曾达到高点

来源：

- [soft_t018 old suite](/home/kyrie/毕设/3_experiments/results/full_suite_best_last_baseline_soft_t018/full_suite_20260309_221949/suite_summary.json)

关键数字（soft t=0.18）：

- baseline_best: HDR `30.8206`
- A1.1_best(old): HDR `33.3438`（高于 baseline）
- A1.1_last(old): HDR `26.0920`，但 SH `11.2283`

额外说明：这说明当时确实存在“图像域明显变好”的 A1.1 checkpoint。

### 4.2 A1.2（2026-03-11 复算）相对旧 A1.1 明显退化

来源：

- [soft_t018 recompute suite](/home/kyrie/毕设/3_experiments/results/full_suite_a11_resume_vs_baseline_soft018_recompute/full_suite_20260311_224321/suite_summary.json)
- [compare report](/home/kyrie/毕设/3_experiments/results/full_suite_a11_resume_vs_baseline_soft018_recompute/full_suite_20260311_224321/compare_report_vs_baseline_and_olda11.json)

关键数字（soft t=0.18）：

- baseline_best: HDR `30.8206`
- A1.2 best（stage02_soft018）: HDR `23.0712`，Benchmark `17.4968`，SH `11.2360`
- A1.2 last: HDR `22.9369`，SH `11.1978`

即：

- 相对 baseline：HDR 约 `-7.75 dB`，但 SH 约 `+6.95 dB`
- 相对旧 A1.1 best：HDR 约 `-10.27 dB`

归因表述修正：

- 该退化对应的是 **A1.2 新策略分支**（loss/选模改动后）的结果
- 不能表述为“resume 操作导致退化”

### 4.3 硬/软口径敏感性（路由口径极其关键）

来源：

- [hard suite](/home/kyrie/毕设/3_experiments/results/full_suite_best_last_baseline_hard/full_suite_20260309_220438/suite_summary.json)
- [soft_t018 suite](/home/kyrie/毕设/3_experiments/results/full_suite_best_last_baseline_soft_t018/full_suite_20260309_221949/suite_summary.json)
- [soft_t020 suite](/home/kyrie/毕设/3_experiments/results/full_suite_best_last_baseline_soft_t020/full_suite_20260309_221311/suite_summary.json)
- [soft_t022 suite](/home/kyrie/毕设/3_experiments/results/full_suite_best_last_baseline_soft_t022/full_suite_20260309_223025/suite_summary.json)

观察：

- A1 系列（尤其旧 A1.1 与 A1.2）在 hard 口径显著差（甚至崩）
- A1 系列在 soft 口径下可明显恢复，且 t=0.18 最优

结论：训练与评估 route profile 若不对齐，会导致“模型选择与真实性能错位”。

---

## 5. 已完成的审计与证据

## 5.1 专家利用率审计（全局计数）

来源（A1.2 soft018 suite 里每 case 的 `expert.*`）：

- [new soft018 suite](/home/kyrie/毕设/3_experiments/results/full_suite_a11_resume_vs_baseline_soft018_recompute/full_suite_20260311_224321/suite_summary.json)

典型值：

- baseline 在 `soft8_t020` 下 `effective_k ≈ 1.00`（几乎单专家）
- A1.2 在 `soft8_t020` 下 `effective_k ≈ 8.99`（专家使用更分散）

解释：A1.2 路由确实“更均匀/更分散”，但这并未自动转化为更高 HDR 质量。

## 5.2 语义漂移审计

同来源：

- `drift.global_rel_l2_to_a ≈ 0.38`（参数变化明显）
- `drift.hard3.b_minus_a_psnr_vs_gt < 0`（hard 口径明显变差）
- `drift.soft8_t020.b_minus_a_psnr_vs_gt > 0`（soft 口径可补偿）

解释：存在“参数漂移 + 口径相关补偿”现象，模型行为强依赖路由 profile。

## 5.3 训练期 Phase0 监控（最近一次）

来源：

- [phase0 metrics jsonl](/home/kyrie/毕设/3_experiments/results/bistro_clean_v2/exp_A1_1_joint_lrs_ema/runtime/phase0_metrics.jsonl)
- [heartbeat](/home/kyrie/毕设/3_experiments/results/bistro_clean_v2/exp_A1_1_joint_lrs_ema/runtime/heartbeat.json)

末期（epoch 1200）观察：

- `val_profile_hard3_img_psnr ≈ 6.14`
- `val_profile_soft8_t018_img_psnr ≈ 15.06`
- `grad_post_routing / grad_post_total` 很高（约 `0.93~0.98`）

解释：优化过程中路由梯度主导明显，hard/soft 验证分裂也非常显著。

---

## 6. 当前“真正的问题”不是一句“模型退化”能概括

### 6.1 已被证实的事实

1. **Oracle 上界在 baseline 上很高**，最初确实是 coeff-limited。
2. **A1 / 旧A1.1 / A1.2 三条链路结果分化明显**：SH 指标提升并不必然带来图像域提升。
3. **路由口径（hard/soft + temperature）会改变结论方向**，且影响“best checkpoint”选择。
4. **A1.2 与旧 A1.1 的差异非常大**（旧 soft 最优 33.34，A1.2 soft 约 23.07）。

### 6.2 尚未被实锤的关键问题

1. 为什么旧 A1.1 的高 HDR 性能没有在新一轮复现？
2. A1.2 的 loss/选模组合是否在推动“SH 拟合最优”而不是“HDR 渲染最优”？
3. 路由均衡/软路由是否对 `U/coeff` 的协同学习产生了副作用（例如过平滑、语义重排）？
4. oracle_monitor 在部分阶段出现“极端高值”现象，是否存在诊断口径或数值路径问题（需单独校验）？

---

## 7. 外部分析者可直接接手的最小任务清单

1. **复核口径一致性**
   - 固定同一 checkpoint，系统扫 `hard3 / soft8@{0.18,0.20,0.22}`，确认 rank 与最优温度是否稳定。
   - 脚本：[run_checkpoint_full_suite.py](/home/kyrie/毕设/3_experiments/scripts/analysis/run_checkpoint_full_suite.py)

2. **复盘 loss 与目标错位**
   - 对比训练 loss 各项占比与 `Falcor periodic` 指标变化是否一致。
   - 数据：[phase0_metrics.jsonl](/home/kyrie/毕设/3_experiments/results/bistro_clean_v2/exp_A1_1_joint_lrs_ema/runtime/phase0_metrics.jsonl)

3. **检查路由均衡副作用**
   - 在相同训练框架下，仅关闭/减弱 `lambda_routing_balance` 做 AB。
   - 查看 expert_utilization 是否仍高、但 HDR 是否回升。

4. **复核旧 A1.1 与 A1.2 可复现性**
   - 旧高点证据在 suite 中，但 checkpoint 已被新训练覆盖；需要保留快照与配置锁定后重训验证。
5. **按新命名复盘 A1.2**
   - 将“proxy image loss + img_psnr选模 + Falcor周期评估”单列为 A1.2 分支，避免与旧 A1.1 混淆。

---

## 8. 一句话总结（给外部顾问）

我们已经从“工具链不完整”走到“证据足够完整”，现在的核心矛盾是：  
**A1.2（新策略分支）显著提升了 SH/coeff 相关指标与专家活跃度，但没有带来 HDR 渲染质量提升，且相对旧 A1.1 出现明显退化；性能结论还对 hard/soft 路由口径高度敏感。**  
接下来的工作重点不是再加新模块，而是先把“优化目标、路由口径、best选择逻辑”三者严格对齐并复现实证。
