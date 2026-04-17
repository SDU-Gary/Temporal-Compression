# A 系实验总整理（Baseline 至 A2.3.3，首轮结构化盘点）

更新时间：2026-04-16  
整理范围：当前仓库内**仍有记录可追溯**的 Baseline 参考实验、A1 系列、A2 系列，覆盖到 `A2.3.3`。  
整理目标：先把实验事实、证据来源、结果口径、时间线和关键数值整理清楚，为下一轮“整体分析”做统一底稿。

## 0. 使用说明

### 0.1 本文如何读

- 本文**先做整理，不做最终归因**。
- 目的不是立刻给出结论，而是把“实验到底做了什么、结果是什么、证据在哪里”统一口径。
- 对每个实验，尽量拆成四件事：
  - `目的`
  - `改动`
  - `结果`
  - `证据来源`

### 0.2 证据分层

- `[文档]`：已有正式文档直接说明，例如时间线、问题陈述、实验计划。
- `[配置]`：可由 YAML 配置直接验证。
- `[结果]`：可由 `falcor_best_summary.json`、`falcor_periodic_eval.jsonl`、`full-suite`、`benchmark_summary.json`、`oracle_diag` 等结果文件直接验证。
- `[推断]`：只能根据命名、配置差异、目录结构推断，后续如有需要应继续追 Git 历史或 train log。

### 0.3 结果口径必须先分开

| 口径 | 含义 | 适用说明 |
| --- | --- | --- |
| `periodic peak` | 训练期周期 Falcor 评估中的峰值 | 最能反映“曾达到过的最佳渲染峰值” |
| `periodic tail` | 最后一次周期 Falcor 评估 | 最能反映“训练末尾是否守住性能” |
| `stage_end_full` | stage 结束时的完整 Falcor 评估 | 与 periodic 是不同采样点，不能混用 |
| `full-suite` | 训练外统一复算 | 跨实验横向对比最可靠 |
| `benchmark_summary.image_metrics.*` | 单次统一评测的图像/SH 指标 | 常用于 baseline / coeff-first 正式参考 |

### 0.4 本文最重要的警告

1. `exp_A1_1_joint_lrs_ema` 目录**被复用过**。  
   03-09 的“旧 A1.1 高点”和 03-11 的“A1.2 新分支结果”不能只看这个目录当前状态，必须依赖 [docs/dev_notes/EXPERIMENT_TIMELINE_BASELINE_TO_A16_20260322.md](docs/dev_notes/EXPERIMENT_TIMELINE_BASELINE_TO_A16_20260322.md) 和相关 `full-suite` 快照。
2. A2 之后的大部分实验都同时存在 `periodic peak / stage_end_full / periodic tail` 三套结论。  
   如果不分口径，几乎一定会得出错误排序。
3. `A2.1_top8_balance` 的“top8 balance”改动**没有在保存下来的 YAML 差异中明确体现**。  
   从当前配置文件看，和 A2 基本只差输出目录与 stage 名字；如果当时存在代码级改动，需要后续追 Git 历史。

## 1. 核心数据源

### 1.1 正式说明文档

- [docs/dev_notes/EXPERIMENT_TIMELINE_BASELINE_TO_A16_20260322.md](docs/dev_notes/EXPERIMENT_TIMELINE_BASELINE_TO_A16_20260322.md)
- [3_experiments/PROBLEM_STATEMENT_A11_20260311.md](3_experiments/PROBLEM_STATEMENT_A11_20260311.md)
- [3_experiments/EXPERIMENT_PLAN_STAGED_TRAINING_AND_COEFF_OPT.md](3_experiments/EXPERIMENT_PLAN_STAGED_TRAINING_AND_COEFF_OPT.md)

### 1.2 本次整理使用的主表

- `docs/dev_notes/exp_a_master_20260416.tsv`
  - 由本次整理从现有结果目录汇总得到
  - 包含：启动时间、完成 epoch、best、stage_end、periodic peak、periodic tail、配置路径

### 1.3 关键结果目录

- `3_experiments/results/bistro_clean_v2/`
- `3_experiments/results/full_suite_*`
- `3_experiments/results/loss_falcor_correlation/`
- `3_experiments/results/loss_falcor_correlation_lag_partial/`

## 2. 总时间线总览

### 2.1 参考基线与早期历史分支

| 实验 | 时间 | 目的 | 核心改动 | 关键结果 | 证据 |
| --- | --- | --- | --- | --- | --- |
| Baseline (`exp_A_K30_r8_baseline`) | 2026-02-27~02-28 | 作为统一起点 | `normal` encoder + hard routing + 单阶段 joint；无 staged / EMA / proxy / periodic Falcor | hard HDR `30.9159`；soft t0.18 HDR `30.8206`；Oracle gap `+11.2823 dB`；诊断为 coeff-limited | [文档][结果] |
| A1 coeff-first (`exp_A1_coeff_first_20260302_202419`) | 2026-03-02 | 先验证 coeff-first 是否能缩小 Oracle gap | stage1 仅训 `coeff+encoder`，stage2 全量 joint | hard HDR `25.2204`；soft t0.18 HDR `30.8200`；Oracle gap 反而变为 `15.3067 dB`；相对 baseline HDR `-5.6956 dB` | [文档][结果] |
| 旧 A1.1（历史快照，不等于目录当前状态） | 2026-03-09 | 验证 soft routing + EMA + 分组 LR staged 主线 | `warmup_joint_fast -> joint_refine`；soft train routing；EMA；**无 proxy / 无 val_profiles / 无 periodic Falcor** | soft t0.18 HDR `33.3438`；hard HDR `21.9068` | [文档][结果] |
| A1.2（文档中“新A1.1”，后更名 A1.2） | 2026-03-11 | 将目标与选模链路转向 proxy 图像与 Falcor 周期评估 | 打开 proxy image loss；加入 val profiles；低频 periodic Falcor；soft018 口径选模 | soft t0.18 HDR `22.9663`；stage 内 best `23.0712`，较旧 A1.1 明显退化 | [文档][结果] |

### 2.2 A1.3 至 A1.6：文档化主干

| 实验 | 启动时间 | 目的 | 核心改动 | 训练内 best | stage_end_full | periodic tail | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1.3 `proxylog_balance_anneal` | 2026-03-13 | 在独立目录重建 A1.2 路线，避免 A1.1 目录污染 | 两阶段 staged；proxy 继续开；balance 退火到 0；Falcor periodic | `27.8670 @ 800` | `21.9367` | `21.1774` | [文档][结果] |
| A1.4 `anchor_globalbest` | 2026-03-15 | 验证锚点 refine + global best 保护 | strict resume；三阶段；anchor refine；global_best 打开；balance 保底 0.005 | `27.3836 @ 150` | `22.9104` | `22.6151` | [文档][结果] |
| A1.5 `proxyfix` | 2026-03-19 | 检查 proxy clamp 数值路径 | `clamp(0,1)` 改为 `clamp(min=0)`；load_model 后单阶段 finetune | `21.7116 @ 200` | `21.7116` | `21.2523` | 明显失败 [文档][结果] |
| A1.5 `bypassA` | 2026-03-19 | 绕过原 light encoder 的信息瓶颈 | `light_encoder_mode=bypass_fixed`；单阶段 400 epoch；冻结 basis | `29.5016 @ 250` | `27.9430` | `24.6192` | 首次明显恢复 [文档][结果] |
| A1.5 `bypassB` | 2026-03-19 | 验证可学习 bypass 投影是否优于固定复制 | `bypass_fixed -> bypass_linear` | `34.3938 @ 150` | `27.5817` | `23.8200` | 第一批高峰值突破 [文档][结果] |
| A1.5 `bypassB_noproxy` | 2026-03-20 | 验证 proxy 是否仍有正收益 | 在 bypassB 上关闭 proxy loss | `31.8811 @ 250` | `26.9880` | `22.8894` | 低于 bypassB，说明 proxy 仍有收益 [文档][结果] |
| A1.6 P0 seed7 | 2026-03-20 | 复现实验，测 seed 稳定性 | 与 bypassB 相同，只改 seed | `34.5215 @ 150` | `26.5145` | `23.7503` | [文档][结果] |
| A1.6 P0 seed19 | 2026-03-20 | 同上 | 同上 | `35.0076 @ 200` | `23.8184` | `23.8989` | 该阶段最高 periodic peak [文档][结果] |
| A1.6 P0 seed73 | 2026-03-20 | 同上 | 同上 | `34.1041 @ 150` | `27.0306` | `24.0158` | [文档][结果] |
| A1.6 P1 proxy λ=0.00 | 2026-03-20 | 扫描 proxy 权重 | 仅改 proxy `lambda=0.00` | `33.6670 @ 150` | `25.2330` | `23.8104` | [文档][结果] |
| A1.6 P1 proxy λ=0.01 | 2026-03-20 | 同上 | `lambda=0.01` | `34.4288 @ 150` | `27.3957` | `23.2149` | [文档][结果] |
| A1.6 P1 proxy λ=0.02 | 2026-03-22 | 同上 | `lambda=0.02` | `34.8678 @ 150` | `24.4009` | `23.8664` | `global_best_falcor` 历史峰值另有 `35.2238 @ 200`，需注明口径差异 [文档][结果] |
| A1.6 P2 normal | 2026-03-21 | encoder 模式对照 | 保持 A1.6 单阶段，只改回 `normal` | `33.7373 @ 150` | `24.0117` | `24.1343` | [文档][结果] |
| A1.6 P2 bypass_fixed | 2026-03-21 | 同上 | `bypass_fixed` | `25.7546 @ 250` | `30.7549` | `24.0064` | stage_end 异常高，但 peak 很低，需分口径看 [文档][结果] |
| A1.6 P2 bypass_linear | 2026-03-21 | 同上 | `bypass_linear` | `34.8626 @ 150` | `23.5470` | `23.8264` | 与 P0/P1 结论一致 [文档][结果] |
| A1.6 P2 bypass_mlp_16_32 | 2026-03-21 | 同上 | `bypass_mlp_16_32` | `34.5837 @ 150` | `23.9586` | `22.6401` | 不如 bypass_linear [文档][结果] |

### 2.3 A1.7 至 A1.9：扩展与诊断分支

| 实验 | 启动时间 | 目的 | 核心改动 | 训练内 best | stage_end_full | periodic tail | 证据级别 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1.7 P0 seed7 | 2026-03-22 | 继续 seed 复现，形成新的稳定快照 | 与 A1.6 P0 基本同路 | `34.5245 @ 150` | `25.1546` | `23.6589` | [结果] |
| A1.7 P0 seed19 | 2026-03-22 | 同上 | 同上 | `34.6552 @ 150` | `25.2341` | `23.8434` | [结果] |
| A1.7 P0 seed73 | 2026-03-22 | 同上 | 同上 | `34.3582 @ 150` | `24.5471` | `23.7336` | [结果] |
| A1.7 P1 `corr_seed19` | 2026-03-23 | `[推断]` 为相关性诊断复跑，增加 Falcor 采样密度 | 与 A1.6 P0 相比，保存下来的 YAML 里唯一实质变化是 `falcor_periodic_eval.every_n_epochs: 50 -> 25`，外加审计 seed 改为 42 | `34.1668 @ 150` | `27.5789` | `23.7691` | [配置][结果][推断] |
| A1.7 P2 `K1` | 2026-03-23 | 容量消融 | `num_gaussians: 30 -> 1` | `19.0629 @ 100` | `19.6733` | `18.9143` | 明显退化 [配置][结果] |
| A1.7 P2 `K30` | 2026-03-23 | 与 K1 对照 | 保持 `K=30` | `34.5224 @ 150` | `25.2882` | `23.7518` | 与 K1 拉开巨大差距 [配置][结果] |
| A1.7 P3 `gap_hard` | 2026-03-23 | 分析 A1.7 P0 `global_best_falcor` 在 hard profile 下的评测落差 | 非训练 run；对 checkpoint 做 hard full-suite | hard HDR `10.5692`；Benchmark `11.1877`；SH `-7.0359` | - | - | 分析目录 [结果] |
| A1.7 P3 `gap_soft` | 2026-03-23 | 同上，但 soft route t0.18 | 非训练 run；soft full-suite | benchmark 失败，`return_code=1` | - | - | 分析目录 [结果] |
| A1.8 `dynamic_softsat_huber_spatial` | 2026-04-10 | 给 proxy 加入更强数值鲁棒性与空间约束 | proxy 改为 Huber；`sampling_mode=per_epoch`；开启 soft saturation；新增 `lambda_spatial=0.001` | `34.2229 @ 150` | `25.8256` | `23.7849` | [配置][结果] |
| A1.8 smoke | 2026-04-10 | smoke only | 1 epoch | - | - | - | [结果] |
| A1.9 `analytic_irradiance` | 2026-04-10 | 将 proxy 换成解析 irradiance 域 | `proxy_image_loss.domain=irradiance`；`mix_mode=linear_only`；固定采样；去掉 soft saturation 和 spatial | `33.9105 @ 125` | `24.8861` | `24.0323` | [配置][结果] |
| A1.9 smoke | 2026-04-10 | smoke only | 1 epoch | - | - | - | [结果] |

### 2.4 A2 至 A2.3.3：proxy→gbuffer 与后续干预

| 实验 | 启动时间 | 目的 | 核心改动 | periodic first | periodic peak | stage_end_full | periodic tail | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A2 `proxy2gbuffer` | 2026-04-14 | 引入 gbuffer 监督，并从 proxy 向 gbuffer handover | 新增 `gbuffer_image_loss`；150-220 epoch handover；打开 `loss_effective_contribution` 与优化监控 | `25.9253 @ 25` | `34.9482 @ 175` | `25.0361` | `22.3510` | 高峰值但后期塌陷 [配置][结果] |
| A2.1 `top8_balance` | 2026-04-15 | 命名上是 top8/balance 分支 | **现存 YAML 未显示除输出目录/stage 名以外的实质差异** | `25.4790 @ 25` | `34.3483 @ 250` | `24.7499` | `23.8283` | 若有代码级改动，需后续追 Git [配置][结果][推断] |
| A2.2-A `temp035` | 2026-04-15 | 抑制路由硬化 | `routing_temp_end: 0.2 -> 0.35` | `21.6401 @ 25` | `34.0087 @ 325` | `23.1302` | `33.9321` | 典型“峰值守住到尾部” [配置][结果] |
| A2.2-B `lrmin1e5` | 2026-04-15 | 提高后期 LR 下限 | `lr_min: 1e-6 -> 1e-5` | `25.5533 @ 25` | `33.4255 @ 250` | `24.9041` | `23.9547` | 只保更新，不保尾部 [配置][结果] |
| A2.2-AB | 2026-04-15 | 联合验证 temp+lr | `temp_end=0.35` + `lr_min=1e-5` | `23.6300 @ 25` | `33.5965 @ 325` | `22.5256` | `32.5737` | 比 A2.2-A 稍弱，但同样显著改善尾部 [配置][结果] |
| A2.3 `temp035_soft_handover_perepoch` | 2026-04-16 | 在 A2.2-A 基础上改 soft handover + per_epoch proxy sampling | `proxy_end_scale: 0 -> 0.1`；proxy sampling 改 `per_epoch`；gbuffer `target_source=dataset_sh` | `24.6808 @ 25` | `34.0034 @ 325` | `24.9364` | `33.8398` | 与 A2.2-A 同属“高尾部稳定” [配置][结果] |
| A2.3.1 `lossweights` | 2026-04-16 | 按偏相关口径重排 loss 权重 | `temporal=1.0`、`gbuffer=1.0`、`recon=0.2`、`linearity=0.001`、`routing_balance=0.01`、`image=0`；保留 `loss_effective_contribution` | `29.9920 @ 25` | `31.0217 @ 50` | `30.3383` | `24.2751` | 前期高、后期未拉升；阶段末反而高于 peak 口径 [配置][结果] |
| A2.3.2 `lossweights_fixed` | 2026-04-16 | 检验上一组权重是否被 `loss_effective_contribution` 扭曲 | 与 A2.3.1 相同，但 `loss_effective_contribution.enabled=false` | `32.4884 @ 25` | `34.6559 @ 200` | `30.2568` | `26.6930` | 相比 A2.3.1 明显恢复峰值 [配置][结果] |
| A2.3.3 `disable_loss_eff` | 2026-04-16 | 只关闭 `loss_effective_contribution`，不改其他 loss 权重 | 以 A2.3 为底，只关 `loss_effective_contribution.enabled` | `25.9268 @ 25` | `34.9235 @ 275` | `25.3089` | `31.9190` | 证明关闭 loss_eff 本身就能显著改善尾部 [配置][结果] |

## 3. 分组详细整理

### 3.1 Baseline 参考组

#### Baseline

- 目的：提供统一起点，先确认当前主瓶颈到底在 `coeff` 还是 `U/routing`。
- 改动：无，作为起点。
- 关键数值：
  - `benchmark_summary.image_metrics.mean_real_render_hdr_psnr = 30.9159`
  - `mean_benchmark_psnr = 23.7824`
  - `mean_sh_psnr = 11.7867`
  - Oracle `all27_psnr_gain_db = 11.2823`
- 当前定位：后续几乎所有 staged / bypass / A2 handover 的论证，都以这个系数上界缺口为起点。

#### A1 coeff-first

- 目的：验证“先训 coeff 再 joint”是否能把模型拉向 Oracle 上界。
- 改动：
  - stage1 只训 `coeff + encoder`
  - stage2 再做 joint
- 关键数值：
  - hard HDR `25.2204`
  - soft018 HDR `30.8200`
  - 相对 baseline HDR `-5.6956 dB`
  - Oracle gap 从 baseline 的 `11.2823 dB` 变为 `15.3067 dB`
- 当前定位：**重要失败前导实验**。证明简单的 coeff-first 路线并不能自动解决 coeff-limited 问题。

### 3.2 A1.1 / A1.2 历史分叉

#### 旧 A1.1

- 目的：建立旧主线，即 soft routing + EMA + staged + 分组 LR。
- 改动：
  - stage1 训 `coeff+encoder+film+routing`
  - stage2 放开 `all`
  - soft routing train：`top8 + temperature anneal`
  - EMA 打开
  - **未开启 proxy / val_profiles / periodic Falcor**
- 关键数值：
  - soft018 HDR `33.3438`
  - hard HDR `21.9068`
- 当前定位：说明“soft 口径下存在明显可用高点”。

#### A1.2

- 目的：把目标函数和选模策略向图像域/Falcor 靠拢。
- 改动：
  - proxy image loss 打开
  - `val_profiles` 打开
  - periodic Falcor 打开
  - 以 soft018 与图像指标参与选模
- 关键数值：
  - soft018 HDR `22.9663`
  - stage 内 best `23.0712`
- 当前定位：说明“目标函数与选模链路增强”不等于性能增强，反而曾导致严重退化。

### 3.3 A1.3~A1.6：从复杂 staged 回到编码链路瓶颈

#### A1.3 / A1.4

- 目的：修正 A1.2 的流程与选模问题。
- 改动：
  - A1.3：重新在独立目录重跑，分两阶段，balance 退火到 0。
  - A1.4：strict resume，三阶段 anchor refine，global best 跨 stage 保护。
- 结果：
  - A1.3 训练内 best `27.8670`
  - A1.4 训练内 best `27.3836`
- 当前定位：复杂 staged 和选模保护**没有**带来新的高峰，问题更像是在编码链路。

#### A1.5 proxyfix

- 目的：验证 proxy 数值路径是否是主因。
- 改动：
  - proxy linear 分支由 `clamp(0,1)` 改为 `clamp(min=0)`
  - 从 A1.4 soft best 加载权重做 finetune
- 结果：训练内 best 只剩 `21.7116`
- 当前定位：单纯改 proxy 数值路径无效，且明显恶化。

#### A1.5 bypassA / bypassB / bypassB_noproxy

- 目的：检查原 light encoder 是否是更大的瓶颈。
- 改动：
  - bypassA：`bypass_fixed`
  - bypassB：`bypass_linear`
  - bypassB_noproxy：在 bypassB 上关掉 proxy
- 结果：
  - bypassA `29.5016`
  - bypassB `34.3938`
  - bypassB_noproxy `31.8811`
- 当前定位：
  - `bypass_linear` 是**第一阶段真正突破点**
  - proxy 在 bypass 链路上仍有正收益

#### A1.6 P0 / P1 / P2

- 目的：
  - P0：测 bypassB 的 seed 稳定性
  - P1：扫 proxy 权重
  - P2：系统对比 encoder 模式
- 关键数值：
  - P0 最好：seed19 `35.0076`
  - P1 最好：`lambda=0.02` 的 recorded peak `34.8678`，另有 `global_best_falcor=35.2238`
  - P2 最好：`bypass_linear = 34.8626`
- 当前定位：
  - `bypass_linear` 的优势有重复性
  - `proxy λ=0.02` 在该分支最稳妥
  - `bypass_fixed` 的 stage_end 很高，但 peak 很低，属于**口径异常点**

### 3.4 A1.7~A1.9：诊断补强与 proxy 变体

#### A1.7

- P0：继续形成 bypassB 的稳定复现快照。
- P1：从现存配置看，核心变化是把 periodic Falcor 频率从 `50` 缩到 `25`；更像是**相关性分析采样更密的诊断复跑**。
- P2：`K1 vs K30` 容量消融非常关键。
  - `K1` stage_end `19.6733`
  - `K30` peak `34.5224`
- P3：不是训练 run，而是 gap-analysis 结果目录。
  - hard route 复算时，A1.7 P0 的 `global_best_falcor` 在 hard full-suite 下只有 HDR `10.5692`
  - soft route t0.18 那次 benchmark 直接失败

#### A1.8

- 目的：增强 proxy 数值鲁棒性。
- 改动：
  - Huber loss
  - per-epoch sampling
  - soft saturation
  - spatial regularization
- 结果：peak `34.2229`
- 当前定位：没有超过 bypass 线的高点，但提供了一个更“工程化”的 proxy 变体。

#### A1.9

- 目的：把 proxy 重写为解析 irradiance 域监督。
- 改动：
  - `domain=irradiance`
  - `mix_mode=linear_only`
  - fixed sampling
  - 去掉 A1.8 的 soft saturation 和 spatial
- 结果：peak `33.9105`
- 当前定位：解析 irradiance 版本没有打穿 bypassB 主线。

### 3.5 A2~A2.3.3：从 proxy→gbuffer 到调度/权重/EMA 口径

#### A2

- 目的：从 proxy 逐步 handover 到 gbuffer，观察是否能把监督更贴近真实渲染。
- 改动：
  - 新增 `gbuffer_image_loss`
  - 150~220 epoch handover
  - `loss_effective_contribution` 打开
  - 优化监控打开
- 结果：
  - first `25.9253`
  - peak `34.9482 @ 175`
  - tail `22.3510`
- 当前定位：**有高峰，但后期塌陷严重**。

#### A2.1

- 目的：命名上是 top8/balance 路线。
- 当前能确定的事实：
  - 结果目录与配置存在
  - 训练结果真实存在
  - 但保存下来的 YAML 与 A2 相比，没有可确认的核心超参差异
- 当前定位：需要后续通过 Git 历史补完“到底改了什么”。

#### A2.2

- A2.2-A：
  - 目的：只抑制路由硬化
  - 改动：`routing_temp_end=0.35`
  - 结果：peak `34.0087`，tail `33.9321`
- A2.2-B：
  - 目的：只抬高 LR 下限
  - 改动：`lr_min=1e-5`
  - 结果：peak `33.4255`，tail `23.9547`
- A2.2-AB：
  - 目的：联合控制
  - 改动：`temp_end=0.35 + lr_min=1e-5`
  - 结果：peak `33.5965`，tail `32.5737`
- 当前定位：
  - **抬温度下限**是 A2 系列第一个明确改善尾部稳定性的干预
  - 单独抬 LR 下限帮助有限

#### A2.3

- 目的：在 A2.2-A 基础上，让 handover 更软，并避免固定方向 proxy 过拟合。
- 改动：
  - `proxy_end_scale: 0 -> 0.1`
  - proxy sampling `fixed -> per_epoch`
  - gbuffer `target_source=dataset_sh`
- 结果：
  - peak `34.0034`
  - tail `33.8398`
- 当前定位：与 A2.2-A 一样，属于**后期性能能守住**的路线。

#### A2.3.1 / A2.3.2 / A2.3.3

- A2.3.1：
  - 目的：按偏相关结果重排 loss 权重
  - 改动：重压 `temporal/gbuffer`，下调 `recon/linearity/routing_balance`，关闭 image，保留 `loss_effective_contribution`
  - 结果：peak `31.0217`，stage_end `30.3383`
  - 定位：稳定但峰值不够
- A2.3.2：
  - 目的：验证 A2.3.1 是否被 `loss_effective_contribution` 机制扭曲
  - 改动：仅关闭 `loss_effective_contribution`
  - 结果：peak `34.6559`，stage_end `30.2568`
  - 定位：性能大幅恢复，说明 loss_eff 本身值得重点审查
- A2.3.3：
  - 目的：从 A2.3 原始权重出发，只关闭 `loss_effective_contribution`
  - 改动：仅 `loss_effective_contribution.enabled=false`
  - 结果：peak `34.9235`，tail `31.9190`
  - 定位：说明**问题并不需要先改 loss 权重，单关 loss_eff 就已产生显著收益**

## 4. 排行与辅助视图

### 4.1 periodic peak 排行（当前记录）

| 排名 | 实验 | periodic peak |
| --- | --- | --- |
| 1 | `exp_A1_6_P0_bypassB_seed19` | `35.0076` |
| 2 | `exp_A2_proxy2gbuffer_seed19` | `34.9482` |
| 3 | `exp_A2_3_3_disable_loss_eff_seed19` | `34.9235` |
| 4 | `exp_A1_6_P1_bypassB_proxy_l002` | `34.8678` |
| 5 | `exp_A1_6_P2_enc_bypass_linear` | `34.8626` |
| 6 | `exp_A2_3_2_lossweights_fixed_seed19` | `34.6559` |
| 7 | `exp_A1_7_P0_bypassB_seed19` | `34.6552` |
| 8 | `exp_A1_6_P2_enc_bypass_mlp1632` | `34.5837` |
| 9 | `exp_A1_7_P0_bypassB_seed7` | `34.5245` |
| 10 | `exp_A1_7_P2_K30_seed19` | `34.5224` |

### 4.2 stage_end_full 排行（当前记录）

| 排名 | 实验 | stage_end_full |
| --- | --- | --- |
| 1 | `exp_A1_6_P2_enc_bypass_fixed` | `30.7549` |
| 2 | `exp_A2_3_1_lossweights_seed19` | `30.3383` |
| 3 | `exp_A2_3_2_lossweights_fixed_seed19` | `30.2568` |
| 4 | `exp_A1_5_bypassA` | `27.9430` |
| 5 | `exp_A1_5_bypassB` | `27.5817` |
| 6 | `exp_A1_7_P1_corr_seed19` | `27.5789` |
| 7 | `exp_A1_6_P1_bypassB_proxy_l001` | `27.3957` |
| 8 | `exp_A1_6_P0_bypassB_seed73` | `27.0306` |
| 9 | `exp_A1_5_bypassB_noproxy` | `26.9880` |
| 10 | `exp_A1_6_P0_bypassB_seed7` | `26.5145` |

### 4.3 periodic tail 排行（当前记录）

| 排名 | 实验 | periodic tail |
| --- | --- | --- |
| 1 | `exp_A2_2_A_temp035_seed19` | `33.9321` |
| 2 | `exp_A2_3_temp035_soft_handover_perepoch_seed19` | `33.8398` |
| 3 | `exp_A2_2_AB_temp035_lrmin1e5_seed19` | `32.5737` |
| 4 | `exp_A2_3_3_disable_loss_eff_seed19` | `31.9190` |
| 5 | `exp_A2_3_2_lossweights_fixed_seed19` | `26.6930` |
| 6 | `exp_A1_5_bypassA` | `24.6192` |
| 7 | `exp_A2_3_1_lossweights_seed19` | `24.2751` |
| 8 | `exp_A1_6_P2_enc_normal` | `24.1343` |
| 9 | `exp_A1_9_analytic_irradiance_seed19` | `24.0323` |
| 10 | `exp_A1_6_P0_bypassB_seed73` | `24.0158` |

### 4.4 这三个排行分别在说什么

- `periodic peak` 更像“这条路最高能冲到哪里”。
- `periodic tail` 更像“训练结束时还剩多少”。
- `stage_end_full` 更像“最后一次完整评估的截图”，常常与 periodic 不一致。

因此：

- A1.6/P0、A2、A2.3.3 是**高峰型**。
- A2.2-A、A2.3、A2.2-AB 是**高尾部稳定型**。
- A2.3.1、A2.3.2、A1.6-P2-bypass_fixed 是**stage_end 高，但不代表 periodic 同样强**。

## 5. 已有分析工件索引

### 5.1 A1 全景 loss/Falcor 相关性快照

- 目录：`3_experiments/results/loss_falcor_correlation/a1_all_snapshot/`
- 已覆盖实验：
  - `A1_1`
  - `A1_3`
  - `A1_4`
  - `A1_5` 全部分支
  - `A1_6` 全部分支
  - `A1_7` P0/P1/P2
  - `A1_8`
  - `A1_9`

### 5.2 A2 系列 lag + partial 结果

- 目录：`3_experiments/results/loss_falcor_correlation_lag_partial/`
- 当前已见集合：
  - `a2_2_a_periodic_focus`
  - `a2_3_periodic`
  - `a2_3_periodic_focus`
  - `a2_3_with_stageend`
  - `a2_3_with_stageend_focus`
  - `a2_3_1_periodic_focus`
  - `a2_3_1_with_stageend_focus`
  - `a2_3_2_periodic_focus`
  - `a2_3_2_with_stageend_focus`
  - `a2_3_3_periodic_focus`
  - `a2_3_3_with_stageend_focus`

### 5.3 gap-analysis 专用目录

- `3_experiments/results/bistro_clean_v2/exp_A1_7_P3_gap_seed19_hard/`
- `3_experiments/results/bistro_clean_v2/exp_A1_7_P3_gap_seed19_soft/`

这两个目录不是训练 run，而是对已有 checkpoint 的补充复算。

## 6. 当前仍未补齐的空白

1. **A1.1 目录复用问题**  
   旧 A1.1 与后来的 A1.2 不能仅凭 `exp_A1_1_joint_lrs_ema/` 当前状态区分，后续若要做更严谨的历史重建，需要继续依赖 full-suite 与 train log。

2. **A2.1 top8_balance 的真实改动**  
   现存 YAML 不足以证明它和 A2 有何超参差异。如果这一步很关键，需要追 commit 或当时的训练日志。

3. **部分早期实验 A1 / A1.2 不再有独立 config 路径**  
   当前只能通过结果快照和文档重建。

4. **smoke run 只保留了启动成功证据，没有分析价值**  
   例如 A1.8 smoke、A1.9 smoke。

## 7. 建议的下一步整理顺序

基于这份盘点，下一轮建议按下面顺序继续：

1. 先做“按阶段归类”的高层分析：
   - 编码链路阶段
   - proxy 变体阶段
   - proxy→gbuffer handover 阶段
   - temp/LR/loss_eff 稳定化阶段
2. 再做“同口径对照”：
   - 只比 `periodic peak`
   - 只比 `periodic tail`
   - 只比 `stage_end_full`
3. 最后做“机制级复盘”：
   - 哪些改动在提升峰值
   - 哪些改动在守住尾部
   - 哪些改动只是改变了评估口径或阶段末观感
