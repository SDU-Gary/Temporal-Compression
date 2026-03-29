# PG-GCPL 开发真相文档（As Of 2026-03-08）

> 本文档只记录当前仓库代码与现存产物中可验证的事实，不复述已失效的历史流程。

## 1. 代码主线结论

- 当前唯一主线模型：`2_src/models/gaussian_physics_unified.py`
  - `LightSetEncoder`（支持光源集合输入 + 强度解耦）
  - `GaussianPhysicsCompressionUnified`（Gaussian routing + low-rank + FiLM）
- 当前训练主入口：`3_experiments/scripts/train.py`
  - `variant` 仅支持 `unified_set`
  - 支持 staged training、EMA、oracle monitor、coeff suite、val profiles、heartbeat
- 当前评估主入口：`3_experiments/scripts/eval.py`
  - 从 checkpoint 自动推断 K/r/embed_dim/light_dim 等
  - 输出 `mae/rmse/sh_psnr/sh_ssim`

## 2. 数据格式与生成真相

- 训练/评估主数据格式：`parametric_tensor.npz`
  - `tensor`: `[P, M, 27]`
  - `probe_positions`: `[P, 3]`
  - `light_configs`: `[M, N, 12]`
  - `light_mask`: `[M, N]`（可选）
- 当前 Falcor 主生成脚本：
  - `1_data_generation/falcor/generate_bistro_temporal_spheres.py`
  - `1_data_generation/falcor/generate_multilight_falcor.py`
- 数据生成编排工具：
  - `tools/run_dataset.py`
  - `tools/workflow_config.py`
  - `1_data_generation/configs/{bistro_smoke,bistro_dev,bistro_prod}.yaml`

## 3. Pipeline 真相

- 当前统一 pipeline 入口：`tools/run_pipeline.py`
- 可用 pipeline 配置：
  - `pipelines/pgcpl_bistro_smoke.yaml`
  - `pipelines/pgcpl_bistro_dev.yaml`
- 执行模式：dataset → train → eval
  - 自动注入 `data_root`
  - eval checkpoint 缺省优先 `best_model.pt`，回退 `last_model.pt`

## 4. 数据库真相（已统一）

- Canonical DB：`metadata/project.db`
- 代码默认 DB：`tools/db_utils.py` 中 `DB_PATH` 现已默认解析到 `metadata/project.db`
- legacy 路径：`project.db`（已不作为默认）
- 可选覆盖：`PGCPL_DB_PATH` 环境变量
- 现存有效记录（当前仓库）：`metadata/project.db` 含 22 条 experiments、8 条 tasks

## 5. 文档与工具状态真相

- `tools/summarize_state.py` 在当前仓库不存在（已移除）
- `docs/dev_notes/project_state.md` 是历史快照，不是当前自动实时状态文件
- 当前实验查询/追踪应直接使用：
  - `python tools/logexp.py query --limit 10`
  - `python tools/logexp.py list-tasks --status todo`

## 6. 训练能力真相（train.py + trainer）

- 已实现能力（代码中可见）：
  - Charbonnier/L1/MSE 重建
  - temporal/linearity/spatial/image/routing-balance 等损失项
  - soft routing 温度退火
  - 参数分组学习率（routing/basis/coeff/encoder/film）
  - EMA eval/save best
  - staged 多阶段训练
  - oracle diagnostics 子进程监控
  - coeff learnability suite
- 产物结构（以 `3_experiments/results/bistro_clean_v2/exp_A1_1_joint_lrs_ema` 为例）：
  - `runtime/heartbeat.json`
  - `runtime/phase0_metrics.jsonl`
  - `staged_training_summary.json`
  - `diagnostics/coeff_suite*/phase1_summary.json`

## 7. 与任务书目标的差距（当前仍待完成）

- 4×3 cascaded volumes：未落地为主线实现
- temporal LRU cache：未见主线实现
- fused CUDA kernels：未见主线实现
- 10-bit quantization：未见主线实现
- energy conservation loss：未见主线实现
- lighting decomposition（direct/indirect）：未见主线实现
- 24-moment 主线训练目标：未形成稳定主线配置闭环

## 8. 当前推荐最小可执行流程

```bash
# 1) 激活环境
source venv/bin/activate

# 2) 训练（主线）
python 3_experiments/scripts/train.py --config 3_experiments/configs/bistro_clean_train.yaml

# 3) 评估（主线）
python 3_experiments/scripts/eval.py \
  --data-root 1_data_generation/output/bistro_clean_v2 \
  --checkpoint 3_experiments/results/bistro_clean_v2/unified_set_K30_r8/best_model.pt \
  --split test

# 4) 查询实验记录（canonical DB）
python tools/logexp.py query --limit 10
```

## 9. 文档使用建议

- 论文目标/任务边界：看 `4_thesis/report/多时刻光照压缩任务书.md`
- 代码真实状态：优先看本文档 + 对应脚本源码
- 历史实验语境：看 `docs/dev_notes/` 下历史报告（不要直接当作当前主线实现说明）
