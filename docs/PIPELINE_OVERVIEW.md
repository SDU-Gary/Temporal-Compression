# 数据生成 → 训练 → 评估：工程结构梳理

下面内容基于当前代码结构与脚本实际行为整理。涵盖数据生成、训练与评估的主线流程，并标注关键入口、输入输出、目录结构与环境依赖。

整体概览

当前主干流程是：
1) Blender（可选）导出发光体轨迹 → 生成 scene.json  
2) Falcor 读取 scene.json/.pyscene → 烘焙 SH 数据 → 输出 parametric_tensor.npz  
3) 训练脚本读取 parametric_tensor.npz → 训练 PG‑GCPL → 输出 best_model.pt  
4) 评估脚本对模型做内插/外插/延迟测试 → 输出图表与 JSON

流程由 YAML 配置驱动（1_data_generation/configs/*.yaml、pipelines/*.yaml），统一入口为 tools/run_dataset.py 与 tools/run_pipeline.py。
注意：评估脚本中仍存在硬编码数据/模型路径。

步骤 0（可选）：Blender 导出发光体轨迹与 scene.json

步骤说明
- 从 Blender 场景中导出发光球轨迹（.npy）与 scene.json，供 Falcor 读取。

代码位置
- tools/blender/export_emissive_trajectories.py（主入口）
- tools/blender/scene_export.py（构建 scene.json）

输入/输出
- 输入：.blend 文件内的发光球体（bpy）
- 输出：scene.json + BallX_traj.npy（路径由 --output-dir 指定）

环境依赖
- 强绑定：Blender + bpy（需在 Blender 中运行）
- 可迁移：在 Mac 上装 Blender 可迁移

步骤 1（可选）：Probe 采样（GBufferRT）

步骤说明
- 用 Falcor GBufferRT 从多个相机采样表面点 → 结合光轨迹权重 → 输出 probe 坐标。

代码位置
- tools/sample_surface_probes.py（主入口）
- tools/probe_sampling.py（权重/去簇）

输入/输出
- 输入：.pyscene 场景，光源轨迹 .npy（可选）
- 输出：probes_surface.npy（探针坐标）+ 可选 .ply

环境依赖
- 强绑定：Falcor + Vulkan + libFalcor.so + NVIDIA GPU
- 可迁移：不易迁移到 Mac

步骤 2：Falcor 数据烘焙（核心）

步骤说明
- 逐帧渲染 SH → 生成 parametric_tensor.npz 数据集。

代码位置
- 生成器：1_data_generation/falcor/generate_bistro_temporal_spheres.py
- Falcor 封装：1_data_generation/falcor/utils/falcor_render.py
- scene.json 接入：1_data_generation/falcor/utils/scene_io.py
- 配置：1_data_generation/configs/bistro_*.yaml
- 入口：tools/run_dataset.py（通过 tools/workflow_config.py 构造命令）

输入/输出
- 输入：
  - .pyscene（如 1_data_generation/falcor/scenes/bistro_exterior_emissive_spheres.pyscene）
  - 可选 scene.json（Blender 导出）
  - 可选 probes_surface.npy
- 输出目录示例：1_data_generation/output/bistro_clean_v2
- 输出文件：
  - parametric_tensor.npz（核心数据）
  - metadata.json
  - run_summary.md（通过 run_dataset 生成）
  - manifest.json（若启用 --write-manifest）

数据格式（parametric_tensor.npz）
- tensor: [P, M, 27]
- probe_positions: [P, 3]
- light_configs: [M, N, 12]
- light_mask: [M, N]
- metadata: dict

环境依赖
- 强绑定：Falcor + libFalcor.so + Vulkan + NVIDIA GPU
- 路径绑定：BISTRO_FBX 默认指向 1_data_generation/scenes/Bistro_v5_2/BistroExterior.fbx
- 可迁移：不易迁移到 Mac

步骤 3（可选但推荐）：数据验证 + Manifest

步骤说明
- 检查数据完整性并写入 manifest（用于训练/追溯）。

代码位置
- 1_data_generation/validate_dataset.py
- tools/manifest_utils.py

输入/输出
- 输入：数据集目录（parametric_tensor.npz）
- 输出：manifest.json、控制台报告

环境依赖
- 可迁移：纯 Python

步骤 4：训练

步骤说明
- 读取 parametric_tensor.npz 训练 PG‑GCPL 模型，输出 best_model.pt。

代码位置
- 入口：3_experiments/scripts/train.py
- 数据集：2_src/data/lightset_dataset.py（多光源）
- 模型：2_src/models/gaussian_physics_unified.py 等
- 训练器：2_src/training/gaussian_physics_trainer.py

输入/输出
- 输入：
  - --data-root <dataset_dir> 或 --manifest <manifest.json>
  - 训练配置参数（variant、num_steps、batch_size 等）
- 输出：
  - best_model.pt
  - last_model.pt
  - 可选 sh_scaler.npz

环境依赖
- 可迁移：纯 Python / PyTorch
- GPU（CUDA）可加速，Mac 可用 CPU/MPS

步骤 5：评估

步骤说明
- 内插/外插/延迟等评估，输出 JSON + 图。

代码位置
- 入口：3_experiments/scripts/evaluation/run_all.py
- 子脚本：
  - 3_experiments/scripts/evaluation/test_interpolation_query_physics.py
  - 3_experiments/scripts/evaluation/test_extrapolation_query_physics.py
  - 3_experiments/scripts/evaluation/test_query_latency_physics.py

输入/输出
- 输出目录：--output-dir 指定
- 注意：部分评估脚本内部硬编码路径
  - test_interpolation_query_physics.py：数据路径写死
  - test_extrapolation_query_physics.py：数据路径写死
  - test_query_latency_physics.py：checkpoint 路径写死

环境依赖
- 可迁移：纯 Python / matplotlib
- 但硬编码路径属于强绑定

强绑定主机环境（不可直接迁移）
- Falcor 相关：1_data_generation/falcor/*、tools/sample_surface_probes.py
- Blender 相关：tools/blender/*
- 评估硬编码路径（需改为参数化）

可迁移到 Mac 的部分
- 训练与评估大多数脚本（2_src/*、3_experiments/scripts/*）
- YAML 管线与 manifest 工具（tools/run_dataset.py、tools/run_pipeline.py）

---

## MacBook 可执行任务（快速迭代 / 小规模 sanity check）

定位
- 不依赖 Falcor
- 不依赖特定 GPU / CUDA（可用 CPU 或 Mac MPS）
- 适合快速试验、小规模验证

可执行脚本/入口
- 训练（小数据或抽样）
  - `3_experiments/scripts/train.py`
  - 调用方式示例：`python 3_experiments/scripts/train.py --variant unified_set --data-root <dataset_dir> --num-steps 100 --batch-size 16`
- 评估（只跑小规模或已有模型）
  - `3_experiments/scripts/evaluation/run_all.py`（根据 config 中 variant 自动选择 physics 或 unified 套件）
  - 调用方式示例：`python 3_experiments/scripts/evaluation/run_all.py --output-dir <out>`
- 可视化/分析（纯 Python）
  - `3_experiments/scripts/visualization/*`
  - `3_experiments/scripts/analysis/*`
- 数据验证与摘要
  - `1_data_generation/validate_dataset.py`
  - `tools/manifest_utils.py`（读写 manifest）

环境条件
- Python + PyTorch（CPU/MPS 可用）
- 常规数值/可视化库（numpy/matplotlib 等）

是否可抽象为「一条命令 + 配置」
- 是。可以用 `tools/run_pipeline.py` 只跑 train/eval（跳过 dataset），或用单独命令调用对应脚本。
- 小提示：pipeline 的 eval 若未显式给 output_dir，会优先使用 **train args 的 output_dir**，或 **train config 里的 output_dir**；若两者都没有，则需要在 eval 配置里手动指定。

## YAML 模板与 Schema（轻量校验）

- 模板：`3_experiments/configs/template_unified_set.yaml`
- Schema：`metadata/schemas/train.schema.json`
- 训练/评估脚本支持：
  - `--schema` 指定 schema 路径
  - `--strict-schema` 在校验失败时直接报错
- 如果未安装 `jsonschema`，会自动跳过校验并给出提示。

## 旧版训练脚本去向

- 旧的 1D/5D 训练脚本已迁移到 `3_experiments/scripts/legacy_training/`
- 当前推荐入口：`3_experiments/scripts/train.py`

---

## Kubuntu 主机必做任务（Falcor + 重计算）

定位
- 依赖 Falcor 进行数据烘焙
- 依赖强 GPU 显存 / 高算力
- 完整数据、完整训练、大场景渲染

必做脚本/入口
- Falcor 数据烘焙（核心）
  - `1_data_generation/falcor/generate_bistro_temporal_spheres.py`
  - 通过配置入口：`tools/run_dataset.py --config 1_data_generation/configs/bistro_prod.yaml`
- Probe 采样（GBufferRT）
  - `tools/sample_surface_probes.py`
- 并行/批量烘焙
  - `tools/run_parallel_bistro_bakes.py`
- 大规模训练（完整数据）
  - `3_experiments/scripts/train.py`（使用高 batch、长训练）
- 大规模评估/渲染
  - `tools/render_probe_video.py` / `tools/render_probe_video_gpu.py`
  - 评估脚本（更大数据、更长运行）

环境条件
- Falcor 已构建（libFalcor.so + Python bindings）
- Vulkan + NVIDIA GPU
- 系统环境变量（LD_LIBRARY_PATH / VK_ICD_FILENAMES / FALCOR_PYTHON）

是否可抽象为「一条命令 + 配置」
- 是。可用：
  - `tools/run_dataset.py`（单数据集）
  - `tools/run_pipeline.py`（dataset + train + eval）
  - 但评估脚本中的硬编码路径仍建议参数化

---

## 未来期望的使用方式（从你的视角）

目标：在 Mac 上做快速 sanity check，在 Kubuntu 上做正式烘焙/训练/评估。下面是“理想命令风格”的例子，便于你后续封装 CLI。

Mac 上的典型命令（快速迭代）
1) `pg-gcpl local-test --dataset <dataset_dir> --steps 100`
   - 动作：用小批量快速跑训练（CPU/MPS），验证损失是否正常。
2) `pg-gcpl local-eval --output <out_dir>`
   - 动作：跑轻量评估脚本与可视化，输出图表。
3) `pg-gcpl local-validate --dataset <dataset_dir>`
   - 动作：检查数据集完整性、统计分布、是否有 NaN/Inf。

Kubuntu 上的典型命令（正式产出）
1) `pg-gcpl remote-bake --config 1_data_generation/configs/bistro_prod.yaml`
   - 动作：Falcor 烘焙完整数据集（可带 radiance clamp、spp、cube_res）。
2) `pg-gcpl remote-train --dataset <dataset_dir> --variant unified_set`
   - 动作：大规模训练，输出 best_model.pt。
3) `pg-gcpl remote-eval --output <eval_dir>`
   - 动作：完整评估 + 生成图表与指标。

以上命令可对应到现有脚本：
- local-test → `3_experiments/scripts/train.py`
- local-eval → `3_experiments/scripts/evaluation/run_all.py`
- local-validate → `1_data_generation/validate_dataset.py`
- remote-bake → `tools/run_dataset.py` + Falcor generator
- remote-train/eval → `tools/run_pipeline.py` 或直接调用对应脚本

---
