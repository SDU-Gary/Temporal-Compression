# ZeroDay 数据生成管线真相（As Of 2026-03-22）

> 口径：只描述当前仓库代码里已经实现并可读到的行为，不描述计划中的未来版本。

## 1. 代码入口与配置

- 生成脚本（动态几何版）：`1_data_generation/falcor/generate_zeroday_temporal.py`
- 生成脚本（静态几何 + 随机光照版）：`1_data_generation/falcor/generate_zeroday_static_randomlights.py`
- 探针有效性工具：`1_data_generation/falcor/utils/probe_validity.py`
- 运行配置（动态几何版）：`1_data_generation/configs/zeroday_prod.yaml`
- 运行配置（静态几何 + 随机光照版）：`1_data_generation/configs/zeroday_static_randomlights_prod.yaml`
- 统一编排入口：`tools/run_dataset.py`

当前默认场景路径（脚本默认值与配置一致）是：

- `1_data_generation/scenes/ZeroDay/MEASURE_SEVEN/MEASURE_SEVEN_COLORED_LIGHTS.fbx`

## 2. ZeroDay 适配后的真实流程

### 2.1 Testbed 结构（分工明确）

脚本会创建两套或三套 Falcor testbed（取决于 `validity_method`）：

1. `PathTracer` testbed：负责最终 SH 烘焙。
2. `GBufferRT` testbed：负责候选点生成与快速有效性筛选。
3. `ZeroDayRayQuery` 1x1 `GBufferRT` testbed（仅 `rayquery_*`）：负责逐 probe 射线复核。

这样避免“用烘焙图直接做几何判定”，几何判定完全基于 `posW/normW/depth`。

### 2.2 动态场景时间推进

- 通过 `_set_scene_time(testbed, frame_idx, fps)` 将动画时间设置为 `frame_idx / fps`。
- `pt_scene` 与 `gb_scene` 都设为 `animated=True`，`loopAnimations=False`。

这意味着候选采样与有效性判断都在多帧动态状态下执行，不是静态单帧近似。

补充：`generate_zeroday_static_randomlights.py` 默认启用 `--freeze-geometry`，会将几何冻结在 `--freeze-frame`（默认 0），同时按帧连续随机更新解析光源参数（`intensity/position/direction`，按光源类型适配）。

### 2.3 多帧 GBuffer 候选池

候选池在 `candidate_frames` 上构建（默认 `candidate_frame_step=5`）：

1. 多相机位姿：`generate_orbit_camera_poses()` 生成环绕 + 顶视相机。
2. 每帧每相机渲染 GBuffer，读取 `posW/normW/depth`。
3. 过滤有效像素（finite + 合法 depth）。
4. 收集表面点与法线，形成全局 surface pool。
5. 在法线正负方向做偏移采样：`make_surface_offset_candidates()`。
6. 体素去重：`dedupe_points_voxel()`。

### 2.4 Probe 采样（adaptive）

`probe_mode=adaptive` 时，最终 probe =

- 一部分体内均匀采样（`probe_uniform_ratio`）
- 一部分来自 surface-offset 候选池

由 `sample_probes_adaptive()` 实现，最后随机打乱。

## 3. Strict `valid_mask` 的当前实现细节

### 3.1 帧内判定（`evaluate_frame_validity`）

对每个验证帧（默认 `validation_frame_step=1`，即全帧）：

1. 采集该帧所有验证相机的 GBuffer 视图。
2. 将 probe 投影到每个相机像素（`project_points`）。
3. 在投影像素处做两类检查：

- 可见性（Visibility）：
  - `d_probe = ||probe - cam_pos||`
  - `d_surf = ||surface(posW_pixel) - cam_pos||`
  - 条件：`d_probe + visibility_margin < d_surf`

- 碰撞/贴面（Collision Proximity）：
  - `dist = ||probe - surface||`
  - `signed = |dot(probe - surface, normal)|`
  - 条件：`dist <= collision_distance` 或 `signed <= collision_signed_epsilon`

4. 帧内有效条件：

- `frame_valid = visible_any & (~collision_any)`

即：至少被一个视角判为可见，且不能被任何视角判为碰撞/贴面。

### 3.2 RayQuery/BVH 复核（新增）

当前 `validity_method` 支持：

- `gbuffer`：仅用屏幕投影 + GBuffer 判定
- `rayquery_hybrid`：先 `gbuffer` 快速筛选，再做逐 probe RayQuery/BVH 复核（默认）
- `rayquery_full`：逐 probe 全量 RayQuery/BVH 判定

RayQuery 复核实现方式：

1. 额外创建 1x1 的 `GBufferRT` testbed（`ZeroDayRayQuery`）。
2. 对每条待测射线，设置相机原点与朝向，渲染 1x1。
3. 读取首个命中点 `posW`，计算命中距离 `hit_dist`。
4. 规则：
   - 碰撞：从 probe 向多方向发射短程射线，若 `hit_dist <= collision_distance` 则判碰撞。
   - 可见性：从验证相机向 probe 发射射线，若首命中在 probe 之后（或无命中）则可见。

说明：`GBufferRT` 内部使用 Falcor 场景求交路径（SceneRayQuery/RT 加速结构），因此复核阶段是 BVH/光追求交而非纯投影近似。

### 3.3 跨帧严格交集（all-frame intersection）

最终：

- `valid_mask = Π_t frame_valid[t]`（逐帧布尔交集）

只要某一验证帧失败，该 probe 就被标为无效。

## 4. 烘焙与输出（只烘焙有效探针）

- `valid_indices = np.where(valid_mask > 0.5)`
- 仅对 `valid_indices` 执行 SH 烘焙。
- 无效 probe 在 `tensor[P,M,27]` 中保持零值。

输出文件：

1. `parametric_tensor.npz`
   - `tensor` `[P, M, 27]`
   - `probe_positions` `[P, 3]`
  - `light_configs` `[M, N, 12]`（`N` 为受控光源数量；静态几何随机光照版写入每帧真实光源描述）
  - `light_mask` `[M, N]`
   - `valid_mask` `[P]`
   - `metadata`
2. `metadata.json`
   - 包含 `probe_validation` 与 `candidate_pool` 统计项

## 5. 与“严谨碰撞/可见性检查”诉求的对应关系

当前代码已满足：

- 多帧（时间维度）检查
- 多视角（空间覆盖）检查
- GBuffer 快速筛选
- RayQuery/BVH 逐 probe 复核（`rayquery_hybrid` / `rayquery_full`）
- 严格帧交集 `valid_mask`

当前代码**未实现**：

- 基于签名距离场（SDF）的解析 inside/outside 判定（当前使用多方向射线命中距离规则）

因此当前是“GBuffer + RayQuery/BVH 混合严格判定”，不是纯 SDF 几何体分析。

## 6. 推荐运行方式（当前仓库）

```bash
# Dry-run（检查命令拼装）
python -m tools.run_dataset \
  --config 1_data_generation/configs/zeroday_prod.yaml \
  --preset smoke --dry-run

# 实跑（需可用 Falcor Python 绑定和支持的 GPU 后端）
python -m tools.run_dataset \
  --config 1_data_generation/configs/zeroday_prod.yaml

# 静态几何 + 连续随机光照版
python -m tools.run_dataset \
  --config 1_data_generation/configs/zeroday_static_randomlights_prod.yaml \
  --preset smoke --dry-run

python -m tools.run_dataset \
  --config 1_data_generation/configs/zeroday_static_randomlights_prod.yaml
```

若本机 `import falcor` 失败，需要先配置 Falcor Python 路径（或在配置中指定 `falcor_python_path`）。
