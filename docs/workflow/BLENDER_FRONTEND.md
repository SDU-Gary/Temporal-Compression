# Blender 前端工作流（PG-GCPL）

本指南描述如何把 Blender 作为交互前端，用于**发光球体轨迹设计**与**probe 采样约束**，而 Falcor 仍负责高质量渲染。

## 核心原则
- Blender 只负责交互与导出轨迹/配置，不负责真实渲染。
- Falcor 仍使用 `.pyscene` 作为场景真值源，保证材质与环境光一致。

## 输出文件
- `scene.json`：描述场景路径、帧数、fps、发光球体轨迹、渲染参数。
- `BallX_traj.npy`：每个球体的轨迹，形状 `[F, 3]`。

## 基本流程
1) 在 Blender 中导入 Bistro FBX 仅作参考。
2) 放置球体并设置关键帧动画（命名 `BallRed/Green/Blue`）。
3) 运行导出脚本：

```
blender -b your_scene.blend -P tools/blender/export_emissive_trajectories.py -- \
  --output-dir metadata/blender_exports/bistro \
  --scene 1_data_generation/falcor/scenes/bistro_exterior_emissive_spheres.pyscene
```

4) Falcor 生成时读取 JSON：

```
python 1_data_generation/falcor/generate_bistro_temporal_spheres.py \
  --scene-json metadata/blender_exports/bistro/scene.json \
  --output 1_data_generation/output/bistro_temporal/dev
```

## 约定
- 对象命名带颜色关键字（Red/Green/Blue），默认自动匹配颜色强度。
- 可在 Blender 对象自定义属性中设置：
  - `emissive_radius`
  - `emissive_color`
  - `emissive_intensity`

> 注意：`emissive_color` 建议为**单位色**（例如 [1,0,0]），强度请用 `emissive_intensity` 控制。
