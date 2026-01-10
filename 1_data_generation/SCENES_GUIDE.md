# 使用Benedikt Bitterli场景库指南

## 场景库简介

**Benedikt Bitterli's Rendering Resources** 是计算机图形学研究领域最著名的开源场景库之一，包含32个高质量3D场景。

- **官方网站**: https://benedikt-bitterli.me/resources/
- **Mitsuba 3 Gallery**: https://mitsuba.readthedocs.io/en/stable/src/gallery.html
- **许可证**: 多数允许商业使用，部分不需要署名

## 推荐场景

### 适合多时刻光照研究的场景

| 场景名 | 类型 | 复杂度 | 适用阶段 | 原因 |
|--------|------|--------|----------|------|
| **Victorian House** 🅱️ | 室外建筑 | 高 | Milestone 2-3 | 复杂几何，丰富阴影变化 |
| **Spaceship** 🅱️ | 半室外 | 高 | Milestone 2-3 | 金属材质，镜面反射 |
| **Modern Hall** 🅱️ | 半室外走廊 | 中 | Milestone 1-2 | 直接/间接光照混合 |
| **Japanese Classroom** 🅱️ | 室内采光 | 中 | Milestone 1-2 | 窗户光照，室内外混合 |
| **Cornell Box** 🅱️ | 测试场景 | 低 | 验证 | 经典间接光照测试 |
| **Veach Ajar** 🅱️ | 测试场景 | 低 | 验证 | MIS算法验证 |

### 材质特性

- **漫反射主导**: Bathroom, Kitchen, Bedroom
- **镜面材质**: Spaceship, Car
- **复杂光照**: Water Caustic, Volumetric Caustic
- **简单测试**: Cornell Box, Material Ball

## 下载方式

### 方法1：手动下载（推荐）

1. 访问 [Mitsuba 3 Gallery](https://mitsuba.readthedocs.io/en/stable/src/gallery.html)
2. 找到标记🅱️的场景（表示来自Benedikt Bitterli）
3. 点击场景图片下方的下载链接
4. 解压到 `data_generation/scenes/` 目录

```bash
cd /home/kyrie/毕设/data_generation
mkdir -p scenes
cd scenes

# 下载示例（需要从Gallery页面获取实际链接）
wget https://rgl.s3.eu-central-1.amazonaws.com/scenes/cornell-box.zip
unzip cornell-box.zip
```

### 方法2：使用Python脚本（需要更新URL）

```bash
python download_scenes.py
```

**注意**: 需要先从Mitsuba Gallery手动获取最新的下载链接。

## 使用场景生成数据集

### 步骤1：下载场景

确保场景已下载到 `data_generation/scenes/` 目录：

```
data_generation/
├── scenes/
│   ├── cornell-box/
│   │   └── scene.xml
│   ├── veach-ajar/
│   │   └── scene.xml
│   └── ...
```

### 步骤2：修改配置

编辑 `generate_dataset_with_scene.py` 中的配置：

```python
config = {
    'scene_path': Path(__file__).parent / 'scenes' / 'cornell-box' / 'scene.xml',
    'num_probes': 100,      # 根据需要调整
    'num_moments': 2,
    'num_gt_images': 5,
    'num_sh_samples': 32,
    'spp': 128,
    'output_dir': Path(__file__).parent / 'output' / 'cornell_box_dataset'
}
```

### 步骤3：运行数据生成

```bash
source ../venv/bin/activate
python generate_dataset_with_scene.py
```

## 场景适配注意事项

### 1. 光源修改

**问题**: 外部场景的光源配置可能与任务需求不匹配。

**解决方案**:

需要修改场景XML文件中的光源：

```xml
<!-- 原始场景可能有各种光源 -->
<emitter type="area">
  ...
</emitter>

<!-- 需要添加或替换为定向光（太阳） -->
<emitter type="directional" id="sun">
  <vector name="direction" value="0, 0.866, -0.5"/>  <!-- 可通过代码动态修改 -->
  <rgb name="irradiance" value="2.0, 1.9, 1.7"/>
</emitter>
```

**代码层面修改**（需要实现）：
- 解析XML找到directional光源
- 动态修改direction参数
- 或使用 `mi.load_dict()` + 场景编辑API

### 2. 场景边界分析

使用 `analyze_scene_bounds()` 函数自动获取场景大小：

```python
scene = mi.load_file("scenes/cornell-box/scene.xml")
bbox = scene.bbox()
print(f"Scene bounds: {bbox.min} to {bbox.max}")
```

根据边界框调整：
- 探针采样范围
- 相机位置范围
- 渲染距离

### 3. 探针密度

根据任务书要求：**1探针/m³**

示例计算：
- Cornell Box (约 1m × 1m × 1m) → 100探针
- Victorian House (约 20m × 20m × 10m = 4000m³) → 4000探针

```python
volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
num_probes = int(volume * 1.0)  # 1探针/m³
```

## 场景预览

### 快速渲染预览

```python
import mitsuba as mi

mi.set_variant('cuda_ad_rgb')

# 加载场景
scene = mi.load_file('scenes/cornell-box/scene.xml')

# 快速渲染
image = mi.render(scene, spp=64)

# 保存预览
mi.util.write_bitmap('preview.png', image)
```

### 查看场景信息

```python
import mitsuba as mi

scene = mi.load_file('scenes/cornell-box/scene.xml')

# 边界框
bbox = scene.bbox()
print(f"Bounding box: {bbox}")

# 传感器（相机）
for sensor in scene.sensors():
    print(f"Sensor: {sensor}")

# 光源
for emitter in scene.emitters():
    print(f"Emitter: {emitter}")
```

## 常见问题

### Q1: 场景太大，渲染很慢怎么办？

A: 分阶段降低质量：
- 减少探针数：15000 → 1000 → 100
- 降低SPP：256 → 128 → 64
- 减少SH采样：64 → 32 → 16
- 降低图像分辨率：1024 → 512 → 256

### Q2: 如何修改场景中的太阳方向？

A: 需要实现XML解析或使用场景编辑API（待实现）：

```python
# 方法1：修改XML文件（手动）
# 编辑 scene.xml，找到 <emitter type="directional">

# 方法2：使用Mitsuba 3场景编辑API（推荐，需实现）
params = mi.traverse(scene)
params['sun.direction'] = [0, 0.866, -0.5]  # 根据实际参数名调整
params.update()
```

### Q3: Cornell Box等简单场景是否足够？

A:
- **Milestone 1验证**: ✓ 可以用Cornell Box快速验证流程
- **Milestone 2-3训练**: ✗ 需要更复杂场景（Victorian House等）
- **论文实验**: ✗ 需要多个不同复杂度场景

### Q4: 下载链接失效怎么办？

A:
1. 访问官方 [Mitsuba 3 Gallery](https://mitsuba.readthedocs.io/en/stable/src/gallery.html)
2. 手动查找场景下载链接
3. 更新 `download_scenes.py` 中的URL

或直接从 [Benedikt Bitterli官网](https://benedikt-bitterli.me/resources/) 下载Tungsten格式，然后转换（需要额外工具）。

## 下一步

1. **选择1-2个适合的场景** 下载并测试
2. **实现光源修改功能** 使太阳方向可编程控制
3. **运行快速原型** 验证外部场景数据生成流程
4. **扩展到完整数据集** 使用多个场景 × 多时刻

## 参考资源

- [Benedikt Bitterli's Resources](https://benedikt-bitterli.me/resources/)
- [Mitsuba 3 Gallery](https://mitsuba.readthedocs.io/en/stable/src/gallery.html)
- [Mitsuba 3 Scene Format](https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html)
- [Mitsuba 3 Python API](https://mitsuba.readthedocs.io/en/stable/src/api_reference.html)
