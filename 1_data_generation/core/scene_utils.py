"""
场景工具模块

提供场景分析、发现和元信息提取功能
"""

from pathlib import Path
from typing import Tuple, List, Dict, TYPE_CHECKING
import xml.etree.ElementTree as ET
import mitsuba as mi

if TYPE_CHECKING:
    from mitsuba import Scene


def analyze_scene_bounds(scene: 'Scene') -> Tuple[float, float, float, float, float, float]:
    """
    分析场景边界框

    Args:
        scene: 已加载的Mitsuba场景对象

    Returns:
        bounds: (x_min, x_max, y_min, y_max, z_min, z_max) 场景边界

    Example:
        >>> scene = mi.load_file('scenes/cornell-box/scene.xml')
        >>> bounds = analyze_scene_bounds(scene)
        >>> print(bounds)
        (-1.0, 1.0, -1.0, 1.0, 0.0, 2.0)
    """
    bbox = scene.bbox()

    x_min, y_min, z_min = bbox.min
    x_max, y_max, z_max = bbox.max

    return (x_min, x_max, y_min, y_max, z_min, z_max)


def get_scene_volume(bounds: Tuple[float, float, float, float, float, float]) -> float:
    """
    计算场景体积

    Args:
        bounds: (x_min, x_max, y_min, y_max, z_min, z_max)

    Returns:
        volume: 场景体积（立方单位）
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    return (x_max - x_min) * (y_max - y_min) * (z_max - z_min)


def discover_scenes(scene_dir: Path) -> List[Dict]:
    """
    递归扫描目录，发现所有场景文件

    Args:
        scene_dir: 场景根目录

    Returns:
        scenes: 场景信息列表
            [
                {
                    'path': Path对象,
                    'name': 场景名称,
                    'size_lines': XML文件行数,
                    'has_envmap': 是否有环境贴图,
                    'has_directional': 是否有定向光,
                    'num_area_lights': 面光源数量
                },
                ...
            ]

    Example:
        >>> scenes = discover_scenes(Path('scenes'))
        >>> for s in scenes:
        ...     print(f"{s['name']}: {s['path']}")
    """
    if not scene_dir.exists():
        return []

    scenes = []
    for xml_file in scene_dir.rglob('scene.xml'):
        info = analyze_scene_file(xml_file)
        scenes.append(info)

    return sorted(scenes, key=lambda x: x['name'])


def analyze_scene_file(xml_path: Path) -> Dict:
    """
    分析场景XML文件，提取元信息

    Args:
        xml_path: 场景XML文件路径

    Returns:
        info: 场景信息字典
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 检测光源类型
        envmaps = root.findall(".//emitter[@type='envmap']")
        directionals = root.findall(".//emitter[@type='directional']")
        area_lights = root.findall(".//emitter[@type='area']")

        # 统计XML行数
        with open(xml_path, 'r') as f:
            num_lines = sum(1 for _ in f)

        return {
            'path': xml_path,
            'name': xml_path.parent.name,
            'size_lines': num_lines,
            'has_envmap': len(envmaps) > 0,
            'has_directional': len(directionals) > 0,
            'num_area_lights': len(area_lights)
        }

    except Exception as e:
        return {
            'path': xml_path,
            'name': xml_path.parent.name,
            'size_lines': 0,
            'has_envmap': False,
            'has_directional': False,
            'num_area_lights': 0,
            'error': str(e)
        }


def print_scene_summary(scenes: List[Dict]):
    """
    打印场景摘要

    Args:
        scenes: discover_scenes()返回的场景列表
    """
    print(f"\nFound {len(scenes)} scenes:\n")
    print(f"{'Name':<20} {'Size':<10} {'Envmap':<10} {'Direct':<10} {'Area Lights':<12}")
    print("-" * 72)

    for scene in scenes:
        name = scene['name']
        size = f"{scene['size_lines']} lines"
        envmap = "✓" if scene['has_envmap'] else "✗"
        direct = "✓" if scene['has_directional'] else "✗"
        area = str(scene['num_area_lights'])

        print(f"{name:<20} {size:<10} {envmap:<10} {direct:<10} {area:<12}")

    print()


def print_scene_bounds_info(scene_path: Path):
    """
    打印场景边界信息

    Args:
        scene_path: 场景XML文件路径
    """
    scene = mi.load_file(str(scene_path))
    bounds = analyze_scene_bounds(scene)
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    volume = get_scene_volume(bounds)

    print(f"\nScene: {scene_path.parent.name}")
    print(f"  Bounding box:")
    print(f"    X: [{x_min:.2f}, {x_max:.2f}]  (width: {x_max - x_min:.2f})")
    print(f"    Y: [{y_min:.2f}, {y_max:.2f}]  (depth: {y_max - y_min:.2f})")
    print(f"    Z: [{z_min:.2f}, {z_max:.2f}]  (height: {z_max - z_min:.2f})")
    print(f"  Volume: {volume:.2f} m³")

    # 根据体积建议探针数量（1探针/m³）
    recommended_probes = int(volume * 1.0)
    print(f"  Recommended probes (1/m³): {recommended_probes}")
    print()
