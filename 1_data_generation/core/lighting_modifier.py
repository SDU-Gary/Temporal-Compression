"""
光源自动修改模块

实现混合方案：
1. 优先尝试traverse API修改现有directional光源
2. 对envmap场景，修改旋转角度
3. 最终回退：通过XML解析添加directional光源
"""

from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
import mitsuba as mi
import xml.etree.ElementTree as ET
import tempfile
import warnings

if TYPE_CHECKING:
    from mitsuba import Scene


class LightingModifier:
    """光源修改器类"""

    @staticmethod
    def modify_scene_lighting(
        scene_path: Path,
        sun_direction: np.ndarray,
        sun_intensity: Optional[np.ndarray] = None
    ) -> 'Scene':
        """
        自动修改场景光照以匹配给定的太阳方向

        混合方案策略：
        1. 尝试traverse API修改现有directional光源（高效）
        2. 如果场景有envmap，修改其旋转角度（特殊优化）
        3. 最终回退：XML解析添加directional光源（通用但慢）

        Args:
            scene_path: 场景XML文件路径
            sun_direction: [3] 太阳方向向量（会被自动归一化）
            sun_intensity: [3] 太阳光强度RGB，默认为[2.0, 1.9, 1.7]

        Returns:
            scene: 修改后的Mitsuba场景对象

        Example:
            >>> scene = LightingModifier.modify_scene_lighting(
            ...     Path('scenes/classroom/scene.xml'),
            ...     sun_direction=np.array([0, 0.866, -0.5])
            ... )
        """
        if sun_intensity is None:
            sun_intensity = np.array([2.0, 1.9, 1.7])

        # 归一化太阳方向
        sun_direction = sun_direction / np.linalg.norm(sun_direction)

        # 1. 加载场景
        scene = mi.load_file(str(scene_path))

        # 2. 策略1：尝试traverse修改
        if LightingModifier._try_traverse_modify(scene, sun_direction, sun_intensity):
            return scene

        # 3. 策略2：检测envmap并修改旋转（适用于classroom/house）
        if LightingModifier._has_envmap(scene_path):
            angle = LightingModifier._compute_angle_from_direction(sun_direction)
            return LightingModifier._modify_envmap_rotation(scene_path, angle)

        # 4. 策略3：XML添加directional光源（最终回退）
        return LightingModifier._xml_add_directional(scene_path, sun_direction, sun_intensity)

    @staticmethod
    def _try_traverse_modify(
        scene: 'Scene',
        sun_direction: np.ndarray,
        sun_intensity: np.ndarray
    ) -> bool:
        """
        尝试使用traverse API修改现有directional光源

        Args:
            scene: 已加载的场景
            sun_direction: 归一化的太阳方向
            sun_intensity: 太阳强度

        Returns:
            bool: 成功返回True，失败返回False
        """
        try:
            params = mi.traverse(scene)

            # 搜索directional光源相关参数
            # 可能的参数名：sun.direction, sun.to_world, sun.irradiance等
            sun_keys = [k for k in params.keys() if 'sun' in k.lower() or 'directional' in k.lower()]

            if not sun_keys:
                return False  # 没有directional光源

            # 尝试修改方向参数
            direction_modified = False

            # 优先尝试to_world变换
            if any('to_world' in k for k in sun_keys):
                # 注意：Mitsuba的directional光源需要从原点指向太阳的反方向
                params['sun.to_world'] = mi.ScalarTransform4f.look_at(
                    origin=[0, 0, 0],
                    target=-sun_direction,  # 负方向
                    up=[0, 0, 1]
                )
                direction_modified = True

            # 备选：直接修改direction向量
            elif any('direction' in k for k in sun_keys):
                direction_key = [k for k in sun_keys if 'direction' in k][0]
                params[direction_key] = sun_direction.tolist()
                direction_modified = True

            if not direction_modified:
                return False

            # 尝试修改强度参数
            if any('irradiance' in k for k in sun_keys):
                irradiance_key = [k for k in sun_keys if 'irradiance' in k][0]
                if 'value' in irradiance_key:
                    params[irradiance_key] = sun_intensity.tolist()
                else:
                    params[irradiance_key + '.value'] = sun_intensity.tolist()

            # 应用修改
            params.update()
            return True

        except Exception as e:
            warnings.warn(f"Traverse modification failed: {e}")
            return False

    @staticmethod
    def _has_envmap(scene_path: Path) -> bool:
        """检测场景是否包含envmap光源"""
        try:
            tree = ET.parse(scene_path)
            root = tree.getroot()
            envmaps = root.findall(".//emitter[@type='envmap']")
            return len(envmaps) > 0
        except Exception:
            return False

    @staticmethod
    def _compute_angle_from_direction(sun_direction: np.ndarray) -> float:
        """
        从太阳方向向量计算envmap旋转角度

        假设：envmap的Y轴旋转控制太阳方位角

        Args:
            sun_direction: [3] 太阳方向（x, y, z）

        Returns:
            angle: 旋转角度（度）
        """
        # 计算方位角（azimuth）：在XZ平面的投影角度
        azimuth = np.degrees(np.arctan2(sun_direction[0], -sun_direction[2]))

        # Mitsuba envmap通常使用-180到180度范围
        return azimuth

    @staticmethod
    def _modify_envmap_rotation(scene_path: Path, angle: float) -> 'Scene':
        """
        修改envmap的旋转角度

        Args:
            scene_path: 场景XML路径
            angle: 新的旋转角度（度）

        Returns:
            scene: 修改后的场景
        """
        tree = ET.parse(scene_path)
        root = tree.getroot()

        # 查找envmap emitter
        for emitter in root.findall(".//emitter[@type='envmap']"):
            # 查找或创建transform元素
            transform = emitter.find("transform[@name='to_world']")

            if transform is None:
                # 创建新的transform
                transform = ET.SubElement(emitter, 'transform', name='to_world')

            # 移除旧的rotate元素
            for rotate in list(transform):
                if rotate.tag == 'rotate':
                    transform.remove(rotate)

            # 添加新的rotate
            ET.SubElement(transform, 'rotate', y='1', angle=str(angle))

        # 使用临时文件保存，以便Mitsuba能正确解析相对路径
        import tempfile
        import os

        # 创建临时文件在原场景同目录（保持相对路径有效）
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.xml',
            dir=scene_path.parent,
            text=True
        )

        try:
            # 写入修改后的XML
            tree.write(temp_path, encoding='utf-8', xml_declaration=True)

            # 加载场景
            scene = mi.load_file(temp_path)

        finally:
            # 清理临时文件
            os.close(temp_fd)
            os.unlink(temp_path)

        return scene

    @staticmethod
    def _xml_add_directional(
        scene_path: Path,
        sun_direction: np.ndarray,
        sun_intensity: np.ndarray
    ) -> 'Scene':
        """
        通过XML解析添加directional光源

        这是最终回退方案，适用于只有area lights的场景

        Args:
            scene_path: 场景XML路径
            sun_direction: 太阳方向
            sun_intensity: 太阳强度

        Returns:
            scene: 修改后的场景
        """
        tree = ET.parse(scene_path)
        root = tree.getroot()

        # 移除已有的directional光源（避免冲突）
        for emitter in root.findall("emitter[@type='directional']"):
            root.remove(emitter)

        # 添加新的directional光源
        sun = ET.SubElement(root, 'emitter', type='directional', id='sun')

        # 设置方向
        sun_dir_str = f'{sun_direction[0]}, {sun_direction[1]}, {sun_direction[2]}'
        ET.SubElement(sun, 'vector', name='direction', value=sun_dir_str)

        # 设置强度
        sun_int_str = f'{sun_intensity[0]}, {sun_intensity[1]}, {sun_intensity[2]}'
        ET.SubElement(sun, 'rgb', name='irradiance', value=sun_int_str)

        # 使用临时文件保存，以便Mitsuba能正确解析相对路径（如纹理文件）
        import tempfile
        import os

        # 创建临时文件在原场景同目录（保持相对路径有效）
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.xml',
            dir=scene_path.parent,
            text=True
        )

        try:
            # 写入修改后的XML
            tree.write(temp_path, encoding='utf-8', xml_declaration=True)

            # 加载场景
            scene = mi.load_file(temp_path)

        finally:
            # 清理临时文件
            os.close(temp_fd)
            os.unlink(temp_path)

        return scene


def test_lighting_modifier():
    """测试光源修改器的基本功能"""
    print("Testing LightingModifier...")

    # 测试场景路径
    test_scenes = [
        Path('scenes/cornell-box/scene.xml'),
        Path('scenes/classroom/scene.xml'),
        Path('scenes/house/scene.xml')
    ]

    # 测试太阳方向
    sun_directions = [
        np.array([0, 1, 0]),      # 正午
        np.array([0.5, 0.866, 0]), # 早晨
        np.array([0, 0.5, -0.866]) # 傍晚
    ]

    for scene_path in test_scenes:
        if not scene_path.exists():
            print(f"  ✗ Scene not found: {scene_path}")
            continue

        print(f"\n  Testing {scene_path.name}:")

        for i, sun_dir in enumerate(sun_directions):
            try:
                scene = LightingModifier.modify_scene_lighting(
                    scene_path, sun_dir
                )
                print(f"    ✓ Sun direction {i+1}: Success")
            except Exception as e:
                print(f"    ✗ Sun direction {i+1}: {e}")

    print("\n✓ Testing completed")


if __name__ == "__main__":
    # 设置Mitsuba变体
    try:
        mi.set_variant('cuda_ad_rgb')
    except:
        mi.set_variant('llvm_ad_rgb')

    test_lighting_modifier()
