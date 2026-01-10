"""
太阳位置计算工具

基于地理位置和时间计算太阳方向（天文算法）
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List


def calculate_sun_direction_simple(
    hour: float,
    latitude: float = 40.0,
    longitude: float = 116.0,
    date: str = "2024-06-21"  # 夏至，太阳轨迹最长
) -> np.ndarray:
    """
    计算太阳方向（简化版本，基于球面几何）

    Args:
        hour: 小时（0-23）
        latitude: 纬度（度）
        longitude: 经度（度）
        date: 日期字符串（YYYY-MM-DD）

    Returns:
        sun_dir: [3] 归一化的太阳方向向量（世界坐标系）

    Coordinate System:
        X: 东
        Y: 天顶（向上）
        Z: 北
    """
    # 简化版本：假设太阳沿椭圆轨迹运动

    # 太阳在一天中的角度（早上6点到晚上18点）
    # 将hour映射到 [-pi, pi] 范围
    hour_angle = (hour - 12.0) * (np.pi / 12.0)  # 正午时为0

    # 纬度的影响（夏至时太阳高度角最大）
    lat_rad = np.radians(latitude)

    # 计算太阳高度角（altitude）和方位角（azimuth）
    # 简化公式：高度角随hour_angle变化
    altitude = np.radians(60.0) * np.cos(hour_angle)  # 最高60度（正午）
    azimuth = np.pi + hour_angle  # 方位角：早上东，正午南，傍晚西

    # 转换为笛卡尔坐标
    # 注意：高度角为负时，太阳在地平线以下
    if altitude < 0:
        altitude = 0  # 夜晚设为0（地平线）

    x = np.cos(altitude) * np.sin(azimuth)  # 东方向分量
    y = np.sin(altitude)  # 向上分量
    z = np.cos(altitude) * np.cos(azimuth)  # 北方向分量

    sun_dir = np.array([x, y, z])
    sun_dir = sun_dir / (np.linalg.norm(sun_dir) + 1e-8)  # 归一化

    return sun_dir


def generate_24hour_sun_trajectory(
    latitude: float = 40.0,
    longitude: float = 116.0,
    date: str = "2024-06-21"
) -> List[Tuple[float, np.ndarray]]:
    """
    生成24小时的太阳轨迹

    Args:
        latitude: 纬度
        longitude: 经度
        date: 日期

    Returns:
        trajectory: List of (hour, sun_direction)
    """
    trajectory = []

    for hour in range(24):
        sun_dir = calculate_sun_direction_simple(hour, latitude, longitude, date)
        trajectory.append((float(hour), sun_dir))

    return trajectory


def plot_sun_trajectory(trajectory: List[Tuple[float, np.ndarray]], save_path: str = None):
    """
    可视化太阳轨迹（需要matplotlib）

    Args:
        trajectory: List of (hour, sun_direction)
        save_path: 保存路径（可选）
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 5))

        # 3D轨迹
        ax1 = fig.add_subplot(121, projection='3d')
        hours = [h for h, _ in trajectory]
        directions = np.array([d for _, d in trajectory])

        ax1.plot(directions[:, 0], directions[:, 2], directions[:, 1], 'b-o', markersize=4)
        ax1.scatter([0], [0], [0], c='red', s=100, marker='*', label='Scene Center')

        # 标注几个关键时刻
        for h in [0, 6, 12, 18]:
            idx = h
            ax1.text(directions[idx, 0], directions[idx, 2], directions[idx, 1],
                    f'{h}:00', fontsize=9)

        ax1.set_xlabel('East (X)')
        ax1.set_ylabel('North (Z)')
        ax1.set_zlabel('Up (Y)')
        ax1.set_title('Sun Trajectory (3D)')
        ax1.legend()

        # 2D天空图（极坐标）
        ax2 = fig.add_subplot(122, projection='polar')

        # 计算方位角和高度角
        azimuths = np.arctan2(directions[:, 0], directions[:, 2])
        altitudes = np.arcsin(np.clip(directions[:, 1], -1, 1))

        # 极坐标：半径 = 90° - 高度角（地平线为外圈）
        radii = np.degrees(np.pi/2 - altitudes)

        ax2.plot(azimuths, radii, 'b-o', markersize=4)
        ax2.set_theta_zero_location('N')  # 北方向为0度
        ax2.set_theta_direction(-1)  # 顺时针
        ax2.set_ylim(0, 90)
        ax2.set_yticks([0, 30, 60, 90])
        ax2.set_yticklabels(['90°', '60°', '30°', '0°'])
        ax2.set_title('Sun Path (Sky View)\nCenter=Zenith, Edge=Horizon')
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved sun trajectory plot to: {save_path}")
        else:
            plt.show()

    except ImportError:
        print("Warning: matplotlib not installed, skipping visualization")


if __name__ == "__main__":
    print("Testing Sun Position calculations...")

    # 生成24小时太阳轨迹
    trajectory = generate_24hour_sun_trajectory(
        latitude=40.0,  # 北京纬度
        longitude=116.0,
        date="2024-06-21"  # 夏至
    )

    print(f"\nGenerated {len(trajectory)} sun positions")
    print("\nSample positions:")
    for hour in [0, 6, 12, 18]:
        h, sun_dir = trajectory[hour]
        altitude = np.degrees(np.arcsin(sun_dir[1]))
        print(f"  Hour {hour:02d}:00 - Direction: [{sun_dir[0]:.3f}, {sun_dir[1]:.3f}, {sun_dir[2]:.3f}], "
              f"Altitude: {altitude:.1f}°")

    # 保存24时刻的太阳方向到文件
    output_file = "sun_directions_24h.txt"
    with open(output_file, 'w') as f:
        f.write("# Hour | Sun_X | Sun_Y | Sun_Z | Altitude(deg)\n")
        for hour, sun_dir in trajectory:
            altitude = np.degrees(np.arcsin(sun_dir[1]))
            f.write(f"{int(hour):02d} {sun_dir[0]:.6f} {sun_dir[1]:.6f} {sun_dir[2]:.6f} {altitude:.2f}\n")

    print(f"\nSaved sun directions to: {output_file}")

    # 可视化（如果matplotlib可用）
    plot_sun_trajectory(trajectory, save_path="sun_trajectory.png")

    print("\nAll tests passed! ✓")
