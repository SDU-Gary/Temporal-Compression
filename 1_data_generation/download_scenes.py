"""
下载Benedikt Bitterli场景库中的场景

使用方法：
    python download_scenes.py
"""

import urllib.request
import zipfile
from pathlib import Path
import os

# Mitsuba 3场景的下载链接（从Mitsuba Gallery获取）
# 注意：实际URL可能需要从 https://mitsuba.readthedocs.io/en/stable/src/gallery.html 确认
SCENES = {
    # 推荐的室外/半室外场景
    'veach-ajar': 'https://rgl.s3.eu-central-1.amazonaws.com/scenes/veach-ajar.zip',
    'bathroom': 'https://rgl.s3.eu-central-1.amazonaws.com/scenes/bathroom.zip',
    'cornell-box': 'https://rgl.s3.eu-central-1.amazonaws.com/scenes/cornell-box.zip',

    # 注意：以下URL是示例，需要从官方文档获取实际链接
    # 'victorian-house': 'https://...',
    # 'spaceship': 'https://...',
    # 'modern-hall': 'https://...',
}


def download_scene(name: str, url: str, output_dir: Path):
    """
    下载并解压场景

    Args:
        name: 场景名称
        url: 下载链接
        output_dir: 输出目录
    """
    scene_dir = output_dir / name

    # 检查是否已存在
    if scene_dir.exists():
        print(f"✓ {name} already exists, skipping...")
        return

    print(f"Downloading {name}...")

    # 下载zip文件
    zip_path = output_dir / f"{name}.zip"

    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"  Downloaded to {zip_path}")

        # 解压
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(scene_dir)
        print(f"  Extracted to {scene_dir}")

        # 删除zip文件
        zip_path.unlink()
        print(f"✓ {name} ready!")

    except Exception as e:
        print(f"✗ Failed to download {name}: {e}")


def list_scene_files(scene_dir: Path):
    """列出场景中的XML文件"""
    xml_files = list(scene_dir.glob("**/*.xml"))
    if xml_files:
        print(f"\n  Scene files in {scene_dir.name}:")
        for xml in xml_files:
            print(f"    - {xml.relative_to(scene_dir)}")


def main():
    """主函数"""
    # 设置输出目录
    output_dir = Path(__file__).parent / 'scenes'
    output_dir.mkdir(exist_ok=True)

    print("="*60)
    print("Downloading Benedikt Bitterli Scenes for Mitsuba 3")
    print("="*60)
    print(f"Output directory: {output_dir}\n")

    # 下载所有场景
    for name, url in SCENES.items():
        download_scene(name, url, output_dir)

    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)

    # 列出所有场景文件
    print("\nAvailable scenes:")
    for scene_name in SCENES.keys():
        scene_dir = output_dir / scene_name
        if scene_dir.exists():
            list_scene_files(scene_dir)


if __name__ == "__main__":
    main()
