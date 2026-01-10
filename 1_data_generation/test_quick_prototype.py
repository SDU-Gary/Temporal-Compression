"""
快速原型测试脚本

自动运行数据生成并验证输出质量
"""

import sys
import os
from pathlib import Path
import time

# 添加utils到Python路径
sys.path.insert(0, str(Path(__file__).parent / "utils"))

import numpy as np


def check_cuda_availability():
    """检查CUDA是否可用"""
    print("="*60)
    print("1. Checking CUDA availability...")
    print("="*60)

    import mitsuba as mi

    variants = mi.variants()
    print(f"Available Mitsuba variants: {variants}")

    has_cuda = any('cuda' in v for v in variants)

    if has_cuda:
        try:
            mi.set_variant('cuda_ad_rgb')
            print("✓ CUDA variant successfully activated")
            return True
        except Exception as e:
            print(f"✗ CUDA variant failed: {e}")
            return False
    else:
        print("✗ CUDA variants not available")
        print("  Will use LLVM variant instead (slower but functional)")
        return False


def run_data_generation():
    """运行数据生成"""
    print("\n" + "="*60)
    print("2. Running data generation...")
    print("="*60)

    # 导入并运行main函数
    from generate_dataset import main

    start_time = time.time()

    try:
        main()
        elapsed = time.time() - start_time
        print(f"\n✓ Data generation completed in {elapsed/60:.1f} minutes")
        return True
    except Exception as e:
        print(f"\n✗ Data generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_output():
    """验证输出数据"""
    print("\n" + "="*60)
    print("3. Validating output data...")
    print("="*60)

    output_dir = Path(__file__).parent / 'output' / 'quick_prototype'

    if not output_dir.exists():
        print(f"✗ Output directory not found: {output_dir}")
        return False

    # 检查探针文件
    probe_file = output_dir / "probes.npz"
    if not probe_file.exists():
        print("✗ probes.npz not found")
        return False

    probes = np.load(probe_file)
    probe_positions = probes['positions']
    print(f"✓ Loaded {len(probe_positions)} probe positions")
    print(f"  Shape: {probe_positions.shape}")
    print(f"  Bounds: X[{probe_positions[:, 0].min():.1f}, {probe_positions[:, 0].max():.1f}], "
          f"Y[{probe_positions[:, 1].min():.1f}, {probe_positions[:, 1].max():.1f}], "
          f"Z[{probe_positions[:, 2].min():.1f}, {probe_positions[:, 2].max():.1f}]")

    # 检查每个时刻的数据
    moments_found = []
    for moment_id in [12, 18]:
        moment_dir = output_dir / f"moment_{moment_id:02d}"

        if not moment_dir.exists():
            print(f"✗ Moment {moment_id} directory not found")
            continue

        # 检查球谐系数
        sh_file = moment_dir / "sh_coeffs.npz"
        if not sh_file.exists():
            print(f"✗ Moment {moment_id}: sh_coeffs.npz not found")
            continue

        sh_data = np.load(sh_file)
        sh_coeffs = sh_data['coeffs']
        sun_dir = sh_data['sun_dir']

        print(f"\n✓ Moment {moment_id}:00")
        print(f"  SH coefficients shape: {sh_coeffs.shape}")
        print(f"  Expected: ({len(probe_positions)}, 27)")
        print(f"  Sun direction: [{sun_dir[0]:.3f}, {sun_dir[1]:.3f}, {sun_dir[2]:.3f}]")
        print(f"  Sun altitude: {np.degrees(np.arcsin(sun_dir[1])):.1f}°")

        # 检查SH系数的统计信息
        print(f"  SH coeffs stats:")
        print(f"    Mean: {sh_coeffs.mean():.6f}")
        print(f"    Std:  {sh_coeffs.std():.6f}")
        print(f"    Min:  {sh_coeffs.min():.6f}")
        print(f"    Max:  {sh_coeffs.max():.6f}")

        # 检查图像
        images_dir = moment_dir / "images"
        if images_dir.exists():
            exr_files = list(images_dir.glob("*.exr"))
            png_files = list(images_dir.glob("*.png"))
            print(f"  Images: {len(exr_files)} EXR, {len(png_files)} PNG")
        else:
            print(f"  ✗ Images directory not found")

        moments_found.append(moment_id)

    if len(moments_found) == 2:
        print(f"\n✓ All {len(moments_found)} moments validated successfully")
        return True
    else:
        print(f"\n✗ Only {len(moments_found)}/2 moments found")
        return False


def estimate_sh_quality():
    """估算球谐重建质量"""
    print("\n" + "="*60)
    print("4. Estimating SH reconstruction quality...")
    print("="*60)

    from spherical_harmonics import (
        reconstruct_from_sh,
        fibonacci_sphere
    )

    output_dir = Path(__file__).parent / 'output' / 'quick_prototype'

    # 加载第一个时刻的数据
    sh_file = output_dir / "moment_12" / "sh_coeffs.npz"
    if not sh_file.exists():
        print("✗ Cannot estimate quality: sh_coeffs.npz not found")
        return False

    sh_data = np.load(sh_file)
    sh_coeffs_array = sh_data['coeffs']

    # 随机选择几个探针
    sample_indices = np.random.choice(len(sh_coeffs_array), min(5, len(sh_coeffs_array)), replace=False)

    print(f"Sampling {len(sample_indices)} random probes for quality check...")

    # 测试方向
    test_directions = fibonacci_sphere(64)

    total_energy = 0
    for idx in sample_indices:
        sh_coeffs = sh_coeffs_array[idx]

        # 重建辐射度
        reconstructed = reconstruct_from_sh(sh_coeffs, test_directions, max_order=2)

        # 计算总能量（辐射度的平均值）
        energy = reconstructed.mean()
        total_energy += energy

        print(f"  Probe {idx}: Mean radiance = {energy:.4f}, "
              f"Max = {reconstructed.max():.4f}, "
              f"Min = {reconstructed.min():.4f}")

    avg_energy = total_energy / len(sample_indices)
    print(f"\n✓ Average energy across samples: {avg_energy:.4f}")

    if avg_energy > 0.01:
        print("✓ SH coefficients appear valid (non-zero energy)")
        return True
    else:
        print("⚠ Warning: Very low energy detected, may indicate rendering issues")
        return False


def print_next_steps():
    """打印后续步骤"""
    print("\n" + "="*60)
    print("Next Steps")
    print("="*60)
    print("""
✓ Quick prototype generation successful!

You can now:

1. Inspect the generated data:
   cd data_generation/output/quick_prototype
   ls -lh

2. View rendered images:
   - Check PNG files in: moment_12/images/ and moment_18/images/

3. Load and visualize in Python:
   ```python
   import numpy as np
   probes = np.load('output/quick_prototype/probes.npz')
   sh_data = np.load('output/quick_prototype/moment_12/sh_coeffs.npz')
   print(probes['positions'].shape)  # [100, 3]
   print(sh_data['coeffs'].shape)    # [100, 27]
   ```

4. Scale up to full dataset:
   - Edit generate_dataset.py config:
     'num_probes': 500 (or 2000, or 5000)
     'num_moments': 3 (or 8, or 24)
     'num_sh_samples': 64
     'spp': 256

5. Train Milestone 1 model:
   - Use this data to train single-moment 4-level spatial hierarchy
   - Target: PSNR > 40dB

Need help? Check data_generation/README.md
""")


def main():
    """主测试流程"""
    print("\n" + "="*60)
    print("QUICK PROTOTYPE TEST - Multi-Moment Lighting Dataset")
    print("="*60)
    print("\nThis will:")
    print("  1. Check CUDA availability")
    print("  2. Generate quick prototype dataset (100 probes, 2 moments)")
    print("  3. Validate output data")
    print("  4. Estimate quality")
    print("\nEstimated time: 10-15 minutes\n")

    # Step 1: Check CUDA
    has_cuda = check_cuda_availability()

    # Step 2: Run generation
    success = run_data_generation()

    if not success:
        print("\n✗ Test failed at data generation stage")
        return 1

    # Step 3: Validate
    valid = validate_output()

    if not valid:
        print("\n✗ Test failed at validation stage")
        return 1

    # Step 4: Quality estimation
    quality_ok = estimate_sh_quality()

    # Final summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"CUDA available:     {'✓' if has_cuda else '✗ (using LLVM)'}")
    print(f"Data generation:    {'✓' if success else '✗'}")
    print(f"Data validation:    {'✓' if valid else '✗'}")
    print(f"Quality check:      {'✓' if quality_ok else '⚠'}")
    print("="*60)

    if success and valid:
        print("\n🎉 QUICK PROTOTYPE TEST PASSED!\n")
        print_next_steps()
        return 0
    else:
        print("\n✗ Test failed, please check errors above\n")
        return 1


if __name__ == "__main__":
    exit(main())
