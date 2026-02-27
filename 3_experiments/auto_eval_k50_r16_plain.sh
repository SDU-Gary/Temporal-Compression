#!/bin/bash
# Auto-evaluation script for K50_r16_plain

set -e

CHECKPOINT="3_experiments/results/bistro_clean_v2/unified_set_K50_r16_plain/best_model.pt"
OUTPUT_DIR="3_experiments/results/bistro_clean_v2/unified_set_K50_r16_plain"

echo "=== Waiting for training to complete ==="
while [ ! -f "$CHECKPOINT" ]; do
    echo "Checkpoint not found, waiting..."
    sleep 300  # Check every 5 minutes
done

echo "=== Checkpoint found! Starting evaluation ==="

# 1. Evaluate SH coefficients
echo "Step 1: Evaluating SH coefficient errors..."
python 3_experiments/scripts/eval.py \
    --data-root 1_data_generation/output/bistro_clean_v2 \
    --checkpoint "$CHECKPOINT" \
    --split test

# 2. Benchmark rendered images
echo "Step 2: Benchmarking rendered image PSNR..."
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
PYTHONPATH="Falcor/build/linux-gcc/bin/Debug/python" \
LD_LIBRARY_PATH="Falcor/build/linux-gcc/bin/Debug:Falcor/build/linux-gcc/bin/Debug/python:${LD_LIBRARY_PATH:-}" \
python tools/benchmark_realtime_pipeline.py \
    --dataset 1_data_generation/output/bistro_clean_v2 \
    --scene 1_data_generation/falcor/scenes/bistro_exterior_emissive_spheres.pyscene \
    --route both \
    --checkpoint "$CHECKPOINT" \
    --output-dir "${OUTPUT_DIR}/benchmark_final" \
    --falcor-python-path Falcor/build/linux-gcc/bin/Debug/python \
    --falcor-python-bin Falcor/build/linux-gcc/bin/Debug/pythondist/bin/python3.10 \
    --width 1280 --height 720 \
    --warmup-frames 4 --benchmark-frames 24 \
    --sync-every 1 \
    --compute-image-metrics \
    --save-frame-metrics-every 1 \
    --cosine-mode irradiance

# 3. Extract and display results
echo "Step 3: Extracting results..."
python3 << 'EOF'
import json

# Load results
eval_data = json.load(open("3_experiments/results/bistro_clean_v2/unified_set_K50_r16_plain/eval.json"))
bench_data = json.load(open("3_experiments/results/bistro_clean_v2/unified_set_K50_r16_plain/benchmark_final/benchmark_summary.json"))

print("\n" + "="*80)
print("K50_r16_plain RESULTS")
print("="*80)
print(f"\nSH Coefficient Errors:")
print(f"  MAE:  {eval_data['mae']:.6f}")
print(f"  RMSE: {eval_data['rmse']:.6f}")

im = bench_data["image_metrics"]
print(f"\nRendered Image Quality (GT vs Model):")
print(f"  sRGB PSNR:   {im['mean_psnr']:.2f} dB")
print(f"  sRGB SSIM:   {im['mean_ssim']:.4f}")
print(f"  Linear PSNR: {im['mean_linear_psnr']:.2f} dB")

print(f"\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"  K30_r8 baseline:     20-25 dB")
print(f"  K50_r16 (failed):    15.6 dB")
print(f"  K50_r16_plain (new): {im['mean_psnr']:.2f} dB")

if im['mean_psnr'] >= 25:
    print(f"\n✓ SUCCESS! Capacity increase helps (+{im['mean_psnr']-22.5:.1f} dB)")
    print("  Next: Run Experiment B (weighted_fixed)")
elif im['mean_psnr'] >= 20:
    print(f"\n~ MARGINAL: Similar to baseline")
    print("  Capacity alone doesn't help much")
else:
    print(f"\n✗ FAILURE: Still worse than baseline")
    print("  Architecture needs fundamental changes")
EOF

echo ""
echo "=== Evaluation complete! ==="
