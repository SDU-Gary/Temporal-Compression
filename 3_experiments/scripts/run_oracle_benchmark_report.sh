#!/usr/bin/env bash
set -euo pipefail

# 通用链路：
# 1) 对某个实验 checkpoint 跑 Oracle 诊断
# 2) 跑 benchmark 生成统一指标
# 3) 与 baseline 汇总对比报告
#
# 用法：
#   bash 3_experiments/scripts/run_oracle_benchmark_report.sh \
#     [EXP_DIR] [BASELINE_BENCH_SUMMARY] [BASELINE_ORACLE_SUMMARY]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

EXP_DIR="${1:-3_experiments/results/bistro_clean_v2/exp_A1_coeff_first_20260302_202419}"
BASELINE_BENCH_SUMMARY="${2:-3_experiments/results/bistro_clean_v2/exp_A_K30_r8_baseline/unified_metrics_run_20260301_140800/benchmark_summary.json}"
BASELINE_ORACLE_SUMMARY="${3:-3_experiments/results/bistro_clean_v2/exp_A_K30_r8_baseline/oracle_diag_cpu_verify_fix/summary.json}"

CHECKPOINT="${EXP_DIR}/best_model.pt"
EXP_ORACLE_SUMMARY="${EXP_DIR}/oracle_diag/summary.json"
EXP_ORACLE_PER_SAMPLE="${EXP_DIR}/oracle_diag/per_sample.csv"
REPORT_JSON="${EXP_DIR}/exp_vs_baseline_report.json"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] checkpoint not found: $CHECKPOINT"
  exit 1
fi

if [[ ! -f "$BASELINE_ORACLE_SUMMARY" ]]; then
  echo "[ERROR] baseline oracle summary not found: $BASELINE_ORACLE_SUMMARY"
  exit 1
fi

if [[ ! -f "$BASELINE_BENCH_SUMMARY" ]]; then
  echo "[ERROR] baseline benchmark summary not found: $BASELINE_BENCH_SUMMARY"
  exit 1
fi

echo "[1/3] Run Oracle diagnostics..."
python 3_experiments/scripts/analysis/run_sh_oracle_diagnostics.py \
  --data-root 1_data_generation/output/bistro_clean_v2 \
  --checkpoint "$CHECKPOINT" \
  --output "$EXP_ORACLE_SUMMARY" \
  --save-per-sample-csv "$EXP_ORACLE_PER_SAMPLE" \
  --split test \
  --device cuda

echo "[2/3] Run benchmark..."
FALCOR_PY="${FALCOR_PY:-/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/python}"
FALCOR_BIN="${FALCOR_BIN:-/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/pythondist/bin/python3.10}"
FALCOR_LIB="${FALCOR_LIB:-/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug}"
EXP_BENCH_OUT="${EXP_DIR}/unified_metrics_run_$(date +%Y%m%d_%H%M%S)"

VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}" \
PYTHONPATH="$FALCOR_PY" \
LD_LIBRARY_PATH="$FALCOR_LIB:$FALCOR_PY:${LD_LIBRARY_PATH:-}" \
python tools/benchmark_realtime_pipeline.py \
  --dataset 1_data_generation/output/bistro_clean_v2 \
  --scene 1_data_generation/falcor/scenes/bistro_exterior_emissive_spheres.pyscene \
  --route both \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$EXP_BENCH_OUT" \
  --falcor-python-path "$FALCOR_PY" \
  --falcor-python-bin "$FALCOR_BIN" \
  --device cuda \
  --warmup-frames 8 \
  --benchmark-frames 24 \
  --sync-every 1 \
  --compute-image-metrics \
  --save-frame-metrics-every 1 \
  --cosine-mode irradiance

echo "[3/3] Build experiment vs baseline report..."
python 3_experiments/scripts/analysis/report_experiment_vs_baseline.py \
  --baseline-oracle-summary "$BASELINE_ORACLE_SUMMARY" \
  --experiment-oracle-summary "$EXP_ORACLE_SUMMARY" \
  --baseline-benchmark-summary "$BASELINE_BENCH_SUMMARY" \
  --experiment-benchmark-summary "$EXP_BENCH_OUT/benchmark_summary.json" \
  --output-json "$REPORT_JSON"

echo ""
echo "Done. Outputs:"
echo "  Oracle summary:      $EXP_ORACLE_SUMMARY"
echo "  Benchmark output:    $EXP_BENCH_OUT"
echo "  Final report:        $REPORT_JSON"
