#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

PY_BIN="${PY_BIN:-python}"
DATASET="${DATASET:-1_data_generation/output/bistro_clean_v2}"
SCENE="${SCENE:-1_data_generation/falcor/scenes/bistro_exterior_emissive_spheres.pyscene}"
OUT_ROOT="${OUT_ROOT:-3_experiments/results/full_suite_best_last_baseline}"
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"
BENCH_DEVICE="${BENCH_DEVICE:-cuda}"
AUDIT_DEVICE="${AUDIT_DEVICE:-cpu}"
AUDIT_BATCH_SIZE="${AUDIT_BATCH_SIZE:-512}"
AUDIT_MAX_SAMPLES="${AUDIT_MAX_SAMPLES:-8192}"
WARMUP_FRAMES="${WARMUP_FRAMES:-8}"
BENCHMARK_FRAMES="${BENCHMARK_FRAMES:-24}"
BENCH_SOFT="${BENCH_SOFT:-0}"
BENCH_SOFT_TOPK="${BENCH_SOFT_TOPK:-8}"
BENCH_SOFT_TEMP="${BENCH_SOFT_TEMP:-0.20}"
BENCH_ROUTING_PROFILE_JSON="${BENCH_ROUTING_PROFILE_JSON:-}"

if [[ -z "${PROFILES_JSON:-}" ]]; then
  PROFILES_JSON='{"profiles":[{"name":"hard3","top_k":3,"training_soft_routing":false},{"name":"soft8_t018","top_k":3,"training_soft_routing":true,"routing_soft_topk":8,"routing_temperature":0.18},{"name":"soft8_t020","top_k":3,"training_soft_routing":true,"routing_soft_topk":8,"routing_temperature":0.20},{"name":"soft8_t022","top_k":3,"training_soft_routing":true,"routing_soft_topk":8,"routing_temperature":0.22}]}'
fi

CHECKPOINTS=(
  "3_experiments/results/bistro_clean_v2/exp_A_K30_r8_baseline/best_model.pt"
  "3_experiments/results/bistro_clean_v2/exp_A_K30_r8_baseline/last_model.pt"
  "3_experiments/results/bistro_clean_v2/exp_A1_coeff_first_20260302_202419/best_model.pt"
  "3_experiments/results/bistro_clean_v2/exp_A1_coeff_first_20260302_202419/last_model.pt"
  "3_experiments/results/bistro_clean_v2/exp_A1_1_joint_lrs_ema/best_model.pt"
  "3_experiments/results/bistro_clean_v2/exp_A1_1_joint_lrs_ema/last_model.pt"
)

for ckpt in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "${ckpt}" ]]; then
    echo "[ERR] missing checkpoint: ${ckpt}" >&2
    exit 1
  fi
done

CMD=(
  "${PY_BIN}" "3_experiments/scripts/analysis/run_checkpoint_full_suite.py"
  "--dataset" "${DATASET}"
  "--scene" "${SCENE}"
  "--output-root" "${OUT_ROOT}"
  "--parallel-jobs" "${PARALLEL_JOBS}"
  "--device" "${BENCH_DEVICE}"
  "--route" "both"
  "--compute-image-metrics"
  "--warmup-frames" "${WARMUP_FRAMES}"
  "--benchmark-frames" "${BENCHMARK_FRAMES}"
  "--audit-split" "test"
  "--audit-device" "${AUDIT_DEVICE}"
  "--audit-batch-size" "${AUDIT_BATCH_SIZE}"
  "--audit-num-workers" "0"
  "--audit-max-samples" "${AUDIT_MAX_SAMPLES}"
  "--profiles-json" "${PROFILES_JSON}"
  "--semantic-reference" "3_experiments/results/bistro_clean_v2/exp_A_K30_r8_baseline/best_model.pt"
)

for ckpt in "${CHECKPOINTS[@]}"; do
  CMD+=("--checkpoint" "${ckpt}")
done

if [[ -n "${BENCH_ROUTING_PROFILE_JSON}" ]]; then
  CMD+=("--benchmark-routing-profile-json" "${BENCH_ROUTING_PROFILE_JSON}")
elif [[ "${BENCH_SOFT}" == "1" ]]; then
  CMD+=(
    "--benchmark-training-soft-routing"
    "--benchmark-routing-soft-topk" "${BENCH_SOFT_TOPK}"
    "--benchmark-routing-temperature" "${BENCH_SOFT_TEMP}"
  )
fi

echo "[run] ${CMD[*]}"
"${CMD[@]}"
