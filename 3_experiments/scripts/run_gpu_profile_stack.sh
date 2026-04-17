#!/usr/bin/env bash
set -euo pipefail

# GPU profiling stack runner:
# 1) Nsight Systems (nsys) for system-level bottleneck attribution
# 2) Nsight Compute (ncu) for hotspot-kernel deep dive
#
# Output:
# - A run directory under 3_experiments/results/profiling/<run_tag>/
# - A concise artifact manifest: artifact_paths.txt

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/venv/bin/activate" ]]; then
  # Project convention: always run inside repo venv for reproducibility.
  # shellcheck disable=SC1091
  source "$ROOT_DIR/venv/bin/activate"
fi

CONFIG="${1:-3_experiments/configs/bistro_clean_train.yaml}"
RUN_TAG="${2:-$(date +%Y%m%d_%H%M%S)}"

if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] config not found: $CONFIG"
  exit 1
fi

if ! command -v nsys >/dev/null 2>&1; then
  echo "[ERROR] nsys not found in PATH"
  exit 1
fi
if ! command -v ncu >/dev/null 2>&1; then
  echo "[ERROR] ncu not found in PATH"
  exit 1
fi

OUT_DIR="3_experiments/results/profiling/${RUN_TAG}"
NSYS_DIR="${OUT_DIR}/nsys"
NCU_DIR="${OUT_DIR}/ncu"
EVENT_DIR="${OUT_DIR}/event"
mkdir -p "$NSYS_DIR" "$NCU_DIR" "$EVENT_DIR"

NSYS_BASE="${NSYS_DIR}/train_smoke"
NSYS_REP="${NSYS_BASE}.nsys-rep"
NSYS_STATS_TXT="${NSYS_DIR}/stats.txt"
NSYS_KEY_TXT="${NSYS_DIR}/key_lines.txt"

NCU_BASE="${NCU_DIR}/train_hotspot"
NCU_REP="${NCU_BASE}.ncu-rep"
NCU_SUMMARY_TXT="${NCU_DIR}/summary.txt"
NCU_DETAILS_TXT="${NCU_DIR}/details.txt"
NCU_KEY_TXT="${NCU_DIR}/key_lines.txt"
NCU_SESSION_TXT="${NCU_DIR}/session.txt"
NCU_RUN_LOG="${NCU_DIR}/run.log"
NCU_STATUS_TXT="${NCU_DIR}/status.txt"
NCU_QUERY_TXT="${NCU_DIR}/query_metrics.txt"

ARTIFACT_MANIFEST="${OUT_DIR}/artifact_paths.txt"

EPOCHS_NSYS="${EPOCHS_NSYS:-1}"
MAX_TRAIN_BATCHES_NSYS="${MAX_TRAIN_BATCHES_NSYS:-120}"
MAX_VAL_BATCHES_NSYS="${MAX_VAL_BATCHES_NSYS:-0}"
NUM_WORKERS_NSYS="${NUM_WORKERS_NSYS:-0}"
LINEARITY_EVERY_STEPS_NSYS="${LINEARITY_EVERY_STEPS_NSYS:-4}"
GRAD_NORM_LOG_EVERY_STEPS_NSYS="${GRAD_NORM_LOG_EVERY_STEPS_NSYS:-100}"
TRAIN_ROUTING_PARAM_MODE_NSYS="${TRAIN_ROUTING_PARAM_MODE_NSYS:-gather}"
CONTRACTION_MODE_NSYS="${CONTRACTION_MODE_NSYS:-fused}"
CUDA_GRAPH_TRAIN_NSYS="${CUDA_GRAPH_TRAIN_NSYS:-1}"
CUDA_GRAPH_MODE_NSYS="${CUDA_GRAPH_MODE_NSYS:-dual}"
CUDA_GRAPH_WARMUP_STEPS_NSYS="${CUDA_GRAPH_WARMUP_STEPS_NSYS:-10}"

EPOCHS_NCU="${EPOCHS_NCU:-1}"
MAX_TRAIN_BATCHES_NCU="${MAX_TRAIN_BATCHES_NCU:-80}"
MAX_VAL_BATCHES_NCU="${MAX_VAL_BATCHES_NCU:-0}"
NUM_WORKERS_NCU="${NUM_WORKERS_NCU:-0}"
LINEARITY_EVERY_STEPS_NCU="${LINEARITY_EVERY_STEPS_NCU:-4}"
GRAD_NORM_LOG_EVERY_STEPS_NCU="${GRAD_NORM_LOG_EVERY_STEPS_NCU:-100}"
TRAIN_ROUTING_PARAM_MODE_NCU="${TRAIN_ROUTING_PARAM_MODE_NCU:-gather}"
CONTRACTION_MODE_NCU="${CONTRACTION_MODE_NCU:-fused}"
CUDA_GRAPH_TRAIN_NCU="${CUDA_GRAPH_TRAIN_NCU:-1}"
CUDA_GRAPH_MODE_NCU="${CUDA_GRAPH_MODE_NCU:-dual}"
CUDA_GRAPH_WARMUP_STEPS_NCU="${CUDA_GRAPH_WARMUP_STEPS_NCU:-10}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-20}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-40}"
NCU_SET="${NCU_SET:-basic}"
NCU_USE_SUDO="${NCU_USE_SUDO:-0}"

CUDA_GRAPH_FLAG_NSYS="--no-cuda-graph-train"
if [[ "$CUDA_GRAPH_TRAIN_NSYS" == "1" || "$CUDA_GRAPH_TRAIN_NSYS" == "true" || "$CUDA_GRAPH_TRAIN_NSYS" == "TRUE" ]]; then
  CUDA_GRAPH_FLAG_NSYS="--cuda-graph-train"
fi

CUDA_GRAPH_FLAG_NCU="--no-cuda-graph-train"
if [[ "$CUDA_GRAPH_TRAIN_NCU" == "1" || "$CUDA_GRAPH_TRAIN_NCU" == "true" || "$CUDA_GRAPH_TRAIN_NCU" == "TRUE" ]]; then
  CUDA_GRAPH_FLAG_NCU="--cuda-graph-train"
fi

run_ncu() {
  if [[ "$NCU_USE_SUDO" == "1" ]]; then
    sudo -E env "PATH=$PATH" "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" ncu "$@"
    return $?
  fi
  ncu "$@"
}

echo "[profile-stack] root:   $ROOT_DIR"
echo "[profile-stack] config: $CONFIG"
echo "[profile-stack] run:    $RUN_TAG"
echo "[profile-stack] out:    $OUT_DIR"
echo ""

echo "[1/5] Preflight"
{
  echo "tool.nsys=$(command -v nsys)"
  echo "tool.ncu=$(command -v ncu)"
  echo "ncu.use_sudo=${NCU_USE_SUDO}"
  echo "nsys.linearity_every_steps=${LINEARITY_EVERY_STEPS_NSYS}"
  echo "nsys.grad_norm_log_every_steps=${GRAD_NORM_LOG_EVERY_STEPS_NSYS}"
  echo "nsys.train_routing_param_mode=${TRAIN_ROUTING_PARAM_MODE_NSYS}"
  echo "nsys.contraction_mode=${CONTRACTION_MODE_NSYS}"
  echo "nsys.cuda_graph_train=${CUDA_GRAPH_TRAIN_NSYS}"
  echo "nsys.cuda_graph_mode=${CUDA_GRAPH_MODE_NSYS}"
  echo "nsys.cuda_graph_warmup_steps=${CUDA_GRAPH_WARMUP_STEPS_NSYS}"
  echo "ncu.linearity_every_steps=${LINEARITY_EVERY_STEPS_NCU}"
  echo "ncu.grad_norm_log_every_steps=${GRAD_NORM_LOG_EVERY_STEPS_NCU}"
  echo "ncu.train_routing_param_mode=${TRAIN_ROUTING_PARAM_MODE_NCU}"
  echo "ncu.contraction_mode=${CONTRACTION_MODE_NCU}"
  echo "ncu.cuda_graph_train=${CUDA_GRAPH_TRAIN_NCU}"
  echo "ncu.cuda_graph_mode=${CUDA_GRAPH_MODE_NCU}"
  echo "ncu.cuda_graph_warmup_steps=${CUDA_GRAPH_WARMUP_STEPS_NCU}"
  nsys --version 2>/dev/null || true
  ncu --version 2>/dev/null || true
  python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch_cuda_runtime:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0:", torch.cuda.get_device_name(0))
PY
} | tee "${OUT_DIR}/preflight.txt"

echo ""
echo "[2/5] Run nsys profile"
nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  -o "$NSYS_BASE" \
  python 3_experiments/scripts/train.py \
    --config "$CONFIG" \
    --epochs "$EPOCHS_NSYS" \
    --num-workers "$NUM_WORKERS_NSYS" \
    --linearity-every-steps "$LINEARITY_EVERY_STEPS_NSYS" \
    --grad-norm-log-every-steps "$GRAD_NORM_LOG_EVERY_STEPS_NSYS" \
    --train-routing-param-mode "$TRAIN_ROUTING_PARAM_MODE_NSYS" \
    --contraction-mode "$CONTRACTION_MODE_NSYS" \
    "$CUDA_GRAPH_FLAG_NSYS" \
    --cuda-graph-mode "$CUDA_GRAPH_MODE_NSYS" \
    --cuda-graph-warmup-steps "$CUDA_GRAPH_WARMUP_STEPS_NSYS" \
    --max-train-batches "$MAX_TRAIN_BATCHES_NSYS" \
    --max-val-batches "$MAX_VAL_BATCHES_NSYS" \
    --no-auto-log \
    --no-progress \
    --no-profile-train

echo ""
echo "[3/5] Export nsys stats"
if ! nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,osrt_sum \
  "$NSYS_REP" >"$NSYS_STATS_TXT" 2>/dev/null; then
  nsys stats "$NSYS_REP" >"$NSYS_STATS_TXT" || true
fi

rg -n "CUDA.*Kernel|cudaLaunchKernel|cudaMemcpy|Memcpy|memcpy|cudaDeviceSynchronize|cudaStreamSynchronize" \
  "$NSYS_STATS_TXT" >"$NSYS_KEY_TXT" || true

echo ""
echo "[4/5] Run ncu hotspot profile"
NCU_PERMISSION_ERROR=0
NCU_RC=0

set +e
run_ncu --query-metrics --devices 0 >"$NCU_QUERY_TXT" 2>&1
NCU_QUERY_RC=$?
set -e
if rg -q "ERR_NVGPUCTRPERM" "$NCU_QUERY_TXT"; then
  NCU_PERMISSION_ERROR=1
  {
    echo "[ncu] query-metrics detected ERR_NVGPUCTRPERM."
    echo "[ncu] skip profile launch to avoid wasting a full training pass."
  } | tee "$NCU_RUN_LOG"
elif [[ "$NCU_QUERY_RC" -ne 0 ]]; then
  echo "[ncu] query-metrics failed (rc=${NCU_QUERY_RC}); continue and try profile launch." | tee "$NCU_RUN_LOG"
fi

if [[ "$NCU_PERMISSION_ERROR" != "1" ]]; then
  set +e
  run_ncu \
    --target-processes all \
    --set "$NCU_SET" \
    --kernel-name-base demangled \
    --launch-skip "$NCU_LAUNCH_SKIP" \
    --launch-count "$NCU_LAUNCH_COUNT" \
    --force-overwrite \
    -o "$NCU_BASE" \
    python 3_experiments/scripts/train.py \
      --config "$CONFIG" \
      --epochs "$EPOCHS_NCU" \
      --num-workers "$NUM_WORKERS_NCU" \
      --linearity-every-steps "$LINEARITY_EVERY_STEPS_NCU" \
      --grad-norm-log-every-steps "$GRAD_NORM_LOG_EVERY_STEPS_NCU" \
      --train-routing-param-mode "$TRAIN_ROUTING_PARAM_MODE_NCU" \
      --contraction-mode "$CONTRACTION_MODE_NCU" \
      "$CUDA_GRAPH_FLAG_NCU" \
      --cuda-graph-mode "$CUDA_GRAPH_MODE_NCU" \
      --cuda-graph-warmup-steps "$CUDA_GRAPH_WARMUP_STEPS_NCU" \
      --max-train-batches "$MAX_TRAIN_BATCHES_NCU" \
      --max-val-batches "$MAX_VAL_BATCHES_NCU" \
      --no-auto-log \
      --no-progress \
      --no-profile-train \
    2>&1 | tee -a "$NCU_RUN_LOG"
  NCU_RC=${PIPESTATUS[0]}
  set -e
fi

if rg -q "ERR_NVGPUCTRPERM" "$NCU_RUN_LOG"; then
  NCU_PERMISSION_ERROR=1
  echo "[ncu] permission error detected (ERR_NVGPUCTRPERM)."
elif rg -q "No metrics to collect found in sections" "$NCU_RUN_LOG"; then
  echo "[ncu] selected set '${NCU_SET}' returned no metrics; retry with explicit fallback sections."
  NCU_BASE="${NCU_DIR}/train_hotspot_fallback"
  NCU_REP="${NCU_BASE}.ncu-rep"
  set +e
  run_ncu \
    --target-processes all \
    --section LaunchStats \
    --section Occupancy \
    --section WorkloadDistribution \
    --kernel-name-base demangled \
    --launch-skip "$NCU_LAUNCH_SKIP" \
    --launch-count "$NCU_LAUNCH_COUNT" \
    --force-overwrite \
    -o "$NCU_BASE" \
    python 3_experiments/scripts/train.py \
      --config "$CONFIG" \
      --epochs "$EPOCHS_NCU" \
      --num-workers "$NUM_WORKERS_NCU" \
      --linearity-every-steps "$LINEARITY_EVERY_STEPS_NCU" \
      --grad-norm-log-every-steps "$GRAD_NORM_LOG_EVERY_STEPS_NCU" \
      --train-routing-param-mode "$TRAIN_ROUTING_PARAM_MODE_NCU" \
      --contraction-mode "$CONTRACTION_MODE_NCU" \
      "$CUDA_GRAPH_FLAG_NCU" \
      --cuda-graph-mode "$CUDA_GRAPH_MODE_NCU" \
      --cuda-graph-warmup-steps "$CUDA_GRAPH_WARMUP_STEPS_NCU" \
      --max-train-batches "$MAX_TRAIN_BATCHES_NCU" \
      --max-val-batches "$MAX_VAL_BATCHES_NCU" \
      --no-auto-log \
      --no-progress \
      --no-profile-train \
    2>&1 | tee -a "$NCU_RUN_LOG"
  NCU_RC=${PIPESTATUS[0]}
  set -e
fi

{
  echo "ncu_exit_code=${NCU_RC}"
  echo "ncu_permission_error=${NCU_PERMISSION_ERROR}"
  echo "ncu_report_path=${ROOT_DIR}/${NCU_REP}"
  echo "ncu_set=${NCU_SET}"
  echo "ncu_use_sudo=${NCU_USE_SUDO}"
} >"$NCU_STATUS_TXT"

echo ""
echo "[5/5] Export ncu text reports"
if [[ "$NCU_PERMISSION_ERROR" == "1" ]]; then
  cat >"$NCU_SUMMARY_TXT" <<'EOF'
[ncu] profiling skipped due to ERR_NVGPUCTRPERM (no permission for GPU performance counters).
[ncu] quick workaround:
  NCU_USE_SUDO=1 bash 3_experiments/scripts/run_gpu_profile_stack.sh
[ncu] permanent system fix:
  https://developer.nvidia.com/ERR_NVGPUCTRPERM
EOF
  cp "$NCU_SUMMARY_TXT" "$NCU_DETAILS_TXT"
  cp "$NCU_SUMMARY_TXT" "$NCU_SESSION_TXT"
elif [[ -f "$NCU_REP" ]]; then
  ncu --import "$NCU_REP" --page details --print-details all --print-summary per-kernel >"$NCU_SUMMARY_TXT" 2>/dev/null || true
  ncu --import "$NCU_REP" --page raw --csv --print-summary per-kernel >"$NCU_DETAILS_TXT" 2>/dev/null || true
  ncu --import "$NCU_REP" --page session >"$NCU_SESSION_TXT" 2>/dev/null || true

  if [[ ! -s "$NCU_SUMMARY_TXT" ]]; then
    echo "[ncu] summary export is empty; see run.log and report in Nsight Compute UI." >"$NCU_SUMMARY_TXT"
  fi
  if [[ ! -s "$NCU_DETAILS_TXT" ]]; then
    echo "[ncu] raw csv export is empty; see run.log and report in Nsight Compute UI." >"$NCU_DETAILS_TXT"
  fi
  if [[ ! -s "$NCU_SESSION_TXT" ]]; then
    echo "[ncu] session export is empty; see run.log and report in Nsight Compute UI." >"$NCU_SESSION_TXT"
  fi
else
  cat >"$NCU_SUMMARY_TXT" <<'EOF'
[ncu] no .ncu-rep report generated.
[ncu] check ncu/run.log for errors.
EOF
  cp "$NCU_SUMMARY_TXT" "$NCU_DETAILS_TXT"
  cp "$NCU_SUMMARY_TXT" "$NCU_SESSION_TXT"
fi

rg -n "SM Throughput|Memory Throughput|Occupancy|Tensor|Duration|Kernel Name|Section|Achieved Occupancy|DRAM" \
  "$NCU_SUMMARY_TXT" "$NCU_DETAILS_TXT" >"$NCU_KEY_TXT" || true

# Optional event-monitor artifact placeholders (for future always-on event timing).
EVENT_STEP_JSONL="${ROOT_DIR}/${OUT_DIR}/event/event_timing_step.jsonl"
EVENT_EPOCH_JSONL="${ROOT_DIR}/${OUT_DIR}/event/event_timing_epoch.jsonl"

cat >"$ARTIFACT_MANIFEST" <<EOF
# GPU Profiling Artifact Paths
run_tag: ${RUN_TAG}
config: ${CONFIG}

[preflight]
${ROOT_DIR}/${OUT_DIR}/preflight.txt

[nsys]
${ROOT_DIR}/${NSYS_REP}
${ROOT_DIR}/${NSYS_STATS_TXT}
${ROOT_DIR}/${NSYS_KEY_TXT}

[ncu]
${ROOT_DIR}/${NCU_REP}
${ROOT_DIR}/${NCU_QUERY_TXT}
${ROOT_DIR}/${NCU_RUN_LOG}
${ROOT_DIR}/${NCU_STATUS_TXT}
${ROOT_DIR}/${NCU_SUMMARY_TXT}
${ROOT_DIR}/${NCU_DETAILS_TXT}
${ROOT_DIR}/${NCU_SESSION_TXT}
${ROOT_DIR}/${NCU_KEY_TXT}

[event_monitor_placeholder]
${EVENT_STEP_JSONL}
${EVENT_EPOCH_JSONL}
EOF

echo ""
echo "Done. Artifact manifest:"
echo "  ${ROOT_DIR}/${ARTIFACT_MANIFEST}"
echo ""
echo "Key files:"
echo "  ${ROOT_DIR}/${NSYS_STATS_TXT}"
echo "  ${ROOT_DIR}/${NCU_SUMMARY_TXT}"
echo "  ${ROOT_DIR}/${NCU_DETAILS_TXT}"
