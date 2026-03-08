#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_nsys_all.sh \
    --config 3_experiments/configs/bistro_clean_train.yaml \
    --checkpoint 3_experiments/results/.../best_model.pt \
    --data-root 1_data_generation/output/bistro_clean_v2 \
    --scene <scene_name_or_path> \
    [--exp-dir 3_experiments/results/.../exp_A_K30_r8_baseline] \
    [--device cuda] \
    [--steps 200] [--warmup-steps 30] [--num-workers 0] \
    [--frames 300] [--warmup-frames 30] [--batch-size 1024] \
    [--pipeline-route both] [--pipeline-warmup-frames 30] [--pipeline-benchmark-frames 300]

Notes:
  1) This script uses external Nsight wrapping (nsys profile ... python tools/profile_*.py --stack nsys)
     to avoid torch CUPTI activity warnings in --stack torch path.
  2) Run in project root with venv activated.
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] Missing command: $1" >&2
    exit 1
  fi
}

CONFIG=""
CHECKPOINT=""
DATA_ROOT=""
SCENE=""
EXP_DIR=""

DEVICE="cuda"
STEPS=200
WARMUP_STEPS=30
NUM_WORKERS=0

FRAMES=300
WARMUP_FRAMES=30
BATCH_SIZE=1024

PIPELINE_ROUTE="both"
PIPELINE_WARMUP_FRAMES=30
PIPELINE_BENCHMARK_FRAMES=300

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --scene) SCENE="$2"; shift 2 ;;
    --exp-dir) EXP_DIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --warmup-steps) WARMUP_STEPS="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --frames) FRAMES="$2"; shift 2 ;;
    --warmup-frames) WARMUP_FRAMES="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --pipeline-route) PIPELINE_ROUTE="$2"; shift 2 ;;
    --pipeline-warmup-frames) PIPELINE_WARMUP_FRAMES="$2"; shift 2 ;;
    --pipeline-benchmark-frames) PIPELINE_BENCHMARK_FRAMES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$CONFIG" || -z "$CHECKPOINT" || -z "$DATA_ROOT" || -z "$SCENE" ]]; then
  echo "[ERROR] Required args missing: --config --checkpoint --data-root --scene" >&2
  usage
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] Config not found: $CONFIG" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERROR] Data root not found: $DATA_ROOT" >&2
  exit 1
fi

if [[ -z "$EXP_DIR" ]]; then
  EXP_DIR="$(dirname "$CHECKPOINT")"
fi

require_cmd python
require_cmd nsys

TS="$(date +%Y%m%d_%H%M%S)"

OUT_TRAIN="$EXP_DIR/profiler_training_nsys_$TS"
OUT_INF_MODEL="$EXP_DIR/profiler_inference_model_nsys_$TS"
OUT_INF_PIPE="$EXP_DIR/profiler_inference_pipeline_nsys_$TS"

mkdir -p "$OUT_TRAIN" "$OUT_INF_MODEL" "$OUT_INF_PIPE"

echo "[1/3] Training Nsight profiling..."
nsys profile -t cuda,nvtx,osrt --sample=none --force-overwrite=true \
  -o "$OUT_TRAIN/nsys_training" \
  python tools/profile_training.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    --device "$DEVICE" \
    --num-workers "$NUM_WORKERS" \
    --steps "$STEPS" \
    --warmup-steps "$WARMUP_STEPS" \
    --stack nsys \
    --output-dir "$OUT_TRAIN"

nsys stats "$OUT_TRAIN/nsys_training.nsys-rep" > "$OUT_TRAIN/nsys_training_stats.txt"

echo "[2/3] Inference model-level Nsight profiling..."
nsys profile -t cuda,nvtx,osrt --sample=none --force-overwrite=true \
  -o "$OUT_INF_MODEL/nsys_inference_model" \
  python tools/profile_inference.py \
    --checkpoint "$CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    --split test \
    --device "$DEVICE" \
    --profile-level model \
    --frames "$FRAMES" \
    --warmup-frames "$WARMUP_FRAMES" \
    --batch-size "$BATCH_SIZE" \
    --stack nsys \
    --output-dir "$OUT_INF_MODEL"

nsys stats "$OUT_INF_MODEL/nsys_inference_model.nsys-rep" > "$OUT_INF_MODEL/nsys_inference_model_stats.txt"

echo "[3/3] Inference pipeline-level Nsight profiling..."
nsys profile -t cuda,nvtx,osrt --sample=none --force-overwrite=true \
  -o "$OUT_INF_PIPE/nsys_inference_pipeline" \
  python tools/profile_inference.py \
    --checkpoint "$CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    --device "$DEVICE" \
    --profile-level pipeline \
    --pipeline-run \
    --scene "$SCENE" \
    --pipeline-route "$PIPELINE_ROUTE" \
    --pipeline-warmup-frames "$PIPELINE_WARMUP_FRAMES" \
    --pipeline-benchmark-frames "$PIPELINE_BENCHMARK_FRAMES" \
    --stack nsys \
    --output-dir "$OUT_INF_PIPE"

nsys stats "$OUT_INF_PIPE/nsys_inference_pipeline.nsys-rep" > "$OUT_INF_PIPE/nsys_inference_pipeline_stats.txt"

cat <<EOF

[DONE] Nsight profiling finished.

Training:
  $OUT_TRAIN
  - training_profile_summary.json
  - training_profile_report.json
  - nsys_training.nsys-rep
  - nsys_training_stats.txt

Inference (model-level):
  $OUT_INF_MODEL
  - inference_model_summary.json
  - inference_profile_report.json
  - nsys_inference_model.nsys-rep
  - nsys_inference_model_stats.txt

Inference (pipeline-level):
  $OUT_INF_PIPE
  - inference_pipeline_summary.json
  - inference_profile_report.json
  - nsys_inference_pipeline.nsys-rep
  - nsys_inference_pipeline_stats.txt

EOF

