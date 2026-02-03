#!/usr/bin/env bash
set -euo pipefail

# End-to-end production run: generate 200-config dataset + FiLM ablation.
# Usage:
#   bash 3_experiments/scripts/run_prod_film_ablation.sh
#
# Optional overrides (env vars):
#   MAX_CONFIGS=200 GRID_RES=6 MAX_PROBES=128
#   SH_MODE=cubemap CUBE_RES=32 NUM_SH_SAMPLES=512 NUM_FRAMES=32 SPP=512
#   FALCOR_PROBE_MODE=adaptive FALCOR_PROBE_UNIFORM_RATIO=0.7
#   FALCOR_PROBE_SURFACE_OFFSET_MIN=0.05 FALCOR_PROBE_SURFACE_OFFSET_MAX=0.5
#   OUT_ROOT=/path/to/data EXP_ROOT=/path/to/experiments
#   SKIP_DATA_GEN=1  (skip Falcor dataset generation)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DEFAULT_FALCOR_PY="$ROOT_DIR/Falcor/build/linux-gcc/bin/Debug/python"
DEFAULT_FALCOR_PYTHON_BIN="$ROOT_DIR/Falcor/build/linux-gcc/bin/Debug/pythondist/bin/python3.10"
DEFAULT_FALCOR_MOGWAI="$ROOT_DIR/Falcor/build/linux-gcc/bin/Debug/Mogwai"
DEFAULT_FALCOR_PY_SITE="$ROOT_DIR/Falcor/build/linux-gcc/bin/Debug/pythondist/lib/python3.10/site-packages"

if [[ -z "${FALCOR_PYTHON_PATH:-}" && -d "$DEFAULT_FALCOR_PY" ]]; then
  FALCOR_PYTHON_PATH="$DEFAULT_FALCOR_PY"
else
  FALCOR_PYTHON_PATH="${FALCOR_PYTHON_PATH:-}"
fi

if [[ -x "$DEFAULT_FALCOR_PYTHON_BIN" ]]; then
  FALCOR_PYTHON_BIN="${FALCOR_PYTHON_BIN:-$DEFAULT_FALCOR_PYTHON_BIN}"
else
  FALCOR_PYTHON_BIN="${FALCOR_PYTHON_BIN:-python}"
fi

if [[ -x "$DEFAULT_FALCOR_MOGWAI" ]]; then
  FALCOR_MOGWAI_BIN="${FALCOR_MOGWAI_BIN:-$DEFAULT_FALCOR_MOGWAI}"
else
  FALCOR_MOGWAI_BIN="${FALCOR_MOGWAI_BIN:-Mogwai}"
fi

if [[ -n "$FALCOR_PYTHON_PATH" ]]; then
  FALCOR_LIB_DIR="$(dirname "$FALCOR_PYTHON_PATH")"
  SYS_LIB_1="/usr/lib/x86_64-linux-gnu"
  SYS_LIB_2="/lib/x86_64-linux-gnu"
  export LD_LIBRARY_PATH="$SYS_LIB_1:$SYS_LIB_2:$FALCOR_LIB_DIR:${LD_LIBRARY_PATH:-}"
  if [[ -f "$SYS_LIB_1/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$SYS_LIB_1/libstdc++.so.6:${LD_PRELOAD:-}"
  elif [[ -f "$SYS_LIB_2/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$SYS_LIB_2/libstdc++.so.6:${LD_PRELOAD:-}"
  fi
fi

SCENE="${SCENE:-$ROOT_DIR/1_data_generation/falcor/scenes/cornell_box_sun_point.pyscene}"

OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/1_data_generation/output/prod_200config}"
DATA_ROOT="${DATA_ROOT:-$OUT_ROOT/multilight}"
EXP_ROOT="${EXP_ROOT:-$ROOT_DIR/3_experiments/output/prod_200config}"
FALCOR_HOME_DIR="${FALCOR_HOME_DIR:-$ROOT_DIR/.falcor_home}"

MOGWAI_ARGS="${MOGWAI_ARGS:-}"
MOGWAI_DEVICE="${MOGWAI_DEVICE:-vulkan}"
MOGWAI_GPU="${MOGWAI_GPU:-0}"
VK_LOADER_LAYERS_DISABLE="${VK_LOADER_LAYERS_DISABLE:-VK_LAYER_KHRONOS_validation}"
VK_VALIDATION_FEATURE_DISABLES="${VK_VALIDATION_FEATURE_DISABLES:-VK_VALIDATION_FEATURE_DISABLE_DEBUG_PRINTF_EXT}"
if [[ -z "$MOGWAI_ARGS" ]]; then
  MOGWAI_ARGS="--device-type $MOGWAI_DEVICE --gpu $MOGWAI_GPU"
fi

GRID_RES="${GRID_RES:-6}"
SH_MODE="${SH_MODE:-cubemap}"
CUBE_RES="${CUBE_RES:-32}"
NUM_SH_SAMPLES="${NUM_SH_SAMPLES:-512}"
NUM_FRAMES="${NUM_FRAMES:-32}"
SPP="${SPP:-512}"
MAX_CONFIGS="${MAX_CONFIGS:-200}"
MAX_PROBES="${MAX_PROBES:-128}"

EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_GAUSSIANS="${NUM_GAUSSIANS:-20}"
RANK="${RANK:-5}"

if [[ "${SKIP_DATA_GEN:-0}" != "1" ]]; then
  echo "Stage 1: Generate multi-light dataset (Mogwai) — configs=${MAX_CONFIGS}"
  mkdir -p "$FALCOR_HOME_DIR"
  HOME="$FALCOR_HOME_DIR" \
  VK_LOADER_LAYERS_DISABLE="$VK_LOADER_LAYERS_DISABLE" \
  VK_VALIDATION_FEATURE_DISABLES="$VK_VALIDATION_FEATURE_DISABLES" \
  PYTHONPATH="$FALCOR_PYTHON_PATH:$DEFAULT_FALCOR_PY_SITE:${PYTHONPATH:-}" \
  FALCOR_OUTPUT="$DATA_ROOT" \
  FALCOR_SCENE="$SCENE" \
  FALCOR_GRID_RES="$GRID_RES" \
  FALCOR_SH_MODE="$SH_MODE" \
  FALCOR_CUBE_RES="$CUBE_RES" \
  FALCOR_NUM_SH_SAMPLES="$NUM_SH_SAMPLES" \
  FALCOR_NUM_FRAMES="$NUM_FRAMES" \
  FALCOR_SPP="$SPP" \
  FALCOR_FIXED_SEED="${FALCOR_FIXED_SEED:-1}" \
  FALCOR_PROBE_MODE="${FALCOR_PROBE_MODE:-grid}" \
  FALCOR_PROBE_UNIFORM_RATIO="${FALCOR_PROBE_UNIFORM_RATIO:-0.7}" \
  FALCOR_PROBE_SURFACE_OFFSET_MIN="${FALCOR_PROBE_SURFACE_OFFSET_MIN:-0.05}" \
  FALCOR_PROBE_SURFACE_OFFSET_MAX="${FALCOR_PROBE_SURFACE_OFFSET_MAX:-0.5}" \
  FALCOR_CONFIG_MODE="${FALCOR_CONFIG_MODE:-mix}" \
  FALCOR_UNIFORM_RATIO="${FALCOR_UNIFORM_RATIO:-0.7}" \
  FALCOR_MAX_CONFIGS="$MAX_CONFIGS" \
  FALCOR_MAX_PROBES="$MAX_PROBES" \
  FALCOR_PYTHON_PATH="$FALCOR_PYTHON_PATH" \
  "$FALCOR_MOGWAI_BIN" --headless --script "$ROOT_DIR/1_data_generation/falcor/generate_multilight_falcor.py" $MOGWAI_ARGS
else
  echo "Stage 1: Skipping dataset generation (SKIP_DATA_GEN=1)"
fi

echo "Stage 2: FiLM ablation (OFF)"
python "$ROOT_DIR/3_experiments/scripts/train.py" \
  --variant unified_set \
  --data-root "$DATA_ROOT" \
  --output-dir "$EXP_ROOT/film_off" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-gaussians "$NUM_GAUSSIANS" \
  --rank "$RANK" \
  --enable-sh-scaler \
  --lambda-linearity 0.0 \
  --linearity-aug-pairs 0 \
  --disable-film

echo "Stage 2: FiLM ablation (ON)"
python "$ROOT_DIR/3_experiments/scripts/train.py" \
  --variant unified_set \
  --data-root "$DATA_ROOT" \
  --output-dir "$EXP_ROOT/film_on" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-gaussians "$NUM_GAUSSIANS" \
  --rank "$RANK" \
  --enable-sh-scaler \
  --lambda-linearity 0.0 \
  --linearity-aug-pairs 0

echo "Done. Outputs:"
echo "  Dataset: $DATA_ROOT"
echo "  Experiments: $EXP_ROOT"
