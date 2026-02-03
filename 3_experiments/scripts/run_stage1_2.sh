#!/usr/bin/env bash
set -euo pipefail

# One-click runner for Stage 1 (data validation) + Stage 2 (FiLM ablation)
# Usage:
#   FALCOR_PYTHON_PATH=/path/to/falcor/python \
#   bash 3_experiments/scripts/run_stage1_2.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DEFAULT_FALCOR_PY="/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/python"
DEFAULT_FALCOR_PYTHON_BIN="/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/pythondist/bin/python3.10"
DEFAULT_FALCOR_MOGWAI="/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/Mogwai"
DEFAULT_FALCOR_PY_SITE="/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/pythondist/lib/python3.10/site-packages"
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
  # Prefer system libstdc++ over conda to satisfy GLIBCXX version
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

OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/1_data_generation/output/stage1_2}"
DATA_ROOT="${DATA_ROOT:-$OUT_ROOT/multilight}"
EXP_ROOT="${EXP_ROOT:-$ROOT_DIR/3_experiments/output/stage1_2}"
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
SH_MODE="${SH_MODE:-dir}"
CUBE_RES="${CUBE_RES:-32}"
NUM_SH_SAMPLES="${NUM_SH_SAMPLES:-512}"
NUM_FRAMES="${NUM_FRAMES:-32}"
SPP="${SPP:-512}"
MAX_CONFIGS="${MAX_CONFIGS:-18}"
MAX_PROBES="${MAX_PROBES:-128}"
PROBE_MODE="${PROBE_MODE:-grid}"
PROBE_UNIFORM_RATIO="${PROBE_UNIFORM_RATIO:-0.7}"
PROBE_SURFACE_OFFSET_MIN="${PROBE_SURFACE_OFFSET_MIN:-0.05}"
PROBE_SURFACE_OFFSET_MAX="${PROBE_SURFACE_OFFSET_MAX:-0.5}"

EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_GAUSSIANS="${NUM_GAUSSIANS:-20}"
RANK="${RANK:-5}"

FALCOR_PATH_ARG=()
if [[ -n "$FALCOR_PYTHON_PATH" ]]; then
  FALCOR_PATH_ARG=(--falcor-python-path "$FALCOR_PYTHON_PATH")
fi

echo "Stage 1: Linearity sanity check (Mogwai)"
mkdir -p "$FALCOR_HOME_DIR"
HOME="$FALCOR_HOME_DIR" \
VK_LOADER_LAYERS_DISABLE="$VK_LOADER_LAYERS_DISABLE" \
VK_VALIDATION_FEATURE_DISABLES="$VK_VALIDATION_FEATURE_DISABLES" \
PYTHONPATH="$FALCOR_PYTHON_PATH:$DEFAULT_FALCOR_PY_SITE:${PYTHONPATH:-}" \
FALCOR_SCENE="$SCENE" \
FALCOR_NUM_SH_SAMPLES="$NUM_SH_SAMPLES" \
FALCOR_SH_MODE="$SH_MODE" \
FALCOR_CUBE_RES="$CUBE_RES" \
FALCOR_NUM_FRAMES="$NUM_FRAMES" \
FALCOR_SPP="$SPP" \
FALCOR_FIXED_SEED="${FALCOR_FIXED_SEED:-1}" \
FALCOR_USE_RUSSIAN_ROULETTE="${FALCOR_USE_RUSSIAN_ROULETTE:-0}" \
FALCOR_PYTHON_PATH="$FALCOR_PYTHON_PATH" \
"$FALCOR_MOGWAI_BIN" --headless --script "$ROOT_DIR/1_data_generation/falcor/linearity_check.py" $MOGWAI_ARGS

echo "Stage 1: Generate multi-light dataset (Mogwai)"
HOME="$FALCOR_HOME_DIR" \
VK_LOADER_LAYERS_DISABLE="$VK_LOADER_LAYERS_DISABLE" \
VK_VALIDATION_FEATURE_DISABLES="$VK_VALIDATION_FEATURE_DISABLES" \
PYTHONPATH="$FALCOR_PYTHON_PATH:$DEFAULT_FALCOR_PY_SITE:${PYTHONPATH:-}" \
FALCOR_OUTPUT="$DATA_ROOT" \
FALCOR_SCENE="$SCENE" \
FALCOR_GRID_RES="$GRID_RES" \
FALCOR_NUM_SH_SAMPLES="$NUM_SH_SAMPLES" \
FALCOR_SH_MODE="$SH_MODE" \
FALCOR_CUBE_RES="$CUBE_RES" \
FALCOR_NUM_FRAMES="$NUM_FRAMES" \
FALCOR_SPP="$SPP" \
FALCOR_FIXED_SEED="${FALCOR_FIXED_SEED:-1}" \
FALCOR_USE_RUSSIAN_ROULETTE="${FALCOR_USE_RUSSIAN_ROULETTE:-0}" \
FALCOR_PROBE_MODE="$PROBE_MODE" \
FALCOR_PROBE_UNIFORM_RATIO="$PROBE_UNIFORM_RATIO" \
FALCOR_PROBE_SURFACE_OFFSET_MIN="$PROBE_SURFACE_OFFSET_MIN" \
FALCOR_PROBE_SURFACE_OFFSET_MAX="$PROBE_SURFACE_OFFSET_MAX" \
FALCOR_MAX_CONFIGS="$MAX_CONFIGS" \
FALCOR_MAX_PROBES="$MAX_PROBES" \
FALCOR_PYTHON_PATH="$FALCOR_PYTHON_PATH" \
"$FALCOR_MOGWAI_BIN" --headless --script "$ROOT_DIR/1_data_generation/falcor/generate_multilight_falcor.py" $MOGWAI_ARGS

echo "Stage 1: Probe field slicer (directional check)"
python "$ROOT_DIR/3_experiments/scripts/visualization/probe_field_slicer.py" \
  --data-root "$DATA_ROOT" \
  --config-idx 0 \
  --axis y \
  --value 0.0 \
  --tolerance 0.1 \
  --mode dir \
  --direction 1 0 0 \
  --save-png

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
