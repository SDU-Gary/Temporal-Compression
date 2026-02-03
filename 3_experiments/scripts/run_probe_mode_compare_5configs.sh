#!/usr/bin/env bash
set -euo pipefail

# Generate two 5-config datasets: grid vs adaptive probe sampling.
# Usage:
#   bash 3_experiments/scripts/run_probe_mode_compare_5configs.sh
#
# Optional overrides (env vars):
#   MAX_CONFIGS=5 MAX_PROBES=128 GRID_RES=6
#   SH_MODE=cubemap CUBE_RES=32 NUM_FRAMES=32 SPP=512
#   OUT_ROOT=/path/to/output
#   PARALLEL=1  (run grid/adaptive in parallel)

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
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/1_data_generation/output/probe_mode_compare_5configs}"
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
MAX_CONFIGS="${MAX_CONFIGS:-5}"
MAX_PROBES="${MAX_PROBES:-128}"


mkdir -p "$FALCOR_HOME_DIR"

run_job() {
  local mode="$1"
  local out="$2"
  local home_dir="$3"
  mkdir -p "$home_dir"
  echo "Generating ${mode} probes dataset (configs=${MAX_CONFIGS})..."
  VK_LOADER_LAYERS_DISABLE="$VK_LOADER_LAYERS_DISABLE" \
  VK_VALIDATION_FEATURE_DISABLES="$VK_VALIDATION_FEATURE_DISABLES" \
  PYTHONPATH="$FALCOR_PYTHON_PATH:$DEFAULT_FALCOR_PY_SITE:${PYTHONPATH:-}" \
  FALCOR_SCENE="$SCENE" \
  FALCOR_GRID_RES="$GRID_RES" \
  FALCOR_SH_MODE="$SH_MODE" \
  FALCOR_CUBE_RES="$CUBE_RES" \
  FALCOR_NUM_SH_SAMPLES="$NUM_SH_SAMPLES" \
  FALCOR_NUM_FRAMES="$NUM_FRAMES" \
  FALCOR_SPP="$SPP" \
  FALCOR_FIXED_SEED="${FALCOR_FIXED_SEED:-1}" \
  FALCOR_CONFIG_MODE="${FALCOR_CONFIG_MODE:-mix}" \
  FALCOR_UNIFORM_RATIO="${FALCOR_UNIFORM_RATIO:-0.7}" \
  FALCOR_MAX_CONFIGS="$MAX_CONFIGS" \
  FALCOR_MAX_PROBES="$MAX_PROBES" \
  FALCOR_PYTHON_PATH="$FALCOR_PYTHON_PATH" \
  FALCOR_OUTPUT="$out" \
  HOME="$home_dir" \
  FALCOR_PROBE_MODE="$mode" \
  FALCOR_PROBE_UNIFORM_RATIO="0.7" \
  FALCOR_PROBE_SURFACE_OFFSET_MIN="0.05" \
  FALCOR_PROBE_SURFACE_OFFSET_MAX="0.5" \
  "$FALCOR_MOGWAI_BIN" --headless --script "$ROOT_DIR/1_data_generation/falcor/generate_multilight_falcor.py" $MOGWAI_ARGS
}

if [[ "${PARALLEL:-0}" == "1" ]]; then
  mkdir -p "$FALCOR_HOME_DIR/grid" "$FALCOR_HOME_DIR/adaptive"
  run_job "grid" "$OUT_ROOT/grid/multilight" "$FALCOR_HOME_DIR/grid" &
  pid_grid=$!
  run_job "adaptive" "$OUT_ROOT/adaptive/multilight" "$FALCOR_HOME_DIR/adaptive" &
  pid_adaptive=$!
  wait "$pid_grid"
  wait "$pid_adaptive"
else
  run_job "grid" "$OUT_ROOT/grid/multilight" "$FALCOR_HOME_DIR"
  run_job "adaptive" "$OUT_ROOT/adaptive/multilight" "$FALCOR_HOME_DIR"
fi

echo "Done."
echo "  GRID dataset:     $OUT_ROOT/grid/multilight"
echo "  ADAPTIVE dataset: $OUT_ROOT/adaptive/multilight"
