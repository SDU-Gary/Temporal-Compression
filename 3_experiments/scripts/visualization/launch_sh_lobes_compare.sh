#!/usr/bin/env bash
set -euo pipefail

# Launch two Polyscope windows to compare grid vs adaptive probe datasets.
# Usage:
#   bash 3_experiments/scripts/visualization/launch_sh_lobes_compare.sh
#
# Optional overrides (env vars):
#   GRID_ROOT=/path/to/grid/multilight
#   ADAPTIVE_ROOT=/path/to/adaptive/multilight
#   CONFIG_IDX=0 STRIDE=1 SUBDIVISIONS=2 BASE_RADIUS=0.03 SCALE=0.08
#   VIS_MODE=log VIS_CLIP=0.95

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

GRID_ROOT="${GRID_ROOT:-$ROOT_DIR/1_data_generation/output/probe_mode_compare_5configs/grid/multilight}"
ADAPTIVE_ROOT="${ADAPTIVE_ROOT:-$ROOT_DIR/1_data_generation/output/probe_mode_compare_5configs/adaptive/multilight}"

CONFIG_IDX="${CONFIG_IDX:-0}"
STRIDE="${STRIDE:-1}"
SUBDIVISIONS="${SUBDIVISIONS:-2}"
BASE_RADIUS="${BASE_RADIUS:-0.03}"
SCALE="${SCALE:-0.08}"
VIS_MODE="${VIS_MODE:-log}"
VIS_CLIP="${VIS_CLIP:-0.95}"

PY_SCRIPT="$ROOT_DIR/3_experiments/scripts/visualization/visualize_sh_lobes_interactive.py"

python "$PY_SCRIPT" \
  --data-root "$GRID_ROOT" \
  --config-idx "$CONFIG_IDX" \
  --stride "$STRIDE" \
  --subdivisions "$SUBDIVISIONS" \
  --base-radius "$BASE_RADIUS" \
  --scale "$SCALE" \
  --vis-mode "$VIS_MODE" \
  --vis-clip "$VIS_CLIP" \
  --title "SH Lobes - GRID" &

python "$PY_SCRIPT" \
  --data-root "$ADAPTIVE_ROOT" \
  --config-idx "$CONFIG_IDX" \
  --stride "$STRIDE" \
  --subdivisions "$SUBDIVISIONS" \
  --base-radius "$BASE_RADIUS" \
  --scale "$SCALE" \
  --vis-mode "$VIS_MODE" \
  --vis-clip "$VIS_CLIP" \
  --title "SH Lobes - ADAPTIVE" &

wait
