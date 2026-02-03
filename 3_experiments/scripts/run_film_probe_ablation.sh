#!/usr/bin/env bash
set -euo pipefail

# One-click FiLM ablation across grid vs adaptive probe datasets.
# Usage:
#   bash 3_experiments/scripts/run_film_probe_ablation.sh
#
# Optional overrides (env vars):
#   GRID_ROOT=/path/to/grid/multilight
#   ADAPTIVE_ROOT=/path/to/adaptive/multilight
#   EXP_ROOT=/path/to/output
#   EPOCHS=300 BATCH_SIZE=256 NUM_GAUSSIANS=20 RANK=5
#   RUN_TRAIN=1 RUN_ANALYSIS=1 RUN_SLICER=0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

GRID_ROOT="${GRID_ROOT:-$ROOT_DIR/1_data_generation/output/probe_mode_compare_5configs/grid/multilight}"
ADAPTIVE_ROOT="${ADAPTIVE_ROOT:-$ROOT_DIR/1_data_generation/output/probe_mode_compare_5configs/adaptive/multilight}"
EXP_ROOT="${EXP_ROOT:-$ROOT_DIR/3_experiments/output/ablation_probe_mode}"

EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_GAUSSIANS="${NUM_GAUSSIANS:-20}"
RANK="${RANK:-5}"
TOP_K="${TOP_K:-3}"
NUM_WORKERS="${NUM_WORKERS:-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
LR="${LR:-}"
LR_SCHEDULER="${LR_SCHEDULER:-none}"
LR_MIN="${LR_MIN:-1e-4}"
LAMBDA_LINEARITY="${LAMBDA_LINEARITY:-0.0}"
LINEARITY_AUG_PAIRS="${LINEARITY_AUG_PAIRS:-0}"
LAMBDA_SPATIAL="${LAMBDA_SPATIAL:-0.0}"
SPATIAL_K="${SPATIAL_K:-1}"
LOAD_MODEL="${LOAD_MODEL:-}"
SH_SCALER_PATH="${SH_SCALER_PATH:-}"
NO_INIT="${NO_INIT:-0}"
ENABLE_RERUN="${ENABLE_RERUN:-0}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
RUN_SLICER="${RUN_SLICER:-0}"
ONLY_GROUP="${ONLY_GROUP:-}"
SLICER_CONFIG_IDX="${SLICER_CONFIG_IDX:-0}"
SLICER_AXIS="${SLICER_AXIS:-y}"
SLICER_VALUE="${SLICER_VALUE:-0.0}"
SLICER_TOL="${SLICER_TOL:-0.1}"
SLICER_COEFF_IDX="${SLICER_COEFF_IDX:-1}"

train_one() {
  local data_root="$1"
  local out_dir="$2"
  local disable_film="$3"
  local tag="$4"
  echo "== Train: ${tag} =="
  mkdir -p "$out_dir"
  local rerun_args=()
  local extra_args=()
  if [[ "$ENABLE_RERUN" == "1" ]]; then
    rerun_args=(--enable-rerun --rerun-save-path "$out_dir/train.rrd")
  fi
  if [[ -n "$LR" ]]; then
    extra_args+=(--lr "$LR")
  fi
  if [[ -n "$LOAD_MODEL" ]]; then
    extra_args+=(--load-model "$LOAD_MODEL")
  fi
  if [[ -n "$SH_SCALER_PATH" ]]; then
    extra_args+=(--sh-scaler-path "$SH_SCALER_PATH")
  fi
  if [[ "$NO_INIT" == "1" ]]; then
    extra_args+=(--no-init)
  fi
  if [[ "$disable_film" == "1" ]]; then
    python "$ROOT_DIR/3_experiments/scripts/train.py" \
      --variant unified_set \
      --data-root "$data_root" \
      --output-dir "$out_dir" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --num-gaussians "$NUM_GAUSSIANS" \
      --rank "$RANK" \
      --top-k "$TOP_K" \
      --num-workers "$NUM_WORKERS" \
      --weight-decay "$WEIGHT_DECAY" \
      --lr-scheduler "$LR_SCHEDULER" \
      --lr-min "$LR_MIN" \
      --enable-sh-scaler \
      --lambda-linearity "$LAMBDA_LINEARITY" \
      --linearity-aug-pairs "$LINEARITY_AUG_PAIRS" \
      --lambda-spatial "$LAMBDA_SPATIAL" \
      --spatial-k "$SPATIAL_K" \
      --disable-film \
      "${extra_args[@]}" \
      "${rerun_args[@]}"
  else
    python "$ROOT_DIR/3_experiments/scripts/train.py" \
      --variant unified_set \
      --data-root "$data_root" \
      --output-dir "$out_dir" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --num-gaussians "$NUM_GAUSSIANS" \
      --rank "$RANK" \
      --top-k "$TOP_K" \
      --num-workers "$NUM_WORKERS" \
      --weight-decay "$WEIGHT_DECAY" \
      --lr-scheduler "$LR_SCHEDULER" \
      --lr-min "$LR_MIN" \
      --enable-sh-scaler \
      --lambda-linearity "$LAMBDA_LINEARITY" \
      --linearity-aug-pairs "$LINEARITY_AUG_PAIRS" \
      --lambda-spatial "$LAMBDA_SPATIAL" \
      --spatial-k "$SPATIAL_K" \
      "${extra_args[@]}" \
      "${rerun_args[@]}"
  fi
}

analyze_one() {
  local data_root="$1"
  local out_dir="$2"
  local disable_film="$3"
  local tag="$4"
  local ckpt="$out_dir/best_model.pt"
  local scaler="$out_dir/sh_scaler.npz"
  local out_json="$out_dir/analysis.json"
  local pred_out=""
  if [[ "$RUN_SLICER" == "1" ]]; then
    pred_out="$out_dir/pred_sh_config${SLICER_CONFIG_IDX}.npy"
  fi
  if [[ ! -f "$ckpt" ]]; then
    echo "Missing checkpoint for ${tag}: $ckpt"
    return 1
  fi
  echo "== Analyze: ${tag} =="
  if [[ "$disable_film" == "1" ]]; then
    python "$ROOT_DIR/3_experiments/scripts/analysis/run_ablation_analysis.py" \
      --data-root "$data_root" \
      --checkpoint "$ckpt" \
      --sh-scaler "$scaler" \
      --output "$out_json" \
      --num-gaussians "$NUM_GAUSSIANS" \
      --rank "$RANK" \
      --top-k "$TOP_K" \
      ${pred_out:+--save-pred-config "$SLICER_CONFIG_IDX"} \
      ${pred_out:+--save-pred-out "$pred_out"} \
      --disable-film
  else
    python "$ROOT_DIR/3_experiments/scripts/analysis/run_ablation_analysis.py" \
      --data-root "$data_root" \
      --checkpoint "$ckpt" \
      --sh-scaler "$scaler" \
      --output "$out_json" \
      --num-gaussians "$NUM_GAUSSIANS" \
      --rank "$RANK" \
      --top-k "$TOP_K" \
      ${pred_out:+--save-pred-config "$SLICER_CONFIG_IDX"} \
      ${pred_out:+--save-pred-out "$pred_out"}
  fi
}

if [[ "$RUN_TRAIN" == "1" ]]; then
  case "$ONLY_GROUP" in
    grid_off)
      train_one "$GRID_ROOT" "$EXP_ROOT/grid/film_off" 1 "GRID + FiLM OFF"
      ;;
    grid_on)
      train_one "$GRID_ROOT" "$EXP_ROOT/grid/film_on" 0 "GRID + FiLM ON"
      ;;
    adaptive_off)
      train_one "$ADAPTIVE_ROOT" "$EXP_ROOT/adaptive/film_off" 1 "ADAPTIVE + FiLM OFF"
      ;;
    adaptive_on)
      train_one "$ADAPTIVE_ROOT" "$EXP_ROOT/adaptive/film_on" 0 "ADAPTIVE + FiLM ON"
      ;;
    "")
      train_one "$GRID_ROOT" "$EXP_ROOT/grid/film_off" 1 "GRID + FiLM OFF"
      train_one "$GRID_ROOT" "$EXP_ROOT/grid/film_on" 0 "GRID + FiLM ON"
      train_one "$ADAPTIVE_ROOT" "$EXP_ROOT/adaptive/film_off" 1 "ADAPTIVE + FiLM OFF"
      train_one "$ADAPTIVE_ROOT" "$EXP_ROOT/adaptive/film_on" 0 "ADAPTIVE + FiLM ON"
      ;;
    *)
      echo "Unknown ONLY_GROUP: $ONLY_GROUP (use grid_off|grid_on|adaptive_off|adaptive_on)"
      exit 1
      ;;
  esac
fi

if [[ "$RUN_ANALYSIS" == "1" ]]; then
  analyze_one "$GRID_ROOT" "$EXP_ROOT/grid/film_off" 1 "GRID + FiLM OFF"
  analyze_one "$GRID_ROOT" "$EXP_ROOT/grid/film_on" 0 "GRID + FiLM ON"
  analyze_one "$ADAPTIVE_ROOT" "$EXP_ROOT/adaptive/film_off" 1 "ADAPTIVE + FiLM OFF"
  analyze_one "$ADAPTIVE_ROOT" "$EXP_ROOT/adaptive/film_on" 0 "ADAPTIVE + FiLM ON"

  EXP_ROOT="$EXP_ROOT" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ.get("EXP_ROOT", "."))
rows = []
labels = [
    ("GRID + FiLM OFF", root / "grid/film_off/analysis.json"),
    ("GRID + FiLM ON", root / "grid/film_on/analysis.json"),
    ("ADAPTIVE + FiLM OFF", root / "adaptive/film_off/analysis.json"),
    ("ADAPTIVE + FiLM ON", root / "adaptive/film_on/analysis.json"),
]

for label, path in labels:
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    rows.append({
        "label": label,
        "rmse": data.get("rmse"),
        "mae": data.get("mae"),
        "l0_rmse": data.get("l0_rmse"),
        "ho_rmse": data.get("ho_rmse"),
        "near_rmse": data.get("near_rmse"),
        "open_rmse": data.get("open_rmse"),
        "ang_deg": data.get("angular_error_deg"),
        "grad_rmse": data.get("shadow_gradient_rmse"),
    })

print("\nAblation Summary:")
print("{:<20} {:>9} {:>9} {:>9} {:>9} {:>11} {:>11} {:>10} {:>11}".format(
    "Group", "RMSE", "MAE", "L0", "HO", "NearRMSE", "OpenRMSE", "Ang(deg)", "GradRMSE"
))
for r in rows:
    print("{:<20} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} {:>11.4f} {:>11.4f} {:>10.2f} {:>11.4f}".format(
        r["label"], r["rmse"], r["mae"], r["l0_rmse"], r["ho_rmse"],
        r["near_rmse"], r["open_rmse"], r["ang_deg"], r["grad_rmse"]
    ))

# Write CSV + LaTeX table for paper
csv_path = root / "ablation_summary.csv"
tex_path = root / "ablation_summary.tex"
root.mkdir(parents=True, exist_ok=True)

with open(csv_path, "w") as f:
    header = ["Group", "RMSE", "MAE", "L0_RMSE", "HO_RMSE", "Near_RMSE", "Open_RMSE", "Angular_Deg", "Grad_RMSE"]
    f.write(",".join(header) + "\n")
    for r in rows:
        f.write("{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
            r["label"], r["rmse"], r["mae"], r["l0_rmse"], r["ho_rmse"],
            r["near_rmse"], r["open_rmse"], r["ang_deg"], r["grad_rmse"]
        ))

def fmt(x, nd=4):
    return f"{x:.{nd}f}"

with open(tex_path, "w") as f:
    f.write("\\begin{table}[t]\n")
    f.write("\\centering\n")
    f.write("\\small\n")
    f.write("\\begin{tabular}{lrrrrrrrr}\n")
    f.write("\\toprule\n")
    f.write("Group & RMSE & MAE & L0 & HO & Near & Open & Ang($^\\circ$) & Grad \\\\\n")
    f.write("\\midrule\n")
    for r in rows:
        f.write(
            f"{r['label']} & {fmt(r['rmse'])} & {fmt(r['mae'])} & "
            f"{fmt(r['l0_rmse'])} & {fmt(r['ho_rmse'])} & "
            f"{fmt(r['near_rmse'])} & {fmt(r['open_rmse'])} & "
            f"{fmt(r['ang_deg'],2)} & {fmt(r['grad_rmse'])} \\\\\n"
        )
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\caption{Ablation results for probe sampling and FiLM.}\n")
    f.write("\\label{tab:ablation-probe-film}\n")
    f.write("\\end{table}\n")

print(f"\nSaved CSV: {csv_path}")
print(f"Saved LaTeX: {tex_path}")
PY
fi

if [[ "$RUN_SLICER" == "1" ]]; then
  echo "Generating slicer comparisons..."
  for mode in grid adaptive; do
    if [[ "$mode" == "grid" ]]; then
      data_root="$GRID_ROOT"
    else
      data_root="$ADAPTIVE_ROOT"
    fi
    pred_on="$EXP_ROOT/$mode/film_on/pred_sh_config${SLICER_CONFIG_IDX}.npy"
    pred_off="$EXP_ROOT/$mode/film_off/pred_sh_config${SLICER_CONFIG_IDX}.npy"
    # GT slice
    python "$ROOT_DIR/3_experiments/scripts/visualization/probe_field_slicer.py" \
      --data-root "$data_root" \
      --config-idx "$SLICER_CONFIG_IDX" \
      --axis "$SLICER_AXIS" --value "$SLICER_VALUE" --tolerance "$SLICER_TOL" \
      --mode coeff --coeff-idx "$SLICER_COEFF_IDX" \
      --output "$EXP_ROOT/$mode/gt_slice_c${SLICER_COEFF_IDX}.npz" --save-png

    # FiLM OFF
    if [[ -f "$pred_off" ]]; then
      python "$ROOT_DIR/3_experiments/scripts/visualization/probe_field_slicer.py" \
        --data-root "$data_root" \
        --config-idx "$SLICER_CONFIG_IDX" \
        --pred-sh "$pred_off" \
        --axis "$SLICER_AXIS" --value "$SLICER_VALUE" --tolerance "$SLICER_TOL" \
        --mode coeff --coeff-idx "$SLICER_COEFF_IDX" \
        --output "$EXP_ROOT/$mode/film_off_slice_c${SLICER_COEFF_IDX}.npz" --save-png
    fi

    # FiLM ON
    if [[ -f "$pred_on" ]]; then
      python "$ROOT_DIR/3_experiments/scripts/visualization/probe_field_slicer.py" \
        --data-root "$data_root" \
        --config-idx "$SLICER_CONFIG_IDX" \
        --pred-sh "$pred_on" \
        --axis "$SLICER_AXIS" --value "$SLICER_VALUE" --tolerance "$SLICER_TOL" \
        --mode coeff --coeff-idx "$SLICER_COEFF_IDX" \
        --output "$EXP_ROOT/$mode/film_on_slice_c${SLICER_COEFF_IDX}.npz" --save-png
    fi
  done
fi

echo "Done. Outputs:"
echo "  Experiments: $EXP_ROOT"
