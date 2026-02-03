#!/usr/bin/env bash
set -euo pipefail

# Progressive FiLM training (Stage 1: OFF -> Stage 2: ON) with analysis.
# Usage:
#   bash 3_experiments/scripts/run_film_progressive_ablation.sh
#
# Optional overrides (env vars):
#   DATA_ROOT=/path/to/multilight
#   EXP_ROOT=/path/to/output
#   EPOCHS_STAGE1=150 EPOCHS_STAGE2=450
#   BATCH_SIZE=256 NUM_GAUSSIANS=30 RANK=8 TOP_K=3 NUM_WORKERS=2
#   LR_STAGE1=1e-3 LR_STAGE2=1e-4
#   LR_SCHEDULER_STAGE1=cosine LR_MIN_STAGE1=1e-4
#   LR_SCHEDULER_STAGE2=cosine LR_MIN_STAGE2=1e-5
#   WEIGHT_DECAY=0.0 LAMBDA_LINEARITY=0.01 LINEARITY_AUG_PAIRS=0
#   ENABLE_RERUN=1 RERUN_DISABLE_RENDER=1
#   RUN_ANALYSIS=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/1_data_generation/output/probe_mode_compare_200configs/adaptive/multilight}"
EXP_ROOT="${EXP_ROOT:-$ROOT_DIR/3_experiments/output/ablation_probe_mode_200_big_opt2}"

EPOCHS_STAGE1="${EPOCHS_STAGE1:-150}"
EPOCHS_STAGE2="${EPOCHS_STAGE2:-450}"

BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_GAUSSIANS="${NUM_GAUSSIANS:-30}"
RANK="${RANK:-8}"
TOP_K="${TOP_K:-3}"
NUM_WORKERS="${NUM_WORKERS:-2}"

LR_STAGE1="${LR_STAGE1:-}"
LR_STAGE2="${LR_STAGE2:-}"
LR_SCHEDULER_STAGE1="${LR_SCHEDULER_STAGE1:-cosine}"
LR_MIN_STAGE1="${LR_MIN_STAGE1:-1e-4}"
LR_SCHEDULER_STAGE2="${LR_SCHEDULER_STAGE2:-cosine}"
LR_MIN_STAGE2="${LR_MIN_STAGE2:-1e-5}"

WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
LAMBDA_LINEARITY="${LAMBDA_LINEARITY:-0.01}"
LINEARITY_AUG_PAIRS="${LINEARITY_AUG_PAIRS:-0}"
LAMBDA_SPATIAL="${LAMBDA_SPATIAL:-0.0}"
SPATIAL_K="${SPATIAL_K:-1}"

ENABLE_RERUN="${ENABLE_RERUN:-0}"
RERUN_DISABLE_RENDER="${RERUN_DISABLE_RENDER:-0}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

OFF_DIR="$EXP_ROOT/film_off"
ON_DIR="$EXP_ROOT/film_on"
SCALER_PATH="$OFF_DIR/sh_scaler.npz"
CKPT_PATH="$OFF_DIR/best_model.pt"

export RERUN_DISABLE_RENDER="$RERUN_DISABLE_RENDER"

rerun_args=()
if [[ "$ENABLE_RERUN" == "1" ]]; then
  rerun_args=(--enable-rerun)
fi

echo "== Stage 1: FiLM OFF (bone) =="
mkdir -p "$OFF_DIR"
python "$ROOT_DIR/3_experiments/scripts/train.py" \
  --variant unified_set \
  --data-root "$DATA_ROOT" \
  --output-dir "$OFF_DIR" \
  --epochs "$EPOCHS_STAGE1" \
  --batch-size "$BATCH_SIZE" \
  --num-gaussians "$NUM_GAUSSIANS" \
  --rank "$RANK" \
  --top-k "$TOP_K" \
  --num-workers "$NUM_WORKERS" \
  --weight-decay "$WEIGHT_DECAY" \
  --lr-scheduler "$LR_SCHEDULER_STAGE1" \
  --lr-min "$LR_MIN_STAGE1" \
  --enable-sh-scaler \
  --sh-scaler-path "$SCALER_PATH" \
  --lambda-linearity "$LAMBDA_LINEARITY" \
  --linearity-aug-pairs "$LINEARITY_AUG_PAIRS" \
  --lambda-spatial "$LAMBDA_SPATIAL" \
  --spatial-k "$SPATIAL_K" \
  --disable-film \
  ${LR_STAGE1:+--lr "$LR_STAGE1"} \
  ${rerun_args[@]} \
  ${rerun_args[@]/--enable-rerun/--enable-rerun --rerun-save-path "$OFF_DIR/train.rrd"}

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Missing Stage1 checkpoint: $CKPT_PATH"
  exit 1
fi

echo "== Stage 2: FiLM ON (expression) =="
mkdir -p "$ON_DIR"
python "$ROOT_DIR/3_experiments/scripts/train.py" \
  --variant unified_set \
  --data-root "$DATA_ROOT" \
  --output-dir "$ON_DIR" \
  --epochs "$EPOCHS_STAGE2" \
  --batch-size "$BATCH_SIZE" \
  --num-gaussians "$NUM_GAUSSIANS" \
  --rank "$RANK" \
  --top-k "$TOP_K" \
  --num-workers "$NUM_WORKERS" \
  --weight-decay "$WEIGHT_DECAY" \
  --lr-scheduler "$LR_SCHEDULER_STAGE2" \
  --lr-min "$LR_MIN_STAGE2" \
  --enable-sh-scaler \
  --sh-scaler-path "$SCALER_PATH" \
  --lambda-linearity "$LAMBDA_LINEARITY" \
  --linearity-aug-pairs "$LINEARITY_AUG_PAIRS" \
  --lambda-spatial "$LAMBDA_SPATIAL" \
  --spatial-k "$SPATIAL_K" \
  --load-model "$CKPT_PATH" \
  --no-init \
  ${LR_STAGE2:+--lr "$LR_STAGE2"} \
  ${rerun_args[@]} \
  ${rerun_args[@]/--enable-rerun/--enable-rerun --rerun-save-path "$ON_DIR/train.rrd"}

if [[ "$RUN_ANALYSIS" == "1" ]]; then
  echo "== Analyze: FiLM OFF (Stage 1) =="
  python "$ROOT_DIR/3_experiments/scripts/analysis/run_ablation_analysis.py" \
    --data-root "$DATA_ROOT" \
    --checkpoint "$OFF_DIR/best_model.pt" \
    --sh-scaler "$SCALER_PATH" \
    --output "$OFF_DIR/analysis.json" \
    --num-gaussians "$NUM_GAUSSIANS" \
    --rank "$RANK" \
    --top-k "$TOP_K" \
    --disable-film

  echo "== Analyze: FiLM ON (Stage 2) =="
  python "$ROOT_DIR/3_experiments/scripts/analysis/run_ablation_analysis.py" \
    --data-root "$DATA_ROOT" \
    --checkpoint "$ON_DIR/best_model.pt" \
    --sh-scaler "$SCALER_PATH" \
    --output "$ON_DIR/analysis.json" \
    --num-gaussians "$NUM_GAUSSIANS" \
    --rank "$RANK" \
    --top-k "$TOP_K"

  EXP_ROOT="$EXP_ROOT" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ.get("EXP_ROOT", "."))
rows = []
labels = [
    ("FiLM OFF (Stage1)", root / "film_off/analysis.json"),
    ("FiLM ON (Stage2)", root / "film_on/analysis.json"),
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

def fmt(x, nd=4):
    return f"{x:.{nd}f}"

print("\nProgressive FiLM Summary:")
print("{:<18} {:>9} {:>9} {:>9} {:>9} {:>11} {:>11} {:>10} {:>11}".format(
    "Group", "RMSE", "MAE", "L0", "HO", "NearRMSE", "OpenRMSE", "Ang(deg)", "GradRMSE"
))
for r in rows:
    print("{:<18} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} {:>11.4f} {:>11.4f} {:>10.2f} {:>11.4f}".format(
        r["label"], r["rmse"], r["mae"], r["l0_rmse"], r["ho_rmse"],
        r["near_rmse"], r["open_rmse"], r["ang_deg"], r["grad_rmse"]
    ))

if len(rows) == 2:
    off, on = rows
    diff = {k: on[k] - off[k] for k in ("rmse","mae","l0_rmse","ho_rmse","near_rmse","open_rmse","ang_deg","grad_rmse")}
    print("{:<18} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} {:>11.4f} {:>11.4f} {:>10.2f} {:>11.4f}".format(
        "Delta(ON-OFF)", diff["rmse"], diff["mae"], diff["l0_rmse"], diff["ho_rmse"],
        diff["near_rmse"], diff["open_rmse"], diff["ang_deg"], diff["grad_rmse"]
    ))

csv_path = root / "progressive_summary.csv"
tex_path = root / "progressive_summary.tex"
root.mkdir(parents=True, exist_ok=True)

with open(csv_path, "w") as f:
    header = ["Group", "RMSE", "MAE", "L0_RMSE", "HO_RMSE", "Near_RMSE", "Open_RMSE", "Angular_Deg", "Grad_RMSE"]
    f.write(",".join(header) + "\n")
    for r in rows:
        f.write("{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
            r["label"], r["rmse"], r["mae"], r["l0_rmse"], r["ho_rmse"],
            r["near_rmse"], r["open_rmse"], r["ang_deg"], r["grad_rmse"]
        ))
    if len(rows) == 2:
        f.write("{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
            "Delta(ON-OFF)", diff["rmse"], diff["mae"], diff["l0_rmse"], diff["ho_rmse"],
            diff["near_rmse"], diff["open_rmse"], diff["ang_deg"], diff["grad_rmse"]
        ))

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
    if len(rows) == 2:
        f.write(
            f"Delta(ON-OFF) & {fmt(diff['rmse'])} & {fmt(diff['mae'])} & "
            f"{fmt(diff['l0_rmse'])} & {fmt(diff['ho_rmse'])} & "
            f"{fmt(diff['near_rmse'])} & {fmt(diff['open_rmse'])} & "
            f"{fmt(diff['ang_deg'],2)} & {fmt(diff['grad_rmse'])} \\\\\n"
        )
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\caption{Progressive FiLM: OFF->ON with delta.}\n")
    f.write("\\label{tab:progressive-film}\n")
    f.write("\\end{table}\n")

print(f"\nSaved CSV: {csv_path}")
print(f"Saved LaTeX: {tex_path}")
PY
fi

echo "Done. Outputs:"
echo "  Experiments: $EXP_ROOT"
