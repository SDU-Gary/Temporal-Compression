#!/usr/bin/env bash
set -euo pipefail

# One-pass pipeline:
# 1) Train with Phase-0 monitoring (heartbeat/oracle monitor/grad metrics)
# 2) Auto-run Phase-1 coeff learnability suite after training

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-3_experiments/configs/bistro_clean_train_A1_1.yaml}"

if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] config not found: $CONFIG"
  exit 1
fi

EXP_DIR="$(python - "$CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1])
cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8')) or {}
exp = cfg.get('experiment', {}) if isinstance(cfg, dict) else {}
out = exp.get('output_dir')
if not out:
    raise SystemExit('missing experiment.output_dir in config')
print(out)
PY
)"

echo "[Phase0+1] config: $CONFIG"
echo "[Phase0+1] output: $EXP_DIR"

python 3_experiments/scripts/train.py --config "$CONFIG"

HEARTBEAT="$EXP_DIR/runtime/heartbeat.json"
PHASE0_JSONL="$EXP_DIR/runtime/phase0_metrics.jsonl"
SUITE_SUMMARY="$EXP_DIR/diagnostics/coeff_suite/phase1_summary.json"

echo ""
echo "[Check] heartbeat: $HEARTBEAT"
[[ -f "$HEARTBEAT" ]] || { echo "[ERROR] heartbeat missing"; exit 1; }

echo "[Check] phase0 metrics: $PHASE0_JSONL"
[[ -f "$PHASE0_JSONL" ]] || { echo "[ERROR] phase0 metrics missing"; exit 1; }

echo "[Check] phase1 summary: $SUITE_SUMMARY"
[[ -f "$SUITE_SUMMARY" ]] || { echo "[ERROR] phase1 summary missing"; exit 1; }

echo ""
echo "Done. Key artifacts:"
echo "  heartbeat:      $HEARTBEAT"
echo "  phase0 metrics: $PHASE0_JSONL"
echo "  phase1 summary: $SUITE_SUMMARY"

