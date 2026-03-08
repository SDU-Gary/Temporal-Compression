#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${1:-3_experiments/configs/bistro_clean_train_A1_1.yaml}"
SESSION_NAME="${2:-a11_train_$(date +%m%d_%H%M%S)}"

STALE_SECONDS="${STALE_SECONDS:-1800}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
STARTUP_GRACE_SECONDS="${STARTUP_GRACE_SECONDS:-900}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux 未安装或不在 PATH 中。"
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "[ERROR] 配置文件不存在: $CONFIG_PATH"
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] tmux session 已存在: $SESSION_NAME"
  echo "        请换一个 session 名，或先执行: tmux kill-session -t $SESSION_NAME"
  exit 1
fi

OUTPUT_DIR_REL="$(python - "$CONFIG_PATH" <<'PY'
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1])
with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

exp = cfg.get("experiment") if isinstance(cfg, dict) else {}
out = (exp or {}).get("output_dir")
if not out:
    print("3_experiments/results/unified_set/tmux_run")
else:
    print(str(out))
PY
)"

OUTPUT_DIR_ABS="$ROOT_DIR/$OUTPUT_DIR_REL"
HEARTBEAT_PATH="$OUTPUT_DIR_ABS/runtime/heartbeat.json"

mkdir -p "$OUTPUT_DIR_ABS"

TRAIN_CMD="cd '$ROOT_DIR' && \
if [ -f venv/bin/activate ]; then source venv/bin/activate; fi && \
python 3_experiments/scripts/train.py --config '$CONFIG_PATH' 2>&1 | tee '$OUTPUT_DIR_ABS/train.log'"

WATCHDOG_CMD="cd '$ROOT_DIR' && \
if [ -f venv/bin/activate ]; then source venv/bin/activate; fi && \
python tools/watchdog_train_heartbeat.py \
  --heartbeat '$HEARTBEAT_PATH' \
  --stale-seconds '$STALE_SECONDS' \
  --check-interval-seconds '$CHECK_INTERVAL_SECONDS' \
  --startup-grace-seconds '$STARTUP_GRACE_SECONDS' \
  --action warn_exit 2>&1 | tee '$OUTPUT_DIR_ABS/watchdog.log'"

tmux new-session -d -s "$SESSION_NAME" -c "$ROOT_DIR"
tmux send-keys -t "$SESSION_NAME:0.0" "$TRAIN_CMD" C-m

tmux split-window -h -t "$SESSION_NAME:0" -c "$ROOT_DIR"
tmux send-keys -t "$SESSION_NAME:0.1" "$WATCHDOG_CMD" C-m

tmux select-layout -t "$SESSION_NAME:0" even-horizontal

echo "[OK] tmux 会话已启动: $SESSION_NAME"
echo "     配置文件: $CONFIG_PATH"
echo "     输出目录: $OUTPUT_DIR_REL"
echo "     心跳文件: $HEARTBEAT_PATH"
echo ""
echo "Attach: tmux attach -t $SESSION_NAME"
echo "停止会话: tmux kill-session -t $SESSION_NAME"
