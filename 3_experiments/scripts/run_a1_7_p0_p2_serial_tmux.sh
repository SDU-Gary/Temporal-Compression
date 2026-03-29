#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kyrie/毕设"
LOG_DIR="$ROOT/3_experiments/results"
RESULT_ROOT="$ROOT/3_experiments/results/bistro_clean_v2"
POLL_SECONDS=30

if ! command -v tmux >/dev/null 2>&1; then
  echo "[error] tmux not found in PATH."
  exit 1
fi

run_tmux_train() {
  local session="$1"
  local config="$2"
  local log_tag="$3"
  local log_file="$LOG_DIR/tmux_${log_tag}_$(date +%Y%m%d_%H%M%S).log"

  local cmd="cd \"$ROOT\" && source venv/bin/activate && PYTHONUNBUFFERED=1 python 3_experiments/scripts/train.py --config \"$config\" --show-progress 2>&1 | tee \"$log_file\""
  tmux new-session -d -s "$session" "$cmd"
  echo "[started] session=$session"
  echo "         config=$config"
  echo "         log=$log_file"
}

is_finished() {
  local exp_dir="$1"
  local hb="$exp_dir/runtime/heartbeat.json"
  [[ -f "$hb" ]] && rg -q '"state"\s*:\s*"finished"' "$hb"
}

wait_until_done() {
  local session="$1"
  local exp_dir="$2"

  while tmux has-session -t "$session" 2>/dev/null; do
    echo "[wait] session=$session running... (poll ${POLL_SECONDS}s)"
    sleep "$POLL_SECONDS"
  done

  if is_finished "$exp_dir"; then
    echo "[done] $exp_dir finished."
    return 0
  fi

  echo "[error] session=$session exited, but experiment is not marked finished:"
  echo "        $exp_dir/runtime/heartbeat.json"
  return 1
}

echo "=== A1.7 P0-2 serial queue ==="
echo "Order: P0(seed7) -> P0(seed19) -> P0(seed73) -> P1(corr) -> P2(K1) -> P2(K30)"
echo "Behavior: skip finished, wait current session, then launch next."

declare -a NAMES=(
  "A1.7-P0-seed7"
  "A1.7-P0-seed19"
  "A1.7-P0-seed73"
  "A1.7-P1-corr-seed19"
  "A1.7-P2-K1-seed19"
  "A1.7-P2-K30-seed19"
)
declare -a SESSIONS=(
  "a17_p0_s7"
  "a17_p0_s19"
  "a17_p0_s73"
  "a17_p1_corr"
  "a17_p2_k1"
  "a17_p2_k30"
)
declare -a CONFIGS=(
  "3_experiments/configs/bistro_clean_train_A1_7_P0_bypassB_seed7.yaml"
  "3_experiments/configs/bistro_clean_train_A1_7_P0_bypassB_seed19.yaml"
  "3_experiments/configs/bistro_clean_train_A1_7_P0_bypassB_seed73.yaml"
  "3_experiments/configs/bistro_clean_train_A1_7_P1_corr_seed19.yaml"
  "3_experiments/configs/bistro_clean_train_A1_7_P2_K1_seed19.yaml"
  "3_experiments/configs/bistro_clean_train_A1_7_P2_K30_seed19.yaml"
)
declare -a LOG_TAGS=(
  "a17_p0_s7"
  "a17_p0_s19"
  "a17_p0_s73"
  "a17_p1_corr"
  "a17_p2_k1"
  "a17_p2_k30"
)
declare -a EXP_DIRS=(
  "$RESULT_ROOT/exp_A1_7_P0_bypassB_seed7"
  "$RESULT_ROOT/exp_A1_7_P0_bypassB_seed19"
  "$RESULT_ROOT/exp_A1_7_P0_bypassB_seed73"
  "$RESULT_ROOT/exp_A1_7_P1_corr_seed19"
  "$RESULT_ROOT/exp_A1_7_P2_K1_seed19"
  "$RESULT_ROOT/exp_A1_7_P2_K30_seed19"
)

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  session="${SESSIONS[$i]}"
  config="${CONFIGS[$i]}"
  log_tag="${LOG_TAGS[$i]}"
  exp_dir="${EXP_DIRS[$i]}"

  echo
  echo "==> [$name]"

  if is_finished "$exp_dir"; then
    echo "[skip] already finished: $exp_dir"
    continue
  fi

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[resume] session already exists: $session"
  else
    run_tmux_train "$session" "$config" "$log_tag"
  fi

  wait_until_done "$session" "$exp_dir"
done

echo
echo "All queued jobs processed."
