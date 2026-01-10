#!/bin/bash
# Monitor rendering progress in real-time

LOG_FILE="/home/kyrie/毕设/data_generation/logs/generation.log"

echo "=========================================="
echo "Rendering Progress Monitor"
echo "=========================================="
echo ""

# Check if process is running
if pgrep -f "generate_transfer_tensor.py" > /dev/null; then
    echo "✓ Rendering process is RUNNING"

    # Show process info
    ps aux | grep generate_transfer_tensor | grep -v grep | awk '{printf "  PID: %s, CPU: %s%%, MEM: %s%%\n", $2, $3, $4}'
    echo ""
else
    echo "✗ Rendering process NOT FOUND"
    echo ""
fi

# Show latest progress
echo "Latest progress:"
tail -3 "$LOG_FILE" | grep -v "^$"
echo ""

# Estimate completion time
CURRENT_LIGHT=$(tail -50 "$LOG_FILE" | grep "Light positions:" | tail -1 | grep -oP '\d+/\d+' | head -1)
if [ ! -z "$CURRENT_LIGHT" ]; then
    echo "Current light position: $CURRENT_LIGHT"
fi

# Check for checkpoints
CHECKPOINT_COUNT=$(ls -1 /home/kyrie/毕设/data_generation/output/transfer_tensor_validation/checkpoint_*.npz 2>/dev/null | wc -l)
if [ $CHECKPOINT_COUNT -gt 0 ]; then
    echo "Checkpoints saved: $CHECKPOINT_COUNT"
    ls -lh /home/kyrie/毕设/data_generation/output/transfer_tensor_validation/checkpoint_*.npz 2>/dev/null | tail -3
fi

echo ""
echo "=========================================="
echo "Commands:"
echo "  Watch live: tail -f $LOG_FILE"
echo "  This monitor: bash monitor_rendering.sh"
echo "=========================================="
