#!/bin/bash
# SessionStart Hook - Auto-load experiment memory system after conversation compact
# This hook runs every time a new Claude Code session starts

# Exit on error
set -e

# Change to project root
cd /home/kyrie/毕设

# Step 1: Regenerate state snapshot (in case DB was updated)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 EXPERIMENT MEMORY SYSTEM - Auto-loading context"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Regenerate state (silent mode)
python tools/summarize_state.py > /dev/null 2>&1

# Step 2: Display critical sections of project_state.md
echo "📍 Current Working Directory: $(pwd)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 ACTIVE TASKS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
sed -n '/## \[TASKS\]/,/## \[MILESTONES\]/p' project_state.md | head -n -2
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏆 MILESTONES & BASELINE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
sed -n '/## \[MILESTONES\]/,/## \[RECENT\]/p' project_state.md | head -n -2
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📈 RECENT EXPERIMENTS (Last 5)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
sed -n '/## \[RECENT\]/,/## \[MAPS\]/p' project_state.md | head -n -2
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💾 DATABASE TOOLS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "• Log experiment:  python tools/logexp.py log --phase ... --results ..."
echo "• Query database:  python tools/logexp.py query --phase Phase2_PGCPL"
echo "• List tasks:      python tools/logexp.py list-tasks"
echo "• Set baseline:    python tools/logexp.py set-baseline <exp_id>"
echo "• Full state:      cat project_state.md"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚠️  CRITICAL REMINDERS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. NEVER guess dataset paths - query database or check project_state.md"
echo "2. Log experiments IMMEDIATELY after completion"
echo "3. Regenerate state after DB updates: python tools/summarize_state.py"
echo "4. Read full context: cat project_state.md (500 words)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 3: Check if there are pending tasks
pending_count=$(sqlite3 project.db "SELECT COUNT(*) FROM tasks WHERE status != 'done'" 2>/dev/null || echo "0")
if [ "$pending_count" -gt 0 ]; then
    echo "📋 You have $pending_count pending task(s). Run: python tools/logexp.py list-tasks"
    echo ""
fi

# Success marker
echo "✅ Memory system loaded successfully"
echo ""
