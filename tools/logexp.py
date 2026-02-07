#!/usr/bin/env python3
"""
CLI tool for experiment logging, task management, and baseline setting.

Usage:
    python tools/logexp.py log --phase Phase2_PGCPL --stage training ...
    python tools/logexp.py set-baseline EXP-20260104-001
    python tools/logexp.py add-task "Implement energy conservation loss" --priority 3
    python tools/logexp.py list-tasks --status todo
    python tools/logexp.py complete-task 5
    python tools/logexp.py query --phase Phase2_PGCPL --min-psnr 30
"""

import argparse
import json
import sys
import time
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.db_utils import get_db, get_next_run_order, get_next_exp_id


def log_experiment(
    phase: str,
    stage: str,
    script_id: str,
    dataset_id: str,
    scene_id: Optional[str] = None,
    hyperparams: Optional[Dict[str, Any]] = None,
    results: Optional[Dict[str, Any]] = None,
    checkpoint_path: Optional[str] = None,
    notes: Optional[str] = None,
    is_baseline: bool = False,
) -> str:
    """
    Log experiment to database with auto-incrementing run_order.

    Args:
        phase: Experiment phase (e.g. Phase2_PGCPL)
        stage: Experiment stage (validation, training, ablation, visualization, evaluation)
        script_id: Script identifier (short name from scripts table)
        dataset_id: Dataset identifier (short name from datasets table)
        scene_id: Scene identifier (optional)
        hyperparams: Dictionary of hyperparameters
        results: Dictionary of result metrics
        checkpoint_path: Path to checkpoint directory
        notes: Free-form notes
        is_baseline: Whether this is the baseline

    Returns:
        Generated exp_id
    """
    hyperparams = hyperparams or {}
    results = results or {}

    # Generate exp_id and run_order
    exp_id = get_next_exp_id()
    run_order = get_next_run_order()

    # Extract core hyperparameters
    num_gaussians = hyperparams.get('num_gaussians') or hyperparams.get('K')
    rank = hyperparams.get('rank') or hyperparams.get('r')
    latent_dim = hyperparams.get('latent_dim')
    learning_rate = hyperparams.get('learning_rate') or hyperparams.get('lr')
    num_epochs = hyperparams.get('num_epochs') or hyperparams.get('epochs')

    # Extract core results
    psnr = results.get('psnr')
    ssim = results.get('ssim')
    mae = results.get('mae')
    rmse = results.get('rmse')
    compress_ratio = results.get('compress_ratio') or results.get('compression_ratio')
    param_count = results.get('param_count') or results.get('params')
    query_latency_ms = results.get('query_latency_ms') or results.get('latency')

    # If baseline, unmark previous baseline in same phase
    if is_baseline:
        with get_db() as conn:
            conn.execute(
                "UPDATE experiments SET is_baseline = 0 WHERE phase = ? AND is_baseline = 1",
                (phase,)
            )

    # Insert experiment
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO experiments (
                exp_id, run_order, phase, stage, script_id, dataset_id, scene_id,
                num_gaussians, rank, latent_dim, learning_rate, num_epochs,
                psnr, ssim, mae, rmse, compress_ratio, param_count, query_latency_ms,
                is_baseline, args_json, results_json, checkpoint_path, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                exp_id, run_order, phase, stage, script_id, dataset_id, scene_id,
                num_gaussians, rank, latent_dim, learning_rate, num_epochs,
                psnr, ssim, mae, rmse, compress_ratio, param_count, query_latency_ms,
                int(is_baseline), json.dumps(hyperparams), json.dumps(results),
                checkpoint_path, notes, int(time.time())
            )
        )

    return exp_id


def set_baseline(exp_id: str):
    """Mark experiment as baseline, unmark previous baseline in same phase."""
    with get_db() as conn:
        # Get phase of exp_id
        cursor = conn.execute("SELECT phase FROM experiments WHERE exp_id = ?", (exp_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Experiment {exp_id} not found")

        phase = row['phase']

        # Unmark old baseline
        conn.execute(
            "UPDATE experiments SET is_baseline = 0 WHERE phase = ? AND is_baseline = 1",
            (phase,)
        )

        # Mark new baseline
        conn.execute("UPDATE experiments SET is_baseline = 1 WHERE exp_id = ?", (exp_id,))

    print(f"✅ Set {exp_id} as baseline for {phase}")


def add_task(desc: str, priority: int = 1, related_exp_id: Optional[str] = None):
    """Add task to todo list."""
    with get_db() as conn:
        conn.execute(
            "INSERT INTO tasks (task_desc, status, priority, related_exp_id, created_at) VALUES (?, ?, ?, ?, ?)",
            (desc, 'todo', priority, related_exp_id, int(time.time()))
        )

    priority_emoji = {1: '🟢', 2: '🟡', 3: '🔴'}
    print(f"✅ Added task: {priority_emoji.get(priority, '')} {desc}")


def complete_task(task_id: int):
    """Mark task as done."""
    with get_db() as conn:
        cursor = conn.execute("SELECT task_desc FROM tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Task {task_id} not found")

        task_desc = row['task_desc']

        conn.execute(
            "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?",
            (int(time.time()), task_id)
        )

    print(f"✅ Completed task #{task_id}: {task_desc}")


def list_tasks(status: Optional[str] = None):
    """List tasks, optionally filtered by status."""
    with get_db() as conn:
        if status:
            cursor = conn.execute(
                "SELECT * FROM v_active_tasks WHERE status = ? ORDER BY priority DESC, created",
                (status,)
            )
        else:
            cursor = conn.execute("SELECT * FROM v_active_tasks ORDER BY priority DESC, created")

        tasks = cursor.fetchall()

    if not tasks:
        print("No tasks found")
        return

    priority_emoji = {1: '🟢', 2: '🟡', 3: '🔴'}
    status_emoji = {'todo': '⏳', 'doing': '🔄', 'done': '✅'}

    print(f"\n{'ID':<5} {'Pri':<5} {'Status':<8} {'Task':<50} {'Created':<20}")
    print("=" * 90)
    for task in tasks:
        pri_sym = priority_emoji.get(task['priority'], '')
        status_sym = status_emoji.get(task['status'], '')
        print(
            f"{task['id']:<5} {pri_sym:<5} {status_sym} {task['status']:<6} {task['task_desc']:<50} {task['created']:<20}"
        )


def update_env():
    """Auto-detect and update environment config."""
    env_updates = {
        'working_dir': os.getcwd(),
        'python_path': sys.executable,
        'conda_env': os.environ.get('CONDA_DEFAULT_ENV', 'N/A'),
        'virtual_env': os.environ.get('VIRTUAL_ENV', 'N/A'),
    }

    with get_db() as conn:
        for key, value in env_updates.items():
            conn.execute(
                "INSERT OR REPLACE INTO env_config (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, int(time.time()))
            )

    print("✅ Environment config updated:")
    for key, value in env_updates.items():
        print(f"   {key}: {value}")


def query_experiments(
    phase: Optional[str] = None,
    dataset_id: Optional[str] = None,
    min_psnr: Optional[float] = None,
    baseline_only: bool = False,
    limit: int = 10,
):
    """Query experiments with filters."""
    conditions = []
    params = []

    if phase:
        conditions.append("phase = ?")
        params.append(phase)

    if dataset_id:
        conditions.append("d.dataset_id = ?")
        params.append(dataset_id)

    if min_psnr is not None:
        conditions.append("psnr >= ?")
        params.append(min_psnr)

    if baseline_only:
        conditions.append("is_baseline = 1")

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    with get_db() as conn:
        cursor = conn.execute(
            f"""
            SELECT
                e.exp_id,
                e.run_order,
                e.phase,
                e.stage,
                s.filename as script,
                d.dataset_id,
                e.psnr,
                e.ssim,
                e.compress_ratio,
                e.is_baseline,
                datetime(e.created_at, 'unixepoch') as created
            FROM experiments e
            JOIN scripts s ON e.script_id = s.script_id
            JOIN datasets d ON e.dataset_id = d.dataset_id
            WHERE {where_clause}
            ORDER BY e.run_order DESC
            LIMIT ?
            """,
            params + [limit]
        )

        results = cursor.fetchall()

    if not results:
        print("No experiments found")
        return

    print(f"\n{'Run':<5} {'Exp ID':<18} {'Phase':<16} {'Dataset':<12} {'PSNR':<8} {'SSIM':<8} {'Comp':<8} {'Baseline'}")
    print("=" * 100)
    for row in results:
        baseline_mark = "⭐" if row['is_baseline'] else ""
        psnr_str = f"{row['psnr']:.2f}" if row['psnr'] else "-"
        ssim_str = f"{row['ssim']:.3f}" if row['ssim'] else "-"
        comp_str = f"{row['compress_ratio']:.1f}×" if row['compress_ratio'] else "-"

        print(
            f"{row['run_order']:<5} {row['exp_id']:<18} {row['phase']:<16} {row['dataset_id']:<12} "
            f"{psnr_str:<8} {ssim_str:<8} {comp_str:<8} {baseline_mark}"
        )


def main():
    parser = argparse.ArgumentParser(description="Experiment logging and task management CLI")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # log command
    log_parser = subparsers.add_parser('log', help='Log experiment')
    log_parser.add_argument('--phase', required=True, help='Experiment phase')
    log_parser.add_argument('--stage', required=True, help='Experiment stage')
    log_parser.add_argument('--script', required=True, help='Script ID')
    log_parser.add_argument('--dataset', required=True, help='Dataset ID')
    log_parser.add_argument('--scene', help='Scene ID (optional)')
    log_parser.add_argument('--hyperparams', help='Hyperparameters JSON string')
    log_parser.add_argument('--results', help='Results JSON string')
    log_parser.add_argument('--checkpoint', help='Checkpoint path')
    log_parser.add_argument('--notes', help='Notes')
    log_parser.add_argument('--baseline', action='store_true', help='Mark as baseline')

    # set-baseline command
    baseline_parser = subparsers.add_parser('set-baseline', help='Set experiment as baseline')
    baseline_parser.add_argument('exp_id', help='Experiment ID')

    # add-task command
    addtask_parser = subparsers.add_parser('add-task', help='Add task')
    addtask_parser.add_argument('desc', help='Task description')
    addtask_parser.add_argument('--priority', type=int, default=1, help='Priority (1=low, 2=medium, 3=high)')
    addtask_parser.add_argument('--related-exp', help='Related experiment ID')

    # complete-task command
    complete_parser = subparsers.add_parser('complete-task', help='Complete task')
    complete_parser.add_argument('task_id', type=int, help='Task ID')

    # list-tasks command
    listtasks_parser = subparsers.add_parser('list-tasks', help='List tasks')
    listtasks_parser.add_argument('--status', choices=['todo', 'doing', 'done'], help='Filter by status')

    # update-env command
    subparsers.add_parser('update-env', help='Update environment config')

    # query command
    query_parser = subparsers.add_parser('query', help='Query experiments')
    query_parser.add_argument('--phase', help='Filter by phase')
    query_parser.add_argument('--dataset', help='Filter by dataset')
    query_parser.add_argument('--min-psnr', type=float, help='Minimum PSNR')
    query_parser.add_argument('--baseline-only', action='store_true', help='Show only baselines')
    query_parser.add_argument('--limit', type=int, default=10, help='Max results')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == 'log':
            hyperparams = json.loads(args.hyperparams) if args.hyperparams else None
            results = json.loads(args.results) if args.results else None

            exp_id = log_experiment(
                phase=args.phase,
                stage=args.stage,
                script_id=args.script,
                dataset_id=args.dataset,
                scene_id=args.scene,
                hyperparams=hyperparams,
                results=results,
                checkpoint_path=args.checkpoint,
                notes=args.notes,
                is_baseline=args.baseline,
            )
            print(f"✅ Logged experiment: {exp_id} (run_order: {get_next_run_order() - 1})")

        elif args.command == 'set-baseline':
            set_baseline(args.exp_id)

        elif args.command == 'add-task':
            add_task(args.desc, args.priority, args.related_exp)

        elif args.command == 'complete-task':
            complete_task(args.task_id)

        elif args.command == 'list-tasks':
            list_tasks(args.status)

        elif args.command == 'update-env':
            update_env()

        elif args.command == 'query':
            query_experiments(
                phase=args.phase,
                dataset_id=args.dataset,
                min_psnr=args.min_psnr,
                baseline_only=args.baseline_only,
                limit=args.limit,
            )

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':  # pragma: no cover
    main()
