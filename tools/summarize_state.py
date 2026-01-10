#!/usr/bin/env python3
"""
Generate project_state.md from project.db.

This creates a deterministic, <500-word summary for hot-start context
after conversation compaction.
"""

import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.db_utils import get_db, get_db_last_modified

DB_PATH = Path(__file__).parent.parent / "project.db"
OUTPUT_PATH = Path(__file__).parent.parent / "project_state.md"


def generate_state():
    """Generate project_state.md from database."""

    with get_db() as conn:
        # 1. Environment config
        cursor = conn.execute("SELECT key, value FROM env_config")
        env = {row['key']: row['value'] for row in cursor.fetchall()}

        # 2. Active tasks
        cursor = conn.execute("SELECT * FROM v_active_tasks LIMIT 10")
        tasks = cursor.fetchall()

        # 3. Milestones (baseline + best)
        cursor = conn.execute("SELECT * FROM v_milestones")
        milestones = cursor.fetchall()

        # 4. Recent experiments
        cursor = conn.execute("SELECT * FROM v_recent_experiments LIMIT 5")
        recent = cursor.fetchall()

        # 5. Top datasets by usage
        cursor = conn.execute("""
            SELECT
                d.dataset_id,
                d.full_name,
                d.num_probes,
                d.num_moments,
                d.spp,
                COUNT(e.exp_id) as exp_count
            FROM datasets d
            LEFT JOIN experiments e ON d.dataset_id = e.dataset_id
            GROUP BY d.dataset_id
            ORDER BY exp_count DESC, d.dataset_id
            LIMIT 5
        """)
        datasets = cursor.fetchall()

        # 6. Script shortcuts
        cursor = conn.execute("""
            SELECT script_id, filename, method
            FROM scripts
            ORDER BY
                CASE method
                    WHEN 'PG-GCPL' THEN 1
                    WHEN 'TemporalMLP' THEN 2
                    WHEN 'TPE' THEN 3
                    ELSE 4
                END,
                script_id
            LIMIT 6
        """)
        scripts = cursor.fetchall()

    # Generate markdown
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db_modified = datetime.fromtimestamp(get_db_last_modified()).strftime("%Y-%m-%d %H:%M:%S")

    content = f"""# Project State Snapshot

**Generated**: {now}
**DB Last Updated**: {db_modified}

---

## [ENV] Environment

- **Working Dir**: {env.get('working_dir', 'N/A')}
- **Python**: {env.get('python_path', 'N/A')}
- **Conda Env**: {env.get('conda_env', 'N/A')}
- **Virtual Env**: {env.get('virtual_env', 'N/A')}

---

## [TASKS] Active Tasks ({len(tasks)} pending)

"""

    if tasks:
        content += "| Priority | Status | Task | Created |\n"
        content += "|----------|--------|------|----------|\n"
        priority_emoji = {1: '🟢 Low', 2: '🟡 Medium', 3: '🔴 High'}
        status_emoji = {'todo': '⏳', 'doing': '🔄'}
        for task in tasks:
            pri = priority_emoji.get(task['priority'], str(task['priority']))
            status_sym = status_emoji.get(task['status'], '')
            created = task['created'].split()[0] if task['created'] else 'N/A'  # Date only
            content += f"| {pri} | {status_sym} {task['status']} | {task['task_desc'][:50]} | {created} |\n"
    else:
        content += "*No active tasks*\n"

    content += "\n---\n\n## [MILESTONES] Key Results\n\n"

    baseline = [m for m in milestones if m['type'] == 'BASELINE']
    best = [m for m in milestones if m['type'] == 'BEST_PSNR']

    if baseline:
        b = baseline[0]
        psnr_str = f"{b['psnr']:.2f}" if b['psnr'] is not None else 'N/A'
        ssim_str = f"{b['ssim']:.3f}" if b['ssim'] is not None else 'N/A'
        comp_str = f"{b['compress_ratio']:.2f}" if b['compress_ratio'] is not None else 'N/A'
        param_str = f"{b['param_count']:,}" if b['param_count'] is not None else 'N/A'
        content += f"""### Current Baseline
- **{b['exp_id']}** ({b['phase']})
- PSNR: {psnr_str} dB, SSIM: {ssim_str}
- Compression: {comp_str}×, Params: {param_str}

"""
    else:
        content += "### Current Baseline\n*No baseline set*\n\n"

    if best:
        b = best[0]
        is_same = baseline and b['exp_id'] == baseline[0]['exp_id']
        if is_same:
            content += "### Historical Best\n*Same as current baseline*\n\n"
        else:
            psnr_str = f"{b['psnr']:.2f}" if b['psnr'] is not None else 'N/A'
            ssim_str = f"{b['ssim']:.3f}" if b['ssim'] is not None else 'N/A'
            content += f"""### Historical Best
- **{b['exp_id']}** ({b['phase']})
- PSNR: {psnr_str} dB, SSIM: {ssim_str}

"""
    else:
        content += "### Historical Best\n*No experiments with PSNR recorded*\n\n"

    content += "---\n\n## [RECENT] Last 5 Experiments\n\n"

    if recent:
        content += "| Run | Exp ID | Phase | Dataset | PSNR | SSIM | Status |\n"
        content += "|-----|--------|-------|---------|------|------|--------|\n"
        for row in recent:
            psnr_str = f"{row['psnr']:.1f}" if row['psnr'] else "-"
            ssim_str = f"{row['ssim']:.3f}" if row['ssim'] else "-"
            status = "⭐ BASELINE" if row['is_baseline'] else "✅"
            phase_short = row['phase'].split('_')[-1] if row['phase'] else 'N/A'
            content += f"| {row['run_order']} | {row['exp_id']} | {phase_short} | {row['dataset_id']} | {psnr_str} | {ssim_str} | {status} |\n"
    else:
        content += "*No experiments logged yet*\n"

    content += "\n---\n\n## [MAPS] Dataset-Scene Mapping (Top 5 Used)\n\n"

    if datasets:
        for d in datasets:
            probes = f"{d['num_probes']:,}" if d['num_probes'] else 'N/A'
            moments = d['num_moments'] if d['num_moments'] else 'N/A'
            spp = d['spp'] if d['spp'] else 'N/A'
            exp_count = d['exp_count']
            content += f"- **{d['dataset_id']}** ({d['full_name']}): {probes} probes × {moments} moments, SPP={spp} ({exp_count} exps)\n"
    else:
        content += "*No datasets registered*\n"

    content += "\n---\n\n## [QUICK_REF] Script Shortcuts\n\n"

    if scripts:
        for s in scripts:
            method_tag = f"[{s['method']}]" if s['method'] else ""
            filename_short = Path(s['filename']).name
            content += f"- `{s['script_id']}`: {filename_short} {method_tag}\n"
    else:
        content += "*No scripts registered*\n"

    content += """
---

## Action Protocol

**After conversation compact**:
1. Run: `python tools/summarize_state.py` to regenerate this file
2. Run: `cat project_state.md` to read context
3. Read [ENV], [TASKS], [MILESTONES], [RECENT] sections
4. **NEVER guess dataset paths** - query DB or check this file

**After completing experiment**:
1. Log immediately: `python tools/logexp.py log --phase ... --results ...`
2. If baseline: `python tools/logexp.py set-baseline <exp_id>`
3. Regenerate: `python tools/summarize_state.py`

**For task planning**:
1. Add task: `python tools/logexp.py add-task "..." --priority 2`
2. Complete: `python tools/logexp.py complete-task <id>`
"""

    # Write to file
    with open(OUTPUT_PATH, 'w') as f:
        f.write(content)

    return OUTPUT_PATH


if __name__ == '__main__':
    output = generate_state()
    print(f"✅ Generated {output}")

    # Check file size
    size = output.stat().st_size
    words = len(output.read_text().split())
    print(f"   Size: {size:,} bytes ({words} words)")

    if words > 500:
        print(f"   ⚠️  Warning: Exceeds 500-word target")
    else:
        print(f"   ✅ Within 500-word target")
