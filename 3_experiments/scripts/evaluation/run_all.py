"""Unified evaluation runner (dry-run friendly)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List


def build_eval_steps(output_dir: str | Path) -> List[Dict[str, str]]:
    out_dir = Path(output_dir)
    steps = [
        {
            "name": "interpolation",
            "script": "3_experiments/scripts/evaluation/test_interpolation_query_physics.py",
            "args": ["--output", str(out_dir / "interpolation")],
        },
        {
            "name": "extrapolation",
            "script": "3_experiments/scripts/evaluation/test_extrapolation_query_physics.py",
            "args": ["--output", str(out_dir / "extrapolation")],
        },
        {
            "name": "latency",
            "script": "3_experiments/scripts/evaluation/test_query_latency_physics.py",
            "args": ["--output", str(out_dir / "latency")],
        },
    ]
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified evaluation suite")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    steps = build_eval_steps(args.output_dir)
    for step in steps:
        cmd = [step["script"], *step["args"]]
        if args.python:
            cmd = [args.python] + cmd
        if args.dry_run:
            print(f"DRY-RUN [{step['name']}]", " ".join(cmd))
            continue
        import subprocess
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
