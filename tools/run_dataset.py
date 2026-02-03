"""Unified dataset generation entrypoint."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List

from tools.manifest_utils import get_git_commit, read_parametric_stats, write_manifest
from tools.run_summary import write_run_summary
from tools.workflow_config import apply_preset, build_generator_command, ensure_python, load_yaml, select_python, validate_dataset_config


def build_plan(config_path: str | Path, preset: str | None) -> tuple[Path, List[str], dict]:
    config = load_yaml(config_path)
    apply_preset(config, preset)
    validate_dataset_config(config)
    output_dir, cmd = build_generator_command(config)
    return output_dir, cmd, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dataset generation from YAML config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preset", default=None)
    parser.add_argument("--python", default=None, help="Python executable for generator")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()

    output_dir, cmd, config = build_plan(args.config, args.preset)
    python_bin = args.python or select_python(config)
    cmd = ensure_python(cmd, python_bin)

    if args.dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return

    start_time = datetime.now().isoformat()
    subprocess.run(cmd, check=True)
    end_time = datetime.now().isoformat()

    if args.write_manifest:
        extra = read_parametric_stats(output_dir)
        dataset_id = config.get("name", output_dir.name)
        write_manifest(
            output_dir=output_dir,
            dataset_id=dataset_id,
            generator=config["generator"]["script"],
            config_path=config.get("_config_path", args.config),
            extra=extra,
        )

    summary_items = {
        "config": str(args.config),
        "preset": str(args.preset) if args.preset else "none",
        "output_dir": str(output_dir),
        "python": str(python_bin) if python_bin else "default",
        "started_at": start_time,
        "finished_at": end_time,
        "git_commit": str(get_git_commit() or "N/A"),
        "command": " ".join(cmd),
    }
    write_run_summary(output_dir / "run_summary.md", "Dataset Run Summary", summary_items)


if __name__ == "__main__":
    main()
