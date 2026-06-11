"""Unified dataset generation entrypoint."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.manifest_utils import get_git_commit, read_parametric_stats, write_manifest
from tools.run_summary import write_run_summary
from tools.workflow_config import apply_preset, build_generator_command, ensure_python, load_yaml, select_python, validate_dataset_config


def build_plan(config_path: str | Path, preset: str | None) -> tuple[Path, List[str], dict]:
    config = load_yaml(config_path)
    apply_preset(config, preset)
    validate_dataset_config(config)
    output_dir, cmd = build_generator_command(config)
    return output_dir, cmd, config


def _is_falcor_generator(cmd: List[str]) -> bool:
    if not cmd:
        return False
    script = Path(cmd[0])
    parts = set(script.parts)
    return "1_data_generation" in parts and "falcor" in parts


def _has_cli_flag(cmd: List[str], flag: str) -> bool:
    return flag in cmd


def _infer_falcor_python_path(python_bin: str | None) -> str | None:
    env_path = os.environ.get("FALCOR_PYTHON_PATH")
    if env_path:
        return env_path

    if python_bin:
        python_path = Path(python_bin).resolve()
        # Falcor's bundled interpreter lives under:
        #   <build>/bin/{Debug,Release}/pythondist/bin/python3.10
        # while its Python extension package lives under:
        #   <build>/bin/{Debug,Release}/python
        parents = list(python_path.parents)
        if len(parents) >= 3 and parents[1].name == "pythondist":
            candidate = parents[2] / "python"
            if candidate.exists():
                return str(candidate)

    for build_type in ("Release", "Debug"):
        candidate = ROOT / "Falcor" / "build" / "linux-gcc" / "bin" / build_type / "python"
        if candidate.exists():
            return str(candidate)

    return None


def _config_falcor_python_path(config: dict) -> str | None:
    for container in (
        config,
        config.get("generator", {}) if isinstance(config.get("generator"), dict) else {},
    ):
        value = container.get("falcor_python_path")
        if value:
            return str(value)
    args = config.get("generator", {}).get("args", {}) if isinstance(config.get("generator"), dict) else {}
    value = args.get("falcor_python_path")
    return str(value) if value else None


def _prepare_falcor_command(cmd: List[str], config: dict, python_bin: str | None, cli_path: str | None) -> tuple[List[str], str | None]:
    if not _is_falcor_generator(cmd):
        return cmd, None

    falcor_python_path = cli_path or _config_falcor_python_path(config) or _infer_falcor_python_path(python_bin)
    if falcor_python_path and not _has_cli_flag(cmd, "--falcor-python-path"):
        cmd = [*cmd, "--falcor-python-path", falcor_python_path]
    return cmd, falcor_python_path


def _build_child_env(falcor_python_path: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    if falcor_python_path:
        falcor_path = Path(falcor_python_path).resolve()
        env["PYTHONPATH"] = str(falcor_path) + os.pathsep + env["PYTHONPATH"]
        lib_dir = str(falcor_path.parent)
        env["LD_LIBRARY_PATH"] = lib_dir + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")

    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dataset generation from YAML config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preset", default=None)
    parser.add_argument("--python", default=None, help="Python executable for generator")
    parser.add_argument("--falcor-python-path", default=None, help="Falcor Python extension package path for Falcor generators")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()

    output_dir, cmd, config = build_plan(args.config, args.preset)
    python_bin = args.python or select_python(config)
    cmd, falcor_python_path = _prepare_falcor_command(cmd, config, python_bin, args.falcor_python_path)
    cmd = ensure_python(cmd, python_bin)
    child_env = _build_child_env(falcor_python_path)

    if args.dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return

    start_time = datetime.now().isoformat()
    subprocess.run(cmd, check=True, env=child_env)
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
        "falcor_python_path": str(falcor_python_path) if falcor_python_path else "none",
        "started_at": start_time,
        "finished_at": end_time,
        "git_commit": str(get_git_commit() or "N/A"),
        "command": " ".join(cmd),
    }
    write_run_summary(output_dir / "run_summary.md", "Dataset Run Summary", summary_items)


if __name__ == "__main__":  # pragma: no cover
    main()
