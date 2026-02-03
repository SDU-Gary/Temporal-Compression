"""Manifest helpers for dataset generation."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _git_commit(root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def get_git_commit(root: Path | None = None) -> Optional[str]:
    root = root or Path(__file__).resolve().parents[1]
    return _git_commit(root)


def read_parametric_stats(output_dir: Path) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    tensor_path = output_dir / "parametric_tensor.npz"
    if tensor_path.exists():
        data = np.load(tensor_path, allow_pickle=True)
        tensor = data["tensor"]
        stats["num_probes"] = int(tensor.shape[0])
        stats["num_configs"] = int(tensor.shape[1])
        if "metadata" in data:
            metadata = data["metadata"].item()
            if isinstance(metadata, dict):
                stats["num_frames"] = int(metadata.get("num_frames", tensor.shape[1]))
                stats["spp"] = int(metadata.get("spp", 0))
                stats["scene"] = metadata.get("scene")
                stats["scene_json"] = metadata.get("scene_json")
    return stats


def write_manifest(
    output_dir: Path,
    dataset_id: str,
    generator: str,
    config_path: str,
    extra: Optional[Dict[str, Any]] = None,
    root: Optional[Path] = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = root or Path(__file__).resolve().parents[1]
    payload: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "output_dir": str(output_dir.resolve()),
        "generator": generator,
        "config_path": str(config_path),
        "created_at": datetime.now().isoformat(),
        "git_commit": _git_commit(root),
    }
    if extra:
        payload.update(extra)
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(payload, f, indent=2)
    return manifest_path


def load_manifest(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r") as f:
        return json.load(f)
