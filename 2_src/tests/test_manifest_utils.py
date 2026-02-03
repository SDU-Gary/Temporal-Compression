import json
from pathlib import Path

import numpy as np

from tools.manifest_utils import load_manifest, read_parametric_stats, write_manifest


def test_read_parametric_stats(tmp_path: Path):
    tensor = np.zeros((2, 3, 27), dtype=np.float32)
    probes = np.zeros((2, 3), dtype=np.float32)
    metadata = {"num_frames": 3, "spp": 16, "scene": "scene.pyscene"}
    np.savez(tmp_path / "parametric_tensor.npz", tensor=tensor, probe_positions=probes, metadata=metadata)

    stats = read_parametric_stats(tmp_path)
    assert stats["num_probes"] == 2
    assert stats["num_configs"] == 3
    assert stats["num_frames"] == 3
    assert stats["spp"] == 16


def test_write_manifest_and_load(tmp_path: Path):
    path = write_manifest(tmp_path, "TEST", "gen.py", "cfg.yaml", extra={"num_probes": 5}, root=tmp_path)
    data = load_manifest(path)
    assert data["dataset_id"] == "TEST"
    assert data["generator"] == "gen.py"
    assert data["num_probes"] == 5
    assert data["git_commit"] is None
