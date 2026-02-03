import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = ROOT / "1_data_generation"
if str(VALIDATOR_PATH) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_PATH))

from validate_dataset import DatasetValidator


def _make_parametric_dataset(tmp_path: Path, num_probes: int = 4, num_configs: int = 3) -> None:
    tensor = np.zeros((num_probes, num_configs, 27), dtype=np.float32)
    probes = np.zeros((num_probes, 3), dtype=np.float32)
    metadata = {"num_probes": num_probes, "num_configs": num_configs, "spp": 16, "scene": "scene.pyscene"}
    np.savez(tmp_path / "parametric_tensor.npz", tensor=tensor, probe_positions=probes, metadata=metadata)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))


def test_parametric_dataset_pass(tmp_path: Path):
    _make_parametric_dataset(tmp_path)
    manifest = {"dataset_id": "TEST", "output_dir": str(tmp_path), "generator": "gen.py", "created_at": "now",
                "num_probes": 4, "num_configs": 3}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    validator = DatasetValidator(tmp_path)
    report = validator.validate()
    assert report["status"] == "PASS"
    assert report["stats"]["num_probes"] == 4


def test_manifest_mismatch_flags_error(tmp_path: Path):
    _make_parametric_dataset(tmp_path)
    manifest = {"dataset_id": "TEST", "output_dir": str(tmp_path), "generator": "gen.py", "created_at": "now",
                "num_probes": 5, "num_configs": 3}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    validator = DatasetValidator(tmp_path)
    report = validator.validate()
    assert report["status"] == "FAIL"
    assert any("manifest.json" in err for err in report["errors"])
