import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "3_experiments" / "scripts"
if str(TRAIN_PATH) not in sys.path:
    sys.path.insert(0, str(TRAIN_PATH))

from train import resolve_data_root


def test_resolve_data_root_direct():
    assert resolve_data_root("/tmp/data", None) == "/tmp/data"


def test_resolve_data_root_from_manifest(tmp_path: Path):
    manifest = {"output_dir": str(tmp_path)}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert resolve_data_root(None, str(path)) == str(tmp_path)


def test_resolve_data_root_missing_manifest_raises():
    with pytest.raises(ValueError):
        resolve_data_root(None, None)
