from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


common = _load_module("experiment_driver_common_test", ROOT / "3_experiments" / "experiment_driver_common.py")


def test_set_nested_value() -> None:
    payload = {}
    common.set_nested_value(payload, "model.rank", 8)
    assert payload == {"model": {"rank": 8}}


def test_modify_config_and_read_json(tmp_path: Path) -> None:
    cfg = tmp_path / "base.yaml"
    out = tmp_path / "tmp.yaml"
    cfg.write_text(
        """
experiment:
  output_dir: __OUT__
model:
  rank: 4
""".replace("__OUT__", str(tmp_path / "results")),
        encoding="utf-8",
    )

    merged = common.modify_config(str(cfg), {"model.rank": 12}, str(out))
    assert out.exists()
    assert merged["model"]["rank"] == 12
    assert (tmp_path / "results").exists()

    missing = common.read_json_if_exists(tmp_path / "missing.json", default={"ok": False})
    assert missing == {"ok": False}

    sample_json = tmp_path / "sample.json"
    sample_json.write_text(json.dumps({"ok": True}), encoding="utf-8")
    loaded = common.read_json_if_exists(sample_json)
    assert loaded == {"ok": True}


def test_run_command_writes_log(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    code = common.run_command([sys.executable, "-c", "print('ok')"], "unit-test", str(log_path))
    assert code == 0
    text = log_path.read_text(encoding="utf-8")
    assert "Command:" in text
    assert "Exit code:" in text

