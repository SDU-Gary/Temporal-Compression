from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_watchdog_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "tools" / "watchdog_train_heartbeat.py"
    spec = importlib.util.spec_from_file_location("watchdog_train_heartbeat", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load watchdog_train_heartbeat.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_watchdog_parses_iso_timestamp_with_z_suffix():
    mod = _load_watchdog_module()
    ts = mod._parse_iso_timestamp("2026-03-04T10:00:00Z")
    assert ts.year == 2026
    assert ts.month == 3
    assert ts.day == 4


def test_watchdog_staleness_seconds_handles_invalid_timestamp():
    mod = _load_watchdog_module()
    stale = mod._staleness_seconds({"timestamp": "invalid"}, now=100.0)
    assert stale == float("inf")


def test_watchdog_status_line_contains_core_fields(tmp_path: Path):
    mod = _load_watchdog_module()
    hb = {
        "timestamp": "2026-03-04T10:00:00",
        "state": "running",
        "last_event": "epoch_end",
        "stage_name": "joint",
        "epoch": 12,
        "num_epochs": 2000,
    }
    line = mod._status_line(hb, 31.5)
    assert "state=running" in line
    assert "event=epoch_end" in line
    assert "stage=joint" in line
    assert "epoch=12/2000" in line
    assert "stale=31.5s" in line


def test_watchdog_load_heartbeat_reads_json(tmp_path: Path):
    mod = _load_watchdog_module()
    path = tmp_path / "heartbeat.json"
    path.write_text(json.dumps({"state": "running"}), encoding="utf-8")
    payload = mod._load_heartbeat(path)
    assert payload["state"] == "running"

