from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


train_hooks = _load_module(
    "train_hooks_test",
    ROOT / "3_experiments" / "scripts" / "train_hooks.py",
)


def test_heartbeat_hook_lifecycle_events() -> None:
    calls = []

    def _update(**kwargs):
        calls.append(kwargs)

    manager = train_hooks.HookManager()
    manager.register(train_hooks.HeartbeatHook(update_heartbeat=_update))
    ev = train_hooks.TrainingEvent(
        stage_index=1,
        stage_name="single_stage",
        epoch=2,
        num_epochs=5,
        run_context={"run_id": "x"},
        metrics={"val_mae": 0.1},
    )
    manager.on_stage_start(ev)
    manager.on_epoch_end(ev)
    manager.on_run_end(ev)

    assert calls[0]["last_event"] == "stage_start"
    assert calls[1]["last_event"] == "epoch_end"
    assert calls[2]["last_event"] == "train_end"
    assert calls[1]["metrics"]["val_mae"] == 0.1


def test_phase0_hook_writes_epoch_metrics(tmp_path: Path) -> None:
    rows = []

    def _append(path: Path, payload):
        rows.append((path, payload))

    hook = train_hooks.Phase0MetricsHook(
        jsonl_path=tmp_path / "phase0.jsonl",
        append_jsonl=_append,
        now_iso=lambda: "2026-01-01T00:00:00",
    )
    manager = train_hooks.HookManager()
    manager.register(hook)

    manager.on_epoch_end(
        train_hooks.TrainingEvent(
            stage_index=2,
            stage_name="stage2",
            epoch=3,
            num_epochs=10,
            run_context={"run_id": "abc"},
            metrics={"train_total": 1.2},
        )
    )

    assert len(rows) == 1
    path, payload = rows[0]
    assert path.name == "phase0.jsonl"
    assert payload["kind"] == "epoch_end"
    assert payload["stage_name"] == "stage2"
    assert payload["metrics"]["train_total"] == 1.2
