"""Train lifecycle hook primitives used by train.py orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Protocol


@dataclass(frozen=True)
class TrainingEvent:
    stage_index: int
    stage_name: str
    epoch: int
    num_epochs: int
    run_context: Dict[str, Any]
    metrics: Dict[str, float] | None = None
    extra: Dict[str, Any] | None = None


class TrainHook(Protocol):
    def on_stage_start(self, event: TrainingEvent) -> None: ...

    def on_epoch_end(self, event: TrainingEvent) -> None: ...

    def on_epoch_audit(self, event: TrainingEvent) -> None: ...

    def on_stage_end(self, event: TrainingEvent) -> None: ...

    def on_run_audits(self, event: TrainingEvent) -> None: ...

    def on_cleanup(self, event: TrainingEvent) -> None: ...

    def on_run_end(self, event: TrainingEvent) -> None: ...


class HookManager:
    def __init__(self) -> None:
        self._hooks: List[TrainHook] = []

    def register(self, hook: TrainHook) -> None:
        self._hooks.append(hook)

    def on_stage_start(self, event: TrainingEvent) -> None:
        for hook in self._hooks:
            fn = getattr(hook, "on_stage_start", None)
            if callable(fn):
                fn(event)

    def on_epoch_end(self, event: TrainingEvent) -> None:
        for hook in self._hooks:
            fn = getattr(hook, "on_epoch_end", None)
            if callable(fn):
                fn(event)

    def on_epoch_audit(self, event: TrainingEvent) -> None:
        for hook in self._hooks:
            fn = getattr(hook, "on_epoch_audit", None)
            if callable(fn):
                fn(event)

    def on_stage_end(self, event: TrainingEvent) -> None:
        for hook in self._hooks:
            fn = getattr(hook, "on_stage_end", None)
            if callable(fn):
                fn(event)

    def on_run_audits(self, event: TrainingEvent) -> None:
        for hook in self._hooks:
            fn = getattr(hook, "on_run_audits", None)
            if callable(fn):
                fn(event)

    def on_cleanup(self, event: TrainingEvent) -> None:
        for hook in self._hooks:
            fn = getattr(hook, "on_cleanup", None)
            if callable(fn):
                fn(event)

    def on_run_end(self, event: TrainingEvent) -> None:
        for hook in self._hooks:
            fn = getattr(hook, "on_run_end", None)
            if callable(fn):
                fn(event)


class HeartbeatHook:
    """Lifecycle heartbeat hook that delegates heartbeat writes to callback."""

    def __init__(
        self,
        *,
        update_heartbeat: Callable[..., None],
    ) -> None:
        self._update = update_heartbeat

    def on_stage_start(self, event: TrainingEvent) -> None:
        self._update(
            state="running",
            last_event="stage_start",
            stage_index=event.stage_index,
            stage_name=event.stage_name,
            epoch=event.epoch,
            num_epochs=event.num_epochs,
            metrics=None,
            extra=event.extra,
        )

    def on_epoch_end(self, event: TrainingEvent) -> None:
        self._update(
            state="running",
            last_event="epoch_end",
            stage_index=event.stage_index,
            stage_name=event.stage_name,
            epoch=event.epoch,
            num_epochs=event.num_epochs,
            metrics=event.metrics,
            extra=event.extra,
        )

    def on_run_end(self, event: TrainingEvent) -> None:
        self._update(
            state="finished",
            last_event="train_end",
            stage_index=event.stage_index,
            stage_name=event.stage_name,
            epoch=event.epoch,
            num_epochs=event.num_epochs,
            metrics=event.metrics,
            extra=event.extra,
        )


class Phase0MetricsHook:
    """Hook that writes epoch-end phase0 metrics into jsonl stream."""

    def __init__(
        self,
        *,
        jsonl_path: Path,
        append_jsonl: Callable[[Path, Dict[str, Any]], None],
        now_iso: Callable[[], str],
    ) -> None:
        self._path = jsonl_path
        self._append = append_jsonl
        self._now_iso = now_iso

    def on_epoch_end(self, event: TrainingEvent) -> None:
        self._append(
            self._path,
            {
                "timestamp": self._now_iso(),
                "kind": "epoch_end",
                "stage_index": int(event.stage_index),
                "stage_name": str(event.stage_name),
                "epoch": int(event.epoch),
                "num_epochs": int(event.num_epochs),
                "metrics": dict(event.metrics or {}),
            },
        )
