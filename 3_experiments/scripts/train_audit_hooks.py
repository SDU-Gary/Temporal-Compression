"""Pluggable audit hooks extracted from train.py orchestration."""

from __future__ import annotations

import atexit
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

from train_hooks import TrainingEvent


@dataclass
class PostTrainingAuditOutputs:
    val_profile_test_compare: Dict[str, Any] | None = None
    coeff_suite_summary: Dict[str, Any] | None = None
    coeff_suite_multi_summary: Dict[str, Any] | None = None
    expert_util_summary: Dict[str, Any] | None = None
    semantic_drift_summary: Dict[str, Any] | None = None


class FalcorPeriodicAuditHook:
    """Runs Falcor periodic and stage-end-full evaluations as a pluggable hook."""

    def __init__(
        self,
        *,
        model: Any,
        eval_cfg: Dict[str, Any],
        phase0_metrics_path: Path,
        falcor_periodic_history_path: Path,
        now_iso: Callable[[], str],
        append_jsonl: Callable[[Path, Dict[str, Any]], None],
        update_heartbeat: Callable[..., None],
        run_falcor_eval: Callable[[Path, Path], Dict[str, Any] | None],
        save_checkpoint: Callable[[Path, int, Dict[str, Any]], None],
        extract_metrics_from_summary: Callable[[Dict[str, Any]], Dict[str, float]],
        primary_metric_name: str,
        primary_metric_value: Callable[[Dict[str, float]], float],
        is_better_than_best: Callable[[float], bool],
        update_best_checkpoint: Callable[..., None],
    ) -> None:
        self._model = model
        self._cfg = eval_cfg
        self._phase0_metrics_path = phase0_metrics_path
        self._history_path = falcor_periodic_history_path
        self._now_iso = now_iso
        self._append_jsonl = append_jsonl
        self._update_heartbeat = update_heartbeat
        self._run_falcor_eval = run_falcor_eval
        self._save_checkpoint = save_checkpoint
        self._extract_metrics_from_summary = extract_metrics_from_summary
        self._primary_metric_name = str(primary_metric_name)
        self._primary_metric_value = primary_metric_value
        self._is_better_than_best = is_better_than_best
        self._update_best_checkpoint = update_best_checkpoint

        self._async_enabled = bool(self._cfg.get("enabled", False)) and bool(self._cfg.get("async_mode", True))
        self._executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1) if self._async_enabled else None
        )
        self._pending: List[Dict[str, Any]] = []
        self._cleanup_done = False
        atexit.register(self._exit_guard)

    def _heartbeat(
        self,
        *,
        event: TrainingEvent,
        state: str,
        last_event: str,
        metrics: Dict[str, float] | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        self._update_heartbeat(
            state=state,
            last_event=last_event,
            stage_index=int(event.stage_index),
            stage_name=str(event.stage_name),
            epoch=int(event.epoch),
            num_epochs=int(event.num_epochs),
            metrics=metrics,
            extra=extra,
        )

    def _falcor_task_runner(self, *, checkpoint_path: Path, output_dir: Path) -> Dict[str, Any]:
        try:
            summary = self._run_falcor_eval(checkpoint_path, output_dir)
            return {"status": "success", "summary": summary}
        except subprocess.TimeoutExpired as timeout_error:
            return {"status": "timeout", "error": str(timeout_error)}
        except Exception as falcor_error:
            return {"status": "error", "error": str(falcor_error)}

    def _schedule_task(
        self,
        *,
        event: TrainingEvent,
        checkpoint_path: Path,
        run_dir: Path,
    ) -> None:
        if self._executor is None:
            return
        future: Future = self._executor.submit(
            self._falcor_task_runner,
            checkpoint_path=checkpoint_path,
            output_dir=run_dir,
        )
        self._pending.append(
            {
                "future": future,
                "stage_index": int(event.stage_index),
                "stage_name": str(event.stage_name),
                "epoch": int(event.epoch),
                "num_epochs": int(event.num_epochs),
                "checkpoint_path": str(checkpoint_path),
                "run_dir": str(run_dir),
            }
        )

    def _append_success_payloads(
        self,
        *,
        stage_index: int,
        stage_name: str,
        epoch: int,
        num_epochs: int,
        checkpoint_path: Path,
        summary_path: Path,
        primary: float,
        metrics: Dict[str, float],
        async_mode: bool,
        kind: str,
    ) -> None:
        self._append_jsonl(
            self._history_path,
            {
                "timestamp": self._now_iso(),
                "kind": kind,
                "stage_index": int(stage_index),
                "stage_name": str(stage_name),
                "epoch": int(epoch),
                "metric_name": self._primary_metric_name,
                "metric_value": float(primary) if np.isfinite(primary) else None,
                "metrics": metrics,
                "summary": str(summary_path),
                "checkpoint": str(checkpoint_path),
                "async": bool(async_mode),
            },
        )

        if kind == "periodic":
            self._append_jsonl(
                self._phase0_metrics_path,
                {
                    "timestamp": self._now_iso(),
                    "kind": "falcor_periodic",
                    "stage_index": int(stage_index),
                    "stage_name": str(stage_name),
                    "epoch": int(epoch),
                    "num_epochs": int(num_epochs),
                    "metrics": {
                        "falcor_primary": float(primary) if np.isfinite(primary) else 0.0,
                        **metrics,
                    },
                    "summary": str(summary_path),
                    "async": bool(async_mode),
                },
            )

    def _handle_success(
        self,
        *,
        stage_index: int,
        stage_name: str,
        epoch: int,
        num_epochs: int,
        checkpoint_path: Path,
        summary_path: Path,
        summary: Dict[str, Any] | None,
        async_mode: bool,
        best_kind: str,
        history_kind: str,
    ) -> None:
        metrics = self._extract_metrics_from_summary(summary or {})
        primary = self._primary_metric_value(metrics)
        self._append_success_payloads(
            stage_index=stage_index,
            stage_name=stage_name,
            epoch=epoch,
            num_epochs=num_epochs,
            checkpoint_path=checkpoint_path,
            summary_path=summary_path,
            primary=primary,
            metrics=metrics,
            async_mode=async_mode,
            kind=history_kind,
        )
        if self._is_better_than_best(primary):
            self._update_best_checkpoint(
                checkpoint_path=checkpoint_path,
                summary_path=summary_path,
                metric_value=float(primary),
                stage_index=int(stage_index),
                stage_name=str(stage_name),
                epoch=int(epoch),
                kind=best_kind,
            )
        self._update_heartbeat(
            state="running",
            last_event=("falcor_periodic_success" if history_kind == "periodic" else "falcor_stage_end_success"),
            stage_index=int(stage_index),
            stage_name=str(stage_name),
            epoch=int(epoch),
            num_epochs=int(num_epochs),
            metrics={"falcor_primary": float(primary) if np.isfinite(primary) else 0.0},
            extra={"summary": str(summary_path), "async": bool(async_mode)},
        )

    def _handle_timeout(
        self,
        *,
        event: TrainingEvent,
        run_dir: Path,
        error_msg: str,
        stage_end: bool,
        async_mode: bool,
    ) -> None:
        timeout_file = run_dir / "timeout.json"
        timeout_file.write_text(
            json.dumps(
                {
                    "kind": "falcor_stage_end_timeout" if stage_end else "falcor_periodic_timeout",
                    "stage": str(event.stage_name),
                    "epoch": int(event.epoch),
                    "timeout_seconds": int(self._cfg.get("timeout_seconds", 1800)),
                    "error": str(error_msg),
                    "timestamp": self._now_iso(),
                    "async": bool(async_mode),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._heartbeat(
            event=event,
            state="running",
            last_event="falcor_stage_end_timeout" if stage_end else "falcor_periodic_timeout",
            extra={"timeout_file": str(timeout_file), "async": bool(async_mode)},
        )
        if stage_end:
            print(
                "Warning: Falcor stage-end eval timeout "
                f"at stage={event.stage_name} timeout={self._cfg.get('timeout_seconds', 1800)}s"
            )
        else:
            print(
                "Warning: Falcor periodic eval timeout "
                f"at stage={event.stage_name} epoch={event.epoch} "
                f"timeout={self._cfg.get('timeout_seconds', 1800)}s"
            )
        if bool(self._cfg.get("fail_on_timeout", False)):
            if stage_end:
                raise RuntimeError(f"Falcor stage-end eval timed out at stage={event.stage_name}")
            raise RuntimeError(
                f"Falcor periodic eval timed out at stage={event.stage_name} epoch={event.epoch}"
            )

    def _handle_error(
        self,
        *,
        event: TrainingEvent,
        error_msg: str,
        stage_end: bool,
        async_mode: bool,
    ) -> None:
        self._heartbeat(
            event=event,
            state="running",
            last_event="falcor_stage_end_error" if stage_end else "falcor_periodic_error",
            extra={"error": str(error_msg), "async": bool(async_mode)},
        )
        if stage_end:
            print(f"Warning: Falcor stage-end eval failed at stage={event.stage_name}: {error_msg}")
        else:
            print(
                f"Warning: Falcor periodic eval failed "
                f"at stage={event.stage_name} epoch={event.epoch}: {error_msg}"
            )
        if bool(self._cfg.get("fail_on_error", False)):
            raise RuntimeError(str(error_msg))

    def _handle_task_done(self, task: Dict[str, Any], result: Dict[str, Any]) -> None:
        event = TrainingEvent(
            stage_index=int(task["stage_index"]),
            stage_name=str(task["stage_name"]),
            epoch=int(task["epoch"]),
            num_epochs=int(task["num_epochs"]),
            run_context={},
        )
        checkpoint_path = Path(str(task["checkpoint_path"]))
        run_dir = Path(str(task["run_dir"]))
        summary_path = run_dir / "benchmark_summary.json"
        status = str(result.get("status", "error"))
        if status == "success":
            self._handle_success(
                stage_index=int(event.stage_index),
                stage_name=str(event.stage_name),
                epoch=int(event.epoch),
                num_epochs=int(event.num_epochs),
                checkpoint_path=checkpoint_path,
                summary_path=summary_path,
                summary=result.get("summary", None),
                async_mode=True,
                best_kind="periodic_async",
                history_kind="periodic",
            )
            return
        if status == "timeout":
            self._handle_timeout(
                event=event,
                run_dir=run_dir,
                error_msg=str(result.get("error", "timeout")),
                stage_end=False,
                async_mode=True,
            )
            return
        self._handle_error(
            event=event,
            error_msg=str(result.get("error", "unknown error")),
            stage_end=False,
            async_mode=True,
        )

    def _drain_tasks(self, *, block: bool = False) -> None:
        while True:
            pending_now = list(self._pending)
            done_any = False
            for task in pending_now:
                future = task["future"]
                if block:
                    result = future.result()
                    self._pending.remove(task)
                    self._handle_task_done(task, result)
                    done_any = True
                    continue
                if future.done():
                    result = future.result()
                    self._pending.remove(task)
                    self._handle_task_done(task, result)
                    done_any = True
            if not block or not self._pending:
                break
            if not done_any:
                time.sleep(0.2)

    def _run_periodic_sync(
        self,
        *,
        event: TrainingEvent,
        checkpoint_path: Path,
        run_dir: Path,
    ) -> None:
        summary_path = run_dir / "benchmark_summary.json"
        try:
            summary = self._run_falcor_eval(checkpoint_path, run_dir)
            self._handle_success(
                stage_index=int(event.stage_index),
                stage_name=str(event.stage_name),
                epoch=int(event.epoch),
                num_epochs=int(event.num_epochs),
                checkpoint_path=checkpoint_path,
                summary_path=summary_path,
                summary=summary,
                async_mode=False,
                best_kind="periodic_sync",
                history_kind="periodic",
            )
        except subprocess.TimeoutExpired as timeout_error:
            self._handle_timeout(
                event=event,
                run_dir=run_dir,
                error_msg=str(timeout_error),
                stage_end=False,
                async_mode=False,
            )
        except Exception as falcor_error:
            self._handle_error(
                event=event,
                error_msg=str(falcor_error),
                stage_end=False,
                async_mode=False,
            )

    def _run_stage_end_sync(self, *, event: TrainingEvent, checkpoint_path: Path, run_dir: Path) -> None:
        summary_path = run_dir / "benchmark_summary.json"
        try:
            self._heartbeat(
                event=event,
                state="falcor_eval_running",
                last_event="falcor_stage_end_start",
                extra={"checkpoint": str(checkpoint_path), "output_dir": str(run_dir)},
            )
            summary = self._run_falcor_eval(checkpoint_path, run_dir)
            self._handle_success(
                stage_index=int(event.stage_index),
                stage_name=str(event.stage_name),
                epoch=int(event.epoch),
                num_epochs=int(event.num_epochs),
                checkpoint_path=checkpoint_path,
                summary_path=summary_path,
                summary=summary,
                async_mode=False,
                best_kind="stage_end_full",
                history_kind="stage_end_full",
            )
        except subprocess.TimeoutExpired as timeout_error:
            self._handle_timeout(
                event=event,
                run_dir=run_dir,
                error_msg=str(timeout_error),
                stage_end=True,
                async_mode=False,
            )
        except Exception as falcor_error:
            self._handle_error(
                event=event,
                error_msg=str(falcor_error),
                stage_end=True,
                async_mode=False,
            )

    def on_epoch_audit(self, event: TrainingEvent) -> None:
        self._drain_tasks(block=False)
        if not bool(self._cfg.get("enabled", False)):
            return
        every_n = int(self._cfg.get("every_n_epochs", 0))
        if every_n <= 0 or int(event.epoch) <= 0 or int(event.epoch) % every_n != 0:
            return
        extra = event.extra if isinstance(event.extra, dict) else {}
        stage_dir_raw = extra.get("stage_dir", None)
        if stage_dir_raw is None:
            return
        stage_dir = Path(str(stage_dir_raw))
        checkpoint_meta = extra.get("checkpoint_meta", None)
        if not isinstance(checkpoint_meta, dict):
            checkpoint_meta = {}

        run_dir = stage_dir / "falcor_periodic_eval" / f"epoch_{int(event.epoch):04d}"
        checkpoint_path = run_dir / "checkpoint.pt"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._save_checkpoint(checkpoint_path, int(event.epoch), checkpoint_meta)

        if self._async_enabled and self._executor is not None:
            self._heartbeat(
                event=event,
                state="falcor_eval_running",
                last_event="falcor_periodic_start",
                extra={"checkpoint": str(checkpoint_path), "output_dir": str(run_dir), "async": True},
            )
            self._schedule_task(
                event=event,
                checkpoint_path=checkpoint_path,
                run_dir=run_dir,
            )
            return

        self._run_periodic_sync(
            event=event,
            checkpoint_path=checkpoint_path,
            run_dir=run_dir,
        )

    def on_stage_end(self, event: TrainingEvent) -> None:
        self._drain_tasks(block=True)
        if not bool(self._cfg.get("enabled", False)) or not bool(self._cfg.get("stage_end_full", True)):
            return
        extra = event.extra if isinstance(event.extra, dict) else {}
        stage_dir_raw = extra.get("stage_dir", None)
        if stage_dir_raw is None:
            return
        stage_dir = Path(str(stage_dir_raw))
        stage_falcor_dir = stage_dir / "falcor_periodic_eval" / "stage_end_full"
        stage_falcor_dir.mkdir(parents=True, exist_ok=True)
        stage_best = stage_dir / "best_model.pt"
        stage_last = stage_dir / "last_model.pt"
        stage_ckpt = stage_best if stage_best.exists() else stage_last
        if not stage_ckpt.exists():
            return

        self._run_stage_end_sync(
            event=event,
            checkpoint_path=stage_ckpt,
            run_dir=stage_falcor_dir,
        )

    def finalize(self, *, block: bool = True) -> None:
        if self._cleanup_done:
            return
        try:
            self._drain_tasks(block=block)
        except Exception as cleanup_error:
            print(f"Warning: falcor pending task cleanup failed: {cleanup_error}")
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self._cleanup_done = True

    def _exit_guard(self) -> None:
        self.finalize(block=True)

    def on_cleanup(self, event: TrainingEvent) -> None:
        self.finalize(block=True)


class OracleAuditHook:
    """Runs Oracle monitor periodic + stage-end audits as a pluggable hook."""

    def __init__(
        self,
        *,
        args: Any,
        model: Any,
        oracle_cfg: Dict[str, Any],
        phase0_metrics_path: Path,
        now_iso: Callable[[], str],
        append_jsonl: Callable[[Path, Dict[str, Any]], None],
        update_heartbeat: Callable[..., None],
        save_checkpoint: Callable[[Path, int, Dict[str, Any]], None],
        run_oracle_monitor: Callable[..., Dict[str, Any] | None],
        metrics_from_summary: Callable[[Dict[str, Any]], Dict[str, float]],
        rerun_logger: Any,
    ) -> None:
        self._args = args
        self._model = model
        self._cfg = oracle_cfg
        self._phase0_metrics_path = phase0_metrics_path
        self._now_iso = now_iso
        self._append_jsonl = append_jsonl
        self._update_heartbeat = update_heartbeat
        self._save_checkpoint = save_checkpoint
        self._run_oracle_monitor = run_oracle_monitor
        self._metrics_from_summary = metrics_from_summary
        self._rerun_logger = rerun_logger

    def _heartbeat(
        self,
        *,
        event: TrainingEvent,
        state: str,
        last_event: str,
        metrics: Dict[str, float] | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        self._update_heartbeat(
            state=state,
            last_event=last_event,
            stage_index=int(event.stage_index),
            stage_name=str(event.stage_name),
            epoch=int(event.epoch),
            num_epochs=int(event.num_epochs),
            metrics=metrics,
            extra=extra,
        )

    def on_epoch_audit(self, event: TrainingEvent) -> None:
        if not bool(self._cfg.get("enabled", False)):
            return
        period_epochs = int(self._cfg.get("period_epochs", 0))
        if period_epochs <= 0:
            return
        if int(event.epoch) <= 0 or int(event.epoch) % period_epochs != 0:
            return

        extra = event.extra if isinstance(event.extra, dict) else {}
        stage_dir_raw = extra.get("stage_dir", None)
        if stage_dir_raw is None:
            return
        stage_dir = Path(str(stage_dir_raw))
        stage_slug = str(extra.get("stage_slug", "main"))
        checkpoint_meta = extra.get("checkpoint_meta", None)
        if not isinstance(checkpoint_meta, dict):
            checkpoint_meta = {}

        monitor_dir = stage_dir / "oracle_monitor"
        checkpoint_path = monitor_dir / f"epoch_{int(event.epoch):04d}.pt"
        summary_path = monitor_dir / f"epoch_{int(event.epoch):04d}_light.json"
        monitor_dir.mkdir(parents=True, exist_ok=True)
        self._save_checkpoint(checkpoint_path, int(event.epoch), checkpoint_meta)

        oracle_runs = extra.get("oracle_runs", None)
        metrics: Dict[str, float] = {}

        try:
            self._heartbeat(
                event=event,
                state="oracle_running",
                last_event="oracle_light_start",
                extra={"checkpoint": str(checkpoint_path), "summary": str(summary_path)},
            )
            summary = self._run_oracle_monitor(
                checkpoint_path=checkpoint_path,
                output_json_path=summary_path,
                args=self._args,
                oracle_cfg=self._cfg,
                max_samples=int(self._cfg.get("lightweight_max_samples", 4096)),
                sample_seed=int(self._cfg.get("lightweight_sample_seed", self._args.seed)) + int(event.epoch),
                timeout_seconds=int(self._cfg.get("timeout_seconds", 180)),
            )
            if summary is not None:
                metrics = self._metrics_from_summary(summary)
                if isinstance(oracle_runs, list):
                    oracle_runs.append(
                        {
                            "kind": "lightweight",
                            "epoch": int(event.epoch),
                            "summary": str(summary_path),
                            "metrics": metrics,
                        }
                    )
                if self._rerun_logger is not None:
                    self._rerun_logger.log_scalars(int(event.epoch), metrics, f"oracle/{stage_slug}/light")
                self._append_jsonl(
                    self._phase0_metrics_path,
                    {
                        "timestamp": self._now_iso(),
                        "kind": "oracle_light",
                        "stage_index": int(event.stage_index),
                        "stage_name": str(event.stage_name),
                        "epoch": int(event.epoch),
                        "num_epochs": int(event.num_epochs),
                        "metrics": metrics,
                        "summary": str(summary_path),
                    },
                )

            self._heartbeat(
                event=event,
                state="running",
                last_event="oracle_light_success",
                metrics=metrics,
            )
        except subprocess.TimeoutExpired as timeout_error:
            timeout_path = monitor_dir / f"epoch_{int(event.epoch):04d}_light_timeout.json"
            timeout_payload = {
                "kind": "lightweight_timeout",
                "stage": str(event.stage_name),
                "epoch": int(event.epoch),
                "timeout_seconds": int(self._cfg.get("timeout_seconds", 180)),
                "error": str(timeout_error),
                "timestamp": self._now_iso(),
            }
            timeout_path.write_text(json.dumps(timeout_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            if isinstance(oracle_runs, list):
                oracle_runs.append(
                    {
                        "kind": "lightweight_timeout",
                        "epoch": int(event.epoch),
                        "summary": str(timeout_path),
                        "metrics": {},
                    }
                )
            self._heartbeat(
                event=event,
                state="running",
                last_event="oracle_light_timeout",
                extra={"timeout_file": str(timeout_path)},
            )
            print(
                f"Warning: Oracle monitor timeout at stage={event.stage_name} "
                f"epoch={event.epoch} timeout={self._cfg.get('timeout_seconds', 180)}s"
            )
            if bool(self._cfg.get("fail_on_timeout", False)):
                raise RuntimeError(
                    f"Oracle monitor timed out at stage={event.stage_name} epoch={event.epoch}"
                ) from timeout_error
        except Exception as monitor_error:
            self._heartbeat(
                event=event,
                state="running",
                last_event="oracle_light_error",
                extra={"error": str(monitor_error)},
            )
            print(
                f"Warning: Oracle monitor failed at stage={event.stage_name} "
                f"epoch={event.epoch}: {monitor_error}"
            )

    def on_stage_end(self, event: TrainingEvent) -> None:
        if not bool(self._cfg.get("enabled", False)) or not bool(self._cfg.get("stage_end_full", True)):
            return

        extra = event.extra if isinstance(event.extra, dict) else {}
        stage_dir_raw = extra.get("stage_dir", None)
        if stage_dir_raw is None:
            return
        stage_dir = Path(str(stage_dir_raw))
        stage_slug = str(extra.get("stage_slug", "main"))
        oracle_runs = extra.get("oracle_runs", None)

        monitor_dir = stage_dir / "oracle_monitor"
        monitor_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = stage_dir / "best_model.pt"
        last_ckpt = stage_dir / "last_model.pt"
        final_ckpt = best_ckpt if best_ckpt.exists() else last_ckpt
        if not final_ckpt.exists():
            return

        full_summary_path = monitor_dir / "stage_end_full.json"
        try:
            self._heartbeat(
                event=event,
                state="oracle_running",
                last_event="oracle_stage_end_start",
                extra={"checkpoint": str(final_ckpt), "summary": str(full_summary_path)},
            )
            full_summary = self._run_oracle_monitor(
                checkpoint_path=final_ckpt,
                output_json_path=full_summary_path,
                args=self._args,
                oracle_cfg=self._cfg,
                max_samples=int(self._cfg.get("full_max_samples", 0)),
                sample_seed=int(self._cfg.get("lightweight_sample_seed", self._args.seed)),
                timeout_seconds=int(self._cfg.get("timeout_seconds", 180)),
            )
            if full_summary is not None:
                full_metrics = self._metrics_from_summary(full_summary)
                if isinstance(oracle_runs, list):
                    oracle_runs.append(
                        {
                            "kind": "stage_end_full",
                            "epoch": int(event.epoch),
                            "summary": str(full_summary_path),
                            "metrics": full_metrics,
                        }
                    )
                if self._rerun_logger is not None:
                    self._rerun_logger.log_scalars(int(event.epoch), full_metrics, f"oracle/{stage_slug}/full")
                self._heartbeat(
                    event=event,
                    state="running",
                    last_event="oracle_stage_end_success",
                    metrics=full_metrics,
                )
        except subprocess.TimeoutExpired as timeout_error:
            timeout_path = monitor_dir / "stage_end_full_timeout.json"
            timeout_payload = {
                "kind": "stage_end_full_timeout",
                "stage": str(event.stage_name),
                "epoch": int(event.epoch),
                "timeout_seconds": int(self._cfg.get("timeout_seconds", 180)),
                "error": str(timeout_error),
                "timestamp": self._now_iso(),
            }
            timeout_path.write_text(json.dumps(timeout_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            if isinstance(oracle_runs, list):
                oracle_runs.append(
                    {
                        "kind": "stage_end_full_timeout",
                        "epoch": int(event.epoch),
                        "summary": str(timeout_path),
                        "metrics": {},
                    }
                )
            self._heartbeat(
                event=event,
                state="running",
                last_event="oracle_stage_end_timeout",
                extra={"timeout_file": str(timeout_path)},
            )
            print(
                f"Warning: Oracle stage-end monitor timeout at stage={event.stage_name} "
                f"timeout={self._cfg.get('timeout_seconds', 180)}s"
            )
            if bool(self._cfg.get("fail_on_timeout", False)):
                raise RuntimeError(f"Oracle stage-end monitor timed out at stage={event.stage_name}") from timeout_error
        except Exception as monitor_error:
            self._heartbeat(
                event=event,
                state="running",
                last_event="oracle_stage_end_error",
                extra={"error": str(monitor_error)},
            )
            print(f"Warning: Oracle stage-end monitor failed at stage={event.stage_name}: {monitor_error}")


class PostTrainingAuditHook:
    """Runs post-training audit suites and exposes structured outputs."""

    def __init__(
        self,
        *,
        args: Any,
        output_dir: Path,
        val_profiles_cfg: Dict[str, Any],
        coeff_suite_cfg: Dict[str, Any],
        expert_util_cfg: Dict[str, Any],
        semantic_drift_cfg: Dict[str, Any],
        global_best_track_falcor: bool,
        global_best_track_soft: bool,
        global_best_falcor_ckpt_path: Path,
        global_best_soft_ckpt_path: Path,
        phase0_metrics_path: Path,
        now_iso: Callable[[], str],
        append_jsonl: Callable[[Path, Dict[str, Any]], None],
        update_heartbeat: Callable[..., None],
        run_profile_test_compare: Callable[..., Dict[str, Any] | None],
        run_coeff_suite: Callable[..., Dict[str, Any] | None],
        coeff_suite_metrics_from_summary: Callable[[Dict[str, Any]], Dict[str, float]],
        run_expert_utilization_audit: Callable[..., Dict[str, Any] | None],
        expert_util_metrics_from_summary: Callable[[Dict[str, Any]], Dict[str, float]],
        run_semantic_drift_audit: Callable[..., Dict[str, Any] | None],
        resolve_audit_profiles: Callable[[List[Dict[str, Any]] | None], List[Dict[str, Any]]],
        sanitize_stage_name: Callable[[str], str],
        rerun_logger: Any,
    ) -> None:
        self._args = args
        self._output_dir = output_dir
        self._val_profiles_cfg = val_profiles_cfg
        self._coeff_suite_cfg = coeff_suite_cfg
        self._expert_util_cfg = expert_util_cfg
        self._semantic_drift_cfg = semantic_drift_cfg
        self._global_best_track_falcor = bool(global_best_track_falcor)
        self._global_best_track_soft = bool(global_best_track_soft)
        self._global_best_falcor_ckpt_path = global_best_falcor_ckpt_path
        self._global_best_soft_ckpt_path = global_best_soft_ckpt_path
        self._phase0_metrics_path = phase0_metrics_path
        self._now_iso = now_iso
        self._append_jsonl = append_jsonl
        self._update_heartbeat = update_heartbeat
        self._run_profile_test_compare = run_profile_test_compare
        self._run_coeff_suite_fn = run_coeff_suite
        self._coeff_suite_metrics_from_summary = coeff_suite_metrics_from_summary
        self._run_expert_utilization_audit = run_expert_utilization_audit
        self._expert_util_metrics_from_summary = expert_util_metrics_from_summary
        self._run_semantic_drift_audit = run_semantic_drift_audit
        self._resolve_audit_profiles = resolve_audit_profiles
        self._sanitize_stage_name = sanitize_stage_name
        self._rerun_logger = rerun_logger

        self.outputs = PostTrainingAuditOutputs()

    def _heartbeat(
        self,
        *,
        event: TrainingEvent,
        state: str,
        last_event: str,
        metrics: Dict[str, float] | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        self._update_heartbeat(
            state=state,
            last_event=last_event,
            stage_index=int(event.stage_index),
            stage_name=str(event.stage_name),
            epoch=int(event.epoch),
            num_epochs=int(event.num_epochs),
            metrics=metrics,
            extra=extra,
        )

    def _resolve_suite_checkpoint(self, raw_name: str) -> Path:
        token = str(raw_name).strip()
        alias = token.lower()
        alias_map = {
            "global_best_falcor": self._global_best_falcor_ckpt_path,
            "global_best_falcor.pt": self._global_best_falcor_ckpt_path,
            "global_best_soft": self._global_best_soft_ckpt_path,
            "global_best_soft_profile": self._global_best_soft_ckpt_path,
            "global_best_soft.pt": self._global_best_soft_ckpt_path,
            "best_model": self._output_dir / "best_model.pt",
            "best_model.pt": self._output_dir / "best_model.pt",
            "last_model": self._output_dir / "last_model.pt",
            "last_model.pt": self._output_dir / "last_model.pt",
        }
        if alias in alias_map:
            return alias_map[alias]
        ckpt = Path(token)
        if not ckpt.is_absolute():
            ckpt = self._output_dir / ckpt
        return ckpt

    def _run_val_profile_test_compare(self, *, event: TrainingEvent, trainer: Any, model: Any, test_loader: Any) -> None:
        if not (
            trainer is not None
            and bool(self._val_profiles_cfg.get("enabled", False))
            and bool(self._val_profiles_cfg.get("run_test_compare", False))
        ):
            return
        try:
            self._heartbeat(
                event=event,
                state="profile_test_compare_running",
                last_event="profile_test_compare_start",
            )
            self.outputs.val_profile_test_compare = self._run_profile_test_compare(
                trainer=trainer,
                model=model,
                test_loader=test_loader,
                output_dir=self._output_dir,
                profile_cfg=self._val_profiles_cfg,
            )
            if self.outputs.val_profile_test_compare is not None:
                self._heartbeat(
                    event=event,
                    state="running",
                    last_event="profile_test_compare_success",
                    extra={"summary": str(self._output_dir / "val_profile_test_compare.json")},
                )
        except Exception as compare_error:
            self._heartbeat(
                event=event,
                state="running",
                last_event="profile_test_compare_error",
                extra={"error": str(compare_error)},
            )
            print(f"Warning: val profile test compare failed: {compare_error}")

    def _run_coeff_suite_block(self, *, event: TrainingEvent) -> None:
        if not bool(self._coeff_suite_cfg.get("enabled", False)) or not bool(
            self._coeff_suite_cfg.get("run_after_training", True)
        ):
            return

        timeout_seconds = int(self._coeff_suite_cfg.get("timeout_seconds", 1800))
        configured_ckpts = [str(x) for x in self._coeff_suite_cfg.get("checkpoints", [])]
        auto_ckpts: List[str] = []
        if self._global_best_track_falcor:
            auto_ckpts.append(str(self._global_best_falcor_ckpt_path.name))
        if self._global_best_track_soft:
            auto_ckpts.append(str(self._global_best_soft_ckpt_path.name))
        auto_ckpts.extend(["best_model.pt", "last_model.pt"])
        raw_candidates = configured_ckpts if configured_ckpts else auto_ckpts

        suite_runs: List[Dict[str, Any]] = []
        seen_suite_paths: set[str] = set()
        successful_runs = 0

        for raw_name in raw_candidates:
            suite_ckpt = self._resolve_suite_checkpoint(raw_name)
            if suite_ckpt.exists():
                key = str(suite_ckpt.resolve())
            else:
                key = str(suite_ckpt)
            if key in seen_suite_paths:
                continue
            seen_suite_paths.add(key)

            label_base = str(Path(str(raw_name)).stem or str(raw_name))
            label = self._sanitize_stage_name(label_base)

            if not suite_ckpt.exists():
                suite_runs.append(
                    {
                        "name": label,
                        "checkpoint": str(suite_ckpt),
                        "exists": False,
                        "status": "missing",
                    }
                )
                continue

            if successful_runs == 0:
                suite_out_dir = self._output_dir / "diagnostics" / "coeff_suite"
            else:
                suite_out_dir = (
                    self._output_dir / "diagnostics" / "coeff_suite_multi" / f"{successful_runs:02d}_{label}"
                )

            try:
                self._heartbeat(
                    event=event,
                    state="coeff_suite_running",
                    last_event="coeff_suite_start",
                    extra={
                        "checkpoint": str(suite_ckpt),
                        "output_dir": str(suite_out_dir),
                        "name": label,
                    },
                )
                summary_obj = self._run_coeff_suite_fn(
                    checkpoint_path=suite_ckpt,
                    output_dir=suite_out_dir,
                    args=self._args,
                    suite_cfg=self._coeff_suite_cfg,
                    timeout_seconds=timeout_seconds,
                )
                metrics_obj = self._coeff_suite_metrics_from_summary(summary_obj or {})
                if self._rerun_logger is not None and metrics_obj:
                    self._rerun_logger.log_scalars(int(event.epoch), metrics_obj, f"coeff_suite/{label}")
                self._append_jsonl(
                    self._phase0_metrics_path,
                    {
                        "timestamp": self._now_iso(),
                        "kind": "coeff_suite",
                        "stage_index": int(event.stage_index),
                        "stage_name": str(event.stage_name),
                        "epoch": int(event.epoch),
                        "num_epochs": int(event.num_epochs),
                        "metrics": metrics_obj,
                        "summary": str(suite_out_dir / "phase1_summary.json"),
                        "checkpoint": str(suite_ckpt),
                        "name": label,
                    },
                )
                self._heartbeat(
                    event=event,
                    state="running",
                    last_event="coeff_suite_success",
                    metrics=metrics_obj,
                    extra={
                        "summary": str(suite_out_dir / "phase1_summary.json"),
                        "checkpoint": str(suite_ckpt),
                        "name": label,
                    },
                )

                suite_runs.append(
                    {
                        "name": label,
                        "checkpoint": str(suite_ckpt),
                        "exists": True,
                        "status": "success",
                        "summary": str(suite_out_dir / "phase1_summary.json"),
                        "metrics": metrics_obj,
                    }
                )
                if self.outputs.coeff_suite_summary is None:
                    self.outputs.coeff_suite_summary = summary_obj
                successful_runs += 1
            except subprocess.TimeoutExpired as timeout_error:
                timeout_path = self._output_dir / "diagnostics" / "coeff_suite_multi" / f"{label}_timeout.json"
                timeout_payload = {
                    "kind": "coeff_suite_timeout",
                    "timeout_seconds": timeout_seconds,
                    "error": str(timeout_error),
                    "checkpoint": str(suite_ckpt),
                    "name": label,
                    "timestamp": self._now_iso(),
                }
                timeout_path.parent.mkdir(parents=True, exist_ok=True)
                timeout_path.write_text(json.dumps(timeout_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                self._heartbeat(
                    event=event,
                    state="running",
                    last_event="coeff_suite_timeout",
                    extra={
                        "timeout_file": str(timeout_path),
                        "checkpoint": str(suite_ckpt),
                        "name": label,
                    },
                )
                suite_runs.append(
                    {
                        "name": label,
                        "checkpoint": str(suite_ckpt),
                        "exists": True,
                        "status": "timeout",
                        "timeout_file": str(timeout_path),
                    }
                )
                print(f"Warning: coeff suite timed out timeout={timeout_seconds}s checkpoint={suite_ckpt}")
                if bool(self._coeff_suite_cfg.get("fail_on_timeout", False)):
                    raise RuntimeError("Coeff suite timed out") from timeout_error
            except Exception as coeff_error:
                self._heartbeat(
                    event=event,
                    state="running",
                    last_event="coeff_suite_error",
                    extra={"error": str(coeff_error), "checkpoint": str(suite_ckpt), "name": label},
                )
                suite_runs.append(
                    {
                        "name": label,
                        "checkpoint": str(suite_ckpt),
                        "exists": True,
                        "status": "error",
                        "error": str(coeff_error),
                    }
                )
                print(f"Warning: coeff suite failed checkpoint={suite_ckpt}: {coeff_error}")

        if suite_runs:
            self.outputs.coeff_suite_multi_summary = {
                "timestamp": self._now_iso(),
                "configured_checkpoints": configured_ckpts,
                "resolved_default_checkpoints": auto_ckpts,
                "runs": suite_runs,
            }
            multi_summary_path = (
                self._output_dir / "diagnostics" / "coeff_suite_multi" / "coeff_suite_multi_summary.json"
            )
            multi_summary_path.parent.mkdir(parents=True, exist_ok=True)
            multi_summary_path.write_text(
                json.dumps(self.outputs.coeff_suite_multi_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _run_expert_utilization_audit_block(self, *, event: TrainingEvent) -> None:
        if not bool(self._expert_util_cfg.get("enabled", False)) or not bool(
            self._expert_util_cfg.get("run_after_training", True)
        ):
            return

        best_ckpt = self._output_dir / "best_model.pt"
        last_ckpt = self._output_dir / "last_model.pt"
        cfg_ckpt = self._expert_util_cfg.get("checkpoint", None)
        if cfg_ckpt:
            ckpt_path = Path(str(cfg_ckpt))
            if not ckpt_path.is_absolute():
                ckpt_path = self._output_dir / ckpt_path
        else:
            ckpt_path = best_ckpt if best_ckpt.exists() else last_ckpt
        if not ckpt_path.exists():
            return

        audit_profiles = self._resolve_audit_profiles(list(self._expert_util_cfg.get("profiles", [])))
        out_path = self._output_dir / "diagnostics" / "expert_utilization" / "expert_utilization_summary.json"
        timeout_seconds = int(self._expert_util_cfg.get("timeout_seconds", 900))
        try:
            self._heartbeat(
                event=event,
                state="expert_utilization_audit_running",
                last_event="expert_utilization_audit_start",
                extra={"checkpoint": str(ckpt_path), "summary": str(out_path)},
            )
            self.outputs.expert_util_summary = self._run_expert_utilization_audit(
                checkpoint_path=ckpt_path,
                output_json_path=out_path,
                args=self._args,
                audit_cfg=self._expert_util_cfg,
                profiles=audit_profiles,
                timeout_seconds=timeout_seconds,
            )
            expert_metrics = self._expert_util_metrics_from_summary(self.outputs.expert_util_summary or {})
            if self._rerun_logger is not None and expert_metrics:
                self._rerun_logger.log_scalars(int(event.epoch), expert_metrics, "expert_utilization")
            self._append_jsonl(
                self._phase0_metrics_path,
                {
                    "timestamp": self._now_iso(),
                    "kind": "expert_utilization_audit",
                    "stage_index": int(event.stage_index),
                    "stage_name": str(event.stage_name),
                    "epoch": int(event.epoch),
                    "num_epochs": int(event.num_epochs),
                    "metrics": expert_metrics,
                    "summary": str(out_path),
                },
            )
            self._heartbeat(
                event=event,
                state="running",
                last_event="expert_utilization_audit_success",
                metrics=expert_metrics,
                extra={"summary": str(out_path)},
            )
        except subprocess.TimeoutExpired as timeout_error:
            timeout_path = self._output_dir / "diagnostics" / "expert_utilization_timeout.json"
            timeout_path.parent.mkdir(parents=True, exist_ok=True)
            timeout_path.write_text(
                json.dumps(
                    {
                        "kind": "expert_utilization_timeout",
                        "timeout_seconds": timeout_seconds,
                        "error": str(timeout_error),
                        "timestamp": self._now_iso(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Warning: expert utilization audit timed out timeout={timeout_seconds}s")
            if bool(self._expert_util_cfg.get("fail_on_timeout", False)):
                raise RuntimeError("expert utilization audit timed out") from timeout_error
        except Exception as audit_error:
            print(f"Warning: expert utilization audit failed: {audit_error}")

    def _run_semantic_drift_audit_block(self) -> None:
        if not bool(self._semantic_drift_cfg.get("enabled", False)) or not bool(
            self._semantic_drift_cfg.get("run_after_training", True)
        ):
            return

        best_ckpt = self._output_dir / "best_model.pt"
        last_ckpt = self._output_dir / "last_model.pt"
        pairs = list(self._semantic_drift_cfg.get("pairs", []))
        if not pairs and best_ckpt.exists() and last_ckpt.exists():
            pairs = [
                {
                    "name": "best_vs_last",
                    "checkpoint_a": str(best_ckpt),
                    "checkpoint_b": str(last_ckpt),
                }
            ]

        drift_profiles = self._resolve_audit_profiles(list(self._semantic_drift_cfg.get("profiles", [])))
        pair_summaries: Dict[str, Any] = {}
        timeout_seconds = int(self._semantic_drift_cfg.get("timeout_seconds", 1200))

        for pair in pairs:
            pair_name = str(pair.get("name", "pair"))
            ckpt_a = Path(str(pair.get("checkpoint_a", "")))
            ckpt_b = Path(str(pair.get("checkpoint_b", "")))
            if not ckpt_a.is_absolute():
                ckpt_a = self._output_dir / ckpt_a
            if not ckpt_b.is_absolute():
                ckpt_b = self._output_dir / ckpt_b
            if not ckpt_a.exists() or not ckpt_b.exists():
                print(f"Warning: semantic drift pair skipped (missing ckpt): {pair_name}")
                continue

            out_path = self._output_dir / "diagnostics" / "semantic_drift" / f"{self._sanitize_stage_name(pair_name)}.json"
            try:
                pair_summary = self._run_semantic_drift_audit(
                    checkpoint_a=ckpt_a,
                    checkpoint_b=ckpt_b,
                    output_json_path=out_path,
                    pair_name=pair_name,
                    args=self._args,
                    audit_cfg=self._semantic_drift_cfg,
                    profiles=drift_profiles,
                    timeout_seconds=timeout_seconds,
                )
                if pair_summary is not None:
                    pair_summaries[pair_name] = pair_summary
            except subprocess.TimeoutExpired as timeout_error:
                print(f"Warning: semantic drift audit timed out for {pair_name}: {timeout_error}")
                if bool(self._semantic_drift_cfg.get("fail_on_timeout", False)):
                    raise RuntimeError(f"semantic drift audit timed out for {pair_name}") from timeout_error
            except Exception as drift_error:
                print(f"Warning: semantic drift audit failed for {pair_name}: {drift_error}")

        if pair_summaries:
            self.outputs.semantic_drift_summary = {
                "pairs": pair_summaries,
                "meta": {
                    "num_pairs": int(len(pair_summaries)),
                },
            }
            summary_path = self._output_dir / "diagnostics" / "semantic_drift" / "semantic_drift_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(self.outputs.semantic_drift_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def on_run_audits(self, event: TrainingEvent) -> None:
        self.outputs = PostTrainingAuditOutputs()
        extra = event.extra if isinstance(event.extra, dict) else {}
        trainer = extra.get("trainer", None)
        model = extra.get("model", None)
        test_loader = extra.get("test_loader", None)

        self._run_val_profile_test_compare(
            event=event,
            trainer=trainer,
            model=model,
            test_loader=test_loader,
        )
        self._run_coeff_suite_block(event=event)
        self._run_expert_utilization_audit_block(event=event)
        self._run_semantic_drift_audit_block()
