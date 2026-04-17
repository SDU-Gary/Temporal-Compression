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


train_hooks = _load_module("train_hooks", ROOT / "3_experiments" / "scripts" / "train_hooks.py")
audit_hooks = _load_module("train_audit_hooks_test", ROOT / "3_experiments" / "scripts" / "train_audit_hooks.py")


def test_hook_manager_extended_lifecycle_dispatch() -> None:
    calls = []

    class _Recorder:
        def on_epoch_audit(self, event) -> None:
            calls.append(("epoch_audit", event.epoch))

        def on_stage_end(self, event) -> None:
            calls.append(("stage_end", event.epoch))

        def on_run_audits(self, event) -> None:
            calls.append(("run_audits", event.epoch))

        def on_cleanup(self, event) -> None:
            calls.append(("cleanup", event.epoch))

    manager = train_hooks.HookManager()
    manager.register(_Recorder())
    event = train_hooks.TrainingEvent(
        stage_index=1,
        stage_name="single_stage",
        epoch=3,
        num_epochs=5,
        run_context={"run_id": "r1"},
    )
    manager.on_epoch_audit(event)
    manager.on_stage_end(event)
    manager.on_run_audits(event)
    manager.on_cleanup(event)

    assert calls == [
        ("epoch_audit", 3),
        ("stage_end", 3),
        ("run_audits", 3),
        ("cleanup", 3),
    ]


def test_falcor_audit_hook_periodic_sync(tmp_path: Path) -> None:
    heartbeats = []
    jsonl_rows = []
    best_updates = []

    def _append(path: Path, payload):
        jsonl_rows.append((path, payload))

    def _save_checkpoint(path: Path, epoch: int, meta: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"epoch={epoch}", encoding="utf-8")

    hook = audit_hooks.FalcorPeriodicAuditHook(
        model=object(),
        eval_cfg={
            "enabled": True,
            "every_n_epochs": 1,
            "stage_end_full": True,
            "async_mode": False,
            "timeout_seconds": 30,
            "fail_on_timeout": False,
            "fail_on_error": False,
        },
        phase0_metrics_path=tmp_path / "phase0.jsonl",
        falcor_periodic_history_path=tmp_path / "falcor_hist.jsonl",
        now_iso=lambda: "2026-01-01T00:00:00",
        append_jsonl=_append,
        update_heartbeat=lambda **kwargs: heartbeats.append(kwargs),
        run_falcor_eval=lambda ckpt, out_dir: {"image_metrics": {"mean_real_render_hdr_psnr": 31.5}},
        save_checkpoint=_save_checkpoint,
        extract_metrics_from_summary=lambda s: {
            "mean_real_render_hdr_psnr": float((s.get("image_metrics") or {}).get("mean_real_render_hdr_psnr", 0.0))
        },
        primary_metric_name="mean_real_render_hdr_psnr",
        primary_metric_value=lambda m: float(m.get("mean_real_render_hdr_psnr", 0.0)),
        is_better_than_best=lambda v: float(v) > 0.0,
        update_best_checkpoint=lambda **kwargs: best_updates.append(kwargs),
    )

    event = train_hooks.TrainingEvent(
        stage_index=2,
        stage_name="stage2",
        epoch=1,
        num_epochs=8,
        run_context={"run_id": "abc"},
        extra={
            "stage_dir": str(tmp_path / "stage2"),
            "checkpoint_meta": {"k": "v"},
        },
    )
    hook.on_epoch_audit(event)

    ckpt = tmp_path / "stage2" / "falcor_periodic_eval" / "epoch_0001" / "checkpoint.pt"
    assert ckpt.exists()
    assert any(payload.get("kind") == "periodic" for _, payload in jsonl_rows)
    assert any(payload.get("kind") == "falcor_periodic" for _, payload in jsonl_rows)
    assert best_updates
    assert any(hb.get("last_event") == "falcor_periodic_success" for hb in heartbeats)

    hook.on_cleanup(event)


def test_oracle_audit_hook_periodic_and_stage_end(tmp_path: Path) -> None:
    heartbeats = []
    jsonl_rows = []
    rerun_rows = []
    oracle_runs = []

    class _Rerun:
        def log_scalars(self, epoch: int, metrics, namespace: str) -> None:
            rerun_rows.append((epoch, namespace, metrics))

    def _save_checkpoint(path: Path, epoch: int, meta: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"epoch={epoch}", encoding="utf-8")

    def _run_monitor(**kwargs):
        out = kwargs["output_json_path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}", encoding="utf-8")
        return {
            "baseline": {"all27": {"sh_psnr": 1.0}, "non_l0": {"sh_non_l0_psnr": 2.0}},
            "oracle_timeweights": {"all27": {"sh_psnr": 3.0}, "non_l0": {"sh_non_l0_psnr": 4.0}},
            "delta": {"all27_psnr_gain_db": 2.0, "non_l0_psnr_gain_db": 2.0},
        }

    hook = audit_hooks.OracleAuditHook(
        args=type("Args", (), {"seed": 42})(),
        model=object(),
        oracle_cfg={
            "enabled": True,
            "period_epochs": 1,
            "stage_end_full": True,
            "timeout_seconds": 30,
            "fail_on_timeout": False,
            "lightweight_max_samples": 16,
            "lightweight_sample_seed": 7,
            "full_max_samples": 0,
        },
        phase0_metrics_path=tmp_path / "phase0.jsonl",
        now_iso=lambda: "2026-01-01T00:00:00",
        append_jsonl=lambda p, payload: jsonl_rows.append((p, payload)),
        update_heartbeat=lambda **kwargs: heartbeats.append(kwargs),
        save_checkpoint=_save_checkpoint,
        run_oracle_monitor=_run_monitor,
        metrics_from_summary=lambda _: {"oracle_all27_psnr": 3.0},
        rerun_logger=_Rerun(),
    )

    stage_dir = tmp_path / "stage"
    event_epoch = train_hooks.TrainingEvent(
        stage_index=1,
        stage_name="stage1",
        epoch=1,
        num_epochs=5,
        run_context={},
        extra={
            "stage_dir": str(stage_dir),
            "stage_slug": "s1",
            "checkpoint_meta": {},
            "oracle_runs": oracle_runs,
        },
    )
    hook.on_epoch_audit(event_epoch)

    assert any(item.get("kind") == "lightweight" for item in oracle_runs)
    assert any(payload.get("kind") == "oracle_light" for _, payload in jsonl_rows)
    assert any(hb.get("last_event") == "oracle_light_success" for hb in heartbeats)
    assert rerun_rows and rerun_rows[0][1] == "oracle/s1/light"

    (stage_dir / "best_model.pt").write_text("ok", encoding="utf-8")
    event_stage_end = train_hooks.TrainingEvent(
        stage_index=1,
        stage_name="stage1",
        epoch=5,
        num_epochs=5,
        run_context={},
        extra={
            "stage_dir": str(stage_dir),
            "stage_slug": "s1",
            "oracle_runs": oracle_runs,
        },
    )
    hook.on_stage_end(event_stage_end)
    assert any(item.get("kind") == "stage_end_full" for item in oracle_runs)


def test_post_training_audit_hook_profile_compare_output(tmp_path: Path) -> None:
    heartbeats = []

    hook = audit_hooks.PostTrainingAuditHook(
        args=type("Args", (), {"epochs": 2, "top_k": 3, "seed": 42})(),
        output_dir=tmp_path,
        val_profiles_cfg={"enabled": True, "run_test_compare": True, "profiles": []},
        coeff_suite_cfg={"enabled": False},
        expert_util_cfg={"enabled": False},
        semantic_drift_cfg={"enabled": False},
        global_best_track_falcor=False,
        global_best_track_soft=False,
        global_best_falcor_ckpt_path=tmp_path / "global_best_falcor.pt",
        global_best_soft_ckpt_path=tmp_path / "global_best_soft.pt",
        phase0_metrics_path=tmp_path / "phase0.jsonl",
        now_iso=lambda: "2026-01-01T00:00:00",
        append_jsonl=lambda *_args, **_kwargs: None,
        update_heartbeat=lambda **kwargs: heartbeats.append(kwargs),
        run_profile_test_compare=lambda **kwargs: {"results": {"p": {"metrics": {"mae": 1.0}}}},
        run_coeff_suite=lambda **kwargs: None,
        coeff_suite_metrics_from_summary=lambda _: {},
        run_expert_utilization_audit=lambda **kwargs: None,
        expert_util_metrics_from_summary=lambda _: {},
        run_semantic_drift_audit=lambda **kwargs: None,
        resolve_audit_profiles=lambda profiles: profiles or [],
        sanitize_stage_name=lambda x: x,
        rerun_logger=None,
    )

    hook.on_run_audits(
        train_hooks.TrainingEvent(
            stage_index=1,
            stage_name="single_stage",
            epoch=2,
            num_epochs=2,
            run_context={},
            extra={"trainer": object(), "model": object(), "test_loader": object()},
        )
    )

    assert hook.outputs.val_profile_test_compare is not None
    assert any(hb.get("last_event") == "profile_test_compare_start" for hb in heartbeats)
    assert any(hb.get("last_event") == "profile_test_compare_success" for hb in heartbeats)
