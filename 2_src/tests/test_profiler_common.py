from __future__ import annotations

from tools.profiler_common import (
    Thresholds,
    build_ncu_command,
    build_nsys_command,
    recommend_cuda,
    summarize_latency,
    top3_concentration_pct,
)


def test_summarize_latency_basic_stats() -> None:
    summary = summarize_latency([1.0, 2.0, 3.0, 4.0])
    assert summary["count"] == 4
    assert abs(summary["mean"] - 2.5) < 1e-6
    assert summary["p50"] >= 2.0


def test_top3_concentration_pct() -> None:
    pct = top3_concentration_pct(
        [
            {"self_cuda_time_total": 60.0},
            {"self_cuda_time_total": 20.0},
            {"self_cuda_time_total": 10.0},
            {"self_cuda_time_total": 10.0},
        ]
    )
    assert abs(pct - 90.0) < 1e-6


def test_recommend_cuda_decisions() -> None:
    thresholds = Thresholds(op_top3_recommend_pct=60.0, train_non_kernel_defer_pct=40.0, pipeline_model_share_min_pct=35.0)

    rec = recommend_cuda(
        {
            "inference_model": {"top3_ops_pct": 75.0},
            "pipeline": {"model_share_pct": 50.0},
        },
        thresholds=thresholds,
    )
    assert rec["decision"] == "recommend"

    defer = recommend_cuda(
        {
            "training": {"dataloader_wait_pct": 25.0, "python_overhead_pct": 30.0, "top3_ops_pct": 75.0},
            "pipeline": {"model_share_pct": 50.0},
        },
        thresholds=thresholds,
    )
    assert defer["decision"] == "defer"

    not_needed = recommend_cuda(
        {
            "inference_model": {"top3_ops_pct": 80.0},
            "pipeline": {"model_share_pct": 20.0},
        },
        thresholds=thresholds,
    )
    assert not_needed["decision"] == "not_needed"


def test_nsys_and_ncu_command_builders() -> None:
    target = ["python", "script.py", "--x", "1"]
    nsys = build_nsys_command(target, "out/nsys_run")
    ncu = build_ncu_command(target, "out/ncu_run")
    assert nsys[:2] == ["nsys", "profile"]
    assert "--trace" in nsys
    assert ncu[0] == "ncu"
    assert "--export" in ncu

