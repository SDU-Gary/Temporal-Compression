from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_report_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "3_experiments" / "scripts" / "analysis" / "report_experiment_vs_baseline.py"
    spec = importlib.util.spec_from_file_location("report_experiment", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load report_experiment_vs_baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_report_gap_and_metric_deltas():
    mod = _load_report_module()

    baseline_oracle = {
        "delta": {
            "all27_psnr_gain_db": 10.0,
            "non_l0_psnr_gain_db": 8.0,
        }
    }
    exp_oracle = {
        "delta": {
            "all27_psnr_gain_db": 6.0,
            "non_l0_psnr_gain_db": 5.0,
        }
    }
    baseline_bench = {
        "image_metrics": {
            "mean_benchmark_psnr": 20.0,
            "mean_benchmark_ssim": 0.80,
            "mean_hdr_psnr": 18.0,
            "mean_hdr_ssim": 0.75,
        }
    }
    exp_bench = {
        "image_metrics": {
            "mean_benchmark_psnr": 24.5,
            "mean_benchmark_ssim": 0.85,
            "mean_hdr_psnr": 21.0,
            "mean_hdr_ssim": 0.79,
        }
    }

    report = mod.build_report(
        baseline_oracle_summary=baseline_oracle,
        experiment_oracle_summary=exp_oracle,
        baseline_benchmark_summary=baseline_bench,
        experiment_benchmark_summary=exp_bench,
        baseline_refs={"oracle_summary": "base_oracle.json", "benchmark_summary": "base_bench.json"},
        experiment_refs={"oracle_summary": "exp_oracle.json", "benchmark_summary": "exp_bench.json"},
    )

    gap = report["oracle_gap"]
    assert gap["gap_shrink_all27_db"] == 4.0
    assert gap["gap_shrink_non_l0_db"] == 3.0
    assert abs(gap["gap_shrink_ratio_all27"] - 0.4) < 1e-9

    psnr_delta = report["psnr_gain_vs_baseline"]
    ssim_delta = report["ssim_gain_vs_baseline"]
    assert psnr_delta["benchmark_psnr"] == 4.5
    assert psnr_delta["hdr_psnr"] == 3.0
    assert abs(ssim_delta["benchmark_ssim"] - 0.05) < 1e-9
    assert abs(ssim_delta["hdr_ssim"] - 0.04) < 1e-9

    diag = report["final_diagnosis"]
    assert diag["oracle_gap_improved"] is True
    assert "benchmark_psnr" in diag["render_psnr_improved_domains"]
