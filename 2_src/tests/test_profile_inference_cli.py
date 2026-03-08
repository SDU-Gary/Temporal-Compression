from __future__ import annotations

import json
from pathlib import Path

from tools.profile_inference import build_arg_parser, run_profile


def test_profile_inference_dry_run_outputs_files(tmp_path: Path) -> None:
    args = build_arg_parser().parse_args(
        [
            "--checkpoint",
            str(tmp_path / "model.pt"),
            "--data-root",
            str(tmp_path / "dataset"),
            "--profile-level",
            "both",
            "--stack",
            "both",
            "--frames",
            "5",
            "--warmup-frames",
            "2",
            "--scene",
            "1_data_generation/falcor/scenes/bistro_exterior_emissive_spheres.pyscene",
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    assert args.pipeline_route == "both"

    report = run_profile(args)
    out_dir = Path(report["run_dir"])

    assert report["dry_run"] is True
    assert (out_dir / "inference_model_summary.json").exists()
    assert (out_dir / "inference_pipeline_summary.json").exists()
    assert (out_dir / "inference_profile_report.json").exists()
    assert (out_dir / "cuda_recommendation_inference.json").exists()

    pipeline_summary = json.loads((out_dir / "inference_pipeline_summary.json").read_text(encoding="utf-8"))
    assert pipeline_summary.get("status") == "not_run"
    command = pipeline_summary.get("command", [])
    route_idx = command.index("--route")
    assert command[route_idx + 1] == "both"

