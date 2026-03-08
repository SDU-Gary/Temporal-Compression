from __future__ import annotations

from pathlib import Path

from tools.profile_training import build_arg_parser, run_profile


def test_profile_training_dry_run_outputs_files(tmp_path: Path) -> None:
    cfg = tmp_path / "train.yaml"
    cfg.write_text("experiment:\n  variant: unified_set\n", encoding="utf-8")

    args = build_arg_parser().parse_args(
        [
            "--config",
            str(cfg),
            "--stack",
            "both",
            "--steps",
            "5",
            "--warmup-steps",
            "2",
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    report = run_profile(args)
    out_dir = Path(report["run_dir"])

    assert report["dry_run"] is True
    assert (out_dir / "training_profile_summary.json").exists()
    assert (out_dir / "training_profile_report.json").exists()
    assert (out_dir / "nsys_command.sh").exists()
    assert (out_dir / "ncu_command.sh").exists()

