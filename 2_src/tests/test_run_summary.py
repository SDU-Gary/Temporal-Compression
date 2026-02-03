from tools.run_summary import build_run_summary_md


def test_build_run_summary_md():
    md = build_run_summary_md(
        "Pipeline Run Summary",
        {"pipeline_id": "TEST", "started_at": "t0"},
        sections=[{"name": "dataset", "command": "cmd"}],
    )
    assert "# Pipeline Run Summary" in md
    assert "- pipeline_id: TEST" in md
    assert "## dataset" in md
