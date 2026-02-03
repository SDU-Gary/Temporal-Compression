from pathlib import Path

from tools.run_dataset import build_plan


def test_build_plan(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
name: test
output_dir: out

generator:
  script: gen.py
  args:
    num_frames: 5
    use_analytic_lights: true
"""
    )
    output_dir, cmd, cfg = build_plan(cfg_path, None)
    assert output_dir.name == "out"
    assert "--num-frames" in cmd
    assert "--use-analytic-lights" in cmd
    assert cfg["name"] == "test"
