from pathlib import Path

from tools.run_pipeline import build_pipeline_steps


def test_build_pipeline_steps(tmp_path: Path):
    dataset_cfg = tmp_path / "dataset.yaml"
    dataset_cfg.write_text(
        """
output_dir: out

generator:
  script: gen.py
  args:
    num_frames: 3
"""
    )
    pipeline = {
        "dataset": {"config": str(dataset_cfg)},
        "python_train": "/env/train/python",
        "python_eval": "/env/eval/python",
        "train": {"entry": "train.py", "args": {"variant": "5d", "batch_size": 4}},
        "eval": {"entry": "eval.py", "args": {"output_dir": "out_eval"}},
    }
    steps = build_pipeline_steps(pipeline)
    assert steps[0]["name"] == "dataset"
    assert "--num-frames" in steps[0]["cmd"]
    assert steps[1]["name"] == "train"
    assert "--batch-size" in steps[1]["cmd"]
    assert "--data-root" in steps[1]["cmd"]
    assert steps[1]["python"] == "/env/train/python"
    assert steps[2]["name"] == "eval"
    assert steps[2]["python"] == "/env/eval/python"
