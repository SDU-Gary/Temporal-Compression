from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
EVAL_PATH = ROOT / "3_experiments" / "scripts" / "evaluation"
if str(EVAL_PATH) not in sys.path:
    sys.path.insert(0, str(EVAL_PATH))

from run_all import build_eval_steps


def test_build_eval_steps(tmp_path: Path):
    steps = build_eval_steps(tmp_path)
    assert len(steps) == 3
    assert steps[0]["name"] == "interpolation"
    assert str(tmp_path) in steps[0]["args"][1]
