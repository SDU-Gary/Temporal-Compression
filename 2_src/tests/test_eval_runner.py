from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
EVAL_PATH = ROOT / "3_experiments" / "scripts" / "evaluation"
if str(EVAL_PATH) not in sys.path:
    sys.path.insert(0, str(EVAL_PATH))

from run_all import build_eval_steps, build_unified_steps


def test_build_eval_steps(tmp_path: Path):
    args = SimpleNamespace(
        data_root=tmp_path / "data",
        manifest=None,
        checkpoint=tmp_path / "best.pt",
        device="cpu",
        rank=8,
        train_mae=0.123,
        n_samples=5,
        seed=7,
        probe_idx=0,
    )
    steps = build_eval_steps(tmp_path, args)
    assert len(steps) == 3
    assert steps[0]["name"] == "interpolation"
    assert steps[0]["args"][0] == "--output"
    assert steps[0]["args"][1] == str(tmp_path / "interpolation")


def test_build_unified_steps(tmp_path: Path):
    args = SimpleNamespace(
        data_root=tmp_path / "data",
        checkpoint=tmp_path / "best.pt",
        device="cpu",
        batch_size=4,
        num_workers=0,
        train_ratio=0.7,
        val_ratio=0.2,
        seed=42,
        split="test",
        num_gaussians=10,
        rank=4,
        top_k=3,
        light_dim=12,
        embed_dim=16,
        intensity_dim=3,
        intensity_offset=1,
        disable_film=False,
        sh_scaler=None,
        near_threshold=0.2,
        k_neighbors=5,
        save_pred_config=None,
        save_pred_out=None,
    )
    steps = build_unified_steps(tmp_path, args)
    assert len(steps) == 1
    assert steps[0]["name"] == "unified_ablation"
    assert steps[0]["args"][0] == "--output"
    assert steps[0]["args"][1] == str(tmp_path / "ablation_results.json")
