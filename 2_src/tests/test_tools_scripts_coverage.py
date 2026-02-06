import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import importlib.util

import tools.db_utils as db_utils
import tools.logexp as logexp
from tools.ema_smooth_dataset import _apply_ema, _ema_backward, _ema_forward, main as ema_main
from tools.median_filter_dataset import _median_filter_time, main as median_main
from tools.probe_sampling import (
    load_light_trajectories,
    compute_light_weights,
    farthest_point_sampling,
    sample_with_decluster,
)
from tools.run_summary import build_run_summary_md, write_run_summary
from tools.workflow_config import (
    load_yaml,
    apply_preset,
    validate_dataset_config,
    build_generator_command,
    ensure_python,
    select_python,
)
from tools.run_dataset import build_plan as build_dataset_plan, main as run_dataset_main
from tools.run_pipeline import _load_pipeline, _load_train_output_dir, build_pipeline_steps, main as run_pipeline_main
from tools.manifest_utils import write_manifest, load_manifest
from tools.blender.scene_export import build_scene_payload

ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


run_all = _load_module("run_all", ROOT / "3_experiments" / "scripts" / "evaluation" / "run_all.py")
train_script = _load_module("train_script", ROOT / "3_experiments" / "scripts" / "train.py")


def _init_temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "project.db"
    monkeypatch.setattr(db_utils, "DB_PATH", db_path)
    schema_path = Path("metadata/schema.sql")
    db_utils.init_db(schema_path)
    return db_path


def _seed_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _init_temp_db(tmp_path, monkeypatch)
    with db_utils.get_db() as conn:
        conn.execute(
            "INSERT INTO datasets(dataset_id, full_name, root_path) VALUES (?, ?, ?)",
            ("D0", "dummy", "/tmp"),
        )
        conn.execute(
            "INSERT INTO scripts(script_id, filename) VALUES (?, ?)",
            ("train", "scripts/train.py"),
        )


def test_ema_and_median_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tensor = np.arange(12, dtype=np.float32).reshape(1, 4, 3)
    forward = _ema_forward(tensor, 0.5)
    backward = _ema_backward(tensor, 0.5)
    both = _apply_ema(tensor, 0.5, True)
    assert forward.shape == tensor.shape
    assert backward.shape == tensor.shape
    assert both.shape == tensor.shape

    with pytest.raises(ValueError):
        _apply_ema(np.zeros((4, 3)), 0.5, False)

    npz_path = tmp_path / "data.npz"
    np.savez(npz_path, tensor=tensor, metadata={"a": 1})
    out_path = tmp_path / "out.npz"
    monkeypatch.setattr(sys, "argv", ["ema", "--input", str(npz_path), "--output", str(out_path)])
    ema_main()
    assert out_path.exists()

    med = _median_filter_time(tensor, 3, pad_mode="edge")
    assert med.shape == tensor.shape
    with pytest.raises(ValueError):
        _median_filter_time(np.zeros((4, 3)), 3)
    with pytest.raises(ValueError):
        _median_filter_time(tensor, 2)

    out_path2 = tmp_path / "out2.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        ["median", "--input", str(npz_path), "--output", str(out_path2), "--kernel", "3"],
    )
    median_main()
    assert out_path2.exists()

    bad_npz = tmp_path / "bad.npz"
    np.savez(bad_npz, metadata=np.array([1, 2, 3]))
    monkeypatch.setattr(sys, "argv", ["ema", "--input", str(bad_npz), "--output", str(tmp_path / "bad_out.npz")])
    with pytest.raises(ValueError):
        ema_main()

    np.savez(bad_npz, tensor=tensor, metadata=np.array([1, 2, 3]))
    monkeypatch.setattr(sys, "argv", ["ema", "--input", str(bad_npz), "--output", str(tmp_path / "bad_out2.npz")])
    ema_main()

    monkeypatch.setattr(sys, "argv", ["ema", "--input", str(npz_path), "--output", str(tmp_path / "bad_out3.npz"), "--alpha", "0"])
    with pytest.raises(ValueError):
        ema_main()

    np.savez(bad_npz, metadata=np.array([1, 2, 3]))
    monkeypatch.setattr(sys, "argv", ["median", "--input", str(bad_npz), "--output", str(tmp_path / "med_out.npz")])
    with pytest.raises(ValueError):
        median_main()

    np.savez(bad_npz, tensor=tensor, metadata=np.array([1, 2, 3]))
    monkeypatch.setattr(sys, "argv", ["median", "--input", str(bad_npz), "--output", str(tmp_path / "med_out2.npz")])
    median_main()


def test_probe_sampling(tmp_path: Path) -> None:
    traj = np.zeros((5, 3), dtype=np.float32)
    traj_path = tmp_path / "traj.npy"
    np.save(traj_path, traj)
    trajectories = load_light_trajectories([str(traj_path)])
    assert len(trajectories) == 1

    traj_multi = np.zeros((4, 2, 3), dtype=np.float32)
    traj_multi_path = tmp_path / "traj_multi.npy"
    np.save(traj_multi_path, traj_multi)
    trajectories_multi = load_light_trajectories([str(traj_multi_path)])
    assert len(trajectories_multi) == 2

    points = np.random.rand(10, 3).astype(np.float32)
    weights = compute_light_weights(points, trajectories, step=1, eps=0.1, temporal_weight=0.2, chunk=5)
    assert weights.shape[0] == 10
    weights2 = compute_light_weights(points, [], step=1)
    assert np.allclose(weights2, np.ones_like(weights2))

    with pytest.raises(FileNotFoundError):
        load_light_trajectories([str(tmp_path / "missing.npy")])

    with pytest.raises(ValueError):
        compute_light_weights(np.zeros((4, 2)), trajectories)
    with pytest.raises(ValueError):
        compute_light_weights(points, [np.zeros((2, 2), dtype=np.float32)])
    with pytest.raises(ValueError):
        load_light_trajectories([])
    with pytest.raises(ValueError):
        bad_path = tmp_path / "bad.npy"
        np.save(bad_path, np.zeros((4, 2), dtype=np.float32))
        load_light_trajectories([str(bad_path)])

    with pytest.raises(ValueError):
        farthest_point_sampling(np.zeros((4, 2)), 2)
    with pytest.raises(ValueError):
        farthest_point_sampling(np.zeros((4, 3)), 0)
    pts = farthest_point_sampling(np.random.rand(5, 3), 10)
    assert pts.shape[1] == 3

    rng = np.random.default_rng(0)
    sampled = sample_with_decluster(points, weights, total=5, adaptive_ratio=0.5, rng=rng, decluster=True)
    assert sampled.shape[0] == 5
    sampled2 = sample_with_decluster(points, weights, total=5, adaptive_ratio=0.0, rng=rng, decluster=False)
    assert sampled2.shape[0] == 5
    with pytest.raises(ValueError):
        sample_with_decluster(points, weights, total=0, adaptive_ratio=0.5, rng=rng)


def test_workflow_config_and_run_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "output_dir: out\n"
        "generator:\n"
        "  script: gen.py\n"
        "  args:\n"
        "    foo: 1\n"
        "    items:\n"
        "      - a\n"
        "      - b\n"
        "presets:\n"
        "  smoke:\n"
        "    args:\n"
        "      foo: 2\n",
        encoding="utf-8",
    )
    config = load_yaml(cfg_path)
    apply_preset(config, "smoke")
    validate_dataset_config(config)
    out_dir, cmd = build_generator_command(config)
    assert out_dir.name == "out"
    assert "--foo" in cmd
    assert "--items" in cmd

    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "missing.yaml")
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text("- list", encoding="utf-8")
    with pytest.raises(ValueError):
        load_yaml(bad_cfg)

    with pytest.raises(ValueError):
        apply_preset(config, "missing")

    with pytest.raises(ValueError):
        validate_dataset_config({"generator": {}})
    with pytest.raises(ValueError):
        validate_dataset_config({"generator": {"script": "x"}})

    with pytest.raises(ValueError):
        ensure_python([])
    assert "python" in Path(ensure_python(["script.py"], python_bin=None)[0]).name
    assert ensure_python(["tool"], python_bin="python-bin")[0] == "python-bin"

    monkeypatch.delenv("FALCOR_PYTHON", raising=False)
    assert select_python(config) is None
    assert select_python("bad") is None
    monkeypatch.setenv("FALCOR_PYTHON", "/bin/python")
    assert select_python(config) == "/bin/python"
    config["python"] = "/usr/bin/python"
    assert select_python(config) == "/usr/bin/python"
    assert select_python({"generator": {"python": "/opt/py"}}) == "/opt/py"

    with pytest.raises(ValueError):
        build_scene_payload("scene", frames=0, fps=30, emissive_objects=[])
    with pytest.raises(ValueError):
        build_scene_payload("scene", frames=1, fps=0, emissive_objects=[])

    summary = build_run_summary_md("Title", {"a": "1"}, sections=[{"name": "step", "b": "2"}])
    assert "Title" in summary and "step" in summary
    out_path = write_run_summary(tmp_path / "summary.md", "Title", {"a": "1"})
    assert out_path.exists()


def test_run_dataset_and_pipeline_build(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "output_dir: out\n"
        "generator:\n"
        "  script: gen.py\n",
        encoding="utf-8",
    )
    out_dir, cmd, config = build_dataset_plan(cfg_path, None)
    assert out_dir.name == "out"
    assert cmd[0].endswith("gen.py")

    pipe_path = tmp_path / "pipe.yaml"
    pipe_path.write_text(
        "id: demo\n"
        "run_dir: run\n"
        "dataset:\n"
        f"  config: {cfg_path}\n"
        "train:\n"
        "  entry: train.py\n",
        encoding="utf-8",
    )
    pipeline = _load_pipeline(pipe_path)
    steps = build_pipeline_steps(pipeline)
    assert steps[0]["name"] == "dataset"

    bad_pipe = tmp_path / "bad_pipe.yaml"
    bad_pipe.write_text("- list", encoding="utf-8")
    with pytest.raises(ValueError):
        _load_pipeline(bad_pipe)


def test_run_dataset_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    output_dir = tmp_path / "out"
    cfg_path.write_text(
        f"output_dir: {output_dir}\n"
        "generator:\n"
        "  script: gen.py\n",
        encoding="utf-8",
    )

    def fake_run(cmd, check=True):
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(output_dir / "parametric_tensor.npz", tensor=np.zeros((1, 1, 27), dtype=np.float32))

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_dataset.py", "--config", str(cfg_path), "--write-manifest"],
    )
    run_dataset_main()

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_dataset.py", "--config", str(cfg_path), "--dry-run"],
    )
    run_dataset_main()


def test_run_pipeline_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    output_dir = tmp_path / "out"
    cfg_path.write_text(
        f"output_dir: {output_dir}\n"
        "generator:\n"
        "  script: gen.py\n",
        encoding="utf-8",
    )
    pipe_path = tmp_path / "pipe.yaml"
    pipe_path.write_text(
        "id: demo\n"
        "dataset:\n"
        f"  config: {cfg_path}\n"
        "train:\n"
        "  entry: train.py\n"
        "eval:\n"
        "  entry: eval.py\n",
        encoding="utf-8",
    )

    def fake_run(cmd, check=True):
        return None

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--config", str(pipe_path), "--dry-run", "--python", "py"])
    run_pipeline_main()
    assert "DRY-RUN" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--config", str(pipe_path)])
    run_pipeline_main()


def test_db_utils_and_logexp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    _seed_db(tmp_path, monkeypatch)
    seeded_db = db_utils.DB_PATH
    assert db_utils.get_next_run_order() == 1
    assert db_utils.get_next_exp_id("20990101").startswith("EXP-20990101")
    stats = db_utils.verify_db()
    assert "experiments" in stats

    with pytest.raises(FileNotFoundError):
        db_utils.init_db(tmp_path / "missing.sql")

    temp_db = tmp_path / "temp.db"
    monkeypatch.setattr(db_utils, "DB_PATH", temp_db)
    assert db_utils.get_db_last_modified() is None

    with pytest.raises(RuntimeError):
        with db_utils.get_db() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS t(x INTEGER)")
            raise RuntimeError("boom")

    monkeypatch.setattr(db_utils, "DB_PATH", seeded_db)

    exp_id = logexp.log_experiment(
        phase="Phase2_PGCPL",
        stage="training",
        script_id="train",
        dataset_id="D0",
        hyperparams={"num_gaussians": 3, "rank": 2},
        results={"psnr": 30.0},
        is_baseline=True,
    )
    assert exp_id.startswith("EXP-")
    logexp.set_baseline(exp_id)
    with pytest.raises(ValueError):
        logexp.set_baseline("EXP-00000000-000")
    logexp.list_tasks(status="todo")
    logexp.add_task("task", priority=2)
    logexp.list_tasks()
    logexp.complete_task(1)
    with pytest.raises(ValueError):
        logexp.complete_task(999)
    logexp.list_tasks(status="done")
    logexp.update_env()
    logexp.query_experiments(phase="Phase2_PGCPL", limit=1)
    logexp.query_experiments(baseline_only=True, limit=1)
    logexp.query_experiments(dataset_id="D0", min_psnr=1.0, limit=1)
    logexp.query_experiments(dataset_id="missing", min_psnr=1.0, limit=1)


def test_db_utils_default_schema_and_modified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    schema_path = ROOT / "schema.sql"
    schema_path.write_text("CREATE TABLE IF NOT EXISTS t(x INTEGER);", encoding="utf-8")
    try:
        db_path = tmp_path / "db.sqlite"
        monkeypatch.setattr(db_utils, "DB_PATH", db_path)
        db_utils.init_db()
        assert db_utils.get_db_last_modified() is not None
    finally:
        schema_path.unlink(missing_ok=True)


def test_run_all_main_non_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess as _subprocess
    monkeypatch.setattr(_subprocess, "run", lambda *a, **k: None)
    out_dir = tmp_path / "eval"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all",
            "--output-dir",
            str(out_dir),
            "--variant",
            "5d",
            "--python",
            "python",
        ],
    )
    run_all.main()


def test_run_all_import_inserts_sys_path(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "2_src"
    sys_path = list(sys.path)
    sys.path = [p for p in sys.path if p not in (str(root), str(src))]
    try:
        module = _load_module("run_all_reimport", root / "3_experiments" / "scripts" / "evaluation" / "run_all.py")
        assert str(root) in sys.path
        assert str(src) in sys.path
        assert module is not None
    finally:
        sys.path = sys_path


def test_run_pipeline_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train_cfg = tmp_path / "train.yaml"
    train_cfg.write_text("experiment:\n  output_dir: out\n", encoding="utf-8")
    train_cfg_root = tmp_path / "train_root.yaml"
    train_cfg_root.write_text("output_dir: out_root\n", encoding="utf-8")
    train_cfg_empty = tmp_path / "train_empty.yaml"
    train_cfg_empty.write_text("experiment:\n  name: x\n", encoding="utf-8")
    train_cfg_list = tmp_path / "train_list.yaml"
    train_cfg_list.write_text("- item\n", encoding="utf-8")
    pipeline_cfg = tmp_path / "pipeline.yaml"
    pipeline_cfg.write_text(
        "dataset:\n  config: data.yaml\n"
        "train:\n  entry: train.py\n  args:\n    output_dir: out_train\n    flag: true\n"
        "eval:\n  entry: eval.py\n  args:\n    flag: true\n"
        "python_train: /usr/bin/python\n"
        "python_eval: /usr/bin/python\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("tools.run_pipeline.build_plan", lambda *a, **k: (Path("out"), ["gen.py"], {"generator": {"script": "gen.py"}}))
    pipeline = _load_pipeline(pipeline_cfg)
    steps = build_pipeline_steps(pipeline)
    assert steps[0]["name"] == "dataset"

    pipeline2 = {
        "train": {"entry": "train.py", "args": {"config": str(train_cfg)}},
        "eval": {"entry": "eval.py", "args": {}},
    }
    steps2 = build_pipeline_steps(pipeline2)
    assert any(step["name"] == "train" for step in steps2)
    pipeline3 = {
        "train": {"entry": "train.py", "config": str(train_cfg), "args": {}},
        "eval": {"entry": "eval.py", "config": str(train_cfg), "args": {}},
    }
    build_pipeline_steps(pipeline3)

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- list", encoding="utf-8")
    with pytest.raises(ValueError):
        _load_pipeline(bad_yaml)

    assert _load_train_output_dir(tmp_path / "missing.yaml") is None
    assert _load_train_output_dir(train_cfg_root) == "out_root"
    assert _load_train_output_dir(train_cfg_empty) is None
    assert _load_train_output_dir(train_cfg_list) is None
    assert _load_train_output_dir(tmp_path) is None

    monkeypatch.setattr(sys, "argv", ["run_pipeline", "--config", str(pipeline_cfg), "--python", "python"])
    import subprocess as _subprocess
    monkeypatch.setattr(_subprocess, "run", lambda *a, **k: None)
    run_pipeline_main()


def test_logexp_cli_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_db(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["logexp"])
    with pytest.raises(SystemExit):
        logexp.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "logexp",
            "log",
            "--phase",
            "Phase2_PGCPL",
            "--stage",
            "training",
            "--script",
            "train",
            "--dataset",
            "D0",
            "--hyperparams",
            '{"a":1}',
            "--results",
            '{"psnr":30}',
            "--baseline",
        ],
    )
    logexp.main()

    monkeypatch.setattr(sys, "argv", ["logexp", "set-baseline", "EXP-20990101-001"])
    with pytest.raises(SystemExit):
        logexp.main()

    monkeypatch.setattr(sys, "argv", ["logexp", "add-task", "desc", "--priority", "2"])
    logexp.main()
    monkeypatch.setattr(sys, "argv", ["logexp", "complete-task", "1"])
    logexp.main()
    monkeypatch.setattr(sys, "argv", ["logexp", "list-tasks"])
    logexp.main()
    monkeypatch.setattr(sys, "argv", ["logexp", "update-env"])
    logexp.main()
    monkeypatch.setattr(sys, "argv", ["logexp", "query", "--dataset", "D0"])
    logexp.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "logexp",
            "log",
            "--phase",
            "Phase2_PGCPL",
            "--stage",
            "training",
            "--script",
            "train",
            "--dataset",
            "D0",
            "--results",
            "{bad_json}",
        ],
    )
    with pytest.raises(SystemExit):
        logexp.main()


def test_manifest_utils(tmp_path: Path) -> None:
    out_dir = tmp_path / "dataset"
    out_dir.mkdir()
    write_manifest(out_dir, "D1", "gen.py", "cfg.yaml")
    manifest = load_manifest(out_dir / "manifest.json")
    assert manifest["dataset_id"] == "D1"


def _dummy_train_args(tmp_path: Path) -> argparse.Namespace:
    return SimpleNamespace(
        variant="unified_set",
        config=None,
        schema=None,
        strict_schema=False,
        data_root=str(tmp_path),
        manifest=None,
        output_dir=str(tmp_path / "out"),
        device="cpu",
        run_id=None,
        num_gaussians=2,
        rank=2,
        top_k=1,
        epochs=1,
        batch_size=1,
        num_workers=0,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
        lr=1e-3,
        weight_decay=0.0,
        lr_scheduler="none",
        lr_min=1e-4,
        recon_loss="mse",
        charbonnier_eps=1e-3,
        lambda_temporal=0.0,
        grad_clip=1.0,
        lambda_linearity=0.0,
        linearity_aug_pairs=0,
        lambda_spatial=0.0,
        spatial_k=1,
        light_dim=12,
        embed_dim=16,
        intensity_dim=3,
        intensity_offset=1,
        disable_film=False,
        load_model=None,
        enable_sh_scaler=False,
        sh_scaler_path=None,
        sh_scaler_max_samples=10,
        no_init=True,
        print_history=False,
        enable_rerun=False,
        rerun_log_freq=10,
        rerun_save_path=None,
        no_auto_log=True,
        log_phase="Phase2_PGCPL",
        log_stage="training",
        log_script_id="train",
        log_dataset_id=None,
        log_notes=None,
        show_progress=False,
    )


def test_train_script_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("variant: unified_set\n", encoding="utf-8")
    cfg = train_script._load_yaml(cfg_path)
    assert cfg["variant"] == "unified_set"

    args = _dummy_train_args(tmp_path)
    defaults = _dummy_train_args(tmp_path)
    train_script._apply_config(args, defaults, {"experiment": {"variant": "unified_set"}})

    with pytest.raises(FileNotFoundError):
        train_script._load_yaml(tmp_path / "missing.yaml")

    assert train_script._device_from_arg("cpu").type == "cpu"

    # resolve_data_root
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"output_dir": "root"}), encoding="utf-8")
    assert train_script.resolve_data_root(None, str(manifest_path)) == "root"


def test_train_script_run_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Monkeypatch build_variant and trainer to avoid heavy training
    class DummyModel(torch.nn.Module):
        def forward(self, positions, params, top_k=1, light_mask=None):
            return torch.zeros((positions.shape[0], 27))

    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            return {"train": {"total": [0.0], "recon": [0.0], "coeff_l1": [0.0], "linearity": [0.0], "spatial": [0.0]},
                    "val": {"mae": [0.0], "rmse": [0.0], "charbonnier": [0.0], "superposition": [0.0]}}

        def validate_epoch(self, *args, **kwargs):
            return {"mae": 0.0, "rmse": 0.0, "charbonnier": 0.0, "superposition": 0.0}

    dummy_loader = []
    monkeypatch.setattr(train_script, "GaussianPhysicsTrainer", DummyTrainer)
    monkeypatch.setattr(
        train_script,
        "build_variant",
        lambda variant, args, device: (DummyModel(), None, (dummy_loader, dummy_loader, dummy_loader), {"init_fn": lambda m, l: None}),
    )

    args = _dummy_train_args(tmp_path)
    train_script.run_training(args)


def test_run_all_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "experiment:\n  variant: unified_set\n"
        "evaluation:\n  output_dir: out\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_all.py", "--config", str(cfg_path), "--dry-run"],
    )
    run_all.main()
    assert "DRY-RUN" in capsys.readouterr().out

    # Explicit physics suite
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_all.py", "--output-dir", str(tmp_path / "out"), "--suite", "physics", "--dry-run"],
    )
    run_all.main()

    monkeypatch.setenv("PYTHONPATH", "dummy")
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_all.py", "--output-dir", str(tmp_path / "out2"), "--suite", "physics", "--dry-run"],
    )
    run_all.main()

    # Missing output dir should error
    monkeypatch.setattr(sys, "argv", ["run_all.py"])
    with pytest.raises(ValueError):
        run_all.main()


def test_run_all_steps_variants(tmp_path: Path) -> None:
    args = SimpleNamespace(
        data_root=str(tmp_path / "data"),
        manifest=str(tmp_path / "manifest.json"),
        checkpoint=str(tmp_path / "ckpt.pt"),
        device="cpu",
        rank=8,
        train_mae=0.1,
        n_samples=5,
        seed=123,
        probe_idx=0,
        batch_size=4,
        num_workers=0,
        train_ratio=0.7,
        val_ratio=0.2,
        split="test",
        num_gaussians=10,
        top_k=3,
        light_dim=12,
        embed_dim=16,
        intensity_dim=3,
        intensity_offset=1,
        disable_film=True,
        sh_scaler="scaler.npz",
        near_threshold=0.2,
        k_neighbors=5,
        save_pred_config=1,
        save_pred_out="pred.npy",
    )
    steps = run_all.build_unified_steps(tmp_path, args)
    assert steps[0]["name"] == "unified_ablation"

    steps_eval = run_all.build_eval_steps(tmp_path, args)
    assert steps_eval[0]["name"] == "interpolation"
