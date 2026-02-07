from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import builtins
import importlib
import sys

import numpy as np
import pytest
import torch

import importlib.util


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


train_script = _load_module("train_script_cov", ROOT / "3_experiments" / "scripts" / "train.py")


class DummyDataset1D:
    def get_full_probe_positions(self, normalized: bool = True):
        return np.zeros((2, 3), dtype=np.float32)


class DummyDataset5D:
    def __init__(self):
        self.tensor = np.zeros((2, 2, 27), dtype=np.float32)
        self.probe_positions = np.zeros((2, 3), dtype=np.float32)
        self.light_configs_subset = np.zeros((2, 2, 12), dtype=np.float32)


class DummyLoader:
    def __init__(self, dataset):
        self.dataset = dataset


class DummyModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def initialize_from_probes(self, *args, **kwargs) -> None:
        return None

    def init_from_kmeans(self, *args, **kwargs) -> None:
        return None

    def load_state_dict(self, *args, **kwargs):
        return ["missing"], ["unexpected"]


class DummyAdapter:
    def __init__(self, params_key=None, mask_key=None, params_transform=None):
        self.params_key = params_key
        self.mask_key = mask_key
        self.params_transform = params_transform
        self.target_transform = None
        self.target_inverse = None


class DummyTrainer:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def fit(self, *args, **kwargs):
        return {"train": {"mae": [1.0]}, "val": {"mae": [2.0]}}

    def validate_epoch(self, *args, **kwargs):
        return {"mae": 0.1}


class DummyScaler:
    def __init__(self):
        self.l0_mean = [0.0]
        self.l0_std = [1.0]
        self.ho_rms = [0.5]

    @classmethod
    def fit_from_dataset(cls, *args, **kwargs):
        return cls()

    @classmethod
    def load(cls, *args, **kwargs):
        return cls()

    def save(self, path: Path) -> None:
        Path(path).write_text("ok", encoding="utf-8")

    def transform(self, x):
        return x

    def inverse(self, x):
        return x


def _patch_train_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_script, "GaussianPhysicsCompressionUnified", DummyModel)
    monkeypatch.setattr(train_script, "BatchAdapter", DummyAdapter)
    monkeypatch.setattr(train_script, "GaussianPhysicsTrainer", DummyTrainer)
    monkeypatch.setattr(train_script, "create_dataloaders_lightset", lambda **_: (DummyLoader(DummyDataset5D()), DummyLoader(DummyDataset5D()), DummyLoader(DummyDataset5D())))


def _build_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        variant="unified_set",
        data_root=str(tmp_path / "data"),
        manifest=None,
        output_dir=None,
        device="cpu",
        run_id="run",
        show_progress=True,
        no_init=False,
        enable_sh_scaler=True,
        sh_scaler_path=None,
        sh_scaler_max_samples=10,
        load_model=str(tmp_path / "ckpt.pt"),
        enable_rerun=True,
        rerun_save_path=None,
        rerun_log_freq=1,
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        lr_scheduler="none",
        lr_min=1e-4,
        recon_loss="mse",
        charbonnier_eps=1e-3,
        lambda_temporal=0.0,
        top_k=3,
        grad_clip=1.0,
        lambda_linearity=0.0,
        linearity_aug_pairs=0,
        lambda_spatial=0.0,
        spatial_k=1,
        batch_size=1,
        num_workers=0,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
        num_gaussians=2,
        rank=1,
        light_dim=5,
        embed_dim=4,
        intensity_dim=1,
        intensity_offset=0,
        disable_film=False,
        print_history=True,
        no_auto_log=False,
        log_phase="Phase2_PGCPL",
        log_stage="training",
        log_script_id="train",
        log_dataset_id=None,
        log_notes=None,
    )


def test_train_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("experiment:\n  variant: unified_set\n", encoding="utf-8")
    data = train_script._load_yaml(cfg_path)
    assert data["experiment"]["variant"] == "unified_set"
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text("- list", encoding="utf-8")
    with pytest.raises(ValueError):
        train_script._load_yaml(bad_path)

    args = SimpleNamespace(variant=None, data_root=None)
    defaults = SimpleNamespace(variant=None, data_root=None)
    train_script._apply_config(args, defaults, {"experiment": {"variant": "unified_set"}, "data": {"data_root": "x"}})
    assert args.variant == "unified_set"
    assert args.data_root == "x"

    assert train_script._device_from_arg("cpu").type == "cpu"
    assert train_script._device_from_arg("").type in ("cpu", "cuda")
    assert train_script.resolve_data_root("x", None) == "x"

    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"output_dir": "out"}', encoding="utf-8")
    assert train_script.resolve_data_root(None, str(manifest)) == "out"
    with pytest.raises(ValueError):
        train_script.resolve_data_root(None, None)
    bad_manifest = tmp_path / "bad_manifest.json"
    bad_manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        train_script.resolve_data_root(None, str(bad_manifest))


def test_build_variant_and_init(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_train_deps(monkeypatch)
    args = _build_args(Path("/tmp"))
    device = torch.device("cpu")

    for variant in ("unified_set",):
        args.variant = variant
        model, adapter, loaders, helpers = train_script.build_variant(variant, args, device)
        assert model is not None
        assert adapter is not None
        assert helpers["init_fn"] is not None
        if adapter.params_transform is not None:
            adapter.params_transform(torch.zeros((1, 2, 12)))

    with pytest.raises(ValueError):
        train_script.build_variant("bad", args, device)

    train_script._init_5d(DummyModel(), DummyLoader(DummyDataset5D()))


def test_run_training_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_train_deps(monkeypatch)

    monkeypatch.setattr(train_script, "get_git_commit", lambda: "abc")
    monkeypatch.setattr(train_script, "log_experiment", lambda **_: "EXP-TEST")
    monkeypatch.setattr(train_script, "load_manifest", lambda *_: {"dataset_id": "D0"})
    monkeypatch.setattr(train_script.torch, "load", lambda *a, **k: {"model_state_dict": {}})

    import data.sh_scaler as sh_scaler
    monkeypatch.setattr(sh_scaler, "AdaptiveSHScaler", DummyScaler)

    class DummyRerunLogger:
        def __init__(self, *args, **kwargs):
            self.args = args

    sys.modules["utils.rerun_logger"] = SimpleNamespace(RerunLogger=DummyRerunLogger)

    args = _build_args(tmp_path)
    args.manifest = str(tmp_path / "manifest.json")
    args.output_dir = str(tmp_path / "out_fit")
    args.sh_scaler_path = str(tmp_path / "scaler_fit.npz")
    train_script.run_training(args)

    args2 = _build_args(tmp_path)
    args2.enable_rerun = True
    args2.no_auto_log = True
    args2.no_init = True
    args2.output_dir = str(tmp_path / "out")
    args2.sh_scaler_path = str(tmp_path / "scaler.npz")
    Path(args2.sh_scaler_path).write_text("x", encoding="utf-8")

    def _raise_import(name, *a, **k):
        if name == "utils.rerun_logger":
            raise ImportError("nope")
        return importlib.__import__(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _raise_import)
    train_script.run_training(args2)

    monkeypatch.setattr(train_script, "log_experiment", lambda **_: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(train_script, "load_manifest", lambda *_: (_ for _ in ()).throw(RuntimeError("bad")))
    args3 = _build_args(tmp_path)
    args3.no_init = True
    args3.enable_sh_scaler = False
    args3.enable_rerun = False
    args3.manifest = "bad.json"
    train_script.run_training(args3)

    args4 = _build_args(tmp_path)
    args4.no_init = True
    args4.enable_sh_scaler = True
    args4.enable_rerun = False
    args4.no_auto_log = True
    args4.output_dir = str(tmp_path / "out_none")
    args4.sh_scaler_path = None
    train_script.run_training(args4)


def test_train_parser_defaults_and_apply_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = train_script.build_arg_parser()
    args = parser.parse_args(["--variant", "unified_set"])
    train_script.apply_variant_defaults(args)
    assert args.num_gaussians is not None


def test_train_sys_path_insert() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "2_src"
    sys_path = list(sys.path)
    sys.path = [p for p in sys.path if p not in (str(root), str(src))]
    try:
        module = _load_module("train_script_reimport", root / "3_experiments" / "scripts" / "train.py")
        assert str(root) in sys.path
        assert str(src) in sys.path
        assert module is not None
    finally:
        sys.path = sys_path
