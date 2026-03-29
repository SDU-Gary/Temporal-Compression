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
        resume=None,
        resume_save_every=1,
        resume_checkpoint_name="resume_latest.pt",
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
        lambda_image=0.0,
        image_loss_warmup_epochs=0,
        image_loss_type="mse",
        image_samples=64,
        image_sample_seed=42,
        image_loss_space="linear",
        enable_weighted_sh_loss=False,
        sh_loss_weights=None,
        sh_weight_mode="basis",
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


def test_resolve_val_profiles_config() -> None:
    args = SimpleNamespace(
        top_k=3,
        _config_obj={
            "training": {
                "val_profiles": {
                    "enabled": True,
                    "best_metric": "img_psnr",
                    "profiles": [
                        {"name": "hard3", "top_k": 3, "training_soft_routing": False},
                        {"name": "soft8_t020", "top_k": 3, "training_soft_routing": True, "routing_soft_topk": 8, "routing_temperature": 0.2},
                    ],
                },
                "test_compare_profiles": {"enabled": True},
            }
        },
    )
    cfg = train_script._resolve_val_profiles_config(args)
    assert cfg["enabled"] is True
    assert cfg["best_metric"] == "img_psnr"
    assert cfg["run_test_compare"] is True
    assert len(cfg["profiles"]) == 2
    assert cfg["profiles"][0]["name"] == "hard3"
    assert cfg["profiles"][1]["routing_temperature"] == 0.2


def test_resolve_proxy_and_falcor_periodic_config() -> None:
    args = SimpleNamespace(
        top_k=3,
        seed=42,
        batch_size=512,
        lambda_image=0.0,
        image_loss_warmup_epochs=0,
        device="cuda",
        _config_obj={
            "training": {
                "proxy_image_loss": {
                    "enabled": True,
                    "lambda": 0.1,
                    "warmup_epochs": 50,
                    "enable_val_image_metrics": True,
                    "mix_mode": "linear_log_mix",
                    "log_mix_weight": 0.3,
                },
                "falcor_periodic_eval": {
                    "enabled": True,
                    "scene": "dummy_scene.pyscene",
                    "every_n_epochs": 20,
                    "stage_end_full": True,
                    "profile": {
                        "name": "soft8_t018",
                        "top_k": 3,
                        "training_soft_routing": True,
                        "routing_soft_topk": 8,
                        "routing_temperature": 0.18,
                    },
                },
            }
        },
    )

    proxy_cfg = train_script._resolve_proxy_image_loss_config(args)
    assert proxy_cfg["enabled"] is True
    assert proxy_cfg["lambda"] == 0.1
    assert proxy_cfg["warmup_epochs"] == 50
    assert proxy_cfg["enable_val_image_metrics"] is True
    assert proxy_cfg["mix_mode"] == "linear_log_mix"
    assert proxy_cfg["log_mix_weight"] == 0.3

    falcor_cfg = train_script._resolve_falcor_periodic_eval_config(args)
    assert falcor_cfg["enabled"] is True
    assert falcor_cfg["every_n_epochs"] == 20
    assert falcor_cfg["profile"]["name"] == "soft8_t018"
    assert falcor_cfg["profile"]["routing_temperature"] == 0.18


def test_resolve_routing_balance_anneal_config() -> None:
    args = SimpleNamespace(
        lambda_routing_balance=0.02,
        _config_obj={
            "training": {
                "routing_balance_anneal": {
                    "enabled": True,
                    "start": 0.02,
                    "end": 0.0,
                    "decay_end_ratio": 0.6,
                }
            }
        },
    )
    cfg = train_script._resolve_routing_balance_anneal_config(args)
    assert cfg["enabled"] is True
    assert cfg["start"] == 0.02
    assert cfg["end"] == 0.0
    assert cfg["decay_end_ratio"] == 0.6


def test_resolve_coeff_suite_config_with_checkpoints() -> None:
    args = SimpleNamespace(
        seed=42,
        batch_size=512,
        _config_obj={
            "training": {
                "coeff_suite": {
                    "enabled": True,
                    "run_after_training": True,
                    "split": "test",
                    "checkpoints": [
                        "global_best_falcor.pt",
                        "global_best_val_soft8_t018.pt",
                        "best_model.pt",
                        "last_model.pt",
                    ],
                }
            }
        },
    )
    cfg = train_script._resolve_coeff_suite_config(args)
    assert cfg["enabled"] is True
    assert cfg["run_after_training"] is True
    assert cfg["split"] == "test"
    assert cfg["checkpoints"] == [
        "global_best_falcor.pt",
        "global_best_val_soft8_t018.pt",
        "best_model.pt",
        "last_model.pt",
    ]


def test_resolve_global_best_config() -> None:
    args = SimpleNamespace(
        _config_obj={
            "training": {
                "global_best": {
                    "enabled": True,
                    "track_falcor": True,
                    "track_soft_profile": True,
                    "soft_profile_name": "soft8_t018",
                    "soft_metric": "img_psnr",
                    "soft_metric_maximize": True,
                }
            }
        },
    )
    cfg = train_script._resolve_global_best_config(args)
    assert cfg["enabled"] is True
    assert cfg["track_falcor"] is True
    assert cfg["track_soft_profile"] is True
    assert cfg["soft_profile_name"] == "soft8_t018"
    assert cfg["soft_metric"] == "img_psnr"
    assert cfg["soft_metric_maximize"] is True


def test_resolve_performance_config_amp_mode_bool_off() -> None:
    args = SimpleNamespace(
        amp_mode="off",
        torch_compile=False,
        torch_compile_mode="reduce-overhead",
        torch_compile_dynamic=False,
        grad_norm_log_every_steps=1,
        enable_soft_profile_sharing=False,
        soft_profile_equiv_check_batches=2,
        soft_profile_mae_tolerance=1e-6,
        soft_profile_img_psnr_tolerance=5e-4,
        _config_obj={
            "training": {
                "performance": {
                    "amp_mode": False,
                }
            }
        },
    )
    cfg = train_script._resolve_performance_config(args)
    assert cfg["amp_mode"] == "off"


def test_resolve_expert_and_drift_audit_config() -> None:
    args = SimpleNamespace(
        top_k=3,
        seed=42,
        batch_size=512,
        _config_obj={
            "training": {
                "expert_utilization_audit": {
                    "enabled": True,
                    "run_after_training": True,
                    "split": "test",
                    "device": "cpu",
                    "max_samples": 4096,
                    "profiles": [
                        {"name": "hard3", "top_k": 3, "training_soft_routing": False},
                        {"name": "soft8_t020", "top_k": 3, "training_soft_routing": True, "routing_soft_topk": 8, "routing_temperature": 0.2},
                    ],
                },
                "semantic_drift_audit": {
                    "enabled": True,
                    "run_after_training": True,
                    "split": "test",
                    "device": "cpu",
                    "max_samples": 4096,
                    "pairs": [
                        {
                            "name": "best_vs_last",
                            "checkpoint_a": "best_model.pt",
                            "checkpoint_b": "last_model.pt",
                        }
                    ],
                },
            }
        },
    )
    util_cfg = train_script._resolve_expert_utilization_audit_config(args)
    assert util_cfg["enabled"] is True
    assert util_cfg["max_samples"] == 4096
    assert len(util_cfg["profiles"]) == 2
    assert util_cfg["profiles"][1]["routing_temperature"] == 0.2

    drift_cfg = train_script._resolve_semantic_drift_audit_config(args)
    assert drift_cfg["enabled"] is True
    assert drift_cfg["max_samples"] == 4096
    assert len(drift_cfg["pairs"]) == 1
    assert drift_cfg["pairs"][0]["name"] == "best_vs_last"


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


def test_run_training_resume_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_train_deps(monkeypatch)

    monkeypatch.setattr(train_script, "get_git_commit", lambda: "abc")
    monkeypatch.setattr(train_script, "log_experiment", lambda **_: "EXP-TEST")
    monkeypatch.setattr(train_script, "load_manifest", lambda *_: {"dataset_id": "D0"})

    resume_ckpt = tmp_path / "resume_latest.pt"
    resume_ckpt.write_text("x", encoding="utf-8")

    fake_ckpt = {
        "epoch": 3,
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "meta": {"training": {"stage_index": 1, "stage_name": "single_stage"}},
    }
    monkeypatch.setattr(train_script.torch, "load", lambda *a, **k: fake_ckpt)

    args = _build_args(tmp_path)
    args.resume = str(resume_ckpt)
    args.load_model = None
    args.no_init = True
    args.enable_sh_scaler = False
    args.enable_rerun = False
    args.no_auto_log = True
    args.output_dir = str(tmp_path / "out_resume")
    train_script.run_training(args)


def test_train_parser_defaults_and_apply_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = train_script.build_arg_parser()
    args = parser.parse_args(["--variant", "unified_set"])
    train_script.apply_variant_defaults(args)
    assert args.num_gaussians is not None


def test_resume_path_and_stage_helpers(tmp_path: Path) -> None:
    output_dir = tmp_path / "exp"
    stage_dir = output_dir / "stage01_warmup_joint_fast"
    stage_dir.mkdir(parents=True, exist_ok=True)
    ckpt = stage_dir / "resume_latest.pt"
    ckpt.write_text("x", encoding="utf-8")

    resolved = train_script._resolve_resume_checkpoint_path(str(ckpt), output_dir)
    assert resolved == ckpt.resolve()
    assert train_script._infer_stage_index_from_checkpoint_path(ckpt) == 1


def test_normalize_train_args_fills_missing_fields() -> None:
    args = SimpleNamespace(variant="unified_set", data_root="/tmp/d")
    train_script.normalize_train_args(args)
    assert args.val_image_metrics is False
    assert args.val_superposition is False
    assert args.enable_rerun is False


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
