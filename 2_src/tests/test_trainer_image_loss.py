from __future__ import annotations

import pytest
import torch

from data.gbuffer_supervision_dataset import (
    GBufferSupervisionDataset,
    create_gbuffer_supervision_dataloader,
)
from training.gaussian_physics_trainer import BatchAdapter, GaussianPhysicsTrainer


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(27))

    def forward(self, positions, params, top_k=3, light_mask=None):
        batch = positions.shape[0]
        return self.bias.unsqueeze(0).expand(batch, -1)

    def compute_temporal_smoothness_loss(self):
        return torch.tensor(0.0, device=self.bias.device)


class TinyRoutingModel(torch.nn.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(27))
        self.k = int(k)
        self.last_param_select_mode = None

    def compute_gaussian_routing(
        self,
        positions,
        top_k=3,
        training_soft_routing=False,
        routing_temperature=1.0,
        routing_soft_topk=None,
    ):
        b = int(positions.shape[0])
        kk = self.k
        weights = torch.full((b, kk), 1.0 / float(kk), dtype=positions.dtype, device=positions.device)
        idx = torch.arange(kk, device=positions.device, dtype=torch.long).unsqueeze(0).expand(b, -1)
        return weights, idx

    def forward_with_routing(self, routing, light_params, light_mask=None, *, param_select_mode="gather", contraction_mode=None):
        self.last_param_select_mode = str(param_select_mode)
        b = int(light_params.shape[0])
        return self.bias.unsqueeze(0).expand(b, -1)

    def forward(self, positions, params, top_k=3, light_mask=None, training_soft_routing=False, routing_temperature=1.0, routing_soft_topk=None):
        b = int(positions.shape[0])
        return self.bias.unsqueeze(0).expand(b, -1)

    def compute_temporal_smoothness_loss(self):
        return torch.tensor(0.0, device=self.bias.device)


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, value: float = 0.0):
        self.value = value

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {
            "probe_position": torch.zeros(3, dtype=torch.float32),
            "light_params": torch.zeros((2, 12), dtype=torch.float32),
            "light_mask": torch.ones(2, dtype=torch.float32),
            "sh_coeffs": torch.full((27,), self.value, dtype=torch.float32),
            "probe_idx": torch.tensor(idx % 2, dtype=torch.long),
        }


def _make_trainer(**kwargs) -> GaussianPhysicsTrainer:
    model = TinyModel()
    adapter = BatchAdapter(params_key="light_params", mask_key="light_mask")
    return GaussianPhysicsTrainer(
        model=model,
        device=torch.device("cpu"),
        adapter=adapter,
        lr=1e-3,
        recon_loss="mse",
        temporal_weight=0.0,
        linearity_weight=0.0,
        spatial_weight=0.0,
        show_progress=False,
        **kwargs,
    )


def test_weighted_sh_recon_loss_changes_scale() -> None:
    pred = torch.ones((2, 27), dtype=torch.float32)
    target = torch.zeros((2, 27), dtype=torch.float32)

    trainer_plain = _make_trainer(enable_weighted_sh_loss=False)
    trainer_weighted = _make_trainer(enable_weighted_sh_loss=True)

    plain = trainer_plain._recon_loss(pred, target).item()
    weighted = trainer_weighted._recon_loss(pred, target).item()

    assert plain > 0.0
    assert weighted > plain


def test_image_loss_and_validate_metrics() -> None:
    trainer = _make_trainer(
        image_loss_weight=0.1,
        image_loss_type="mse",
        image_samples=16,
        image_sample_seed=7,
        image_loss_space="linear",
        enable_weighted_sh_loss=True,
    )

    a = torch.zeros((3, 27), dtype=torch.float32)
    b = torch.zeros((3, 27), dtype=torch.float32)
    c = torch.ones((3, 27), dtype=torch.float32)

    assert trainer._compute_image_loss(a, b).item() == 0.0
    assert trainer._compute_image_loss(a, c).item() > 0.0

    loader = torch.utils.data.DataLoader(TinyDataset(value=1.0), batch_size=2)
    val = trainer.validate_epoch(loader)

    assert "img_mae" in val
    assert "img_rmse" in val
    assert "img_psnr" in val


def test_gbuffer_image_loss_train_step(tmp_path) -> None:
    gbuffer_root = tmp_path / "gbuffer"
    gbuffer_root.mkdir(parents=True, exist_ok=True)
    sample = {
        "posW": torch.zeros((4, 4, 3), dtype=torch.float32),
        "normW": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).reshape(1, 1, 3).expand(4, 4, 3),
        "albedo": torch.ones((4, 4, 3), dtype=torch.float32),
        "gt_linear": torch.ones((4, 4, 3), dtype=torch.float32) * 0.5,
        "light_params": torch.zeros((2, 12), dtype=torch.float32),
        "light_mask": torch.ones((2,), dtype=torch.float32),
    }
    torch.save(sample, gbuffer_root / "frame_0000.pt")

    gbuffer_loader = create_gbuffer_supervision_dataloader(
        data_root=gbuffer_root,
        pixel_sample_count=8,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    trainer = _make_trainer(
        gbuffer_image_loss_cfg={
            "enabled": True,
            "lambda": 0.2,
            "warmup_epochs": 0,
            "every_steps": 1,
            "loss_type": "charbonnier",
            "dataset_root": str(gbuffer_root),
            "pixel_sample_count": 8,
        },
        gbuffer_loader=gbuffer_loader,
        gbuffer_probe_min=[0.0, 0.0, 0.0],
        gbuffer_probe_max=[1.0, 1.0, 1.0],
    )

    loader = torch.utils.data.DataLoader(TinyDataset(value=0.0), batch_size=2)
    metrics = trainer.train_epoch(loader)
    assert metrics["gbuffer"] > 0.0
    assert metrics["gbuffer_samples"] > 0.0
    assert metrics["gbuffer_image_loss_weight"] > 0.0


def test_proxy_gbuffer_handover_weights_schedule(tmp_path) -> None:
    gbuffer_root = tmp_path / "gbuffer_handover"
    gbuffer_root.mkdir(parents=True, exist_ok=True)
    sample = {
        "posW": torch.zeros((2, 2, 3), dtype=torch.float32),
        "normW": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).reshape(1, 1, 3).expand(2, 2, 3),
        "albedo": torch.ones((2, 2, 3), dtype=torch.float32),
        "gt_linear": torch.ones((2, 2, 3), dtype=torch.float32) * 0.4,
        "light_params": torch.zeros((2, 12), dtype=torch.float32),
        "light_mask": torch.ones((2,), dtype=torch.float32),
    }
    torch.save(sample, gbuffer_root / "frame_0000.pt")

    gbuffer_loader = create_gbuffer_supervision_dataloader(
        data_root=gbuffer_root,
        pixel_sample_count=4,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    trainer = _make_trainer(
        image_loss_weight=0.2,
        image_loss_warmup_epochs=0,
        gbuffer_image_loss_cfg={
            "enabled": True,
            "lambda": 0.1,
            "warmup_epochs": 0,
            "every_steps": 1,
            "loss_type": "charbonnier",
            "dataset_root": str(gbuffer_root),
            "pixel_sample_count": 4,
        },
        proxy_gbuffer_handover_cfg={
            "enabled": True,
            "start_epoch": 3,
            "end_epoch": 5,
            "proxy_start_scale": 1.0,
            "proxy_end_scale": 0.0,
            "gbuffer_start_scale": 0.0,
            "gbuffer_end_scale": 2.0,
        },
        gbuffer_loader=gbuffer_loader,
        gbuffer_probe_min=[0.0, 0.0, 0.0],
        gbuffer_probe_max=[1.0, 1.0, 1.0],
    )

    w_img_2, w_gbuf_2 = trainer._aux_image_weights_for_epoch(2)
    w_img_4, w_gbuf_4 = trainer._aux_image_weights_for_epoch(4)
    w_img_6, w_gbuf_6 = trainer._aux_image_weights_for_epoch(6)

    assert abs(w_img_2 - 0.2) < 1e-12
    assert abs(w_gbuf_2 - 0.0) < 1e-12
    assert abs(w_img_4 - 0.1) < 1e-12
    assert abs(w_gbuf_4 - 0.1) < 1e-12
    assert abs(w_img_6 - 0.0) < 1e-12
    assert abs(w_gbuf_6 - 0.2) < 1e-12


def test_gbuffer_respects_dense_masked_routing_mode(tmp_path) -> None:
    gbuffer_root = tmp_path / "gbuffer_dense"
    gbuffer_root.mkdir(parents=True, exist_ok=True)
    sample = {
        "posW": torch.zeros((2, 2, 3), dtype=torch.float32),
        "normW": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).reshape(1, 1, 3).expand(2, 2, 3),
        "albedo": torch.ones((2, 2, 3), dtype=torch.float32),
        "gt_linear": torch.ones((2, 2, 3), dtype=torch.float32) * 0.25,
        "light_params": torch.zeros((2, 12), dtype=torch.float32),
        "light_mask": torch.ones((2,), dtype=torch.float32),
    }
    torch.save(sample, gbuffer_root / "frame_0000.pt")

    gbuffer_loader = create_gbuffer_supervision_dataloader(
        data_root=gbuffer_root,
        pixel_sample_count=4,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    model = TinyRoutingModel(k=3)
    adapter = BatchAdapter(params_key="light_params", mask_key="light_mask")
    trainer = GaussianPhysicsTrainer(
        model=model,
        device=torch.device("cpu"),
        adapter=adapter,
        lr=1e-3,
        recon_loss="mse",
        temporal_weight=0.0,
        linearity_weight=0.0,
        spatial_weight=0.0,
        show_progress=False,
        train_routing_param_mode="dense_masked",
        gbuffer_image_loss_cfg={
            "enabled": True,
            "lambda": 0.2,
            "warmup_epochs": 0,
            "every_steps": 1,
            "loss_type": "charbonnier",
            "dataset_root": str(gbuffer_root),
            "pixel_sample_count": 4,
        },
        gbuffer_loader=gbuffer_loader,
        gbuffer_probe_min=[0.0, 0.0, 0.0],
        gbuffer_probe_max=[1.0, 1.0, 1.0],
    )

    loader = torch.utils.data.DataLoader(TinyDataset(value=0.0), batch_size=2)
    metrics = trainer.train_epoch(loader)
    assert metrics["gbuffer"] > 0.0
    assert model.last_param_select_mode == "dense_masked"


def test_gbuffer_dataset_strict_keys_prevents_ambiguous_gt(tmp_path) -> None:
    gbuffer_root = tmp_path / "gbuffer_strict"
    gbuffer_root.mkdir(parents=True, exist_ok=True)
    # Deliberately store GT under a fallback key so strict mode should reject it.
    sample = {
        "posW": torch.zeros((2, 2, 3), dtype=torch.float32),
        "normW": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).reshape(1, 1, 3).expand(2, 2, 3),
        "albedo": torch.ones((2, 2, 3), dtype=torch.float32),
        "gt": torch.full((2, 2, 3), 0.5, dtype=torch.float32),
        "light_params": torch.zeros((2, 12), dtype=torch.float32),
    }
    torch.save(sample, gbuffer_root / "frame_0000.pt")

    dataset = GBufferSupervisionDataset(
        data_root=gbuffer_root,
        pixel_sample_count=4,
        strict_keys=True,
        gt_linear_key="gt_linear",
    )
    with pytest.raises(KeyError):
        _ = dataset[0]


def test_gbuffer_dataset_srgb_target_is_decoded_to_linear(tmp_path) -> None:
    gbuffer_root = tmp_path / "gbuffer_srgb"
    gbuffer_root.mkdir(parents=True, exist_ok=True)
    sample = {
        "world_pos": torch.zeros((1, 1, 3), dtype=torch.float32),
        "world_normal": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).reshape(1, 1, 3),
        "base_color": torch.ones((1, 1, 3), dtype=torch.float32),
        "gt": torch.full((1, 1, 3), 0.5, dtype=torch.float32),
        "light_params": torch.zeros((2, 12), dtype=torch.float32),
    }
    torch.save(sample, gbuffer_root / "frame_0000.pt")

    loader = create_gbuffer_supervision_dataloader(
        data_root=gbuffer_root,
        pixel_sample_count=1,
        strict_keys=False,
        pos_key="posW",
        normal_key="normW",
        albedo_key="albedo",
        gt_linear_key="gt_linear",
        gt_color_space="srgb",
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    batch = next(iter(loader))
    gt_linear = batch["gt_linear"].reshape(-1, 3)
    # sRGB 0.5 maps to linear ~0.214 (inverse sRGB transfer).
    assert torch.allclose(gt_linear, torch.full_like(gt_linear, 0.21404114), atol=1e-5)
