from __future__ import annotations

import torch

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
