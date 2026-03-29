from __future__ import annotations

from pathlib import Path

import torch

from training.gaussian_physics_trainer import BatchAdapter, GaussianPhysicsTrainer


class _GroupedDummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.zeros(2, 3))
        self.log_scale = torch.nn.Parameter(torch.zeros(2, 3))
        self.U = torch.nn.Parameter(torch.zeros(2, 27, 4))
        self.U_l0 = torch.nn.Parameter(torch.zeros(2, 3, 4))
        self.coeffs = torch.nn.Parameter(torch.zeros(2, 4, 8))
        self.coeffs_l0 = torch.nn.Parameter(torch.zeros(2, 4, 8))
        self.light_encoder = torch.nn.Linear(12, 8)
        self.gamma = torch.nn.Sequential(torch.nn.Linear(8, 4))
        self.beta = torch.nn.Sequential(torch.nn.Linear(8, 4))

    def forward(self, positions, params, top_k=3, light_mask=None):
        batch = positions.shape[0]
        return torch.zeros((batch, 27), device=positions.device, dtype=positions.dtype)

    def compute_temporal_smoothness_loss(self):
        return torch.tensor(0.0, device=self.mu.device)




class _RoutingDummyModel(torch.nn.Module):
    def __init__(self, num_gaussians: int = 6, sh_dim: int = 27):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.linspace(-1.0, 1.0, num_gaussians).unsqueeze(-1).repeat(1, 3))
        self.log_scale = torch.nn.Parameter(torch.zeros(num_gaussians, 3))
        self.U = torch.nn.Parameter(torch.randn(num_gaussians, sh_dim, 1) * 0.01)

    def compute_gaussian_routing(
        self,
        positions,
        top_k=3,
        training_soft_routing=False,
        routing_temperature=1.0,
        routing_soft_topk=None,
    ):
        distances = ((positions.unsqueeze(1) - self.mu.unsqueeze(0)) ** 2).sum(dim=-1)
        if training_soft_routing:
            soft_k = self.mu.shape[0] if not routing_soft_topk or routing_soft_topk <= 0 else min(self.mu.shape[0], int(routing_soft_topk))
            values, indices = torch.topk(distances, k=soft_k, largest=False, dim=-1)
            temp = max(float(routing_temperature), 1e-6)
            logits = -values / temp
            weights = torch.softmax(logits, dim=-1)
            return weights, indices

        values, indices = torch.topk(distances, k=min(int(top_k), self.mu.shape[0]), largest=False, dim=-1)
        weights = torch.softmax(-values, dim=-1)
        return weights, indices

    def forward_with_routing(self, routing, params, light_mask=None):
        weights, indices = routing
        selected = self.U[indices, :, 0]
        return torch.einsum('bk,bkd->bd', weights, selected)

    def forward(self, positions, params, top_k=3, light_mask=None, **kwargs):
        routing = self.compute_gaussian_routing(positions, top_k=top_k, **kwargs)
        return self.forward_with_routing(routing, params, light_mask=light_mask)

    def compute_temporal_smoothness_loss(self):
        return torch.tensor(0.0, device=self.mu.device)

class _BiasModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(27))

    def forward(self, positions, params, top_k=3, light_mask=None):
        batch = positions.shape[0]
        return self.bias.unsqueeze(0).expand(batch, -1)

    def compute_temporal_smoothness_loss(self):
        return torch.tensor(0.0, device=self.bias.device)


class _TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        return {
            "probe_position": torch.zeros(3, dtype=torch.float32),
            "light_params": torch.zeros((2, 12), dtype=torch.float32),
            "light_mask": torch.ones(2, dtype=torch.float32),
            "sh_coeffs": torch.ones((27,), dtype=torch.float32),
            "probe_idx": torch.tensor(idx % 2, dtype=torch.long),
        }


def test_param_group_lrs_are_applied() -> None:
    model = _GroupedDummyModel()
    trainer = GaussianPhysicsTrainer(
        model=model,
        device=torch.device("cpu"),
        adapter=BatchAdapter(params_key="light_params", mask_key="light_mask"),
        lr=1e-4,
        weight_decay=0.0,
        recon_loss="mse",
        temporal_weight=0.0,
        linearity_weight=0.0,
        spatial_weight=0.0,
        param_group_lrs={"all": 1e-4, "coeff": 6e-4, "encoder": 5e-4, "film": 4e-4},
        show_progress=False,
    )

    group_lrs = {str(pg.get("group", "")): float(pg["lr"]) for pg in trainer.optimizer.param_groups}
    assert abs(group_lrs["routing"] - 1e-4) < 1e-12
    assert abs(group_lrs["basis"] - 1e-4) < 1e-12
    assert abs(group_lrs["coeff"] - 6e-4) < 1e-12
    assert abs(group_lrs["encoder"] - 5e-4) < 1e-12
    assert abs(group_lrs["film"] - 4e-4) < 1e-12


def test_ema_eval_and_save_best_checkpoint(tmp_path: Path) -> None:
    model = _BiasModel()
    trainer = GaussianPhysicsTrainer(
        model=model,
        device=torch.device("cpu"),
        adapter=BatchAdapter(params_key="light_params", mask_key="light_mask"),
        lr=5e-2,
        weight_decay=0.0,
        recon_loss="mse",
        temporal_weight=0.0,
        linearity_weight=0.0,
        spatial_weight=0.0,
        ema_enabled=True,
        ema_decay=0.95,
        ema_eval=True,
        ema_save_best=True,
        show_progress=False,
    )

    loader = torch.utils.data.DataLoader(_TinyDataset(), batch_size=4, shuffle=False)
    history = trainer.fit(
        train_loader=loader,
        val_loader=loader,
        num_epochs=2,
        output_dir=tmp_path,
        checkpoint_meta={"test": "ema"},
        save_best=True,
        best_metric="mae",
    )
    assert len(history["val"]["mae"]) == 2

    best_path = tmp_path / "best_model.pt"
    assert best_path.exists()
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    assert ckpt.get("saved_with_ema_weights") is True
    assert ckpt.get("ema_enabled") is True
    assert ckpt.get("ema_eval") is True
    assert ckpt.get("model_state_dict_raw") is not None
    assert ckpt.get("ema_state_dict") is not None



def test_routing_balance_metrics_with_soft_routing() -> None:
    model = _RoutingDummyModel()
    trainer = GaussianPhysicsTrainer(
        model=model,
        device=torch.device("cpu"),
        adapter=BatchAdapter(params_key="light_params", mask_key="light_mask"),
        lr=1e-3,
        weight_decay=0.0,
        recon_loss="mse",
        temporal_weight=0.0,
        linearity_weight=0.0,
        spatial_weight=0.0,
        lambda_routing_balance=0.02,
        routing_soft_train=True,
        routing_soft_topk=4,
        routing_temp_start=1.0,
        routing_temp_end=0.2,
        routing_temp_anneal_epochs=10,
        show_progress=False,
    )

    loader = torch.utils.data.DataLoader(_TinyDataset(), batch_size=4, shuffle=False)
    metrics = trainer.train_epoch(loader)

    assert "routing_balance" in metrics
    assert "routing_entropy" in metrics
    assert "routing_nonzero_ratio" in metrics
    assert "routing_temp" in metrics
    assert torch.isfinite(torch.tensor(metrics["routing_balance"]))
    assert 0.0 <= metrics["routing_nonzero_ratio"] <= 1.0


def test_routing_balance_uses_global_expert_indices() -> None:
    model = _RoutingDummyModel(num_gaussians=8)
    trainer = GaussianPhysicsTrainer(
        model=model,
        device=torch.device("cpu"),
        adapter=BatchAdapter(params_key="light_params", mask_key="light_mask"),
        lr=1e-3,
        weight_decay=0.0,
        recon_loss="mse",
        temporal_weight=0.0,
        linearity_weight=0.0,
        spatial_weight=0.0,
        lambda_routing_balance=1.0,
        show_progress=False,
    )

    # Slot usage is perfectly uniform (all weights are 0.5/0.5),
    # but expert usage is imbalanced because expert 0 appears in every sample.
    weights = torch.full((4, 2), 0.5, dtype=torch.float32)
    indices = torch.tensor(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
        ],
        dtype=torch.long,
    )
    loss, stats = trainer._compute_routing_balance_loss((weights, indices))
    assert float(loss.detach().item()) > 0.0
    assert stats["routing_nonzero_ratio"] < 1.0


def test_fit_saves_best_checkpoints_for_val_profiles(tmp_path: Path) -> None:
    model = _RoutingDummyModel()
    trainer = GaussianPhysicsTrainer(
        model=model,
        device=torch.device("cpu"),
        adapter=BatchAdapter(params_key="light_params", mask_key="light_mask"),
        lr=1e-3,
        weight_decay=0.0,
        recon_loss="mse",
        temporal_weight=0.0,
        linearity_weight=0.0,
        spatial_weight=0.0,
        show_progress=False,
    )

    loader = torch.utils.data.DataLoader(_TinyDataset(), batch_size=4, shuffle=False)
    callback_payloads: list[dict] = []

    trainer.fit(
        train_loader=loader,
        val_loader=loader,
        num_epochs=2,
        output_dir=tmp_path,
        checkpoint_meta={"test": "val_profiles"},
        save_best=True,
        best_metric="mae",
        val_profiles=[
            {"name": "hard3", "top_k": 3, "training_soft_routing": False},
            {"name": "soft8_t020", "top_k": 3, "training_soft_routing": True, "routing_soft_topk": 8, "routing_temperature": 0.2},
        ],
        save_best_profiles=True,
        best_metric_per_profile="mae",
        epoch_end_callback=lambda payload: callback_payloads.append(payload),
    )

    assert (tmp_path / "best_model.pt").exists()
    assert (tmp_path / "best_model_val_hard3.pt").exists()
    assert (tmp_path / "best_model_val_soft8_t020.pt").exists()
    assert callback_payloads
    last_payload = callback_payloads[-1]
    assert "val_profiles_metrics" in last_payload
    assert "best_value_by_profile" in last_payload
    assert "hard3" in last_payload["val_profiles_metrics"]
    assert "soft8_t020" in last_payload["val_profiles_metrics"]


def test_metric_direction_and_image_loss_warmup() -> None:
    model = _RoutingDummyModel()
    trainer = GaussianPhysicsTrainer(
        model=model,
        device=torch.device("cpu"),
        adapter=BatchAdapter(params_key="light_params", mask_key="light_mask"),
        lr=1e-3,
        weight_decay=0.0,
        recon_loss="mse",
        temporal_weight=0.0,
        linearity_weight=0.0,
        spatial_weight=0.0,
        image_loss_weight=0.2,
        image_loss_warmup_epochs=4,
        show_progress=False,
    )

    assert trainer._metric_higher_is_better("img_psnr") is True
    assert trainer._metric_higher_is_better("mae") is False
    assert trainer._metric_is_better("img_psnr", 11.0, 10.0) is True
    assert trainer._metric_is_better("mae", 0.4, 0.5) is True
    assert abs(trainer._image_loss_weight_for_epoch(1) - 0.05) < 1e-8
    assert abs(trainer._image_loss_weight_for_epoch(4) - 0.2) < 1e-8


def test_proxy_image_loss_mix_modes() -> None:
    model = _RoutingDummyModel()
    common_kwargs = dict(
        model=model,
        device=torch.device("cpu"),
        adapter=BatchAdapter(params_key="light_params", mask_key="light_mask"),
        lr=1e-3,
        weight_decay=0.0,
        recon_loss="mse",
        temporal_weight=0.0,
        linearity_weight=0.0,
        spatial_weight=0.0,
        image_loss_weight=1.0,
        show_progress=False,
    )
    pred = torch.ones((2, 27), dtype=torch.float32) * 0.3
    target = torch.ones((2, 27), dtype=torch.float32) * 0.15

    trainer_linear = GaussianPhysicsTrainer(
        **common_kwargs,
        proxy_image_loss_mix_mode="linear_only",
    )
    loss_linear, comp_linear = trainer_linear._compute_image_loss_components(pred, target)
    assert torch.allclose(loss_linear, comp_linear["linear"])
    assert torch.allclose(comp_linear["log"], torch.tensor(0.0))

    trainer_log = GaussianPhysicsTrainer(
        **common_kwargs,
        proxy_image_loss_mix_mode="log_only",
    )
    loss_log, comp_log = trainer_log._compute_image_loss_components(pred, target)
    assert torch.allclose(loss_log, comp_log["log"])
    assert torch.allclose(comp_log["linear"], torch.tensor(0.0))

    trainer_mix = GaussianPhysicsTrainer(
        **common_kwargs,
        proxy_image_loss_mix_mode="linear_log_mix",
        proxy_image_log_mix_weight=0.25,
    )
    loss_mix, comp_mix = trainer_mix._compute_image_loss_components(pred, target)
    expected_mix = 0.75 * comp_mix["linear"] + 0.25 * comp_mix["log"]
    assert torch.allclose(loss_mix, expected_mix, atol=1e-6)


def test_routing_balance_anneal_schedule_with_epoch_offset() -> None:
    model = _RoutingDummyModel()
    trainer = GaussianPhysicsTrainer(
        model=model,
        device=torch.device("cpu"),
        adapter=BatchAdapter(params_key="light_params", mask_key="light_mask"),
        lr=1e-3,
        weight_decay=0.0,
        recon_loss="mse",
        temporal_weight=0.0,
        linearity_weight=0.0,
        spatial_weight=0.0,
        lambda_routing_balance=0.02,
        routing_balance_anneal_enabled=True,
        routing_balance_anneal_start=0.02,
        routing_balance_anneal_end=0.0,
        routing_balance_anneal_epochs=10,
        routing_balance_anneal_epoch_offset=5,
        show_progress=False,
    )

    w_epoch1 = trainer._routing_balance_weight_for_epoch(1)
    w_epoch5 = trainer._routing_balance_weight_for_epoch(5)
    w_epoch6 = trainer._routing_balance_weight_for_epoch(6)
    assert abs(w_epoch1 - 0.01) < 1e-8
    assert abs(w_epoch5 - 0.002) < 1e-8
    assert abs(w_epoch6 - 0.0) < 1e-8
