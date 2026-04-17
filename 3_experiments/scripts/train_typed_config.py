"""Typed training config normalization with alias compatibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


def _as_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


@dataclass
class ExperimentConfig:
    variant: str | None = None
    output_dir: str | None = None
    device: str | None = None
    seed: int | None = None


@dataclass
class DataConfig:
    data_root: str | None = None
    manifest: str | None = None
    train_ratio: float | None = None
    val_ratio: float | None = None
    batch_size: int | None = None
    num_workers: int | None = None
    dataloader_timeout_seconds: int | None = None
    dataloader_prefetch_factor: int | None = None
    dataloader_persistent_workers: bool | None = None
    dataloader_multiprocessing_context: str | None = None


@dataclass
class ModelConfig:
    num_gaussians: int | None = None
    rank: int | None = None
    top_k: int | None = None
    light_dim: int | None = None
    embed_dim: int | None = None
    intensity_dim: int | None = None
    intensity_offset: int | None = None
    disable_film: bool | None = None
    light_encoder_mode: str | None = None
    bypass_feature_pairs: Any = None
    bypass_feature_norm_mean: Any = None
    bypass_feature_norm_std: Any = None


@dataclass
class TrainingConfig:
    heartbeat_enabled: bool | None = None
    heartbeat_history: bool | None = None
    epochs: int | None = None
    lr: float | None = None
    weight_decay: float | None = None
    lr_scheduler: str | None = None
    lr_min: float | None = None
    warmup_epochs: int | None = None
    recon_loss: str | None = None
    recon_weight: float | None = None
    charbonnier_eps: float | None = None
    lambda_temporal: float | None = None
    grad_clip: float | None = None
    lambda_linearity: float | None = None
    linearity_aug_pairs: int | None = None
    linearity_every_steps: int | None = None
    lambda_spatial: float | None = None
    spatial_k: int | None = None
    lambda_image: float | None = None
    image_loss_warmup_epochs: int | None = None
    lambda_routing_balance: float | None = None
    routing_soft_train: bool | None = None
    routing_soft_topk: int | None = None
    routing_temp_start: float | None = None
    routing_temp_end: float | None = None
    routing_temp_anneal_epochs: int | None = None
    train_routing_param_mode: str | None = None
    contraction_mode: str | None = None
    cuda_graph_train: bool | None = None
    cuda_graph_mode: str | None = None
    cuda_graph_warmup_steps: int | None = None
    cuda_graph_fallback_eager: bool | None = None
    image_loss_type: str | None = None
    image_samples: int | None = None
    image_sample_seed: int | None = None
    image_loss_space: str | None = None
    enable_weighted_sh_loss: bool | None = None
    sh_loss_weights: Any = None
    sh_weight_mode: str | None = None
    val_image_metrics: bool | None = None
    val_superposition: bool | None = None
    enable_sh_scaler: bool | None = None
    sh_scaler_path: str | None = None
    sh_scaler_max_samples: int | None = None
    no_init: bool | None = None
    load_model: str | None = None
    resume: str | None = None
    resume_save_every: int | None = None
    resume_checkpoint_name: str | None = None
    enable_rerun: bool | None = None
    rerun_log_freq: int | None = None
    rerun_save_path: str | None = None
    show_progress: bool | None = None


@dataclass
class EvaluationConfig:
    output_dir: str | None = None
    checkpoint: str | None = None


@dataclass
class TypedTrainConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    alias_hits: List[str] = field(default_factory=list)

    def to_arg_mapping(self) -> Dict[str, Any]:
        tr = self.training
        return {
            "variant": self.experiment.variant,
            "output_dir": self.experiment.output_dir,
            "device": self.experiment.device,
            "seed": self.experiment.seed,
            "data_root": self.data.data_root,
            "manifest": self.data.manifest,
            "train_ratio": self.data.train_ratio,
            "val_ratio": self.data.val_ratio,
            "batch_size": self.data.batch_size,
            "num_workers": self.data.num_workers,
            "dataloader_timeout_seconds": self.data.dataloader_timeout_seconds,
            "dataloader_prefetch_factor": self.data.dataloader_prefetch_factor,
            "dataloader_persistent_workers": self.data.dataloader_persistent_workers,
            "dataloader_multiprocessing_context": self.data.dataloader_multiprocessing_context,
            "heartbeat_enabled": tr.heartbeat_enabled,
            "heartbeat_history": tr.heartbeat_history,
            "num_gaussians": self.model.num_gaussians,
            "rank": self.model.rank,
            "top_k": self.model.top_k,
            "light_dim": self.model.light_dim,
            "embed_dim": self.model.embed_dim,
            "intensity_dim": self.model.intensity_dim,
            "intensity_offset": self.model.intensity_offset,
            "disable_film": self.model.disable_film,
            "light_encoder_mode": self.model.light_encoder_mode,
            "bypass_feature_pairs": self.model.bypass_feature_pairs,
            "bypass_feature_norm_mean": self.model.bypass_feature_norm_mean,
            "bypass_feature_norm_std": self.model.bypass_feature_norm_std,
            "epochs": tr.epochs,
            "lr": tr.lr,
            "weight_decay": tr.weight_decay,
            "lr_scheduler": tr.lr_scheduler,
            "lr_min": tr.lr_min,
            "warmup_epochs": tr.warmup_epochs,
            "recon_loss": tr.recon_loss,
            "recon_weight": tr.recon_weight,
            "charbonnier_eps": tr.charbonnier_eps,
            "lambda_temporal": tr.lambda_temporal,
            "grad_clip": tr.grad_clip,
            "lambda_linearity": tr.lambda_linearity,
            "linearity_aug_pairs": tr.linearity_aug_pairs,
            "linearity_every_steps": tr.linearity_every_steps,
            "lambda_spatial": tr.lambda_spatial,
            "spatial_k": tr.spatial_k,
            "lambda_image": tr.lambda_image,
            "image_loss_warmup_epochs": tr.image_loss_warmup_epochs,
            "lambda_routing_balance": tr.lambda_routing_balance,
            "routing_soft_train": tr.routing_soft_train,
            "routing_soft_topk": tr.routing_soft_topk,
            "routing_temp_start": tr.routing_temp_start,
            "routing_temp_end": tr.routing_temp_end,
            "routing_temp_anneal_epochs": tr.routing_temp_anneal_epochs,
            "train_routing_param_mode": tr.train_routing_param_mode,
            "contraction_mode": tr.contraction_mode,
            "cuda_graph_train": tr.cuda_graph_train,
            "cuda_graph_mode": tr.cuda_graph_mode,
            "cuda_graph_warmup_steps": tr.cuda_graph_warmup_steps,
            "cuda_graph_fallback_eager": tr.cuda_graph_fallback_eager,
            "image_loss_type": tr.image_loss_type,
            "image_samples": tr.image_samples,
            "image_sample_seed": tr.image_sample_seed,
            "image_loss_space": tr.image_loss_space,
            "enable_weighted_sh_loss": tr.enable_weighted_sh_loss,
            "sh_loss_weights": tr.sh_loss_weights,
            "sh_weight_mode": tr.sh_weight_mode,
            "val_image_metrics": tr.val_image_metrics,
            "val_superposition": tr.val_superposition,
            "enable_sh_scaler": tr.enable_sh_scaler,
            "sh_scaler_path": tr.sh_scaler_path,
            "sh_scaler_max_samples": tr.sh_scaler_max_samples,
            "no_init": tr.no_init,
            "load_model": tr.load_model,
            "resume": tr.resume,
            "resume_save_every": tr.resume_save_every,
            "resume_checkpoint_name": tr.resume_checkpoint_name,
            "enable_rerun": tr.enable_rerun,
            "rerun_log_freq": tr.rerun_log_freq,
            "rerun_save_path": tr.rerun_save_path,
            "show_progress": tr.show_progress,
            "eval_output_dir": self.evaluation.output_dir,
            "eval_checkpoint": self.evaluation.checkpoint,
        }


def parse_typed_train_config(cfg: Dict[str, Any]) -> TypedTrainConfig:
    root = _as_mapping(cfg)
    experiment = _as_mapping(root.get("experiment"))
    data = _as_mapping(root.get("data"))
    model = _as_mapping(root.get("model"))
    training = _as_mapping(root.get("training"))
    loss_weights = _as_mapping(training.get("loss_weights"))
    evaluation = _as_mapping(root.get("evaluation"))
    eval_alias = _as_mapping(root.get("eval"))
    alias_hits: List[str] = []

    # Backward-compatible aliases (legacy keys continue to work but are reported).
    if "dataset_path" in data and "data_root" not in data:
        data = dict(data)
        data["data_root"] = data.get("dataset_path")
        alias_hits.append("data.dataset_path -> data.data_root")
    if "num_epochs" in training and "epochs" not in training:
        training = dict(training)
        training["epochs"] = training.get("num_epochs")
        alias_hits.append("training.num_epochs -> training.epochs")
    if eval_alias and not evaluation:
        evaluation = dict(eval_alias)
        alias_hits.append("eval -> evaluation")
    if "out_dir" in evaluation and "output_dir" not in evaluation:
        evaluation = dict(evaluation)
        evaluation["output_dir"] = evaluation.get("out_dir")
        alias_hits.append("evaluation.out_dir -> evaluation.output_dir")
    if "rerun" in training and "enable_rerun" not in training:
        training = dict(training)
        training["enable_rerun"] = training.get("rerun")
        alias_hits.append("training.rerun -> training.enable_rerun")
    if "progress" in training and "show_progress" not in training:
        training = dict(training)
        training["show_progress"] = training.get("progress")
        alias_hits.append("training.progress -> training.show_progress")

    def _loss_weight_or(field: str, fallback: Any) -> Any:
        if fallback is not None:
            return fallback
        return loss_weights.get(field)

    # Top-level fallback aliases.
    if "variant" in root and "variant" not in experiment:
        experiment = dict(experiment)
        experiment["variant"] = root.get("variant")
        alias_hits.append("variant -> experiment.variant")
    if "output_dir" in root and "output_dir" not in experiment:
        experiment = dict(experiment)
        experiment["output_dir"] = root.get("output_dir")
        alias_hits.append("output_dir -> experiment.output_dir")
    if "device" in root and "device" not in experiment:
        experiment = dict(experiment)
        experiment["device"] = root.get("device")
        alias_hits.append("device -> experiment.device")
    if "seed" in root and "seed" not in experiment:
        experiment = dict(experiment)
        experiment["seed"] = root.get("seed")
        alias_hits.append("seed -> experiment.seed")
    if "data_root" in root and "data_root" not in data:
        data = dict(data)
        data["data_root"] = root.get("data_root")
        alias_hits.append("data_root -> data.data_root")
    if "manifest" in root and "manifest" not in data:
        data = dict(data)
        data["manifest"] = root.get("manifest")
        alias_hits.append("manifest -> data.manifest")

    tr = TrainingConfig(
        heartbeat_enabled=training.get("heartbeat_enabled"),
        heartbeat_history=training.get("heartbeat_history"),
        epochs=training.get("epochs"),
        lr=training.get("lr"),
        weight_decay=training.get("weight_decay"),
        lr_scheduler=training.get("lr_scheduler"),
        lr_min=training.get("lr_min"),
        warmup_epochs=training.get("warmup_epochs"),
        recon_loss=training.get("recon_loss"),
        recon_weight=_loss_weight_or("recon", training.get("recon_weight")),
        charbonnier_eps=training.get("charbonnier_eps"),
        lambda_temporal=_loss_weight_or("temporal", training.get("lambda_temporal")),
        grad_clip=training.get("grad_clip"),
        lambda_linearity=_loss_weight_or("linearity", training.get("lambda_linearity")),
        linearity_aug_pairs=training.get("linearity_aug_pairs"),
        linearity_every_steps=training.get("linearity_every_steps"),
        lambda_spatial=training.get("lambda_spatial"),
        spatial_k=training.get("spatial_k"),
        lambda_image=_loss_weight_or("image", training.get("lambda_image")),
        image_loss_warmup_epochs=training.get("image_loss_warmup_epochs"),
        lambda_routing_balance=_loss_weight_or("routing_balance", training.get("lambda_routing_balance")),
        routing_soft_train=training.get("routing_soft_train"),
        routing_soft_topk=training.get("routing_soft_topk"),
        routing_temp_start=training.get("routing_temp_start"),
        routing_temp_end=training.get("routing_temp_end"),
        routing_temp_anneal_epochs=training.get("routing_temp_anneal_epochs"),
        train_routing_param_mode=training.get("train_routing_param_mode"),
        contraction_mode=training.get("contraction_mode"),
        cuda_graph_train=training.get("cuda_graph_train"),
        cuda_graph_mode=training.get("cuda_graph_mode"),
        cuda_graph_warmup_steps=training.get("cuda_graph_warmup_steps"),
        cuda_graph_fallback_eager=training.get("cuda_graph_fallback_eager"),
        image_loss_type=training.get("image_loss_type"),
        image_samples=training.get("image_samples"),
        image_sample_seed=training.get("image_sample_seed"),
        image_loss_space=training.get("image_loss_space"),
        enable_weighted_sh_loss=training.get("enable_weighted_sh_loss"),
        sh_loss_weights=training.get("sh_loss_weights"),
        sh_weight_mode=training.get("sh_weight_mode"),
        val_image_metrics=training.get("val_image_metrics"),
        val_superposition=training.get("val_superposition"),
        enable_sh_scaler=training.get("enable_sh_scaler"),
        sh_scaler_path=training.get("sh_scaler_path"),
        sh_scaler_max_samples=training.get("sh_scaler_max_samples"),
        no_init=training.get("no_init"),
        load_model=training.get("load_model"),
        resume=training.get("resume"),
        resume_save_every=training.get("resume_save_every"),
        resume_checkpoint_name=training.get("resume_checkpoint_name"),
        enable_rerun=training.get("enable_rerun"),
        rerun_log_freq=training.get("rerun_log_freq"),
        rerun_save_path=training.get("rerun_save_path"),
        show_progress=training.get("show_progress"),
    )

    return TypedTrainConfig(
        experiment=ExperimentConfig(
            variant=experiment.get("variant"),
            output_dir=experiment.get("output_dir"),
            device=experiment.get("device"),
            seed=experiment.get("seed"),
        ),
        data=DataConfig(
            data_root=data.get("data_root"),
            manifest=data.get("manifest"),
            train_ratio=data.get("train_ratio"),
            val_ratio=data.get("val_ratio"),
            batch_size=data.get("batch_size"),
            num_workers=data.get("num_workers"),
            dataloader_timeout_seconds=training.get(
                "dataloader_timeout_seconds",
                data.get("dataloader_timeout_seconds"),
            ),
            dataloader_prefetch_factor=training.get(
                "dataloader_prefetch_factor",
                data.get("dataloader_prefetch_factor"),
            ),
            dataloader_persistent_workers=training.get(
                "dataloader_persistent_workers",
                data.get("dataloader_persistent_workers"),
            ),
            dataloader_multiprocessing_context=training.get(
                "dataloader_multiprocessing_context",
                data.get("dataloader_multiprocessing_context"),
            ),
        ),
        model=ModelConfig(
            num_gaussians=model.get("num_gaussians"),
            rank=model.get("rank"),
            top_k=model.get("top_k"),
            light_dim=model.get("light_dim"),
            embed_dim=model.get("embed_dim"),
            intensity_dim=model.get("intensity_dim"),
            intensity_offset=model.get("intensity_offset"),
            disable_film=model.get("disable_film"),
            light_encoder_mode=model.get("light_encoder_mode"),
            bypass_feature_pairs=model.get("bypass_feature_pairs"),
            bypass_feature_norm_mean=model.get("bypass_feature_norm_mean"),
            bypass_feature_norm_std=model.get("bypass_feature_norm_std"),
        ),
        training=tr,
        evaluation=EvaluationConfig(
            output_dir=evaluation.get("output_dir"),
            checkpoint=evaluation.get("checkpoint"),
        ),
        alias_hits=alias_hits,
    )


def dedupe_alias_hits(alias_hits: List[str]) -> Tuple[str, ...]:
    seen = set()
    out: List[str] = []
    for item in alias_hits:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)
