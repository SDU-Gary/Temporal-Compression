"""
数据增强模块 - Phase 1 Stage 3

提供物理线性保持的增强策略:
1. 光照强度抖动 (RGB × [0.9, 1.1])
2. 探针位置噪声 (Gaussian noise, σ=0.02)
3. 光源随机丢弃 (dropout 10-20%)
"""

import torch
from typing import Tuple, Optional


def augment_intensity(
    light_params: torch.Tensor,
    training: bool = True,
    scale_range: Tuple[float, float] = (0.9, 1.1)
) -> torch.Tensor:
    """
    光照强度抖动 (RGB × uniform[scale_range])

    Args:
        light_params: [N, 12] light descriptors (intensity at dims 1-3)
        training: 是否在训练模式
        scale_range: 强度缩放范围

    Returns:
        增强后的 light_params
    """
    if not training:
        return light_params

    scale = torch.empty(1, device=light_params.device).uniform_(*scale_range)
    light_params = light_params.clone()
    light_params[:, 1:4] *= scale  # RGB channels (dims 1-3)
    return light_params


def augment_position(
    probe_pos: torch.Tensor,
    training: bool = True,
    std: float = 0.02
) -> torch.Tensor:
    """
    探针位置噪声 (Gaussian noise)

    Args:
        probe_pos: [3] probe position
        training: 是否在训练模式
        std: 噪声标准差

    Returns:
        增强后的 probe_pos
    """
    if not training:
        return probe_pos

    noise = torch.randn(3, device=probe_pos.device) * std
    return probe_pos + noise


def augment_dropout(
    light_params: torch.Tensor,
    light_mask: torch.Tensor,
    training: bool = True,
    p: float = 0.15
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    光源随机丢弃

    Args:
        light_params: [N, 12] light descriptors
        light_mask: [N] light validity mask
        training: 是否在训练模式
        p: dropout 概率

    Returns:
        (light_params, updated_mask)
    """
    if not training:
        return light_params, light_mask

    dropout_mask = torch.rand(light_params.shape[0], device=light_params.device) > p
    return light_params, light_mask & dropout_mask


def create_augmentation_transform(config: dict):
    """
    根据配置创建增强函数

    Args:
        config: training 配置字典

    Returns:
        增强函数或 None
    """
    augment_intensity_enabled = config.get('augment_intensity', False)
    augment_position_enabled = config.get('augment_position', False)
    augment_dropout_p = config.get('augment_dropout', 0.0)

    if not any([augment_intensity_enabled, augment_position_enabled, augment_dropout_p > 0]):
        return None

    def transform(light_params: torch.Tensor, training: bool = True) -> torch.Tensor:
        if augment_intensity_enabled:
            light_params = augment_intensity(light_params, training=training)
        return light_params

    return transform
