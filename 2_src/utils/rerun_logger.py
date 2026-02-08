"""Rerun logging utilities for Gaussian-Physics training visualization.

This module provides a unified interface for logging training metrics, 3D spatial
distributions, and SH coefficient reconstructions using Rerun.
"""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Dict, Optional, Tuple


def _lazy_imports():
    import numpy as np
    import torch
    from torch.utils.data import Dataset

    return np, torch, Dataset


# Bind commonly used deps at module scope so helper functions (defined below)
# can use them without NameError during training-time visualization.
np, torch, Dataset = _lazy_imports()

try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False
    print("Warning: rerun-sdk not installed. Install with: pip install rerun-sdk")


class RerunLogger:
    """Rerun logger for Gaussian-Physics training visualization.

    Logs training metrics, 3D spatial distributions, and SH coefficient comparisons.
    """

    def __init__(
        self,
        app_id: str,
        spawn: bool = True,
        save_path: Optional[Path] = None,
        log_frequency: int = 10
    ):
        """Initialize Rerun logger.

        Args:
            app_id: Application identifier for Rerun
            spawn: Whether to auto-spawn viewer window
            save_path: Optional path to save .rrd file
            log_frequency: Log visualizations every N epochs
        """
        if not RERUN_AVAILABLE:
            raise ImportError("rerun-sdk is required. Install with: pip install rerun-sdk")

        np, torch, Dataset = _lazy_imports()

        self.app_id = app_id
        self.log_frequency = log_frequency
        self.save_path = save_path

        # Rerun uses a global recording; guard rr.* calls since we may log from
        # both the training loop and an optional background rendering thread.
        self._rr_lock = threading.Lock()

        # Handle for expensive rendered comparisons (Falcor subprocess).
        self._render_thread: Optional[threading.Thread] = None

        # Initialize Rerun
        rr.init(app_id, spawn=spawn)

        # If save_path provided, also save to file
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            rr.save(str(save_path))

        # Default blueprint: prioritize rendering views so they are visible by default.
        try:
            from rerun import blueprint as bp

            rendering_tabs = bp.Tabs(
                bp.Spatial2DView(name="rendering/best", contents="/rendering/best/**"),
                bp.Spatial2DView(name="rendering/worst", contents="/rendering/worst/**"),
                bp.Spatial2DView(name="rendering/all", contents="/rendering/**"),
                name="rendering",
                active_tab="rendering/best",
            )

            blueprint = bp.Blueprint(
                rendering_tabs,
                auto_views=True,
                auto_layout=True,
            )
            rr.send_blueprint(blueprint, make_active=True, make_default=True)
        except Exception as e:
            print(f"Warning: Could not set Rerun blueprint: {e}")

        print(f"RerunLogger initialized: app_id='{app_id}', spawn={spawn}, log_freq={log_frequency}")

    def set_time(self, timeline: str, value: int):
        """Set timeline value for subsequent logs.

        Args:
            timeline: Timeline name (e.g., "epoch", "step")
            value: Timeline value (sequence number)
        """
        with self._rr_lock:
            rr.set_time(timeline, sequence=value)

    def log_scalars(self, epoch: int, metrics: Dict[str, float], prefix: str):
        """Log scalar metrics (losses, MAE, RMSE).

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metric names to values
            prefix: Prefix for entity path (e.g., "train", "val")
        """
        self.set_time("epoch", epoch)

        with self._rr_lock:
            for name, value in metrics.items():
                entity_path = f"{prefix}/{name}"
                rr.log(entity_path, rr.Scalars(float(value)))

    def log_model_metadata(
        self,
        num_gaussians: int,
        rank: int,
        num_params: int,
        compression_ratio: Optional[float] = None
    ):
        """Log model architecture metadata.

        Args:
            num_gaussians: Number of Gaussians (K)
            rank: Low-rank factorization rank
            num_params: Total parameter count
            compression_ratio: Optional compression ratio
        """
        metadata_text = f"""Model Architecture:
- Gaussians (K): {num_gaussians}
- Rank (r): {rank}
- Total Parameters: {num_params:,}
"""
        if compression_ratio is not None:
            metadata_text += f"- Compression Ratio: {compression_ratio:.2f}×\n"

        with self._rr_lock:
            rr.log("model/metadata", rr.TextDocument(metadata_text))

    def log_gaussian_centers(self, epoch: int, centers: np.ndarray):
        """Log 3D Gaussian center positions.

        Args:
            epoch: Current epoch number
            centers: Gaussian centers [K, 3]
        """
        self.set_time("epoch", epoch)

        K = centers.shape[0]
        with self._rr_lock:
            rr.log("model/gaussian_centers", rr.Points3D(
                positions=centers,
                colors=[[255, 200, 0]] * K,  # Gold color
                radii=0.05,
                labels=[f"G{i}" for i in range(K)]
            ))

    def log_probe_errors(
        self,
        epoch: int,
        positions: np.ndarray,
        errors: np.ndarray,
        split: str = "val"
    ):
        """Log probe positions colored by prediction error.

        Args:
            epoch: Current epoch number
            positions: Probe positions [P, 3]
            errors: Per-probe errors [P]
            split: Dataset split name ("train", "val", "test")
        """
        self.set_time("epoch", epoch)

        # Convert errors to colors (blue=low, red=high)
        colors = error_to_colormap(errors)

        with self._rr_lock:
            rr.log(f"{split}/probe_errors", rr.Points3D(
                positions=positions,
                colors=colors,
                radii=0.03
            ))

        # Log error statistics
        stats_text = f"""Error Statistics (Epoch {epoch}):
- Mean: {errors.mean():.6f}
- Std: {errors.std():.6f}
- Min: {errors.min():.6f}
- Max: {errors.max():.6f}
- Median: {np.median(errors):.6f}
"""
        with self._rr_lock:
            rr.log(f"{split}/error_stats", rr.TextDocument(stats_text))

    def log_gaussian_coverage(
        self,
        epoch: int,
        centers: np.ndarray,
        scales: np.ndarray,
        probe_positions: np.ndarray
    ):
        """Log Gaussian coverage analysis.

        Args:
            epoch: Current epoch number
            centers: Gaussian centers [K, 3]
            scales: Gaussian scales [K, 3]
            probe_positions: Probe positions [P, 3]
        """
        self.set_time("epoch", epoch)

        # Compute nearest Gaussian distance for each probe
        nearest_distances = compute_nearest_gaussian_distances(centers, probe_positions)

        # Log coverage spheres (3-sigma range)
        with self._rr_lock:
            for i, (center, scale) in enumerate(zip(centers, scales)):
                rr.log(f"analysis/gaussian_coverage/sphere_{i}", rr.Ellipsoids3D(
                    centers=[center],
                    half_sizes=[scale * 3],  # 3-sigma range
                    colors=[[255, 255, 0, 64]]  # Semi-transparent yellow
                ))

        # Log probes colored by distance to nearest Gaussian
        colors = distance_to_colormap(nearest_distances)
        with self._rr_lock:
            rr.log("analysis/gaussian_coverage/probes", rr.Points3D(
                positions=probe_positions,
                colors=colors,
                radii=0.02
            ))

        # Compute and log coverage statistics
        stats = compute_gaussian_coverage_stats(centers, scales, probe_positions)
        stats_text = f"""Gaussian Coverage Statistics (Epoch {epoch}):
- Avg distance to nearest Gaussian: {stats['avg_distance_to_nearest']:.4f}
- Coverage radius (mean): {stats['coverage_radius_mean']:.4f}
- Coverage radius (std): {stats['coverage_radius_std']:.4f}
- Probes within 1σ: {stats['probes_within_1sigma']:.1f}%
- Probes within 3σ: {stats['probes_within_3sigma']:.1f}%
"""
        with self._rr_lock:
            rr.log("analysis/gaussian_coverage/stats", rr.TextDocument(stats_text))

    def log_sh_comparison(
        self,
        epoch: int,
        sh_gt: np.ndarray,
        sh_pred: np.ndarray,
        sample_name: str
    ):
        """Log SH coefficient comparison as image.

        Args:
            epoch: Current epoch number
            sh_gt: Ground truth SH coefficients [27]
            sh_pred: Predicted SH coefficients [27]
            sample_name: Sample identifier (e.g., "best", "worst")
        """
        self.set_time("epoch", epoch)

        # Reshape to [3 RGB, 9 bases]
        sh_gt_img = sh_gt.reshape(3, 9)
        sh_pred_img = sh_pred.reshape(3, 9)
        error_img = np.abs(sh_gt_img - sh_pred_img)

        # Concatenate horizontally: GT | Pred | Error
        comparison = np.hstack([sh_gt_img, sh_pred_img, error_img])

        # Convert to RGB image with colormap
        comparison_rgb = colormap_tensor(comparison)

        with self._rr_lock:
            rr.log(f"validation/sh_comparison/{sample_name}", rr.Image(comparison_rgb))

        # Log numerical comparison
        mae = np.abs(sh_gt - sh_pred).mean()
        rmse = np.sqrt(((sh_gt - sh_pred) ** 2).mean())
        stats_text = f"""SH Comparison ({sample_name}, Epoch {epoch}):
- MAE: {mae:.6f}
- RMSE: {rmse:.6f}
- Max Error: {np.abs(sh_gt - sh_pred).max():.6f}
"""
        with self._rr_lock:
            rr.log(f"validation/sh_comparison/{sample_name}_stats", rr.TextDocument(stats_text))

    def log_rendered_comparison(
        self,
        epoch: int,
        sh_gt: np.ndarray,
        sh_pred: np.ndarray,
        sample_name: str,
        scene_path: Optional[str] = None,
        temp_dir: str = '/tmp/rerun_rendering'
    ):
        """Log rendered image comparison (GT vs Pred vs Error) using Falcor.

        Args:
            epoch: Current epoch number
            sh_gt: Ground truth SH coefficients [27]
            sh_pred: Predicted SH coefficients [27]
            sample_name: Sample identifier (e.g., "best", "worst")
            scene_path: Optional Falcor scene (.pyscene) path
            temp_dir: Temporary directory for environment maps
        """
        # Avoid blocking the training loop: run the expensive Falcor rendering in
        # a background thread. If a previous render is still running, skip.
        # Set RERUN_RENDER_SYNC=1 for synchronous behavior (debugging).
        if self._render_thread is not None and self._render_thread.is_alive():
            return

        if os.environ.get("RERUN_RENDER_SYNC", "0") == "1":
            self._render_and_log_comparison(
                epoch,
                sh_gt,
                sh_pred,
                sample_name,
                scene_path=scene_path,
                temp_dir=temp_dir,
            )
            return

        args = (
            int(epoch),
            np.asarray(sh_gt, dtype=np.float32).copy(),
            np.asarray(sh_pred, dtype=np.float32).copy(),
            str(sample_name),
            scene_path,
            str(temp_dir),
        )
        self._render_thread = threading.Thread(
            target=self._render_and_log_comparison,
            args=args,
            daemon=True,
            name="rerun_rendered_comparison",
        )
        self._render_thread.start()

    def _render_and_log_comparison(
        self,
        epoch: int,
        sh_gt: np.ndarray,
        sh_pred: np.ndarray,
        sample_name: str,
        scene_path: Optional[str],
        temp_dir: str,
    ) -> None:
        from utils.rendering_utils import (
            sh_to_envmap,
            render_with_envmap_external,
            compute_image_metrics,
            create_error_heatmap,
            save_exr,
            save_hdr,
            cleanup_temp_files,
            openexr_available,
            compute_auto_exposure,
            tone_map_reinhard,
            srgb_encode,
        )
        import os

        try:
            spp = int(os.environ.get("RERUN_RENDER_SPP", "32"))
            use_exr = openexr_available()
            if not use_exr:
                self._warned_no_openexr = True

            envmap_gt = sh_to_envmap(sh_gt, H=128, W=256)
            envmap_pred = sh_to_envmap(sh_pred, H=128, W=256)

            if os.environ.get("RERUN_AUTO_EXPOSURE", "1") != "0":
                pct = float(os.environ.get("RERUN_EXPOSURE_PERCENTILE", "95"))
                target = float(os.environ.get("RERUN_EXPOSURE_TARGET", "0.6"))
                min_exp = float(os.environ.get("RERUN_EXPOSURE_MIN", "0.05"))
                max_exp = float(os.environ.get("RERUN_EXPOSURE_MAX", "50.0"))
                exposure = compute_auto_exposure(
                    envmap_gt,
                    percentile=pct,
                    target=target,
                    min_exposure=min_exp,
                    max_exposure=max_exp,
                )
                envmap_gt = envmap_gt * exposure
                envmap_pred = envmap_pred * exposure

            os.makedirs(temp_dir, exist_ok=True)
            ext = "exr" if use_exr else "hdr"
            envmap_gt_path = f"{temp_dir}/gt_{epoch}_{sample_name}.{ext}"
            envmap_pred_path = f"{temp_dir}/pred_{epoch}_{sample_name}.{ext}"
            if use_exr:
                save_exr(envmap_gt_path, envmap_gt)
                save_exr(envmap_pred_path, envmap_pred)
            else:
                save_hdr(envmap_gt_path, envmap_gt)
                save_hdr(envmap_pred_path, envmap_pred)

            render_gt = render_with_envmap_external(
                envmap_gt_path,
                scene_path=scene_path,
                spp=spp,
                resolution=(256, 256),
                output_pass="AccumulatePass.output",
                enable_tonemapper=False,
                quiet=True,
            )
            render_pred = render_with_envmap_external(
                envmap_pred_path,
                scene_path=scene_path,
                spp=spp,
                resolution=(256, 256),
                output_pass="AccumulatePass.output",
                enable_tonemapper=False,
                quiet=True,
            )

            if os.environ.get("RERUN_TONEMAP", "1") != "0":
                render_gt = tone_map_reinhard(render_gt)
                render_pred = tone_map_reinhard(render_pred)
            if os.environ.get("RERUN_SRGB", "1") != "0":
                render_gt = srgb_encode(render_gt)
                render_pred = srgb_encode(render_pred)

            metrics = compute_image_metrics(render_gt, render_pred)
            error_heatmap = create_error_heatmap(render_gt, render_pred)

            with self._rr_lock:
                rr.set_time("epoch", sequence=epoch)
                rr.log(f"rendering/{sample_name}/gt", rr.Image(render_gt))
                rr.log(f"rendering/{sample_name}/pred", rr.Image(render_pred))
                rr.log(f"rendering/{sample_name}/error", rr.Image(error_heatmap))
                rr.log(
                    f"rendering_metrics/{sample_name}/psnr",
                    rr.Scalars(float(metrics["psnr"])),
                )
                rr.log(
                    f"rendering_metrics/{sample_name}/ssim",
                    rr.Scalars(float(metrics["ssim"])),
                )

                stats_text = (
                    f"Rendering Quality ({sample_name}, Epoch {epoch}):\n"
                    f"- PSNR: {metrics['psnr']:.2f} dB\n"
                    f"- SSIM: {metrics['ssim']:.4f}\n"
                )
                rr.log(f"rendering/{sample_name}/stats", rr.TextDocument(stats_text))

            cleanup_temp_files(temp_dir)

        except Exception:
            try:
                envmap_gt = sh_to_envmap(sh_gt, H=128, W=256)
                envmap_pred = sh_to_envmap(sh_pred, H=128, W=256)

                H, W = envmap_gt.shape[:2]
                center_gt = envmap_gt[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4]
                center_pred = envmap_pred[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4]

                center_gt_norm = np.clip(center_gt / (center_gt.max() + 1e-8), 0, 1)
                center_pred_norm = np.clip(center_pred / (center_pred.max() + 1e-8), 0, 1)

                with self._rr_lock:
                    rr.set_time("epoch", sequence=epoch)
                    rr.log(
                        f"rendering/{sample_name}/envmap_gt_fallback",
                        rr.Image(center_gt_norm),
                    )
                    rr.log(
                        f"rendering/{sample_name}/envmap_pred_fallback",
                        rr.Image(center_pred_norm),
                    )
            except Exception:
                pass


# ============================================================================
# Helper Functions
# ============================================================================

def error_to_colormap(errors: np.ndarray) -> np.ndarray:
    """Convert error values to RGB colors (blue=low, red=high).

    Args:
        errors: Error values [N]

    Returns:
        colors: RGB colors [N, 3] in range [0, 255]
    """
    # Normalize errors to [0, 1]
    errors_norm = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)

    # Blue (low) to Red (high) colormap
    colors = np.zeros((len(errors), 3), dtype=np.uint8)
    colors[:, 0] = (errors_norm * 255).astype(np.uint8)  # Red channel
    colors[:, 2] = ((1 - errors_norm) * 255).astype(np.uint8)  # Blue channel

    return colors


def distance_to_colormap(distances: np.ndarray) -> np.ndarray:
    """Convert distance values to RGB colors (green=close, red=far).

    Args:
        distances: Distance values [N]

    Returns:
        colors: RGB colors [N, 3] in range [0, 255]
    """
    # Normalize distances to [0, 1]
    distances_norm = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)

    # Green (close) to Red (far) colormap
    colors = np.zeros((len(distances), 3), dtype=np.uint8)
    colors[:, 0] = (distances_norm * 255).astype(np.uint8)  # Red channel
    colors[:, 1] = ((1 - distances_norm) * 255).astype(np.uint8)  # Green channel

    return colors


def colormap_tensor(tensor: np.ndarray) -> np.ndarray:
    """Convert 2D tensor to RGB image using colormap.

    Args:
        tensor: 2D array [H, W]

    Returns:
        rgb_image: RGB image [H, W, 3] in range [0, 255]
    """
    # Normalize to [0, 1]
    tensor_norm = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

    # Apply colormap (viridis-like: blue -> green -> yellow)
    rgb = np.zeros((*tensor.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (tensor_norm * 255).astype(np.uint8)  # Red
    rgb[..., 1] = (tensor_norm * 200).astype(np.uint8)  # Green
    rgb[..., 2] = ((1 - tensor_norm) * 255).astype(np.uint8)  # Blue

    return rgb


def compute_per_probe_errors(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    adapter,
    max_samples: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-probe average errors for coloring.

    Args:
        model: Trained model
        dataset: Validation dataset
        device: Torch device
        adapter: BatchAdapter for unpacking batches
        max_samples: Maximum samples to evaluate per probe

    Returns:
        positions: Probe positions [P, 3]
        errors: Average MAE per probe [P]
    """
    model.eval()

    # Get probe positions
    if hasattr(dataset, 'get_full_probe_positions'):
        positions = dataset.get_full_probe_positions(normalized=True)
    elif hasattr(dataset, 'probe_positions_norm'):
        positions = dataset.probe_positions_norm
    elif hasattr(dataset, 'probe_positions'):
        positions = dataset.probe_positions
    else:
        raise AttributeError("Dataset does not have probe positions attribute")

    # Initialize error accumulator
    probe_errors = {}
    probe_counts = {}

    # Iterate through dataset
    with torch.no_grad():
        for i, batch in enumerate(dataset):
            if i >= max_samples:
                break

            # Get probe index
            probe_idx = batch['probe_idx'].item() if 'probe_idx' in batch else i

            # Unpack batch (add batch dimension)
            batch_expanded = {k: v.unsqueeze(0) for k, v in batch.items()}
            unpacked = adapter.unpack(batch_expanded, device)
            if len(unpacked) == 4:
                pos, params, target, mask = unpacked
            else:
                pos, params, target = unpacked
                mask = None

            # Forward pass
            try:
                pred = model(pos, params, light_mask=mask)
            except TypeError:
                pred = model(pos, params)

            # Compute error
            error = torch.mean(torch.abs(pred - target)).item()

            # Accumulate
            if probe_idx not in probe_errors:
                probe_errors[probe_idx] = 0.0
                probe_counts[probe_idx] = 0
            probe_errors[probe_idx] += error
            probe_counts[probe_idx] += 1

    # Average errors per probe
    P = len(positions)
    errors = np.zeros(P)
    for probe_idx in range(P):
        if probe_idx in probe_errors:
            errors[probe_idx] = probe_errors[probe_idx] / probe_counts[probe_idx]
        else:
            errors[probe_idx] = 0.0  # No samples for this probe

    return positions, errors


def compute_nearest_gaussian_distances(
    centers: np.ndarray,
    probe_positions: np.ndarray
) -> np.ndarray:
    """Compute distance from each probe to nearest Gaussian.

    Args:
        centers: Gaussian centers [K, 3]
        probe_positions: Probe positions [P, 3]

    Returns:
        distances: Distance to nearest Gaussian [P]
    """
    # Compute pairwise distances [P, K]
    distances_matrix = np.linalg.norm(
        probe_positions[:, np.newaxis, :] - centers[np.newaxis, :, :],
        axis=2
    )

    # Find minimum distance for each probe
    nearest_distances = distances_matrix.min(axis=1)

    return nearest_distances


def compute_gaussian_coverage_stats(
    centers: np.ndarray,
    scales: np.ndarray,
    probe_positions: np.ndarray
) -> Dict[str, float]:
    """Compute Gaussian coverage statistics.

    Args:
        centers: Gaussian centers [K, 3]
        scales: Gaussian scales [K, 3]
        probe_positions: Probe positions [P, 3]

    Returns:
        stats: Dictionary of coverage statistics
    """
    # Compute nearest Gaussian distances
    nearest_distances = compute_nearest_gaussian_distances(centers, probe_positions)

    # Compute average coverage radius (mean of 3D scales)
    coverage_radii = scales.mean(axis=1)  # [K]

    # Count probes within 1σ and 3σ of any Gaussian
    probes_within_1sigma = 0
    probes_within_3sigma = 0

    for probe_pos in probe_positions:
        # Check if probe is within any Gaussian's range
        for center, scale in zip(centers, scales):
            dist = np.linalg.norm(probe_pos - center)
            avg_scale = scale.mean()

            if dist <= avg_scale:
                probes_within_1sigma += 1
                probes_within_3sigma += 1
                break
            elif dist <= 3 * avg_scale:
                probes_within_3sigma += 1
                break

    stats = {
        'avg_distance_to_nearest': nearest_distances.mean(),
        'coverage_radius_mean': coverage_radii.mean(),
        'coverage_radius_std': coverage_radii.std(),
        'probes_within_1sigma': (probes_within_1sigma / len(probe_positions)) * 100,
        'probes_within_3sigma': (probes_within_3sigma / len(probe_positions)) * 100,
    }

    return stats
