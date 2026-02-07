# 2_src - CORE SOURCE CODE

**Generated**: 2026-02-05
**Directory**: Stage 2 in workflow (1_data_generation → 2_src → 3_experiments → 4_thesis)

## OVERVIEW
Core algorithms and neural network implementations for hierarchical neural compression of multi-temporal lighting.

## STRUCTURE
```
2_src/
├── models/                 # Neural network architectures
│   └── gaussian_physics_unified.py     # Unified FiLM mainline model
├── training/              # Training infrastructure
│   ├── gaussian_physics_trainer.py     # Main trainer with multiple losses
│   ├── trainer.py                      # Base trainer class
│   ├── losses.py                       # Loss functions (MSE, perceptual)
│   └── metrics.py                      # Evaluation metrics (PSNR, SSIM)
├── data/                  # Dataset loaders and preprocessing
│   ├── lightset_dataset.py             # Main lighting dataset loader
│   └── sh_scaler.py                    # Spherical harmonics scaling
├── utils/                 # Shared utilities
│   ├── spherical_harmonics.py          # SH fitting and reconstruction
│   ├── rendering_utils.py              # SH reconstruction and cubemap
│   ├── coords.py                       # Coordinate transformations
│   ├── probe_field_slicer.py           # Probe field slicing utilities
│   └── rerun_logger.py                 # Rerun visualization logger
└── tests/                 # Unit and integration tests
    ├── test_dataset.py                 # Dataset loading tests
    ├── test_scene_io.py                # Scene I/O tests
    └── 10+ other test files
```

## KEY MODELS
**PG-GCPL Mainline**:
- `GaussianPhysicsCompressionUnified`: FiLM-based unified_set model

## KEY UTILITIES
**Spherical Harmonics**:
- `spherical_harmonics.py`: SH fitting (`fit_sh_coefficients`) and reconstruction
- `rendering_utils.py`: SH to cubemap conversion and visualization

**Coordinate Systems**:
- `coords.py`: Cartesian/spherical conversions, sun position calculations

**Data Processing**:
- `probe_field_slicer.py`: Slice and reshape probe field tensors
- `sh_scaler.py`: Normalize SH coefficients across datasets

## CONVENTIONS
**Model Naming**: Follows `GaussianPhysicsCompression{D}` pattern where D indicates dimensionality.

**Physics Integration**: All models accept `sun_positions` tensor for physics-guided compression.

**Tensor Shapes**: Consistent shape conventions: `[batch, channels, height, width, depth]` for 5D, `[batch, time, channels]` for 1D.

**Loss Functions**: Defined in `training/losses.py` with configurable weighting.

## IMPORTANT NOTES
仓库仅保留最新主线，历史模型/脚本已清理。

**Training Loop**: `GaussianPhysicsTrainer` handles multi-loss optimization with validation metrics.

**Dataset Reality**: Always verify actual probe counts in `probes.npz` - config files may be inaccurate.

**Memory Management**: Large probe fields (~15k probes) require careful batching in data loaders.
