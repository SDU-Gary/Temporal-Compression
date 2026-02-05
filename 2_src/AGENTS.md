# 2_src - CORE SOURCE CODE

**Generated**: 2026-02-05
**Directory**: Stage 2 in workflow (1_data_generation → 2_src → 3_experiments → 4_thesis)

## OVERVIEW
Core algorithms and neural network implementations for hierarchical neural compression of multi-temporal lighting.

## STRUCTURE
```
2_src/
├── models/                 # Neural network architectures
│   ├── gaussian_physics_1D.py          # 1D temporal compression
│   ├── gaussian_physics_5D.py          # 5D physics-guided compression
│   ├── gaussian_physics_unified.py     # Unified 1D/5D model
│   ├── physics_low_rank.py             # Physics-guided low-rank (Tucker)
│   ├── gaussian_mixture.py             # 3D Gaussian spatial representation
│   ├── pg_rglt.py                      # Physics-guided regularization
│   ├── dual_gaussian_fbt.py            # Dual Gaussian forward-backward
│   └── gaussian_physics_compression.py # Legacy base class
├── training/              # Training infrastructure
│   ├── gaussian_physics_trainer.py     # Main trainer with multiple losses
│   ├── trainer.py                      # Base trainer class
│   ├── losses.py                       # Loss functions (MSE, perceptual)
│   └── metrics.py                      # Evaluation metrics (PSNR, SSIM)
├── data/                  # Dataset loaders and preprocessing
│   ├── lightset_dataset.py             # Main lighting dataset loader
│   ├── intensity_modulation_dataset.py # Intensity modulation datasets
│   ├── transfer_tensor_dataset.py      # Transfer tensor datasets
│   └── sh_scaler.py                    # Spherical harmonics scaling
├── utils/                 # Shared utilities
│   ├── spherical_harmonics.py          # SH fitting and reconstruction
│   ├── rendering_utils.py              # SH reconstruction and cubemap
│   ├── coords.py                       # Coordinate transformations
│   ├── probe_field_slicer.py           # Probe field slicing utilities
│   └── rerun_logger.py                 # Rerun visualization logger
└── tests/                 # Unit and integration tests
    ├── test_models.py                  # Model architecture tests
    ├── test_dataset.py                 # Dataset loading tests
    ├── test_scene_io.py                # Scene I/O tests
    └── 10+ other test files
```

## KEY MODELS
**Gaussian-Physics Hybrid Models**:
- `GaussianPhysicsCompression1D`: 1D temporal compression (time-varying latent codes)
- `GaussianPhysicsCompression5D`: 5D physics-guided compression (sun position basis)
- `GaussianPhysicsCompressionUnified`: Unified 1D/5D model with switchable modes

**Physics-Guided Models**:
- `PhysicsLowRank`: Tucker decomposition with physical basis functions
- `GaussianMixture`: 3D Gaussian spatial representation with learnable parameters

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
**Physics Basis**: 5D models use astronomical sun position calculations (azimuth, elevation) as physical basis functions.

**Gaussian Parameters**: `GaussianMixture` stores means, covariances, and weights for spatial representation.

**Low-Rank Structure**: `PhysicsLowRank` implements Tucker decomposition with rank constraints.

**Unified Interface**: All models implement `forward()` with signature `(x, sun_positions=None)`.

**Training Loop**: `GaussianPhysicsTrainer` handles multi-loss optimization with validation metrics.

**Dataset Reality**: Always verify actual probe counts in `probes.npz` - config files may be inaccurate.

**Memory Management**: Large probe fields (~15k probes) require careful batching in data loaders.