# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Graduate thesis project on **hierarchical neural compression for multi-temporal lighting** using low-rank decomposition and FiLM modulation. Extends Gaussian Compression (SIGGRAPH'25) from single-moment to multi-moment scenarios.

**Target Metrics**: 1:17 compression ratio, >38dB PSNR, <0.5ms decompression, 60 FPS @ 1080p

**Stack**: Python 3.13, PyTorch 2.9.1, Mitsuba 3.7.3, Falcor rendering engine

## Architecture Overview

### Three-Stage Pipeline

```
1. DATA GENERATION (1_data_generation/)
   Mitsuba/Falcor → Probe sampling → SH baking → parametric_tensor.npz

2. TRAINING (2_src/ + 3_experiments/)
   LightSetEncoder → GaussianPhysicsCompressionUnified → Multi-loss optimization

3. EVALUATION (3_experiments/scripts/)
   Checkpoint → Metrics (MAE/RMSE/PSNR/SSIM) → JSON report
```

### Core Model Architecture

**Input**: Light descriptors [B, N, 12] + probe positions [B, 3]

**LightSetEncoder** (intensity-decoupled):
- Separates intensity from directional features
- Sum-pooled embedding → Z: [B, 32]
- Ensures physical light superposition: f(A+B) = f(A)+f(B)

**GaussianPhysicsCompressionUnified**:
- **Spatial basis**: K 3D Gaussians with learnable μ, scale, rotation
- **Low-rank basis**: U [K, 27, rank] per Gaussian
- **FiLM modulation**: U' = U ⊙ (1 + γ(Z)) + β(Z)
- **Coefficients**: [K, rank, embed_dim] matrices
- **Output**: SH coefficients [B, 27] (9 bases × 3 RGB)

**Loss Functions**:
- Reconstruction: Charbonnier (smooth L1) or MSE
- Temporal smoothness: L1/L2 on temporal differences
- Linearity: Enforces f(A+B) ≈ f(A)+f(B)
- Spatial: Smoothness across probe neighbors

### Directory Structure (Numbered Workflow)

- `1_data_generation/`: Rendering pipeline (Mitsuba/Falcor)
- `2_src/`: Core algorithms (models, training, data, utils)
- `3_experiments/`: Configs, scripts, results
- `4_thesis/`: Thesis materials
- `tools/`: CLI utilities (logexp, pipeline orchestration)
- `metadata/`: SQLite experiment tracking database
- `pipelines/`: YAML workflow definitions
- `archive/`: Deprecated methods (TemporalMLP, TPE)

## Common Commands

### Environment Setup
```bash
source venv/bin/activate
```

### Data Generation
```bash
cd 1_data_generation
python generate_dataset.py
python validate_dataset.py --data_dir output/method_test_v1
python analyze_sh_quality.py --data_dir output/method_test_v1
```

### Training
```bash
cd 3_experiments
python scripts/train.py --config configs/bistro_clean_train.yaml
```

### Evaluation
```bash
python scripts/eval.py \
  --data-root 1_data_generation/output/bistro_clean_v2 \
  --checkpoint 3_experiments/results/bistro_clean_v2/unified_set_K30_r8/best_model.pt \
  --split test
```

### Pipeline Execution
```bash
python tools/run_pipeline.py --config pipelines/pgcpl_bistro_dev.yaml
```

### Experiment Tracking
```bash
# Log experiment results to database
python tools/logexp.py log --phase Phase2_PGCPL --stage training \
  --script train_pgcpl5d --dataset 5D_param_val \
  --hyperparams '{"num_gaussians": 30, "rank": 8}' \
  --results '{"psnr": 32.5, "ssim": 0.951, "compress_ratio": 16.59}'

# Query historical experiments
python tools/logexp.py query --phase Phase2_PGCPL --limit 5

# Set baseline for comparison
python tools/logexp.py set-baseline --exp-id 42
```

### Testing
```bash
cd 2_src
pytest tests/
pytest tests/test_mainline_smoke.py -v
```

## Key Theoretical Concepts

### Spherical Harmonics (SH)

**2nd-order SH**: 9 basis functions (L=0: 1, L=1: 3, L=2: 5) × 3 RGB = 27 coefficients

**Fitting Process**:
1. Sample radiance in N directions (typically 64+)
2. Evaluate SH basis Y at each direction
3. Solve least-squares: `c = (Y^T Y)^{-1} Y^T L`

**Reconstruction**: `L(direction) = Σ_i c_i * Y_i(direction)`

**Implementation**: `2_src/utils/spherical_harmonics.py`

### Low-Rank Decomposition

**Model**: `SH(p, L) = Σ_j G_j(p) × [U_j(Z) @ (coeffs_j @ Z)]`

**Compression Ratio**:
- Naive: K × T × 27 parameters
- Low-rank: K × (3 + 4 + 27 + rank + rank×embed_dim)
- Typical: K=30, rank=8, embed_dim=32 → ~16.59× compression

### FiLM Modulation

**Dynamic basis adjustment**: `U'_j(Z) = U_j ⊙ (1 + γ_j(Z)) + β_j(Z)`

Adapts low-rank basis to changing light conditions while maintaining physical linearity.

### Light Set Encoding

**Intensity decoupling**: `Embed_i = intensity_i × MLP(features_i)`, then sum-pool to Z

Ensures physical light superposition: `f(Σ_i intensity_i × light_i) = Σ_i intensity_i × f(light_i)`

## Configuration Patterns

### Training Config Structure (YAML)

```yaml
experiment:
  variant: unified_set
  output_dir: 3_experiments/results/{scene}/{variant}_K{K}_r{rank}
  device: cuda
  seed: 42

model:
  num_gaussians: 30        # K: number of spatial Gaussians
  rank: 8                  # r: low-rank basis dimension
  top_k: 3                 # top-K Gaussian selection
  light_dim: 12            # light descriptor dimension
  embed_dim: 32            # pooled embedding dimension
  disable_film: false      # enable FiLM modulation

training:
  epochs: 2000
  lr: 1.0e-3
  recon_loss: charbonnier  # or 'mse'
  lambda_temporal: 0.001   # temporal smoothness weight
  lambda_linearity: 0.1    # linearity constraint weight
  linearity_aug_pairs: 2   # number of light pairs for linearity loss
```

**Naming Convention**: K{num_gaussians}_r{rank} (e.g., K30_r8)

### Dataset Format

**Output**: `parametric_tensor.npz` containing:
- `tensor`: [P, M, 27] (P probes, M moments, 27 SH coefficients)
- `probe_positions`: [P, 3]
- `light_configs`: [M, N, 12] (light descriptors)
- `light_mask`: [M, N] (validity mask)
- `metadata.json`: Scene bounds, probe count, moment info

## Important Conventions

### Numbered Directories
Sequential workflow stages: `1_` (data) → `2_` (src) → `3_` (experiments) → `4_` (thesis)

### Database-Driven Tracking
All experiments logged to `metadata/project.db` (SQLite). Query before assuming dataset paths.

### Bilingual Documentation
Chinese user-facing docs, English code comments and technical docs.

### Virtual Environment
Always activate: `source venv/bin/activate`

## Anti-Patterns to Avoid

1. **Never guess dataset paths** — Query database (`metadata/project.db`) first; use `docs/dev_notes/project_state.md` only as historical snapshot
2. **Never use deprecated methods** — TemporalMLP and TPE are archived; use PG-GCPL (GaussianPhysicsCompressionUnified)
3. **Never modify BSDF physical realism** — Mitsuba rendering parameters are calibrated
4. **Never bypass virtual environment** — Dependencies are version-specific
5. **Never confuse experimental vs. target architecture** — Current implementation is K30_r8; target is 4×3 cascaded volumes

## Current Implementation Status

### Implemented ✅
- Gaussian mixture spatial representation
- Temporal MLP with intensity-decoupled light encoding
- FiLM-modulated low-rank basis
- Multi-loss training (reconstruction, temporal, linearity, spatial)
- Data generation (Mitsuba + Falcor)
- Evaluation metrics and benchmarking
- Database-driven experiment tracking

### Pending ❌
1. 4×3 Cascaded volumes (hierarchical spatiotemporal structure)
2. Temporal LRU cache for real-time decompression
3. Fused CUDA kernels for <0.5ms decompression
4. 10-bit quantization (two-stage training)
5. Energy conservation loss (physics-based soft constraint)
6. Lighting decomposition (direct + indirect for editability)
7. 24-moment training (currently 3-6 moments)

## Key Files Reference

| File | Purpose |
|------|---------|
| `2_src/models/gaussian_physics_unified.py` | Main model architecture |
| `2_src/training/gaussian_physics_trainer.py` | Unified trainer |
| `2_src/data/lightset_dataset.py` | Dataset loader |
| `2_src/utils/spherical_harmonics.py` | SH basis evaluation and fitting |
| `3_experiments/configs/bistro_clean_train.yaml` | Main training config |
| `3_experiments/scripts/train.py` | Training entry point |
| `3_experiments/scripts/eval.py` | Evaluation entry point |
| `tools/logexp.py` | Experiment logging CLI |
| `tools/run_pipeline.py` | Pipeline orchestration |
| `metadata/project.db` | SQLite experiment tracking |

## Additional Documentation

- `docs/dev_notes/DEVELOPMENT_TRUTH_2026-03-08.md`: Current code-truth snapshot (date-versioned)
- `docs/dev_notes/CLAUDE.md`: 588-line comprehensive development guide
- `AGENTS.md`: Project knowledge base
- `docs/dev_notes/project_state.md`: Historical project context snapshot (not auto-generated in current repo)
- `README.md`: Quick start guide
