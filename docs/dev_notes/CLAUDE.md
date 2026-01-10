# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a graduate thesis project on **hierarchical neural compression for multi-temporal lighting** using time-varying latent codes, Gaussian mixture representation, and spatiotemporal cascaded volumes. The goal is to extend Gaussian Compression (SIGGRAPH'25) from single-moment to multi-moment scenarios through four key innovations:

1. **Time-Varying Latent Codes**: Small MLP maps base latent code $F_j^{base}$ + sun direction $sun\_dir(t)$ → time-specific latent code $F_j(t)$, achieving 18× temporal compression
2. **Spatiotemporal Cascaded Volumes**: 4 spatial levels (1m, 4m, 16m, 64m) × 3 temporal levels (5min, 15min, 1hr) for adaptive resolution
3. **Real-Time Decompression**: Temporal LRU cache + fused CUDA kernels targeting <0.5ms decompression, <0.15ms with cache hits
4. **Physical Soft Constraints**: Energy conservation and temporal smoothness losses with tunable weights

**Target Metrics**: 1:17 compression ratio (vs. naive 24× independent models), >38dB PSNR, <0.5ms decompression, 60 FPS @ 1080p

**CRITICAL**: The thesis task specification in `docs/thesis/多时刻光照压缩任务书.md` describes the TARGET architecture. Current implementations include both the target architecture AND exploratory experimental methods. Always refer to the task specification to understand what needs to be built.

## Repository Structure

```
毕设/
├── data_generation/           # Mitsuba 3-based dataset generation
│   ├── generate_dataset.py    # Main generation script
│   ├── core/                  # Core rendering logic
│   ├── utils/                 # SH fitting, sun position calculation
│   └── output/                # Generated datasets
├── multi_time_compression/    # Main compression system
│   ├── configs/               # Training configurations (YAML)
│   ├── data/                  # Dataset loaders
│   ├── models/                # Neural network models
│   │   ├── gaussian_mixture.py        # 3D Gaussian spatial representation
│   │   ├── temporal_mlp.py            # Time-varying latent code MLP (TASK SPEC)
│   │   ├── decoder_mlp.py             # SH coefficient decoder
│   │   ├── full_model.py              # Full integration (TASK SPEC)
│   │   ├── physics_low_rank.py        # Experimental: physics-guided low-rank
│   │   └── gaussian_physics_compression.py  # Experimental: hybrid method
│   ├── training/              # Training loop, losses, metrics
│   ├── scripts/               # Training and evaluation scripts
│   ├── experiments/           # Training outputs and checkpoints
│   ├── docs/experiment_reports/  # 10 detailed experiment reports
│   └── tests/                 # Unit tests
├── docs/                      # Thesis and research notes
│   ├── thesis/多时刻光照压缩任务书.md  # **AUTHORITATIVE TASK SPECIFICATION**
│   └── research_notes/        # Research notes on related work
├── mitsuba-docs/              # Local Mitsuba documentation
└── venv/                      # Python virtual environment
```

## Development Environment

### Python Environment

**Always activate the virtual environment first:**

```bash
source venv/bin/activate
```

**Key dependencies**:
- Python 3.13
- PyTorch 2.9.1 with CUDA 12.8
- Mitsuba 3.7.3 (cuda_ad_rgb variant)
- DrJit for differentiable rendering
- numpy 1.26.4, scipy 1.16.3

### Common Development Commands

#### Data Generation

```bash
cd data_generation

# Generate dataset with multiple time moments
python generate_dataset.py

# Validate generated dataset quality
python validate_dataset.py --data_dir output/method_test_v1

# Analyze SH reconstruction quality
python analyze_sh_quality.py --data_dir output/method_test_v1
```

#### Training

```bash
cd multi_time_compression

# Train full model (task specification architecture)
python scripts/train.py --config configs/baseline.yaml

# Quick diagnostic (checks data loading and model forward pass)
python scripts/quick_diagnose.py

# Detailed training diagnostics
python scripts/diagnose_training.py --config configs/baseline.yaml --steps 100

# Experimental methods (exploratory, not main task)
python scripts/train_physics_low_rank_proper.py
python scripts/train_gaussian_physics.py
```

#### Testing

```bash
cd multi_time_compression

# Test dataset loading
python tests/test_dataset.py

# Test model components
python tests/test_models.py

# Run all tests
pytest tests/
```

## Target Architecture (From Task Specification)

The thesis task describes a 4-layer architecture:

### 1. Time Layer: Time-Varying Latent Codes

**Core Equation**:
```
F_j(t) = MLP_temporal([F_j^base, sun_dir(t)])
```

- **Input**: Base latent code $F_j^{base}$ (6-D, time-invariant) + sun direction (3-D, computed via astronomical algorithm)
- **Output**: Time-specific latent code $F_j(t)$ (9-D)
- **Network**: 2 layers × 32 neurons, ReLU activation
- **Compression**: Store K × 6 parameters + 2K MLP weights instead of K × T × 9 parameters
- **Implementation**: `models/temporal_mlp.py`, `models/full_model.py`

**Compression Ratio (Time Dimension)**:
- Naive: K × T × D parameters (T=24 moments, D=9)
- Time-varying: K × D_base + 2K parameters (D_base=6)
- Ratio: (24 × 9) / (6 + 2) = 27× → ~18× accounting for overheads

### 2. Spatial Layer: Gaussian Mixture Representation

**Core Equation**:
```
F(p,t) = Σ_{j: p ∈ R(G_j)} F_j(t) · G_j(p)
```

- **Purpose**: Spatial distribution of latent codes using K 3D Gaussians
- **Parameters per Gaussian**: μ_j (position, 3D), s_j (scale, 3D), q_j (rotation quaternion, 4D), F_j^base (latent, 6D)
- **Initialization**: K-Means clustering on probe positions
- **Implementation**: `models/gaussian_mixture.py`

### 3. Decoder Layer: SH Coefficient MLP

**Core Equation**:
```
SH_coeff(p,t) = MLP_decoder([F(p,t), p, sun_dir(t)])
```

- **Input**: Latent feature F(p,t) (9-D) + position p (3-D) + sun direction (3-D) = 15-D
- **Output**: Spherical harmonics coefficients (27-D: 9 bases × 3 RGB channels)
- **Network**: 2 layers × 64 neurons, Sigmoid activation
- **Implementation**: `models/decoder_mlp.py`

### 4. Hierarchy Layer: 4×3 Cascaded Volumes (TO BE IMPLEMENTED)

**Spatial Levels** (4 levels):
- Level 0: 1m blocks (near field, <5m from camera)
- Level 1: 4m blocks (mid field, 5-20m)
- Level 2: 16m blocks (far field, 20-80m)
- Level 3: 64m blocks (very far, >80m)

**Temporal Levels** (3 levels):
- Level 0: 5-minute granularity (fast changes, e.g., clouds)
- Level 1: 15-minute granularity (solar trajectory)
- Level 2: 1-hour granularity (slow changes)

**Selection Strategy**:
- Spatial: Based on distance from camera
- Temporal: Based on change rate $||L(p,t) - L(p,t-Δt)||_2 / Δt$
- Total: 12 cascaded volumes (4 × 3)

**Status**: Not yet implemented - this is a key milestone in the thesis task

## Current Implementation Status

### What Exists (Target Architecture)

1. ✅ **Full Model** (`models/full_model.py`): Integrates Gaussian mixture + temporal MLP + decoder MLP
2. ✅ **Temporal MLP** (`models/temporal_mlp.py`): Time-varying latent code generation
3. ✅ **Gaussian Mixture** (`models/gaussian_mixture.py`): Spatial latent code representation
4. ✅ **Decoder MLP** (`models/decoder_mlp.py`): SH coefficient reconstruction
5. ✅ **Data Generation** (`data_generation/`): Mitsuba 3-based multi-moment rendering
6. ✅ **Dataset Loader** (`data/dataset.py`): Multi-temporal lighting data loading
7. ✅ **Training Loop** (`training/trainer.py`): Basic training infrastructure
8. ✅ **Loss Functions** (`training/losses.py`): Reconstruction, temporal smoothness

### What's Missing (Per Task Specification)

1. ❌ **4×3 Cascaded Volumes**: Hierarchical spatiotemporal structure
2. ❌ **Temporal LRU Cache**: Real-time decompression optimization
3. ❌ **Fused CUDA Kernels**: GPU acceleration for <0.5ms decompression
4. ❌ **10-bit Quantization**: Two-stage training (fine-tuning + decoder adaptation)
5. ❌ **Energy Conservation Loss**: Physics-based soft constraint
6. ❌ **Lighting Decomposition**: Direct + indirect light for editability
7. ❌ **24-Moment Training**: Current experiments use 3-6 moments, target is 24

### Experimental Methods (Not Main Task)

The following are exploratory implementations documented in `docs/experiment_reports/`:

1. **TPE Method** (文档1-7): Temporal Perturbation Embedding - **FAILED** (84.9% error contribution)
2. **Physics-Guided Low-Rank** (文档8): SH(t) ≈ U @ (coeffs @ [cos(θ), sin(θ), 1]) - Experimental compression method
3. **Gaussian-Physics Hybrid** (文档9): Spatial Gaussians + temporal physics basis - Experimental method

**Key Learning**: These experiments explored alternative compression strategies but are NOT the target architecture described in the thesis task. They provide valuable insights but should not be confused with the main deliverable.

## Configuration System

All training parameters are in YAML files (`configs/`):

**Key config sections**:
- `data`: Dataset paths, split ratios, batch size
- `model`: num_gaussians, latent dimensions, MLP architectures
- `training`: Optimizer settings, loss weights, learning rate schedule
- `validation`: Visualization and evaluation settings

**To modify training**:
1. Copy `configs/baseline.yaml` to a new file
2. Edit parameters (e.g., num_gaussians, latent_dim_base, loss weights)
3. Run: `python scripts/train.py --config configs/your_config.yaml`

## Important Implementation Notes

### Coordinate Systems

- **World coordinates**: Right-handed, Y-up (Mitsuba convention)
- **Probe positions**: Normalized to [-1, 1]³ in dataset loader
- **Sun directions**: Unit vectors in world coordinates (computed via astronomical algorithms)
- **SH coefficients**: Real spherical harmonics, 2nd order (9 bases), stored as [N, 27] (9 bases × 3 RGB)

### Data Flow

1. **Dataset Generation** (data_generation/):
   - Mitsuba 3 renders scenes at T time moments with different sun positions
   - Generates N probe positions via uniform grid sampling
   - Bakes spherical harmonics (2nd order, 27 coefficients) for each probe/moment
   - Computes sun directions using astronomical algorithms
   - Outputs: probes.npz, moment_XX/sh_coeffs.npz, moment_XX/images/

2. **Training** (multi_time_compression/):
   - Loads multi-temporal data (probe positions, SH coefficients, sun directions)
   - Initializes K Gaussians via K-Means clustering on probe positions
   - For each Gaussian: stores base latent code F_j^base
   - Forward pass: (p, t) → Gaussian → F_base(p) → TemporalMLP → F(p,t) → DecoderMLP → SH(p,t)
   - Backprop: Updates Gaussian parameters, F_j^base, MLP weights

3. **Inference** (planned):
   - Query probe position p + time moment t
   - Compute sun direction from time via astronomical algorithm
   - Look up nearby Gaussians, compute weighted sum of base latent codes
   - Pass through temporal MLP to get F(p,t)
   - Decode to SH coefficients via decoder MLP
   - Evaluate radiance for view direction

### Loss Functions (Task Specification)

Primary losses (`training/losses.py`):

1. **Reconstruction Loss** (Weight: 1.0):
   ```
   L_recon = (1/N·T) Σ ρ(||L_pred(p_i,t) - L_gt(p_i,t)||_2)
   ρ(z) = √(z² + ε²)  (Charbonnier robust L2)
   ```

2. **Temporal Smoothness** (Weight: 0.01-0.2):
   ```
   L_temporal = (1/N·(T-2)) Σ ||L(p_i,t-1) - 2·L(p_i,t) + L(p_i,t+1)||_1
   ```
   Encourages smooth temporal transitions (second-order difference)

3. **Energy Conservation** (Weight: 0.1, **TO BE IMPLEMENTED**):
   ```
   L_energy = (1/N·T) Σ max(0, ∫ L_pred - ∫ L_gt - ε)²
   ```
   Ensures outgoing radiance ≤ incoming radiance

**Important**: Current implementation only has L_recon and L_temporal. Energy conservation is specified in the task but not yet implemented.

### Gaussian Initialization

Gaussians are initialized via K-Means on training probe positions:
- Number of Gaussians (K): Typically 5% of probe count (e.g., 750 for 15K probes)
- Position μ_j: K-Means cluster centers
- Scale s_j: Distance to nearest neighbor Gaussian
- Rotation q_j: Identity quaternion [1,0,0,0]
- Base latent F_j^base: Random initialization from N(0, 0.1²)

**Critical**: K must be ≤ number of training probes. Setting K too large causes overfitting.

### Training Stability

Common issues and solutions:

1. **NaN losses**: Usually from unstable Gaussian scales
   - Solution: Gradient clipping (grad_clip: 1.0 in config)
   - Add Gaussian scale regularization

2. **Poor reconstruction**: Insufficient model capacity
   - Solution: Increase num_gaussians or latent_dim_base
   - Check Gaussian coverage (aim for >95% of probes within influence)

3. **Temporal jitter**: Insufficient temporal smoothness
   - Solution: Increase temporal_smooth loss weight (0.01 → 0.1)
   - Verify sun direction inputs are correct

4. **Overfitting to training moments**: Poor interpolation
   - Solution: Increase training data diversity
   - Add dropout to MLPs
   - Test on held-out time moments

## Development Workflow

### Adding a New Model Component

1. Create new file in `models/` (e.g., `models/cascaded_volume.py`)
2. Inherit from `torch.nn.Module`, implement `__init__` and `forward`
3. Add to `models/__init__.py` for easy import
4. Write unit test in `tests/test_models.py`
5. Run: `pytest tests/test_models.py -v`

### Adding a New Loss Function

1. Add function to `training/losses.py` following existing pattern:
   ```python
   def new_loss(pred, target, weight=1.0):
       """
       Args:
           pred: [B, ...] predicted values
           target: [B, ...] ground truth
           weight: scalar loss weight
       Returns:
           scalar loss value
       """
       loss = ...  # implementation
       return loss * weight
   ```
2. Register in config YAML under `training.loss`
3. Add to `Trainer._compute_loss()` in `training/trainer.py`

### Debugging Training

Use the diagnostic scripts:

```bash
# Quick check: data loading + forward pass
python scripts/quick_diagnose.py

# Detailed: gradient flow, loss curves, data stats
python scripts/diagnose_training.py --config configs/baseline.yaml --steps 100
```

**Key metrics to check**:
- Loss convergence: Should decrease steadily in first 1000 steps
- Gaussian coverage: % of probes within Gaussian influence (aim for >95%)
- Temporal smoothness: Should be low and stable across moments
- Gradient norms: Should be stable (not exploding or vanishing)

## Physics and Mathematical Background

### Spherical Harmonics (SH)

Used to compactly represent directional radiance:
- 2nd order SH (9 bases) captures low-frequency lighting
- Coefficients computed by projecting radiance onto SH basis
- Reconstruction: L(ω) = Σ c_lm · Y_lm(ω)

**Implementation**: `data_generation/utils/spherical_harmonics.py`

### Sun Position Calculation

Astronomical algorithm to compute sun direction from time:
- Input: Time (hour), latitude, longitude, date
- Output: Unit direction vector [x,y,z] + solar zenith angle θ
- Based on standard astronomical calculations

**Implementation**: `data_generation/utils/sun_position.py`

**Usage**:
```python
from data_generation.utils.sun_position import compute_sun_directions
sun_dirs, zeniths = compute_sun_directions(
    hours=[6, 12, 18],
    latitude=40.0,
    longitude=116.0
)
```

### Compression Ratio Analysis

**Naive approach**: Store 24 independent Gaussian Compression models
- Storage: 24 × 50MB = 1200MB

**Time-Varying Latent Code Approach** (Task Specification):
- K Gaussians × (position + scale + rotation + F_base): K × (3+3+4+6) = K × 16 floats
- Temporal MLP weights: ~2K parameters (2 layers × 32 neurons, shared globally)
- Decoder MLP weights: ~7K parameters (2 layers × 64 neurons, shared globally)
- Total (K=750): ~12K params = 0.048MB (FP32)

**With 10-bit quantization** (task specification):
- Gaussian parameters: K × 16 × 10/8 = K × 20 bytes
- MLP weights: Keep FP32
- Target storage: <100MB for 24 moments (including metadata, hierarchies)

**Target compression ratio**: 1200MB / 70MB ≈ **1:17**

## Thesis Roadmap (From Task Specification)

### Milestone 1: Single-Moment 4-Level Spatial Hierarchy (Months 1-3)
- Implement 4-level spatial cascaded volumes (1m, 4m, 16m, 64m)
- Verify distance-based adaptive selection
- Target: PSNR >40dB, 30 FPS @ 1080p
- **Status**: NOT YET STARTED

### Milestone 2: Single-Level 3-Moment Temporal Hierarchy (Months 4-6)
- Implement 3-level temporal cascaded volumes (5min, 15min, 1hr)
- Verify time-varying latent code MLP effectiveness
- Target: 3 moments, PSNR >38dB, compression >2.5×
- **Status**: PARTIAL - temporal MLP exists, but no hierarchies

### Milestone 3: 4×3 Joint Optimization (Months 7-10)
- Integrate 4 spatial × 3 temporal levels = 12 cascaded volumes
- Extend to 24 moments (every hour)
- Add energy conservation loss
- Implement 10-bit quantization
- Target: 1:17 compression, >38dB PSNR
- **Status**: NOT YET STARTED

### Milestone 4: Real-Time Decompression & Editability (Months 11-12)
- Implement Temporal LRU cache
- Implement fused CUDA kernels
- Add lighting decomposition (direct + indirect)
- Target: <0.5ms decompression, 60 FPS rendering
- **Status**: NOT YET STARTED

## Tips for Working with this Codebase

1. **Read the task specification first**: `docs/thesis/多时刻光照压缩任务书.md` is the authoritative source
2. **Understand experimental vs. target**: `docs/experiment_reports/` documents exploratory methods, NOT the main deliverable
3. **Check configs first**: Most hyperparameters are in YAML, not hardcoded
4. **Use the full model**: `models/full_model.py` implements the task specification architecture
5. **Verify sun directions**: Astronomical algorithm is critical for temporal compression
6. **Test temporal interpolation**: Model should generalize to unseen time moments
7. **Monitor Gaussian coverage**: Aim for >95% of probes within Gaussian influence
8. **Version control configs**: When changing parameters, save config with experiment

## Experiment Memory System

This project uses an SQLite-based persistent memory system to maintain experiment history across Claude Code conversation compactions.

### Architecture

**Database**: `project.db` (SQLite)
- `datasets` - Dataset metadata (11 datasets)
- `scenes` - Scene definitions (cornell-box, house, staircase2)
- `scripts` - Script registry with descriptions
- `experiments` - Complete experiment history (17+ experiments)
- `tasks` - Task management (7 pending tasks)
- `env_config` - Environment configuration

**State Summary**: `project_state.md` (auto-generated, <500 words)
- Hot-start context after conversation compaction
- Shows: Active tasks, milestones, recent experiments, dataset mappings
- Regenerated via `python tools/summarize_state.py`

**Validation**: `DATABASE_VERIFICATION_REPORT.md`
- Cross-validation report against source files
- Documents 7 corrections made to historical data
- Establishes four-source validation methodology (JSON > logs > reports > config)

### Automatic Context Loading

**SessionStart Hook**: `.claude/hooks/SessionStart.sh`
- Automatically runs on each new Claude Code session
- Regenerates `project_state.md` from database
- Displays critical context: tasks, milestones, recent experiments
- Shows database tools and reminders

### Workflow Protocol

**After conversation compaction**:
1. Hook auto-runs: regenerates state and displays context
2. Read `project_state.md` sections: [TASKS], [MILESTONES], [RECENT]
3. **NEVER guess dataset paths** - query database or check state file

**After completing experiment**:
```bash
# Log immediately with all results
python tools/logexp.py log \
    --phase Phase2_PGCPL \
    --stage training \
    --script train_pgcpl5d \
    --dataset 5D_param_val \
    --hyperparams '{"num_gaussians": 30, "rank": 8}' \
    --results '{"psnr": 32.5, "ssim": 0.951, "compress_ratio": 16.59}'

# Set as baseline if applicable
python tools/logexp.py set-baseline EXP-20260104-001

# Regenerate state
python tools/summarize_state.py
```

**Task management**:
```bash
# Add task
python tools/logexp.py add-task "Implement energy conservation loss" --priority 3

# List tasks
python tools/logexp.py list-tasks --status todo

# Complete task
python tools/logexp.py complete-task 5
```

**Query experiments**:
```bash
# Query by phase
python tools/logexp.py query --phase Phase2_PGCPL

# Query by dataset
python tools/logexp.py query --dataset 5D_param_val

# Find best PSNR
python tools/logexp.py query --min-psnr 30 --limit 5
```

### Database Tools

**CLI Tool**: `tools/logexp.py`
- `log` - Log new experiment
- `set-baseline` - Mark experiment as baseline
- `add-task`, `complete-task`, `list-tasks` - Task management
- `query` - Search experiments with filters

**State Generator**: `tools/summarize_state.py`
- Generates `project_state.md` from database
- Deterministic output (<500 words)
- Run after any database update

**Database Utilities**: `tools/db_utils.py`
- `get_db()` - Context manager for database connection
- `init_db()` - Initialize from schema

### Data Quality Assurance

**Validation Priority**:
1. JSON results (most reliable, program-generated)
2. Training logs (real-time records)
3. Experiment reports (human summaries, may have rounding)
4. Config files (for verification only)

**Critical Rules**:
- All probe counts use actual `probes.npz` data, not `config.json` declarations
- Training experiments (K30_r8) have MAE/RMSE, not PSNR/SSIM
- Rendering experiments have PSNR/SSIM, not MAE/RMSE
- All historical data verified via four-source cross-validation

### Dataset-Scene Mapping

**Most-Used Datasets** (from database):
- **5D_PARAM** (5D_parametric_validation): 125 probes × 41 moments, SPP=128
- **D2K** (dataset_2k): 1,728 probes × 6 moments, SPP=128
- **MT1** (method_test_v1): 343 probes × 6 moments, SPP=256
- **TPE_CB** (level2_tpe/cornell-box_test): 1 probe × 13 moments, SPP=256
- **TPE_HS** (level2_tpe/house_p0): 1 probe × 13 moments, SPP=256

**Note**: Config files may declare different probe counts (e.g., dataset_15k declares 15K but has 13,824). Always trust actual data.

## References

- **Task specification** (AUTHORITATIVE): `docs/thesis/多时刻光照压缩任务书.md`
- Experiment reports: `multi_time_compression/docs/experiment_reports/` (exploratory work)
- Data generation guide: `data_generation/README.md`
- Mitsuba 3 docs: `mitsuba-docs/` (local copy)
- Gaussian Compression paper: SIGGRAPH 2025 (base architecture)
- K-Planes: CVPR 2023 (temporal smoothness inspiration)
- PBR-NeRF: 2024 (energy conservation inspiration)
- **Memory system**: `DATABASE_VERIFICATION_REPORT.md`, `tools/logexp.py`, `project_state.md`
