# TemporalMLP: Task Specification Baseline Method

**Method**: Gaussian mixture + Temporal MLP + Decoder MLP
**Status**: Deprecated (Baseline for comparison)
**Note**: Original task specification method - replaced by PG-GCPL for better performance

## Method Overview

### Core Idea

Three-stage architecture for multi-temporal lighting compression:
1. **Spatial Compression**: Gaussian mixture represents probe distribution
2. **Temporal Encoding**: MLP maps [base_latent, sun_dir] → time_latent
3. **Decoding**: MLP decodes time_latent → SH coefficients

### Mathematical Formulation

```
# Stage 1: Spatial encoding
spatial_latent = ΣGaussianMixture(probe_position)

# Stage 2: Temporal encoding
time_latent = TemporalMLP(spatial_latent, sun_direction)

# Stage 3: Decoding
SH_coeffs = DecoderMLP(time_latent)
```

### Comparison with PG-GCPL

| Aspect | TemporalMLP (This) | PG-GCPL (Current Best) |
|--------|-------------------|------------------------|
| **Temporal Encoding** | Learned MLP (black box) | Physics basis (interpretable) |
| **Parameters** | More (MLPs add overhead) | Fewer (low-rank factorization) |
| **Compression** | Moderate | 16.59× |
| **Quality** | Baseline | +33.5% vs splines |
| **Physical Consistency** | Implicit (learned) | Explicit (trigonometric) |

## Project Structure

```
TemporalMLP/
├── src/
│   ├── models/
│   │   ├── temporal_mlp.py      # Temporal encoding MLP
│   │   ├── decoder_mlp.py       # SH decoding MLP
│   │   ├── full_model.py        # Complete architecture
│   │   └── time_encoder.py      # Time encoding variant
│   ├── data/
│   │   └── dataset.py           # Dataset loader (copied from shared)
│   ├── scripts/
│   │   └── knn_baseline.py      # KNN baseline for comparison
│   └── training/                # Training utilities (copied)
├── configs/
│   ├── baseline.yaml
│   ├── baseline_2k.yaml
│   ├── baseline_2k_v2.yaml
│   ├── baseline_2k_v3.yaml
│   └── baseline_regularized.yaml
└── experiments/
    ├── baseline/
    ├── baseline_2k/
    ├── baseline_2k_v2/
    └── baseline_2k_v3/
```

## Experimental Results

### Baseline Experiments
- Multiple training runs with different configurations
- Checkpoints and logs available in `experiments/baseline*/`
- Results used as comparison baseline for PG-GCPL method

### KNN Baseline
- Simple K-Nearest Neighbors interpolation baseline
- Results: `experiments/knn_baseline_results.npz`

## Why Deprecated?

The TemporalMLP method was replaced by PG-GCPL for the following reasons:

1. **Physics-Guided Superiority**: PG-GCPL's physics basis (trigonometric functions) outperformed learned embeddings
2. **Better Compression**: Low-rank factorization more parameter-efficient than MLP layers
3. **Interpretability**: Physics basis provides clear physical meaning vs black-box MLP
4. **Empirical Results**: PG-GCPL achieved 33.5% better precision on validation set

## Related Methods (Archived)

### TPE Method (Temporal Perturbation Embedding)
- **Location**: `../../archive/TPE_Method_Archived/`
- **Status**: Failed - 84.9% error contribution
- **Lesson**: Grid interpolation > Implicit neural representations for smooth temporal data

See: `archive/TPE_Method_Archived/docs/` for detailed failure analysis

## Usage (Historical Reference)

```bash
# Train TemporalMLP baseline (deprecated)
# python TemporalMLP/src/scripts/train.py --config TemporalMLP/configs/baseline.yaml

# Run KNN baseline
python TemporalMLP/src/scripts/knn_baseline.py --data_dir ../data_generation/output/method_test_v1
```

**Note**: This method is kept for historical reference and comparison purposes. **Use PG-GCPL for active development.**

## Key Learnings

1. **Explicit physics priors** > Learned implicit representations (for outdoor lighting)
2. **Low-rank factorization** > MLP layers (for temporal compression)
3. **Interpretability matters**: Physics-guided methods easier to debug and improve

## See Also

- **Active Method**: [PG-GCPL/README.md](../PG-GCPL/README.md)
- **Failed Method**: [archive/TPE_Method_Archived/](../archive/TPE_Method_Archived/)
- **Task Specification**: `/home/kyrie/毕设/docs/thesis/多时刻光照压缩任务书.md`
