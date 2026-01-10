# Phase 01: Physics Low-Rank Validation

**Timeline**: Week 1-2
**Goal**: Validate low-rank hypothesis for temporal SH dynamics and compare physics-guided vs spline interpolation

## Experiment Structure

### svd_analysis/
- **Purpose**: SVD analysis of temporal SH coefficient matrices
- **Key Results**: Verified that 95%+ variance captured by rank 5-8 for outdoor lighting
- **Files**: Singular value plots, rank analysis results

### physics_vs_spline/
- **Purpose**: Compare physics-guided low-rank factorization vs cubic spline interpolation
- **Key Results**: Physics method achieved 33.5% precision improvement over splines
- **Files**: Comparison plots, metrics JSON, training logs

### training_results/
- **Purpose**: Training outputs for physics-guided low-rank model
- **Model**: PhysicsLowRank5D (single probe, 170 params, rank=5)
- **Key Results**: 2.34× compression, 38.1 dB PSNR (Cornell Box)

### low_rank_analysis/
- **Purpose**: Additional low-rank decomposition analysis
- **Files**: Low-rank analysis outputs and visualizations

## Key Findings

1. **Low-Rank Hypothesis Confirmed**: Temporal SH dynamics lie on low-dimensional manifold
2. **Physics Basis Superior**: Trigonometric basis [cos(θ), sin(θ), 1] outperforms learned embeddings
3. **Compression-Quality Trade-off**: Modest compression (2.34×) with significant quality improvement (+33.5%)

## Related Documentation

- Report: [PG-GCPL/docs/reports/01_Physics_Low_Rank_Results.md](../../docs/reports/01_Physics_Low_Rank_Results.md)
- Original reports: Experiment Report 08

## Scripts

Training:
- `PG-GCPL/src/scripts/training/train_physics_low_rank.py`
- `PG-GCPL/src/scripts/training/train_physics_low_rank_proper.py`
- `PG-GCPL/src/scripts/training/train_5D_physics_vs_spline.py`

Analysis:
- `PG-GCPL/src/scripts/analysis/verify_5D_low_rank.py`
- `PG-GCPL/src/scripts/analysis/verify_low_rank.py`
