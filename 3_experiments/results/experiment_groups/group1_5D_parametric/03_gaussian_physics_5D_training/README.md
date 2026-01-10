# Phase 03: Gaussian-Physics 5D Training

**Timeline**: Week 5
**Goal**: Scale from single-probe to multi-probe using Gaussian mixture representation

## Experiment Structure

### training_logs/
- **Purpose**: Full training run for GaussianPhysics5D model
- **Model**: K=30 Gaussians, rank=8, 5D parametric space
- **Parameters**: 8,340 total (138,375 → 8,340 = 16.59× compression)
- **Training Duration**: ~906 epochs
- **Files**:
  - `best_model.pt` - Best checkpoint
  - `training_curve.png` - Loss curves
  - `training_results.json` - Final metrics

### gaussian_physics/
- **Purpose**: Gaussian-physics hybrid experiments
- **Variants**: Different K (Gaussian count) and rank configurations
- **Files**: Multiple training runs with varying hyperparameters

## Model Architecture

```
SH(probe_pos, light_params) = Σ_j G_j(probe_pos) × [U_j @ (coeffs_j @ Φ_physics(light_params))]
```

Where:
- `G_j(probe_pos)`: Gaussian weight for probe position
- `U_j`: Low-rank spatial basis (27 × rank)
- `coeffs_j`: Time mixing coefficients (rank × 5)
- `Φ_physics(light_params)`: 5D physics basis [cos(θ), sin(θ), cos(φ), sin(φ), 1]

## 5D Parametric Space

1. **Sun Zenith** (θ): Solar elevation angle
2. **Sun Azimuth** (φ): Solar horizontal angle
3. **Intensity**: Light intensity scale
4. **Color Temperature**: Correlated color temperature
5. **Cloud Cover**: Atmospheric attenuation factor

## Key Results

- **Compression**: 16.59× (138,375 params → 8,340 params)
- **Quality**: Maintains high PSNR across 125 probes × 7 light configs
- **Generalization**: Smooth interpolation in 5D parameter space

## Related Documentation

- Report: [PG-GCPL/docs/reports/02_Gaussian_Physics_Hybrid.md](../../docs/reports/02_Gaussian_Physics_Hybrid.md)
- Report: [PG-GCPL/docs/reports/03_Future_Experiments.md](../../docs/reports/03_Future_Experiments.md)
- Original reports: Experiment Report 09, 10

## Scripts

Training:
- `PG-GCPL/src/scripts/training/train_gaussian_physics.py`
- `PG-GCPL/src/scripts/training/train_gaussian_physics_5D.py`
