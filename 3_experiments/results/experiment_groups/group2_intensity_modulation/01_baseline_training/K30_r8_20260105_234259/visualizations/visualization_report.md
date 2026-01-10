# Visualization Report: 1D Intensity Modulation Baseline (K30_r8)

**Generated**: 2026-01-06 11:17:28
**Experiment**: K30_r8_20260105_234259
**Best Epoch**: 1997

---

## Summary Metrics

| Metric | Value |
|--------|-------|
| **Test MAE** | 0.7241 |
| **Test RMSE** | 1.4145 |
| **Per-coeff MAE (mean)** | 0.5137 |
| **Per-coeff MAE (max)** | 0.6060 |
| **Compression Ratio** | 15.56× |
| **Gaussian Coverage** | 100% |
| **Training Loss Reduction** | 68.50% |

---

## Visualization Files

1. **`1_training_curves.png`** - Training loss, validation MAE/RMSE, per-coefficient MAE, Gaussian coverage (6 subplots)
2. **`2_convergence_analysis.png`** - Early/mid/late phase analysis + log scale view (4 subplots)
3. **`3_gt_vs_pred_rendering.png`** - Environment map rendering comparison: GT | Pred | Diff | Metrics (12 samples)
4. **`4_error_distribution.png`** - Error histogram, absolute error, scatter plot, per-coefficient boxplot

---

## Key Findings

### Training Convergence
- Initial train loss: 1.1669
- Final train loss: 0.3676
- Loss reduction: **68.5%**
- Best validation MAE: **0.7817** @ epoch **1997**

### Error Statistics (12 samples)
- Mean error: 0.009205 (near-zero bias)
- Std error: 0.1546
- Mean absolute error: 0.0997
- Median absolute error: 0.0577
- Max absolute error: 0.6622

### Rendering Quality
- GT vs Pred environment maps show **high visual similarity**
- Difference maps indicate errors concentrated in **low-radiance regions**
- All intensity levels (0.250 - 1.000) reconstructed consistently

---

## Conclusion

✅ **Excellent baseline performance**:
- Test MAE < 1.0 indicates strong reconstruction quality
- Compression ratio 15.56× with minimal quality loss
- 100% Gaussian coverage ensures no spatial blind zones
- Error distribution centered at zero (unbiased prediction)

🔄 **Next Steps**:
- Extend to 2D: intensity + color temperature modulation
- Compare with 5D parametric baseline
- Test on larger datasets (>343 probes)
