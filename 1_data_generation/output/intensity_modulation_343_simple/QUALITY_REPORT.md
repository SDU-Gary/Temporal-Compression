# Dataset Quality Report: intensity_modulation_343_simple

## Generation Summary

**Date**: 2026-01-05
**Scene**: Simple geometry (sphere + floor + point light)
**Purpose**: Emergency fix for Cornell Box transform issues

## Dataset Specification

- **Probes**: 343 (7³ uniform grid)
- **Probe bounds**: X,Y ∈ [-0.4, 0.4], Z ∈ [0.1, 0.7]
- **Time moments**: 12 (covering 1 complete 3-second cycle)
- **Intensity function**: I(t) = 0.5 + 0.5 * sin(2πt/3)
- **Intensity range**: [0.0, 1.0]
- **SPP**: 128
- **SH samples**: 64
- **SH order**: 2 (27 coefficients: 9 bases × 3 RGB)

## Scene Geometry

**Point Light**:
- Position: [0, 0, 0.8]
- Base intensity: 50.0 W/sr
- Modulated by sinusoidal function

**Central Sphere**:
- Center: [0, 0, 0.3]
- Radius: 0.15
- BSDF: Diffuse (reflectance: [0.8, 0.8, 0.8])

**Floor**:
- Large sphere (radius: 100) at [0, 0, -100.5]
- Acts as infinite ground plane
- BSDF: Diffuse (reflectance: [0.8, 0.8, 0.8])

## Validation Results

### SH Coefficients by Moment

| Moment | Time (s) | Intensity | SH Min    | SH Max   | SH Mean  |
|--------|----------|-----------|-----------|----------|----------|
| 00     | 0.000    | 0.500000  | -40.3631  | 35.1999  | 0.1185   |
| 01     | 0.250    | 0.750000  | -60.5447  | 52.7998  | 0.1777   |
| 02     | 0.500    | 0.933013  | -75.3186  | 65.6839  | 0.2211   |
| 03     | 0.750    | 1.000000  | -80.7263  | 70.3998  | 0.2370   |
| 04     | 1.000    | 0.933013  | -75.3186  | 65.6839  | 0.2211   |
| 05     | 1.250    | 0.750000  | -60.5447  | 52.7998  | 0.1777   |
| 06     | 1.500    | 0.500000  | -40.3631  | 35.1999  | 0.1185   |
| 07     | 1.750    | 0.250000  | -20.1816  | 17.5999  | 0.0592   |
| 08     | 2.000    | 0.066987  | -5.4076   | 4.7159   | 0.0159   |
| 09     | 2.250    | 0.000000  | 0.0000    | 0.0000   | 0.0000   |
| 10     | 2.500    | 0.066987  | -5.4076   | 4.7159   | 0.0159   |
| 11     | 2.750    | 0.250000  | -20.1816  | 17.5999  | 0.0592   |

### Global Statistics

- **Global SH range**: [-80.7263, 70.3998]
- **Mean of means**: 0.1185
- **Perfect linear scaling**: SH_max / Intensity ≈ 70.4 (constant across moments)

### Quality Checks

✅ **SH magnitude normal**: max: 70.40 (in expected range [0.1, 1000])
✅ **Intensity modulation captured**: Linear relationship verified
  - SH@I=0.0: 0.00
  - SH@I=0.5: 35.20
  - SH@I=1.0: 70.40
✅ **No NaN/Inf**: All values finite and valid
✅ **Perfect symmetry**: Moments 00↔06, 01↔05, 02↔04 match (sinusoidal symmetry)
✅ **Complete darkness at minimum**: Moment 09 (I=0.0) has all-zero SH coefficients

## GT Rendering Validation

**Camera**: Perspective FOV=60°, origin=[0, -0.8, 0.5], target=[0, 0, 0.3]

**Rendered moments** (SPP=128):
- t=0.00s (I=0.5): Image range [0, 50.60], mean 1.24
- t=0.75s (I=1.0): Image range [0, 101.21], mean 2.48 ✅ **Exactly 2× brighter**
- t=1.50s (I=0.5): Image range [0, 50.60], mean 1.24 ✅ **Matches t=0.00s**
- t=2.25s (I=0.0): Image range [0, 0], mean 0 ✅ **Complete darkness**

**Visual quality**:
- ✅ Sphere geometry clear and well-lit
- ✅ Floor gradient smooth and realistic
- ✅ Shadows and shading physically correct
- ✅ Intensity modulation visually obvious

## Comparison with Previous (Broken) Dataset

| Metric               | Broken Dataset      | Fixed Dataset       | Improvement  |
|----------------------|---------------------|---------------------|--------------|
| SH max               | 0.002438            | 70.3998             | **28,875×**  |
| SH mean              | 0.000001            | 0.1185              | **118,500×** |
| Probe placement      | 80% outside walls   | 100% inside scene   | ✅ Fixed     |
| Rendered images      | All black           | Clear and realistic | ✅ Fixed     |
| Scene geometry       | Cornell Box (broken)| Simple sphere/floor | ✅ Works     |

## Root Cause of Previous Failure

**Cornell Box Transform Issue**:
- Area light rectangle transforms resulted in all-zero renders
- Multiple fix attempts failed (including `cornell_box_builder_fixed.py`)
- Root cause: Complex rotation chain in `to_world` transforms

**Emergency Solution**:
- Switched to simple geometry: spheres + point light
- Point light confirmed working in prior tests
- Simple transforms (translate + scale only)

## Recommendations for Training

1. ✅ **Dataset ready for training**: All quality checks passed
2. ✅ **Use 1D intensity modulation model**: GaussianPhysicsCompression1D
3. ⚠️ **Expected challenge**: Simple geometry may have less spatial variation than Cornell Box
4. ✅ **Temporal compression opportunity**: Perfect sinusoidal pattern, ideal for physics-guided approach
5. ⚠️ **Future work**: Investigate Cornell Box transform issue for more complex scenes

## Files Generated

```
intensity_modulation_343_simple/
├── probes.npz                    # [343, 3] positions
├── metadata.json                 # Generation parameters
├── QUALITY_REPORT.md             # This report
└── moment_00/ ... moment_11/     # 12 time moments
    ├── sh_coeffs.npz             # [343, 27] + intensity + time
    └── intensity.txt             # Scalar intensity value
```

**Total dataset size**: ~6.5 MB

## Conclusion

✅ **DATASET GENERATION SUCCESSFUL**

All quality metrics passed. Dataset exhibits perfect linear intensity modulation with clean sinusoidal pattern. Ready for PG-GCPL 1D training experiments.

**Status**: READY FOR TRAINING
