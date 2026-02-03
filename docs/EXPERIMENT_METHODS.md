# Experiment Methods (Draft)

This document records the planned experimental methodology for the probe sampling and FiLM ablation studies. It is intended as a living reference for the paper write-up and future implementation.

---

## 1. Goals

- Evaluate the impact of **probe sampling strategy** (grid vs. adaptive 70/30).
- Evaluate the impact of **FiLM** (ON vs. OFF) under identical data conditions.
- Provide **high-confidence** quantitative conclusions with reproducible splits and statistics.
- Include **qualitative evidence** (residual maps / slices) to support numerical results.

---

## 2. Datasets

### 2.1 Grid Sampling (Baseline)
- Probe positions from a uniform grid inside the scene bounds.
- If `max_probes` is set, use auto grid resolution and randomized subsampling.
- Probe relocation is applied to avoid obstacles.

### 2.2 Adaptive Sampling (70/30)
- 70% probes from uniform grid distribution (global coverage).
- 30% probes sampled near obstacle AABB surfaces, offset in `[0.05, 0.5]`.
- Probe relocation is applied to avoid obstacles.
- Invalid probes (cannot be relocated) are flagged in `valid_mask` and filtered in training.

### 2.3 Dataset Parameters (Target)
- `num_configs = 200`
- `num_probes = 128`
- Cubemap mode: `cube_res = 32`, `num_frames = 32`
- Fixed seed: `FALCOR_FIXED_SEED = 1`

---

## 3. Model Variants

We evaluate a 2x2 ablation matrix:

| Group | Probe Sampling | FiLM |
|------|----------------|------|
| A | Grid | OFF |
| B | Adaptive 70/30 | OFF |
| C | Grid | ON |
| D | Adaptive 70/30 | ON |

---

## 4. Data Splits and Reproducibility

### 4.1 Primary Split (Default)
- Train / Val / Test = 70% / 15% / 15%
- Fixed random seed for split reproducibility.

### 4.2 Cross-Validation (High-Confidence)
- Repeat training and evaluation with **3 different random seeds** for splits.
- Report **mean ± std** across seeds for all metrics.

---

## 5. Evaluation Metrics

### 5.1 Global Metrics
- **MAE**, **RMSE** over all probes/configs.
- **L0 RMSE** (DC component).
- **HO RMSE** (higher-order terms).

### 5.2 Spatial Specificity (Near vs. Open)
Split probes into groups based on distance to nearest obstacle AABB:
- **Near-Surface**: distance < 0.2
- **Open-Space**: distance ≥ 0.2

Report:
- **Near RMSE**
- **Open RMSE**

### 5.3 Angular & Structural Fidelity
We compute:
- **Angular Error** (degrees) based on L1 SH coefficients.
  - Extract a direction vector from L1 RGB coefficients.
  - Compute mean angular error between prediction and GT.
- **Shadow Gradient RMSE**
  - Compute spatial gradient magnitude of L0-luma across probe neighbors.
  - Compare gradient magnitude between prediction and GT.

---

## 6. Qualitative Diagnostics

### 6.1 Residual Visualization
Produce residual slice maps using `probe_field_slicer.py`:
- `Residual = Pred - GT`
- Focus on coefficients `L1` (indices 1,2,3).
- Compare FiLM ON vs OFF near shadow boundaries.

### 6.2 SH Lobe Visualization
Use Polyscope to compare:
- Grid dataset vs Adaptive dataset.
- FiLM OFF vs FiLM ON (qualitative inspection).

---

## 7. OOD Generalization Test (Optional, High Value)

Define a region in lamp position space that is **excluded** from training:
- Example: hold out configs where lamp position falls in a corner subvolume.

Train on remaining configs and report:
- **OOD RMSE / MAE**
- Compare FiLM ON vs OFF.

---

## 8. Reporting Format

### 8.1 Table (LaTeX / CSV)
Generate a table with the following columns:

| Group | RMSE | MAE | L0 RMSE | HO RMSE | Near RMSE | Open RMSE | Angular (deg) | Grad RMSE |

### 8.2 Statistical Reporting
For each metric, report:
- mean ± std across seeds
- best/worst case (optional)

---

## 9. Implementation Status

### Completed
- Dataset generation (grid / adaptive)
- Probe relocation + valid mask
- FiLM ablation runner
- Metrics analysis (global / near / angular / gradient)
- CSV + LaTeX output

### Planned
- Multi-seed cross-validation wrapper
- Residual visualization automation
- OOD split and evaluation script

---

## 10. Notes

- Very small config counts (e.g., 5) are useful for debugging but **not** reliable for final claims.
- Ensure validation split is non-empty (avoid 0 val configs).
- Confirm that `best_model.pt` is meaningful (val > 0).
