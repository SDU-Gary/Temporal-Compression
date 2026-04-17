# Loss-Falcor Correlation Analyzer

## Purpose
Analyze optimization correlation between training/validation losses (from `runtime/phase0_metrics.jsonl`) and Falcor HDR PSNR (from `runtime/falcor_periodic_eval.jsonl`).

## Important Scope
- Old experiments: only metrics that were actually logged can be analyzed.
- New experiments (after `train.py` phase0 metric expansion): all numeric `train_metrics` / `val_metrics` are persisted and can be analyzed.

## Quick Start

Analyze one experiment:

```bash
python tools/analysis/analyze_loss_falcor_correlation.py \
  --experiment-dir 3_experiments/results/bistro_clean_v2/exp_A1_7_P1_corr_seed19 \
  --include-all-numeric \
  --include-stage-end-full \
  --output-dir 3_experiments/results/loss_falcor_correlation/a1_7_p1_check
```

Analyze multiple experiments:

```bash
python tools/analysis/analyze_loss_falcor_correlation.py \
  --experiments-root 3_experiments/results/bistro_clean_v2 \
  --experiment-glob 'exp_A1_*' \
  --include-all-numeric \
  --include-stage-end-full \
  --output-dir 3_experiments/results/loss_falcor_correlation/a1_all_snapshot
```

## Outputs
For each experiment:
- `loss_falcor_correlation.csv`: per-metric Pearson/Spearman correlations
- `phase0_epoch_metrics.csv`: parsed epoch metrics table
- `falcor_epoch_metrics.csv`: Falcor PSNR timeline
- `summary.json`: top positive/negative correlated metrics

Global:
- `summary_all.json`
- `summary_all.csv`
