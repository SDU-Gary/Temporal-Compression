# Loss-Falcor Lag + Partial Correlation Analyzer

## Purpose
Analyze whether training metrics are related to Falcor HDR PSNR with:
- **Lagged correlation** (metric leads/lags Falcor by N epochs)
- **Partial correlation** (control for confounders such as epoch/routing temp/handover progress)

## Quick Start

Single experiment:

```bash
python tools/analysis/analyze_loss_falcor_lag_partial.py \
  --experiment-dir 3_experiments/results/bistro_clean_v2/exp_A2_2_A_temp035_seed19 \
  --include-all-numeric \
  --control-metric train_routing_temp \
  --control-metric train_proxy_gbuffer_handover_progress \
  --max-lag-epochs 40 \
  --include-stage-end-full \
  --output-dir 3_experiments/results/loss_falcor_correlation_lag_partial/a2_2_a
```

Multiple experiments:

```bash
python tools/analysis/analyze_loss_falcor_lag_partial.py \
  --experiments-root 3_experiments/results/bistro_clean_v2 \
  --experiment-glob 'exp_A2_2_*' \
  --include-all-numeric \
  --control-metric train_routing_temp \
  --control-metric train_proxy_gbuffer_handover_progress \
  --control-metric train_param_update_ratio_shared \
  --max-lag-epochs 40 \
  --output-dir 3_experiments/results/loss_falcor_correlation_lag_partial/a2_2_all
```

## Outputs
Per experiment:
- `loss_falcor_lag_partial_correlation.csv`: all metrics × all lags results
- `loss_falcor_partial_lag0.csv`: lag=0 subset ranked by `abs(partial_pearson)`
- `loss_falcor_partial_best_lag_per_metric.csv`: best lag per metric by `abs(partial_pearson)`
- `aligned_dense_epoch_metrics.csv`: dense aligned timeline used by analysis
- `summary.json`: concise top-k summary and control coverage

Global:
- `summary_all.json`
- `summary_all.csv`
