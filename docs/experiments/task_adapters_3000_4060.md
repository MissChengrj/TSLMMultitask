# Chronos2 task-adapter experiment (RTX 4060 Ti)

## Setup

- Commit: `104171a`
- Sources: ACARS and QAR, using the existing train/validation/test JSONL splits
- Tasks: forecast, interpolation, and anomaly detection
- Adapter: output-head LoRA (`r=4`, `alpha=8`, dropout `0.1`)
- Optimization: 3,000 steps, batch size 2, learning rate `5e-7`
- Stability: normalized values clipped to 50; non-finite gradients sanitized before clipping
- Forecast adapters optimize forecast loss only. Interpolation and anomaly adapters optimize masked reconstruction only.
- Anomaly fusion weight is selected on validation AUROC and evaluated once on the test split.

## Training

| Source | Task | Train loss | Runtime (s) |
| --- | --- | ---: | ---: |
| ACARS | Forecast | 13.8236 | 190.7 |
| ACARS | Interpolation | 1.4438 | 193.4 |
| ACARS | Anomaly detection | 1.5126 | 192.6 |
| QAR | Forecast | 19.1265 | 213.0 |
| QAR | Interpolation | 1.7559 | 214.4 |
| QAR | Anomaly detection | 1.7929 | 205.8 |

All six runs completed without non-finite gradient warnings or GPU memory failures.

## Test results

### Forecast

| Source | Model | MSE | MAE |
| --- | --- | ---: | ---: |
| ACARS | Chronos2 base | 18,143.60 | 11.7217 |
| ACARS | Multitask adapter | 18,143.65 | 11.7217 |
| ACARS | Forecast adapter | 18,143.62 | 11.7217 |
| ACARS | Linear trend | **15,486.23** | **11.0778** |
| QAR | Chronos2 base | 3,548,650.63 | 90.7153 |
| QAR | Multitask adapter | 3,548,658.73 | 90.7181 |
| QAR | Forecast adapter | 3,548,662.83 | 90.7166 |
| QAR | Last value | **3,006,670.51** | **73.9110** |

### Interpolation

| Source | Model | MSE | MAE |
| --- | --- | ---: | ---: |
| ACARS | Chronos2 base | **3.7408** | 0.8729 |
| ACARS | Multitask adapter | 3.7408 | 0.8729 |
| ACARS | Interpolation adapter | 3.7408 | 0.8729 |
| ACARS | Linear interpolation | 4.2477 | **0.7682** |
| QAR | Chronos2 base | 9,177,819.80 | 327.5871 |
| QAR | Multitask adapter | 9,177,824.54 | 327.5860 |
| QAR | Interpolation adapter | 9,177,864.78 | 327.5875 |
| QAR | Linear interpolation | **421,748.28** | **22.8067** |

### Anomaly detection

Chronos2 reconstruction errors are robustly scaled per channel before aggregation. This differs from the earlier raw-error metric and makes channels with different physical units comparable.

| Source | Model | F1 at top-k | AUROC | Chronos weight |
| --- | --- | ---: | ---: | ---: |
| ACARS | Chronos2 base | 0.6025 | 0.9792 | - |
| ACARS | Anomaly adapter | 0.6025 | 0.9792 | - |
| ACARS | Adapter + rolling fusion | 0.6561 | 0.9849 | 0.8 |
| ACARS | Rolling z-score | **0.8424** | **0.9951** | - |
| QAR | Chronos2 base | 0.5234 | 0.9671 | - |
| QAR | Anomaly adapter | 0.5235 | 0.9672 | - |
| QAR | Adapter + rolling fusion | 0.7607 | 0.9837 | 0.5 |
| QAR | Rolling z-score | **0.8661** | **0.9963** | - |

## Decision

Task-specific output-head LoRA is numerically stable but does not materially change Chronos2 predictions or reconstructions. It should remain a reproducible stable baseline, not the main improvement claim.

For the next experiment:

1. Keep per-source and per-task checkpoints and robust per-channel anomaly scaling.
2. Use rolling z-score as the primary anomaly baseline; retain fusion as an ablation, since it does not beat rolling z-score here.
3. For QAR forecasting and interpolation, add channel-aware scaling and evaluate metrics per channel before aggregation.
4. Increase trainable capacity through the output head and feed-forward blocks while keeping attention LoRA disabled until its non-finite gradients are resolved.
5. Tune on validation metrics with early stopping instead of extending output-head-only training beyond 3,000 steps.

Full machine-readable outputs are stored on the training host at `results/task_adapters_3000_stable_4060/`.
