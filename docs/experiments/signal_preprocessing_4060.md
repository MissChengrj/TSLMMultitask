# Signal-aware preprocessing experiment (RTX 4060 Ti)

## Question

The large aggregate MAE, especially on QAR, initially suggested that random
high-frequency noise might dominate the data. The measured signal statistics do
not support that explanation:

- Median lag-1 correlation is `0.954` for QAR and `0.879` for ACARS.
- The median isolated-spike rate and estimated noise floor are both zero.
- Most channels are smooth or piecewise smooth. Large errors are concentrated in
  a few channels with encoding discontinuities or invalid values.

On the raw QAR test forecast split, persistence-error shares are `51.7%` for
`ALT`, `18.1%` for `FF2`, and `18.0%` for `FF1`. On interpolation they are
`59.2%`, `25.1%`, and `7.4%`, respectively. These three channels therefore
explain about `87.8%` of forecast error and `91.7%` of interpolation error.

## Root causes

1. `ALT` contains 16-bit wrap discontinuities, for example values near `-61445`
   adjacent to `4103`. The physical difference is small after adding/subtracting
   the period `65536`, but a linear metric sees an error near `65536`.
2. `FF1` and `FF2` contain invalid codes such as `12288`, well outside the normal
   fuel-flow range.
3. `FAN_IMB_ANG_*` and `LPT_IMB_ANG_*` are circular angles. Crossing `359/0`
   is continuous physically but discontinuous in the raw numeric representation.
4. ACARS has much fewer encoding artifacts. Its forecast error mainly reflects
   real operating-phase transitions and large-scale dynamic channels, so broad
   smoothing would remove useful fault information.

## Preprocessing

The implementation is `scripts/datasets/analyze_and_preprocess_jsonl.py`. It
creates a separate split and leaves the raw JSONL files unchanged.

- Unwrap QAR altitude with period `65536` and map it to a plausible physical
  range.
- Unwrap QAR imbalance angles with period `360`.
- Mask physically invalid values for altitude, speed, Mach, fuel flow, EGT, and
  spool-speed channels, then interpolate only those invalid samples.
- Replace isolated one-sample impulses with a 3-point median only when the local
  deviation exceeds four robust scales. Steps lasting at least two samples are
  retained.
- Transform forecast context and target in one coordinate system without using
  target statistics. Preserve interpolation masks. Preprocess anomaly context,
  then reapply the synthetic anomaly delta so labels and injected faults remain.

On the remote split, QAR invalid-value rates are `0.71%` to `1.03%` for forecast
and `0.56%` for test interpolation. Only about `0.05%` to `0.07%` of QAR points
are treated as isolated impulses. ACARS impulse correction is approximately
`0.03%` to `0.17%` and no ACARS values are removed by the QAR physical ranges.

## Training

All clean-data adapters use output-head LoRA (`r=4`, `alpha=8`, dropout `0.1`),
batch size 2, learning rate `5e-7`, and 3,000 steps.

| Source | Task | Train loss | Runtime (s) |
| --- | --- | ---: | ---: |
| ACARS | Forecast | 13.9917 | 205.3 |
| ACARS | Interpolation | 1.4636 | 204.5 |
| QAR | Forecast | 19.1481 | 224.4 |
| QAR | Interpolation | 1.7537 | 227.9 |

All four output-head runs completed without non-finite gradients. A separate
QAR feed-forward-plus-output LoRA smoke test was stopped because exactly
`368,640` feed-forward adapter gradients became non-finite on every step. The
output-head path remains the only stable adapter path, but it has insufficient
capacity to change predictions materially.

## Results

Each row uses the same first 300 test records before and after preprocessing.
`sMAPE` is included because aggregate MAE is dominated by high-unit channels.

### Forecast

| Source | Model | Raw MAE | Clean MAE | Raw sMAPE | Clean sMAPE |
| --- | --- | ---: | ---: | ---: | ---: |
| QAR | Chronos2 base | 90.7153 | **12.4313** | 0.1412 | 0.1179 |
| QAR | Forecast adapter | 90.7166 | **12.4313** | 0.1412 | 0.1179 |
| QAR | Last value | 73.9110 | 12.8442 | 0.1246 | **0.1047** |
| QAR | Transformer | 205.5196 | 18.3193 | 0.2166 | 0.1684 |
| QAR | LSTM | 347.8625 | 28.9179 | 0.2311 | 0.2084 |
| ACARS | Chronos2 base | 11.7217 | 11.7219 | 0.1046 | 0.1046 |
| ACARS | Forecast adapter | 11.7217 | 11.7218 | 0.1046 | 0.1046 |
| ACARS | Linear trend | **11.0778** | **11.0778** | 0.1404 | 0.1404 |

QAR preprocessing reduces Chronos2 MAE by `86.3%`. Last-value forecasting still
has the best QAR sMAPE, while Chronos2 has the best QAR MAE after invalid-value
correction. ACARS is effectively unchanged, as expected.

### Interpolation

| Source | Model | Raw MAE | Clean MAE | Raw sMAPE | Clean sMAPE |
| --- | --- | ---: | ---: | ---: | ---: |
| QAR | Chronos2 base | 327.5871 | 35.1615 | 0.1857 | 0.1577 |
| QAR | Interpolation adapter | 327.5875 | 35.1615 | 0.1857 | 0.1577 |
| QAR | Linear interpolation | 22.8067 | **2.1619** | 0.0351 | **0.0258** |
| QAR | Transformer AE | 32.6566 | 2.9039 | 0.0673 | 0.0587 |
| QAR | LSTM AE | 103.6831 | 9.9411 | 0.1215 | 0.0897 |
| ACARS | Chronos2 base | 0.8729 | 0.8705 | 0.3208 | 0.3115 |
| ACARS | Interpolation adapter | 0.8729 | 0.8705 | 0.3208 | 0.3115 |
| ACARS | Linear interpolation | **0.7682** | **0.7656** | 0.2381 | 0.2361 |

QAR preprocessing reduces Chronos2 interpolation MAE by `89.3%`, but linear
interpolation remains decisively better. ACARS changes are small (`0.3%` MAE for
linear interpolation), confirming that stronger smoothing is not justified.

### Anomaly detection

Preprocessing does not uniformly improve anomaly ranking. QAR Chronos2 F1 rises
from `0.5234` to `0.5672`, while its rolling-hybrid F1 falls from `0.7608` to
`0.7246`; rolling z-score changes from `0.8661` to `0.8605`. ACARS changes are
minor. Raw and physical-coordinate views should therefore both be retained for
anomaly detection: raw-code anomalies can be operationally meaningful, while
physical coordinates avoid false wrap-boundary alarms.

## Decisions and next experiments

1. Use the signal-clean split for QAR forecasting and interpolation, but retain
   the raw split as an immutable audit source.
2. Do not apply global low-pass smoothing. Keep the current isolated-impulse
   rule and explicitly preserve sustained steps, trends, and anomaly deltas.
3. Report MAE per channel, macro sMAPE, and aggregate MAE together. Aggregate MAE
   alone is not a reliable cross-channel model-selection metric.
4. Use linear interpolation as the production interpolation baseline. Treat
   Chronos2 interpolation as an ablation until it beats this baseline.
5. For anomaly detection, fuse features from both raw and physical-coordinate
   representations and select thresholds on validation data only.
6. Output-head LoRA should remain a stable baseline, not the main fine-tuning
   method. Before training larger adapters, isolate the feed-forward non-finite
   backward path with per-layer gradient hooks and an FP32 adapter computation
   experiment. Do not simply increase steps on the current output head.

Machine-readable outputs are on the training host under
`results/signal_preprocessing_*_4060/`. Clean checkpoints are under
`weights/chronos2_jsonl_{source}_{task}_signal_clean_output_3000/checkpoint-final`.
