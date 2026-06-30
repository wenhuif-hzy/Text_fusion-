# Solar Text Residual Forecasting

Current mainline: `FRTC-BER`

Full name: `Fuzzy Residual Text Correction with Budgeted Extreme-aware Release`

Code version: `frtc_ber_v53_daylight_context_protocol`

This project forecasts photovoltaic `OT` with a 15-minute numerical stream and hourly text streams. The current method is not the old step-by-step data stacking route. It uses a stable numerical foundation first, then lets text act as forecast-time residual evidence.

Default supervised targets cover `[08:00, 19:00)` at 15-minute resolution, so `pred_len=44`. The input context remains `seq_len=96`, i.e. a full 24-hour numerical/NWP history before the forecast origin. This lets the 08:00 forecast use early-morning context such as 06:00-07:45 without training or evaluating on nighttime targets.

## Project Structure

```text
text_fusion/
  train_solar.py              # Main training, evaluation, and paper-mode script
  data_interface.py           # Data loading and time alignment utilities
  enhanced_itransformer.py    # iTransformer backbone components
  text_generation_rules.py    # NWP-to-text rule generation utilities
  diagnose_datasets.py        # Dataset diagnostics
  datasets/                   # Multi-station datasets
  encoder/                    # Local text encoder
  cache/text/                 # Cached text embeddings
  results/main_results/       # Clean main results
```

Each dataset directory uses the same layout:

```text
datasets/<dataset_name>/
  solar.csv
  context_full_day.csv        # Optional 24h numeric/NWP context source
  rt_text1.jsonl
  rt_text2.jsonl
```

Current available datasets include `kongzhaopu`, `daapeng`, `yanfengdong`, `adong`, `aping`, `jiuzhai`, and `hebei_station00` to `hebei_station09`.

## Method Logic

The code follows one integrated framework:

1. Numerical foundation: an iTransformer-style numerical baseline uses past OT, auxiliary meteorology, past NWP, future NWP, periodic priors, and candidate residual branches.
2. Text1 realtime intervention: hourly realtime text is asynchronously aligned to 15-minute horizons and used as residual evidence over the numerical forecast.
3. Text2 forecast-time intervention: forecast-time trend text provides fuzzy risk, extreme-weather, ramp, and shock semantics.
4. Residual release: Text2 does not directly replace the prediction. It proposes bounded residual corrections, then a soft Pareto extreme-aware release policy balances global accuracy, high-residual correction, extreme/ramp gains, and normal-daylight stability.
5. Final forecast: `y_final = y_text1 + released_text2_residual`.

Main training selection and reported MSE/MAE are computed in normalized target
space. For multi-dataset runs, each station has its own target scaler, so
reported errors are comparable across stations and are not dominated by station
capacity. The default supervised forecast window is `[08:00, 19:00)` at
15-minute resolution, while the numerical context window remains 24 hours. If
`context_full_day.csv` exists for a station, it is used as the full-day context
source; `solar.csv` remains the supervised target source. Checkpoints are
accepted for reuse only when the dataset identity, `seq_len`, `pred_len`, and
prediction-window protocol match the current run.

Safety constraints:

```text
Text2 must be available at forecast issue time. If a Text2 file includes
`available_time`, `issue_time`, or `forecast_issue_time`, the loader masks text
that was not yet available at the forecast origin.
Validation residuals are used only for fitting fixed release/calibration profiles.
Test-time release uses observable text, NWP disagreement, fuzzy risk, ramp, shock, and coverage signals.
```

## Default Run

```bash
cd /home/fengwh/text_fusion
python train_solar.py --mode paper --calibration_search --device cuda --batch_size 64 --num_workers 4 --save_arrays
```

With the default dataset `kongzhaopu`, outputs are written to:

```text
results/main_results/kongzhaopu/
```

Main subdirectories:

```text
00_numeric_baseline/          # Numerical foundation
01_realtime_text1/            # Realtime Text1 residual intervention
02_text2_soft_pareto_best/    # Final selected Text2 result
release_scale_sweep/          # Diagnostic release-scale comparison
diagnostics/                  # Diagnostic safety variants
```

## Current Main Result

See:

```text
results/main_results/kongzhaopu/result_summary.csv
results/main_results/kongzhaopu/result_summary.json
results/main_results/kongzhaopu/README.md
```

Historical cleaned mainline on `kongzhaopu` from the archived real-scale run:

```text
Numerical baseline MSE: 0.350361
Text1 MSE:              0.340564
Final Text2 MSE:        0.334160
Final MAE:              0.297378
Gain vs numeric MSE:    0.016289
Text2 gain vs Text1:    0.006676
Extreme-union gain:     0.026479
High-residual gain:     0.040396
Normal daylight degrade:0.002081
Selected policy:        soft_pareto_scale_090
```

## Multi-Dataset Run

Use comma-separated dataset names under `datasets`:

```bash
python train_solar.py \
  --mode paper \
  --datasets daapeng,jiuzhai,yanfengdong,hebei_station04,hebei_station05 \
  --out_dir ./results/main_results/multi_dataset_5_balanced \
  --calibration_search \
  --device cuda \
  --batch_size 64 \
  --num_workers 4 \
  --save_arrays
```

Multi-dataset mode keeps each station as an independent chronological time series. Feature and NWP scalers are fitted on the union of all training partitions, while the target scale is fitted per station so that large-capacity stations do not dominate training or normalized evaluation. In `paper` mode, multi-dataset runs train their own numerical baseline and Text1 checkpoint for the current dataset combination instead of reusing single-station checkpoints.

For multi-dataset numerical training, the robust foundation profile is selected automatically. It enables history/NWP/periodic numerical priors, balanced station sampling, dataset-conditioned residual release, and validation-fitted candidate pooling. When several stations are used together, the candidate pool can fit station-specific horizon weights, so a branch that helps one station is not forced onto every station.

Avoid mixing abnormal-scale stations such as `adong`/`aping` with ordinary stations in the first validation run. Add them later as a separate scale group or after confirming the balanced group is stable.

Before running full text experiments on a new station, use:

```bash
python diagnose_datasets.py --out ./results/dataset_diagnostics.csv
```

This summarizes target scale, daylight volatility, ramp rate, and Text1/Text2 template diversity.
