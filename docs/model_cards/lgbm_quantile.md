# Model card — lgbm_quantile

## What it is

Three LightGBM boosters, one per quantile (P10/P50/P90), each trained
directly on pinball loss (`objective="quantile"`). File: `src/models/gbm.py`.
Quantile crossing removed by post-hoc clipping.

## Inputs

22 features per hour: load lags (48/72/168/336/504/672 h + 7-day mean),
calendar (hour, weekday, month, PL holidays, bridge days, cyclic encodings),
weather (temperature, wind, cloud, radiation, humidity + heating/cooling
degrees), population-weighted over 10 cities.
Does NOT see: lag 24 (post-cutoff = leakage), prices, neighbor countries.

## Training

Rolling 365-day window, refit every 7 days in walk-forward. 500 trees,
lr 0.05, 63 leaves, subsample 0.9. **Not tuned** — these are conservative
defaults; tuning only after this honest row landed.

## Performance (walk-forward, honest lead-2 forecast weather, 8762 h)

| model | MAPE | MAE (MW) | skill vs naive | pinball p10/p50/p90 |
|---|---|---|---|---|
| TSO (benchmark) | 2.31% | 418 | 0.59 | – / 209 / – |
| **lgbm_quantile** | **3.16%** | **579** | **0.43** | 170 / 290 / 203 |
| ridge | 4.03% | 718 | 0.30 | 169 / 359 / 179 |
| seasonal naive | 5.60% | 1025 | 0.00 | 354 / 513 / 282 |

Source: `reports/backtests/2026-07-14_fcst_summary.csv`.

Weak spots (see `reports/figures/backtest_mape_by_*.png`): midday hours
(ramp + peak), and the worst-day tail is still fatter than the TSO's.

## Interpretability

`reports/figures/shap_summary.png`. Top global drivers, plain words:
1. load last week at this hour, 2. load two weeks ago, 3. load four weeks ago.
Holidays cut the forecast by up to ~4 GW; cold (heating degrees) raises it.

## Known failure modes

- Holidays not seen in the 365-day window (rare bridge configurations).
- Weather regimes outside training range (extreme cold snaps).
- Trees cannot extrapolate above the highest load seen in training.

## Status

dev. UAT shadow-run candidate once M9 environments exist.
