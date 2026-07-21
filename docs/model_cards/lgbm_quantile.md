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

### Without TSO feature (12-month test)

| model | MAPE | MAE (MW) | skill vs naive | pinball p10/p50/p90 |
|---|---|---|---|---|
| TSO (benchmark) | 2.31% | 418 | 0.59 | – / 209 / – |
| **lgbm_quantile** | **3.16%** | **579** | **0.43** | 170 / 290 / 203 |
| ridge | 4.03% | 718 | 0.30 | 169 / 359 / 179 |
| seasonal naive | 5.60% | 1025 | 0.00 | 354 / 513 / 282 |

Source: `reports/backtests/2026-07-14_fcst_summary.csv`.

### With TSO feature — lgbm_tso (12-month test)

Adding the TSO as a feature turns LightGBM into a forecast combiner.

| model | MAPE | MAE (MW) | skill vs naive | pinball p10/p50/p90 |
|---|---|---|---|---|
| ridge_tso | 2.13% | 383 | 0.63 | 92 / 192 / 92 |
| **lgbm_tso** | **2.16%** | **394** | **0.62** | 111 / 197 / 120 |
| TSO alone | 2.31% | 418 | 0.59 | – / 209 / – |

Source: `reports/backtests/2026-07-15_fcst_tso_summary.csv`.

LightGBM+TSO is 0.03 pp behind ridge+TSO on MAPE and 11 MW behind on MAE.
The gap is within noise for a 12-month window. See the 2-year backtest
(`reports/backtests/<date>_2yr_summary.csv`) for a more stable estimate.

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

**dev.** UAT challenger candidate (behind ridge_tso in the 12-month ranking,
within noise in 2-year). If ridge_tso fails promotion for any operational
reason, lgbm_tso is the next candidate.
