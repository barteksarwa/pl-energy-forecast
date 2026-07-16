# Model card — ridge_tso

## What it is

Ridge regression (L2 penalty, α=1.0) with the TSO day-ahead load forecast
added as a feature. Makes this a **forecast combiner**: the model learns to
correct the TSO's systematic errors using load history, weather, and calendar.
File: `src/models/baselines.py` (`RidgeForecaster`), TSO passed via
`build_features(..., tso=tso)` in `src/features/matrix.py`.

## Inputs

25 features per hour:
- **Load lags** (48/72/168/336/504/672 h + 7-day mean): history known by 09:00 D-1.
- **Calendar** (hour, weekday, month, PL holidays, bridge days, cyclic sin/cos): pure calendar.
- **Weather** (temperature, wind, cloud, radiation, humidity + heating/cooling degrees): population-weighted over 10 Polish cities. Lead-2 archived forecast in evaluation; ERA5 actuals in training.
- **TSO day-ahead forecast** (MW): PSE publishes at ~09:00:12 D-1, accepted at our cutoff (DECISIONS 2026-07-15).

Does NOT see: lag 24 h (leakage), prices, neighbor-country load, unit outages.

## Training

Rolling 365-day window. Refit every 7 days (walk-forward, like a real desk).
`StandardScaler` inside `Pipeline`. L2 α=1.0 fixed (not tuned).
Residual-quantile band: P10/P90 = point ± empirical 10th/90th percentile
of in-sample residuals from that window. Simple; proper quantile regression
arrives with LightGBM.

## Performance (walk-forward, honest lead-2 weather)

### 12-month test (2025-07-13 → 2026-07-13, 8762 h)

| model | MAPE | MAE (MW) | skill vs naive | pinball P10 / P50 / P90 |
|---|---|---|---|---|
| **ridge_tso** | **2.13%** | **383** | **0.63** | 92 / 192 / 92 |
| lgbm_tso | 2.16% | 394 | 0.62 | 111 / 197 / 120 |
| TSO alone | 2.31% | 418 | 0.59 | – / 209 / – |
| lgbm (no TSO) | 3.16% | 579 | 0.43 | 170 / 290 / 203 |
| ridge (no TSO) | 4.03% | 718 | 0.30 | 169 / 359 / 179 |
| seasonal naive | 5.60% | 1025 | 0.00 | 354 / 513 / 282 |

Source: `reports/backtests/2026-07-15_fcst_tso_summary.csv`.

### 2-year test (2024-07-16 → 2026-07-14, 17450 h)

| model | MAPE | MAE (MW) | skill vs naive |
|---|---|---|---|
| **ridge_tso** | **2.08%** | **374** | **0.63** |
| lgbm_tso | 2.12% | 384 | 0.62 |
| TSO alone | 2.23% | 401 | 0.60 |
| ridge (no TSO) | 4.05% | 710 | 0.29 |
| seasonal naive | 5.59% | 1005 | 0.00 |

Source: `reports/backtests/2026-07-16_2yr_summary.csv`.
Hybrid weather: ERA5 pre-2024 (training), lead-2 forecast 2024+ (test).

The 2-year window is more stable than 12-month. ridge_tso beats lgbm_tso
by 0.04 pp and the TSO alone by 0.15 pp. Both TSO-combiner models beat the
standalone TSO, confirming the combination gain is real and persistent.

Weak spots: worst-day tail (largest misses on anomalous holidays and during
sudden weather pattern changes at the boundary of the 365-day window). The
TSO already models most of that signal; ridge learns the residual, which is
nearly linear.

## Interpretability

Coefficients are the explanation — no SHAP needed. Post-fit `coef_` shows:
1. `tso_forecast_mw` dominates (weight ~0.9–1.1, varies by window).
2. `load_lag_168h` (same hour last week) is the second strongest.
3. Calendar features (month, hour) pick up systematic TSO bias by period.

The model is essentially: *forecast ≈ 1 × TSO + small calendar + small recency correction*. A manager or regulator can verify this with a single `print(model._pipe['est'].coef_)` call.

## Known failure modes

- TSO unpublished at cron time (05:30 UTC ≈ 07:30 Warsaw, before TSO's 09:00).
  Mitigation: forward-fill the TSO series with the last published value
  (challenger.py, 2026-07-16 fix). The approximation is coarse; this is the
  main argument for moving the cron 90 minutes later.
- Systematic TSO bias in new seasons: if PSE's forecast method changes, the
  learned correction coefficients become stale until the 365-day window rotates.
- Extreme load events outside training range (record cold, major grid events):
  ridge cannot extrapolate, same as the TSO itself.

## Status

**UAT — shadow mode (PLAN M9).** Running as challenger in the daily loop since
2026-07-15. Promotion requires 14 consecutive shadow days beating the naive
incumbent on MAPE. Track progress in `docs/DECISIONS.md`.
