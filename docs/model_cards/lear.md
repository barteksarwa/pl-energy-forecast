# Model card — lear (price)

## What it is

LEAR = LASSO-Estimated AutoRegression (Ziel & Weron 2018). The standard
day-ahead electricity price baseline. 24 separate LASSO models, one per
delivery hour. The benchmark every fancier price model must beat.
File: `src/models/price.py` (`PriceLEAR`).

Target: PL day-ahead price, EUR/MWh (ENTSO-E, `price_da_eur.parquet`).

## Inputs

~55 features per hour-model:
- **Full 24-hour price vector of D-1** (`price_d1_h00..h23`): the core
  LEAR input. Tomorrow morning is predicted by yesterday evening's ramp.
- **Same-hour price lags** 1/2/3/7 local days back + 7-day mean.
  Lags shift by LOCAL calendar days — a fixed minus-24h reaches into the
  target day on the 25-hour DST day (leakage, test-proven).
- **Load lags** (48h+, 09:00 D-1 cutoff, same as Phase 1).
- **Calendar** (weekday, month, holidays, bridge days, cyclic encodings).
- **TSO day-ahead load forecast** for day D (published before gate closure).

Does NOT see: wind/solar generation forecasts (next M6 step), fuel/CO2
prices, cross-border flows, outages.

## Timeline / cutoff

Auction for day D clears ~12:00 CET on D-1 (SDAC). All 24 prices of D-1
are known at bid time (fixed at the D-2 auction) — price lag 1d is legal.
Load actuals respect the 09:00 D-1 cutoff.

## Transform — the part that matters

z = asinh((p − median) / MAD), median/MAD from the training window only
(Uniejewski, Weron & Ziel 2018). Quantile band from per-hour training
residuals in z-space, mapped back with the monotone inverse.

Measured failure without the centering: asinh on raw ~100 EUR prices sits
in its log regime; sinh-back amplifies z-errors ~100x. Winter months hit
monthly rMAE 2.64 (Dec 2025). With robust standardization the same months
score 0.7. Full history in `reports/backtests/2026-07-16_price_summary.md`.

## Training

Rolling 365-day window, refit every 7 days, walk-forward.
LassoCV (50 alphas, 5-fold CV inside the training window), per hour.

## Performance (walk-forward, 2024-07-16 → 2026-07-14, 17,480 h)

| model | MAE (EUR/MWh) | RMSE | rMAE | pinball P10/P50/P90 | coverage 80% |
|---|---|---|---|---|---|
| **lear** | **20.8** | **33.4** | **0.744** | **5.0 / 10.4 / 5.4** | 73.4% |
| naive yesterday | 28.0 | 44.2 | 1.000 | 7.4 / 14.0 / 7.1 | 53.1% |
| naive last week | 34.0 | 52.9 | 1.216 | 7.4 / 17.0 / 7.3 | 53.7% |

- rMAE 0.744 sits in the literature range for LEAR vs naive (0.75–0.85).
- Wins or ties naive in **all 25 test months** (worst month rMAE 1.0).
- No MAPE: 798 negative-price hours in the sample make it meaningless.

## Honest limitations

- **Coverage 73.4% vs nominal 80%.** The static residual band is too
  narrow in spike months. Fix candidates: conformal band, quantile
  regression (LightGBM price model, M7).
- No fundamentals yet (wind/solar forecasts are price driver #1).
  This is deliberate: LEAR-on-lags is the floor, fundamentals come next.
- Pooled evaluation hides tail behavior; spike-only evaluation (P90
  coverage on top-decile hours) planned for the M7 writeup.

## Status

- [x] Beats both naives over 2 years, all months
- [ ] Fundamentals features (wind/solar forecast) — next
- [ ] LightGBM quantile challenger — M7
- [ ] Spike-tail evaluation — M7
