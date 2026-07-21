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

- **Wind + solar day-ahead forecast** (added 2026-07-16): TSO series via
  ENTSO-E. Published ~18:00 D-1, after gate closure — accepted as a
  bid-time proxy (DECISIONS 2026-07-16; standard in the EPF literature).

Does NOT see: fuel/CO2 prices, cross-border flows, outages.

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
score 0.7. Full history in `reports/backtests/2026-07-16_price_summary.csv`.

## Training

Rolling 365-day window, refit every 7 days, walk-forward.
LassoCV (50 alphas, 5-fold CV inside the training window), per hour.

## The extrapolation guard (z-clip)

Adding RES features exposed a failure: one weekly-refit hour-19 model
extrapolated on record solar values (solar capacity grows every year, so
2025-26 values exceed any training window) and sinh-back turned that into
38,000 EUR/MWh predictions for a week of May 2025. Fix: predicted z is
clipped to the training z-range ± 0.5 — never beyond every price the
window has seen, with ~65% headroom for genuine spikes. Measured effect:
RMSE 558 → 32.9, MAE 27.2 → 18.5.

## Performance (walk-forward, 2024-07-16 → 2026-07-14, 17,480 h)

With RES features + z-clip (`reports/backtests/2026-07-16_price_res_summary.csv`):

| model | MAE (EUR/MWh) | RMSE | rMAE | coverage 80% | spike MAE | spike P90 cov |
|---|---|---|---|---|---|---|
| lgbm_quantile | 17.8 | 28.7 | 0.638 | 51.4% | 60.6 | 45.1% |
| **lear** | **18.5** | **32.9** | **0.660** | **72.1%** | 71.0 | 49.7% |
| naive yesterday | 28.0 | 44.2 | 1.000 | 53.1% | 77.6 | 37.1% |
| naive last week | 34.0 | 52.9 | 1.216 | 53.7% | 93.0 | 37.2% |

Lags-only variant (no RES, `2026-07-16_price_summary.csv`): MAE 20.8,
rMAE 0.744, wins all 25 test months. Literature range for LEAR vs naive
is 0.75–0.85 — fundamentals push us past it.

- Spike columns: top-5% priciest hours of the test period.
- No MAPE: 798 negative-price hours in the sample make it meaningless.

## Honest limitations

- **Loses to LightGBM on MAE** (18.5 vs 17.8). Kept as THE baseline:
  every future price model must beat LEAR first.
- **Raw coverage 72.1% vs nominal 80%** — FIXED in Phase 2.5 by rolling
  conformal calibration (CQR, 90d trailing window): **79.5%**. The daily
  loop publishes the conformal band (offset +3.6 EUR/MWh each side,
  config/price_conformal.json). Table:
  reports/backtests/2026-07-17_price_conformal_summary.md.
- Spike MAE 71 vs LGBM 60.6 — both models miss spikes badly; tails are
  the weak spot of the whole table.
- **Fuel features (TTF/EUA proxy) adopted 2026-07-17**: MAE 18.5 → 18.24.
  Gain concentrated in high-gas winter 2024/25 (Jan bias −15.9 → −4.6).
  Cross-border flows and outages evaluated; outages FLAT (CI 503 endpoint).

## Status

- [x] Beats both naives over 2 years, all months
- [x] Fundamentals features (wind/solar forecast) + extrapolation guard
- [x] LightGBM quantile challenger — wins MAE, loses coverage
- [x] Spike-tail evaluation — in the summary table
- [x] Conformal calibrated band (Phase 2.5): 72.1% → 79.5% coverage
- [x] Fuel + CO2 proxy features (TTF index + EUA-tracking ETC)
- [x] Shadow run live (2026-07-18, target day 1 of 14)
