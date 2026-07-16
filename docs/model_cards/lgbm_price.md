# Model card — lgbm_quantile (price)

## What it is

The same `LightGBMQuantile` class as the load model (`src/models/gbm.py`),
pointed at the price target. Three boosters (P10/P50/P90), pinball
objective, conservative untuned defaults. Current MAE champion on the
price table.

Target: PL day-ahead price, EUR/MWh (`price_da_eur.parquet`).

## Inputs

Same ~50-feature matrix as LEAR (`src/features/price_matrix.py`):
price lags (local-day shifts), full D-1 price vector, load lags,
calendar, TSO load forecast, wind+solar day-ahead forecast
(bid-time proxy caveat: DECISIONS 2026-07-16).

No transform needed — trees are scale- and monotone-invariant, and they
cannot extrapolate beyond training targets, so the LEAR z-clip pathology
cannot happen here structurally.

## Performance (walk-forward, 2024-07-16 → 2026-07-14, 17,480 h)

`reports/backtests/2026-07-16_price_res_summary.md`:

| metric | lgbm_quantile | lear | naive yesterday |
|---|---|---|---|
| MAE (EUR/MWh) | **17.8** | 18.5 | 28.0 |
| rMAE | **0.638** | 0.660 | 1.000 |
| RMSE | **28.7** | 32.9 | 44.2 |
| coverage [P10,P90] | 51.4% | 72.1% | 53.1% |
| spike MAE (top 5%) | **60.6** | 71.0 | 77.6 |

## Drivers (SHAP, P50 booster, last 90 days)

`reports/sensitivity/shap_importance_price.csv`:

1. solar forecast — 18.7 EUR mean |SHAP| (price driver #1, merit order)
2. price lag 1d — 14.1
3. wind onshore forecast — 8.4
4. TSO load forecast — 7.9
5. price lag 7d — 6.1

Plain words: tomorrow's price is set by how much sun and wind tomorrow
brings, anchored on today's price level and expected demand.

## Honest limitations

- **Coverage 51.4% vs nominal 80% — the band is badly under-dispersed.**
  Untuned quantile boosters overfit their in-sample quantiles. This must
  be fixed (conformal calibration or tuning) before any production use.
  P50 is trustworthy; the band is not.
- Spike MAE 60.6: better than everyone else, still 3x the pooled MAE.
- Untuned. Tuning comes only now that the honest first row exists
  (repo rule).

## Status

- [x] Honest first row: MAE champion, rMAE 0.638
- [x] SHAP drivers artifact
- [ ] Band calibration (conformal) — REQUIRED before shipping
- [ ] Tuning pass
- [ ] Daily-loop shadow integration
