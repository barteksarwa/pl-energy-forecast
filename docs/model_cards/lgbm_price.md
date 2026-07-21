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

`reports/backtests/2026-07-16_price_res_summary.csv`:

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

Cross-checked by retrain ablation (2026-07-17, walk-forward Jan-Jul
2026): dropping the RES group costs +3.54 EUR/MWh MAE — the largest
group cost, above the whole price-lag block (+2.80). SHAP rank and
value-of-information agree here. Caveats: solar-season window; load
lags are dead weight for price (-0.04, pruning candidate). See
`docs/notes/learning/13_shap_vs_ablation.tex` for why the two methods
CAN disagree and what each one means.

## Honest limitations

- **Raw coverage 51.4% vs nominal 80%** — the untuned quantile boosters
  overfit their in-sample quantiles. FIXED in Phase 2.5 by rolling
  conformal calibration: **78.7%** coverage, pinball P90 5.58 -> 4.79
  (offset +10.2 EUR/MWh each side). The raw band stays unusable; only
  the conformal variant qualifies for anything risk-facing.
- Spike MAE 60.6: better than everyone else, still 3x the pooled MAE.
- Untuned. Tuning comes only now that the honest first row exists
  (repo rule).

## Status

- [x] Honest first row: MAE champion, rMAE 0.638
- [x] SHAP drivers artifact
- [x] Band calibration (conformal, Phase 2.5): 51.4% -> 78.7% coverage
- [x] Daily-loop shadow integration (2026-07-17); promotion gate: 14 valid shadow days
- [ ] Tuning pass (after shadow window confirms coverage gap vs LEAR)
