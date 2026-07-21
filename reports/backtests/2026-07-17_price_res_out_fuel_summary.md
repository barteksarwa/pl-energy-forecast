# Price backtest summary — 2026-07-17_price_res_out_fuel

Target: PL day-ahead price, EUR/MWh (ENTSO-E). Test period: 2024-07-16 → 2026-07-15.
rMAE = MAE / MAE(naive-yesterday). No MAPE: prices cross zero.
coverage_80_pct: share of actuals inside [P10, P90]; nominal 80.

| model                 |    mae |   rmse |   pinball_p10 |   pinball_p50 |   pinball_p90 |   coverage_80_pct |   spike_mae |   spike_cover_pct |   n_hours |   rmae |
|:----------------------|-------:|-------:|--------------:|--------------:|--------------:|------------------:|------------:|------------------:|----------:|-------:|
| lgbm_quantile         | 17.871 | 28.753 |         4.665 |         8.935 |         5.415 |            52.268 |      60.287 |            43.95  |     17504 |  0.64  |
| lear                  | 18.241 | 32.749 |         4.247 |         9.121 |         4.704 |            71.858 |      69.466 |            51.598 |     17504 |  0.653 |
| price_naive_yesterday | 27.94  | 44.172 |         7.348 |        13.97  |         7.102 |            53.091 |      77.46  |            36.872 |     17504 |  1     |
| price_naive_week      | 34.053 | 52.889 |         7.417 |        17.027 |         7.32  |            53.639 |      92.899 |            36.986 |     17504 |  1.219 |

## Verdict (fundamentals round, 2026-07-17)

- **Outages: FLAT** (LGBM 17.90 vs 17.85 base). Coarse aggregate +
  revision-history limitation; refinements logged. Research store kept,
  feature NOT adopted, backfill now opt-in (CI 503s on the endpoint).
- **Fuel (TTF + EUA proxy): ADOPTED for the daily incumbent.** LEAR
  18.24 vs 18.46. The gain concentrates exactly where the bias chart
  pointed: winter 2024/25 (high-gas regime) — Jan bias -15.9 -> -4.6,
  MAE -2.5; Feb bias -11.5 -> -1.0. Calm-gas winter 2025/26: no change,
  as expected. Merit-order mechanism, measured.
- LGBM: flat on fuel (trees already carry the slow level via lags).
- Conformal offsets refreshed from the fuel-model errors
  (lear 3.45, lgbm 10.06).
