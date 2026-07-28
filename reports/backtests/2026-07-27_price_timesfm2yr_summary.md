# Price backtest summary — 2026-07-27_price_timesfm2yr

Target: PL day-ahead price, EUR/MWh (ENTSO-E). Test period: 2024-07-16 → 2026-07-24.
rMAE = MAE / MAE(naive-yesterday). No MAPE: prices cross zero.
coverage_80_pct: share of actuals inside [P10, P90]; nominal 80.

| model      |    mae |   rmse |   pinball_p10 |   pinball_p50 |   pinball_p90 |   coverage_80_pct |   winkler |   spike_mae |   spike_cover_pct |   n_hours |
|:-----------|-------:|-------:|--------------:|--------------:|--------------:|------------------:|----------:|------------:|------------------:|----------:|
| timesfm_zs | 22.383 | 35.499 |         5.453 |        11.191 |         6.056 |            80.886 |   115.088 |      73.698 |            44.131 |     17720 |
