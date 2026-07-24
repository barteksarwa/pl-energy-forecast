# Price backtest summary — 2026-07-24_price_timesfm2yr

Target: PL day-ahead price, EUR/MWh (ENTSO-E). Test period: 2024-07-16 → 2026-07-23.
rMAE = MAE / MAE(naive-yesterday). No MAPE: prices cross zero.
coverage_80_pct: share of actuals inside [P10, P90]; nominal 80.

| model      |    mae |   rmse |   pinball_p10 |   pinball_p50 |   pinball_p90 |   coverage_80_pct |   winkler |   spike_mae |   spike_cover_pct |   n_hours |
|:-----------|-------:|-------:|--------------:|--------------:|--------------:|------------------:|----------:|------------:|------------------:|----------:|
| timesfm_zs | 22.515 | 35.681 |         5.471 |        11.257 |         6.096 |            80.679 |   115.666 |      74.187 |            43.616 |     17696 |
