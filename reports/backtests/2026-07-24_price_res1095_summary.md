# Price backtest summary — 2026-07-24_price_res1095

Target: PL day-ahead price, EUR/MWh (ENTSO-E). Test period: 2024-07-16 → 2026-07-23.
rMAE = MAE / MAE(naive-yesterday). No MAPE: prices cross zero.
coverage_80_pct: share of actuals inside [P10, P90]; nominal 80.

| model         |    mae |   rmse |   pinball_p10 |   pinball_p50 |   pinball_p90 |   coverage_80_pct |   winkler |   spike_mae |   spike_cover_pct |   n_hours |
|:--------------|-------:|-------:|--------------:|--------------:|--------------:|------------------:|----------:|------------:|------------------:|----------:|
| lgbm_quantile | 17.353 | 28.276 |         4.368 |         8.677 |         5.46  |            55.368 |    98.276 |      63.309 |            37.175 |     17696 |
| lear          | 18.037 | 34.656 |         4.173 |         9.019 |         4.781 |            76.582 |    89.535 |      72.719 |            40.113 |     17696 |
