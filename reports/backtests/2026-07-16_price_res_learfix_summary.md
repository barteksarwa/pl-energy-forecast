# Price backtest summary — 2026-07-16_price_res_learfix

Target: PL day-ahead price, EUR/MWh (ENTSO-E). Test period: 2024-07-16 → 2026-07-14.
rMAE = MAE / MAE(naive-yesterday). No MAPE: prices cross zero.
coverage_80_pct: share of actuals inside [P10, P90]; nominal 80.

| model   |    mae |   rmse |   pinball_p10 |   pinball_p50 |   pinball_p90 |   coverage_80_pct |   spike_mae |   spike_cover_pct |   n_hours |
|:--------|-------:|-------:|--------------:|--------------:|--------------:|------------------:|------------:|------------------:|----------:|
| lear    | 18.461 | 32.923 |         4.272 |          9.23 |         4.789 |            72.077 |      58.288 |            56.163 |     17480 |
