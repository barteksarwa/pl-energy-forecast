# Price backtest summary — 2026-07-28_price_learfull

Target: PL day-ahead price, EUR/MWh (ENTSO-E). Test period: 2024-07-16 → 2026-07-24.
rMAE = MAE / MAE(naive-yesterday). No MAPE: prices cross zero.
coverage_80_pct: share of actuals inside [P10, P90]; nominal 80.

| model     |    mae |   rmse |   pinball_p10 |   pinball_p50 |   pinball_p90 |   coverage_80_pct |   winkler |   spike_mae |   spike_cover_pct |   n_hours |
|:----------|-------:|-------:|--------------:|--------------:|--------------:|------------------:|----------:|------------:|------------------:|----------:|
| lear      | 18.511 | 32.857 |         4.299 |         9.255 |         4.8   |            71.377 |    90.99  |      70.973 |            48.984 |     17720 |
| lear_full | 19.18  | 34.232 |         4.385 |         9.59  |         5.184 |            69.819 |    95.693 |      77.589 |            43.454 |     17720 |
