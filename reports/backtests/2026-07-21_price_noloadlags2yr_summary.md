# Price backtest summary — 2026-07-21_price_noloadlags2yr

Target: PL day-ahead price, EUR/MWh (ENTSO-E). Test period: 2024-07-16 → 2026-07-17.
rMAE = MAE / MAE(naive-yesterday). No MAPE: prices cross zero.
coverage_80_pct: share of actuals inside [P10, P90]; nominal 80.

| model                 |    mae |   rmse |   pinball_p10 |   pinball_p50 |   pinball_p90 |   coverage_80_pct |   winkler |   spike_mae |   spike_cover_pct |   n_hours |   rmae |
|:----------------------|-------:|-------:|--------------:|--------------:|--------------:|------------------:|----------:|------------:|------------------:|----------:|-------:|
| lgbm_quantile         | 17.755 | 28.721 |         4.638 |         8.878 |         5.568 |            49.766 |   102.058 |      60.926 |            44.761 |     17552 |  0.636 |
| price_naive_yesterday | 27.914 | 44.132 |         7.339 |        13.957 |         7.106 |            53.02  |   144.454 |      77.539 |            36.788 |     17552 |  1     |
