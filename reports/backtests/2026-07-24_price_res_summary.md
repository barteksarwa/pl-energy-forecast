# Price backtest summary — 2026-07-24_price_res

Target: PL day-ahead price, EUR/MWh (ENTSO-E). Test period: 2024-07-16 → 2026-07-23.
rMAE = MAE / MAE(naive-yesterday). No MAPE: prices cross zero.
coverage_80_pct: share of actuals inside [P10, P90]; nominal 80.

| model                 |    mae |   rmse |   pinball_p10 |   pinball_p50 |   pinball_p90 |   coverage_80_pct |   winkler |   spike_mae |   spike_cover_pct |   n_hours |   rmae |
|:----------------------|-------:|-------:|--------------:|--------------:|--------------:|------------------:|----------:|------------:|------------------:|----------:|-------:|
| lgbm_quantile         | 17.842 | 28.597 |         4.656 |         8.921 |         5.582 |            51.339 |   102.379 |      60.449 |            44.972 |     17696 |  0.64  |
| lear                  | 18.462 | 32.835 |         4.265 |         9.231 |         4.777 |            72.061 |    90.42  |      70.762 |            49.718 |     17696 |  0.662 |
| price_naive_yesterday | 27.881 | 44.062 |         7.326 |        13.941 |         7.08  |            53.091 |   144.063 |      77.154 |            37.062 |     17696 |  1     |
| price_naive_week      | 33.947 | 52.72  |         7.386 |        16.973 |         7.301 |            53.701 |   146.864 |      92.529 |            37.175 |     17696 |  1.218 |
