# Price backtest summary — 2026-07-16_price

Target: PL day-ahead price, EUR/MWh (ENTSO-E). Test period: 2024-07-16 → 2026-07-14.
rMAE = MAE / MAE(naive-yesterday). No MAPE: prices cross zero.
coverage_80_pct: share of actuals inside [P10, P90]; nominal 80.

| model                 |    mae |   rmse |   pinball_p10 |   pinball_p50 |   pinball_p90 |   coverage_80_pct |   n_hours |   rmae |
|:----------------------|-------:|-------:|--------------:|--------------:|--------------:|------------------:|----------:|-------:|
| lear                  | 20.812 | 33.405 |         4.976 |        10.406 |         5.383 |            73.415 |     17480 |  0.744 |
| price_naive_yesterday | 27.96  | 44.198 |         7.35  |        13.98  |         7.098 |            53.141 |     17480 |  1     |
| price_naive_week      | 34.01  | 52.862 |         7.418 |        17.005 |         7.316 |            53.69  |     17480 |  1.216 |
