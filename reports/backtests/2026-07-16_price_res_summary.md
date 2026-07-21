# Price backtest summary — 2026-07-16_price_res (final, LEAR z-clip fix)

Target: PL day-ahead price, EUR/MWh (ENTSO-E). Test: 2024-07-16 → 2026-07-14.
Features include TSO wind+solar day-ahead forecast (proxy caveat: DECISIONS 2026-07-16).
rMAE = MAE / MAE(naive-yesterday). No MAPE: prices cross zero.
coverage_80_pct: share of actuals inside [P10, P90]; nominal 80.
spike_*: metrics on the top-5% priciest hours only.

| model                 |    mae |   rmse |   pinball_p10 |   pinball_p50 |   pinball_p90 |   coverage_80_pct |   spike_mae |   spike_cover_pct |   n_hours |   rmae |
|:----------------------|-------:|-------:|--------------:|--------------:|--------------:|------------------:|------------:|------------------:|----------:|-------:|
| lgbm_quantile         | 17.849 | 28.663 |         4.666 |         8.924 |         5.576 |            51.436 |      60.627 |            45.08  |     17480 |  0.638 |
| lear                  | 18.461 | 32.923 |         4.272 |         9.23  |         4.789 |            72.077 |      70.956 |            49.657 |     17480 |  0.66  |
| price_naive_yesterday | 27.96  | 44.198 |         7.35  |        13.98  |         7.098 |            53.141 |      77.558 |            37.071 |     17480 |  1     |
| price_naive_week      | 34.01  | 52.862 |         7.418 |        17.005 |         7.316 |            53.69  |      92.964 |            37.185 |     17480 |  1.216 |
