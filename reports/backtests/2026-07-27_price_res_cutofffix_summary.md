# Price backtest summary — 2026-07-27_price_res_cutofffix

Target: PL day-ahead price, EUR/MWh (ENTSO-E). Test period: 2024-07-16 → 2026-07-24.
rMAE = MAE / MAE(naive-yesterday). No MAPE: prices cross zero.
coverage_80_pct: share of actuals inside [P10, P90]; nominal 80.

| model                 |    mae |   rmse |   pinball_p10 |   pinball_p50 |   pinball_p90 |   coverage_80_pct |   winkler |   spike_mae |   spike_cover_pct |   n_hours |   rmae |
|:----------------------|-------:|-------:|--------------:|--------------:|--------------:|------------------:|----------:|------------:|------------------:|----------:|-------:|
| lgbm_quantile         | 17.95  | 28.961 |         4.663 |         8.975 |         5.644 |            51.067 |   103.065 |      61.763 |             44.47 |     17720 |  0.644 |
| price_naive_yesterday | 27.878 | 44.046 |         7.32  |        13.939 |         7.08  |            53.104 |   143.996 |      77.161 |             37.02 |     17720 |  1     |
