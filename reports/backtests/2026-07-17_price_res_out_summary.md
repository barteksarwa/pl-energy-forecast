# Price backtest summary — 2026-07-17_price_res_out

Target: PL day-ahead price, EUR/MWh (ENTSO-E). Test period: 2024-07-16 → 2026-07-15.
rMAE = MAE / MAE(naive-yesterday). No MAPE: prices cross zero.
coverage_80_pct: share of actuals inside [P10, P90]; nominal 80.

| model                 |    mae |   rmse |   pinball_p10 |   pinball_p50 |   pinball_p90 |   coverage_80_pct |   spike_mae |   spike_cover_pct |   n_hours |   rmae |
|:----------------------|-------:|-------:|--------------:|--------------:|--------------:|------------------:|------------:|------------------:|----------:|-------:|
| lgbm_quantile         | 17.896 | 28.724 |         4.643 |         8.948 |         5.553 |            51.782 |      61.305 |            44.749 |     17504 |  0.641 |
| lear                  | 18.43  | 32.874 |         4.285 |         9.215 |         4.78  |            72.155 |      70.652 |            50.342 |     17504 |  0.66  |
| price_naive_yesterday | 27.94  | 44.172 |         7.348 |        13.97  |         7.102 |            53.091 |      77.46  |            36.872 |     17504 |  1     |
| price_naive_week      | 34.053 | 52.889 |         7.417 |        17.027 |         7.32  |            53.639 |      92.899 |            36.986 |     17504 |  1.219 |
