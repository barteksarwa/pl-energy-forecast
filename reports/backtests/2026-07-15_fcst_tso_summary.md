# Backtest summary — 2026-07-15_fcst_tso

Test period: 2025-07-13 → 2026-07-13. Weather input: archived forecasts, lead 2 days (honest).

| model          |     mae |    rmse |   mape_pct |   pinball_p10 |   pinball_p50 |   pinball_p90 |   n_hours |   skill_vs_naive |
|:---------------|--------:|--------:|-----------:|--------------:|--------------:|--------------:|----------:|-----------------:|
| ridge          |  383.19 |  509.33 |       2.13 |         92.14 |        191.6  |         92.48 |      8762 |             0.63 |
| lgbm_quantile  |  394.01 |  521.63 |       2.16 |        110.81 |        197    |        120.1  |      8762 |             0.62 |
| tso_forecast   |  417.86 |  556.78 |       2.31 |        nan    |        208.93 |        nan    |      8762 |             0.59 |
| seasonal_naive | 1025.14 | 1537.35 |       5.6  |        353.79 |        512.57 |        281.75 |      8762 |             0    |
