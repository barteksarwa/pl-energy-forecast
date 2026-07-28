# Backtest summary — 2026-07-28_fcst_tso_load2yr_cutoff

Test period: 2024-07-16 → 2026-07-24. Weather input: archived forecasts, lead 2 days (honest).

| model          |     mae |    rmse |   mape_pct |   pinball_p10 |   pinball_p50 |   pinball_p90 |   n_hours |   skill_vs_naive |
|:---------------|--------:|--------:|-----------:|--------------:|--------------:|--------------:|----------:|-----------------:|
| ridge          |  386.55 |  514.98 |       2.16 |         94.27 |        193.27 |         91.6  |     17690 |             0.61 |
| lasso_ar       |  389.34 |  522.07 |       2.16 |         96.23 |        194.67 |         92.44 |     17690 |             0.61 |
| tso_forecast   |  403.48 |  545.88 |       2.25 |        nan    |        201.74 |        nan    |     17690 |             0.6  |
| seasonal_naive | 1002.57 | 1530.78 |       5.58 |        357.27 |        501.28 |        250.7  |     17690 |             0    |
