# Rolling-vs-expanding ablation — 2026-07-16

Test period: 2024-07-16 → 2026-07-14. Rolling: 365-day window. Expanding: all history from 2023-01-01.

| model                |     mae |    rmse |   mape_pct |   pinball_p10 |   pinball_p50 |   pinball_p90 |   n_hours |   skill_vs_naive |
|:---------------------|--------:|--------:|-----------:|--------------:|--------------:|--------------:|----------:|-----------------:|
| ridge_tso_rolling365 |  374.08 |  502.85 |       2.08 |         92.8  |        187.04 |         88.33 |     17450 |             0.63 |
| ridge_tso_expanding  |  377.54 |  504.81 |       2.1  |         93.03 |        188.77 |         88.96 |     17450 |             0.62 |
| ridge_expanding      |  709.99 |  974.14 |       4.05 |        175.5  |        355    |        175.28 |     17450 |             0.29 |
| ridge_rolling365     |  710.42 |  965.15 |       4.05 |        171.32 |        355.21 |        175.38 |     17450 |             0.29 |
| seasonal_naive       | 1005.26 | 1537.32 |       5.59 |        359.81 |        502.63 |        252.32 |     17450 |             0    |
