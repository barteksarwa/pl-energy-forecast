# Price backtest summary — 2026-07-24_price_chronos2yr

Target: PL day-ahead price, EUR/MWh (ENTSO-E). Test period: 2024-07-16 → 2026-07-23.
rMAE = MAE / MAE(naive-yesterday). No MAPE: prices cross zero.
coverage_80_pct: share of actuals inside [P10, P90]; nominal 80.

| model           |    mae |   rmse |   pinball_p10 |   pinball_p50 |   pinball_p90 |   coverage_80_pct |   winkler |   spike_mae |   spike_cover_pct |   n_hours |
|:----------------|-------:|-------:|--------------:|--------------:|--------------:|------------------:|----------:|------------:|------------------:|----------:|
| chronos_bolt_zs | 21.931 | 34.781 |         5.382 |        10.965 |         5.514 |             78.34 |   108.959 |      66.928 |            51.864 |     17696 |
