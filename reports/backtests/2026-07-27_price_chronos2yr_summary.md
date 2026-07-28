# Price backtest summary — 2026-07-27_price_chronos2yr

Target: PL day-ahead price, EUR/MWh (ENTSO-E). Test period: 2024-07-16 → 2026-07-24.
rMAE = MAE / MAE(naive-yesterday). No MAPE: prices cross zero.
coverage_80_pct: share of actuals inside [P10, P90]; nominal 80.

| model           |    mae |   rmse |   pinball_p10 |   pinball_p50 |   pinball_p90 |   coverage_80_pct |   winkler |   spike_mae |   spike_cover_pct |   n_hours |
|:----------------|-------:|-------:|--------------:|--------------:|--------------:|------------------:|----------:|------------:|------------------:|----------:|
| chronos_bolt_zs | 21.818 |  34.61 |         5.363 |        10.909 |         5.468 |            78.499 |   108.307 |      66.379 |            52.483 |     17720 |
