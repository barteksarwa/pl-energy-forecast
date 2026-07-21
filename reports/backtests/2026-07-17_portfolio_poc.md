# Portfolio POC — sub-national load vs national TSO forecast — 2026-07-17_portfolio_poc

Synthetic retailer portfolio (BDEW h0/g0/g3 mix, ~500 MW, AR(1)
noise, 2%/yr growth). 12-month walk-forward, weekly refits, no
weather in any design (level playing field).

| design                           |   mape_pct |   mae_mw |   coverage_80_pct |   n_hours |
|:---------------------------------|-----------:|---------:|------------------:|----------:|
| (a) ridge + national TSO feature |       7.78 |    34.58 |             78.58 |      8772 |
| (a-) ridge, no TSO               |       7.89 |    34.89 |             79.29 |      8772 |
| (b) share x TSO forecast         |       8.42 |    38.75 |             80.44 |      8772 |
| (c) seasonal naive               |       9.49 |    42.45 |             53.04 |      8772 |

Reading: (a) vs (a-) prices the national signal; (a) vs (b) decides
feature-vs-share; everything must beat (c). Caveats: synthetic
portfolio inherits BDEW calendar shape — correlation with the
national series is optimistic vs a real portfolio; ratio
non-stationarity here is mild (smooth growth), real churn is lumpier.
## Verdict (2026-07-17)

- **(a) plain-feature wins.** Simplest design, best MAE. Use it.
- **The national signal adds only 0.11pp here** — this portfolio
  correlates with the country only via calendar. A real portfolio with
  weather exposure (heating/cooling customers) would gain more; rerun
  with weather features before citing this number for such a book.
- **(b) share model loses** (+0.6pp vs (a)): it pays multiplicative
  error compounding (share error x national forecast error) and gets
  nothing back at this correlation level. Worth retrying only for a
  portfolio that is a stable, large fraction of the zone.
- Everything beats naive; the noise floor (unforecastable AR(1) churn)
  bounds all designs at ~7.5% MAPE by construction.
