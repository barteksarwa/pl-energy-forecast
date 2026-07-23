# CRPS-weighted ensemble — 2026-07-24_price_ensemble

Members: lgbm, lear, chronos. Weights: inverse trailing-60d
crps3 (mean pinball over the 3 quantiles), equal during warm-up.
FM members conformalized before blending.

| model                 |    mae |   rmse |   pinball_p10 |   pinball_p50 |   pinball_p90 |   coverage_80_pct |   winkler |   spike_mae |   spike_cover_pct |   n_hours |   rmae |
|:----------------------|-------:|-------:|--------------:|--------------:|--------------:|------------------:|----------:|------------:|------------------:|----------:|-------:|
| ens_crps              | 17.339 | 28.21  |         4.063 |         8.669 |         4.454 |            84.209 |    85.168 |      61.064 |            57.42  |     17504 |  0.62  |
| ens_equal             | 17.457 | 28.342 |         4.1   |         8.729 |         4.491 |            84.295 |    85.908 |      61.268 |            56.621 |     17504 |  0.624 |
| lgbm                  | 17.871 | 28.753 |         4.219 |         8.935 |         4.734 |            78.925 |    89.529 |      60.287 |            51.256 |     17504 |  0.639 |
| lear                  | 18.241 | 32.749 |         4.271 |         9.121 |         4.521 |            79.593 |    87.916 |      69.466 |            55.594 |     17504 |  0.652 |
| chronos               | 21.986 | 34.885 |         5.398 |        10.993 |         5.528 |            79.965 |   109.258 |      67.245 |            51.941 |     17504 |  0.786 |
| price_naive_yesterday | 27.96  | 44.198 |         7.35  |        13.98  |         7.098 |            53.068 |   144.475 |      77.612 |            36.872 |     17480 |  1     |
