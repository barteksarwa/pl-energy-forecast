# Price backtest — conformal band calibration — 2026-07-27_price_conformal

Rolling split-conformal (CQR, 90-day trailing window of
out-of-sample errors, walk-forward honest). P50 untouched — only
the band moves. First 30 days keep the raw band.

| model                   |    mae |   rmse |   pinball_p10 |   pinball_p50 |   pinball_p90 |   coverage_80_pct |   winkler |   spike_mae |   spike_cover_pct |   n_hours |   rmae |
|:------------------------|-------:|-------:|--------------:|--------------:|--------------:|------------------:|----------:|------------:|------------------:|----------:|-------:|
| lgbm_quantile           | 17.829 | 28.639 |         4.688 |         8.914 |         5.623 |            50.909 |   103.113 |      60.798 |            43.905 |     17720 |  0.64  |
| lgbm_quantile_conformal | 17.829 | 28.639 |         4.276 |         8.914 |         4.793 |            78.533 |    90.695 |      60.798 |            50.564 |     17720 |  0.64  |
| lear                    | 18.463 | 32.837 |         4.267 |         9.232 |         4.776 |            72.065 |    90.43  |      70.799 |            49.661 |     17720 |  0.662 |
| lear_conformal          | 18.463 | 32.837 |         4.36  |         9.232 |         4.527 |            79.492 |    88.866 |      70.799 |            52.935 |     17720 |  0.662 |
| price_naive_yesterday   | 27.878 | 44.046 |         7.32  |        13.939 |         7.08  |            53.104 |   143.996 |      77.161 |            37.02  |     17720 |  1     |
| price_naive_week        | 33.937 | 52.697 |         7.379 |        16.969 |         7.3   |            53.713 |   146.792 |      92.397 |            37.133 |     17720 |  1.217 |
