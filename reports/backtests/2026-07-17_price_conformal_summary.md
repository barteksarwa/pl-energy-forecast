# Price backtest — conformal band calibration — 2026-07-17_price_conformal

Rolling split-conformal (CQR, 90-day trailing window of
out-of-sample errors, walk-forward honest). P50 untouched — only
the band moves. First 30 days keep the raw band.

| model                   |    mae |   rmse |   pinball_p10 |   pinball_p50 |   pinball_p90 |   coverage_80_pct |   winkler |   spike_mae |   spike_cover_pct |   n_hours |   rmae |
|:------------------------|-------:|-------:|--------------:|--------------:|--------------:|------------------:|----------:|------------:|------------------:|----------:|-------:|
| lgbm_quantile           | 17.877 | 28.765 |         4.667 |         8.938 |         5.414 |            52.239 |   100.819 |      60.366 |            44.165 |     17504 |  0.639 |
| lgbm_quantile_conformal | 17.877 | 28.765 |         4.22  |         8.938 |         4.738 |            78.816 |    89.576 |      60.366 |            51.487 |     17504 |  0.639 |
| lear                    | 18.25  | 32.767 |         4.249 |         9.125 |         4.709 |            71.726 |    89.576 |      69.538 |            51.487 |     17504 |  0.653 |
| lear_conformal          | 18.25  | 32.767 |         4.272 |         9.125 |         4.524 |            79.456 |    87.964 |      69.538 |            55.492 |     17504 |  0.653 |
| price_naive_yesterday   | 27.96  | 44.198 |         7.35  |        13.98  |         7.098 |            53.141 |   144.475 |      77.558 |            37.071 |     17480 |  1     |
| price_naive_week        | 34.01  | 52.862 |         7.418 |        17.005 |         7.316 |            53.69  |   147.337 |      92.964 |            37.185 |     17480 |  1.216 |
