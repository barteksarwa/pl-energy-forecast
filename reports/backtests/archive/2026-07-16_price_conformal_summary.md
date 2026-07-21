# Price backtest — conformal band calibration — 2026-07-16_price_conformal

Rolling split-conformal (CQR, 90-day trailing window of
out-of-sample errors, walk-forward honest). P50 untouched — only
the band moves. First 30 days keep the raw band.

| model                   |    mae |   rmse |   pinball_p10 |   pinball_p50 |   pinball_p90 |   coverage_80_pct |   spike_mae |   spike_cover_pct |   n_hours |   rmae |
|:------------------------|-------:|-------:|--------------:|--------------:|--------------:|------------------:|------------:|------------------:|----------:|-------:|
| lgbm_quantile           | 17.849 | 28.663 |         4.666 |         8.924 |         5.576 |            51.436 |      60.627 |            45.08  |     17480 |  0.638 |
| lgbm_quantile_conformal | 17.849 | 28.663 |         4.257 |         8.924 |         4.789 |            78.673 |      60.627 |            52.174 |     17480 |  0.638 |
| lear                    | 18.461 | 32.923 |         4.272 |         9.23  |         4.789 |            72.077 |      70.956 |            49.657 |     17480 |  0.66  |
| lear_conformal          | 18.461 | 32.923 |         4.365 |         9.23  |         4.54  |            79.497 |      70.956 |            52.746 |     17480 |  0.66  |
| price_naive_yesterday   | 27.96  | 44.198 |         7.35  |        13.98  |         7.098 |            53.141 |      77.558 |            37.071 |     17480 |  1     |
| price_naive_week        | 34.01  | 52.862 |         7.418 |        17.005 |         7.316 |            53.69  |      92.964 |            37.185 |     17480 |  1.216 |
