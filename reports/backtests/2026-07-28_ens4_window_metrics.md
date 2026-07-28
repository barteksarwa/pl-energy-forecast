# ens4 window metrics — 2026-07-28_ens4_window_metrics

ens4 and ens3 scored on the identical hour set (the 4-member
intersection; the TFT member's window ends 2026-07-14). This is
the artifact behind the ens4 gate rows and its P&L capture.
coverage_80_pct: nominal 80. Capture = P&L / perfect foresight,
same days for every row.

| model                 |    mae |   rmse |   pinball_p10 |   pinball_p50 |   pinball_p90 |   coverage_80_pct |   winkler |   spike_mae |   spike_cover_pct |   n_hours |   rmae |   eur_per_day |   capture_rate |   perfect_eur_per_day |   n_days |
|:----------------------|-------:|-------:|--------------:|--------------:|--------------:|------------------:|----------:|------------:|------------------:|----------:|-------:|--------------:|---------------:|----------------------:|---------:|
| ens4_tft              | 16.879 | 27.626 |         3.823 |         8.439 |         4.434 |            80.03  |    82.57  |      63.19  |            53.265 |     17456 |  0.604 |       206.086 |          0.931 |               221.414 |      712 |
| ens3                  | 17.355 | 28.173 |         4.07  |         8.677 |         4.477 |            79.932 |    85.476 |      62.132 |            54.754 |     17456 |  0.621 |       205.333 |          0.927 |               221.414 |      712 |
| price_naive_yesterday | 27.934 | 44.189 |         7.326 |        13.967 |         7.117 |            53.002 |   144.425 |      77.785 |            36.77  |     17456 |  1     |       180.027 |          0.813 |               221.414 |      712 |
