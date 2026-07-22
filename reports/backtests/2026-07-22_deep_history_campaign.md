# Deep-history campaign — 2026-07-22

Data extended to 2015+ (ENTSO-E). This campaign asks what the
extra history buys: longer training windows, crisis-regime
evaluation, and the previously blocked deep re-benchmark on
the full 2-year test.

## 1. LGBM training-window sweep (test 2024-07-16 →)

|   train_days |    mae |   rmae |   spike_mae | source                               |
|-------------:|-------:|-------:|------------:|:-------------------------------------|
|          365 | 17.872 |  0.64  |      60.657 | 2026-07-22_price_win365_summary.csv  |
|          730 | 17.483 |  0.626 |      61.962 | 2026-07-22_price_win730_summary.csv  |
|         1095 | 17.379 |  0.623 |      63.607 | 2026-07-22_price_win1095_summary.csv |
|         1460 | 17.522 |  0.628 |      63.546 | 2026-07-22_price_win1460_summary.csv |

Same model, same test window — only the training window
changes. Per-year tables: the win*_summary.md files.

## 2. Crisis-regime backtest (test 2021-07-16 →, 5 years)

| model                 |    mae |   rmse |   pinball_p10 |   pinball_p50 |   pinball_p90 |   coverage_80_pct |   winkler |   spike_mae |   spike_cover_pct |   n_hours |   rmae |
|:----------------------|-------:|-------:|--------------:|--------------:|--------------:|------------------:|----------:|------------:|------------------:|----------:|-------:|
| lgbm_quantile         | 17.491 | 28.431 |         4.512 |         8.745 |         5.583 |            49.635 |   100.941 |      65.982 |            45.803 |     43832 |  0.644 |
| lear                  | 19.216 | 35.64  |         4.257 |         9.608 |         5.368 |            71.331 |    96.252 |      77.857 |            50.593 |     43832 |  0.707 |
| price_naive_yesterday | 27.168 | 42.478 |         6.968 |        13.584 |         6.681 |            52.984 |   136.493 |      74.433 |            48.495 |     43832 |  1     |
| price_naive_week      | 32.973 | 50.628 |         7     |        16.487 |         6.882 |            53.657 |   138.829 |      89.679 |            48.449 |     43832 |  1.214 |

Per-year breakdown (incl. 2022 crisis): 2026-07-22_price_crisis5yr_summary.md.

## 3. Deep re-benchmark — TFT (730d windows, FULL 2-yr test)

| config          |   seed |    mae |   rmae |   coverage_80_pct |
|:----------------|-------:|-------:|-------:|------------------:|
| tft730_2yr      |     42 | 20.197 |  0.723 |            79.476 |
| tft730_2yr      |      7 | 20.204 |  0.723 |            79.934 |
| tft730_2yr      |   2026 | 20.538 |  0.735 |            78.663 |
| tft730_2yr_ens3 |     -1 | 19.522 |  0.699 |            80.907 |

Per-year MAE:

|   period |   ('tft730_2yr', 7) |   ('tft730_2yr', 42) |   ('tft730_2yr', 2026) |   ('tft730_2yr_ens3', -1) |
|---------:|--------------------:|---------------------:|-----------------------:|--------------------------:|
|     2024 |               22.07 |                21.68 |                  22.88 |                     21.26 |
|     2025 |               18.67 |                18.83 |                  18.37 |                     17.96 |
|     2026 |               21.44 |                21.44 |                  22.53 |                     20.93 |

## 3. Deep re-benchmark — PATCHTST (730d windows, FULL 2-yr test)

| config               |   seed |    mae |   rmae |   coverage_80_pct |
|:---------------------|-------:|-------:|-------:|------------------:|
| patchtst730_2yr      |     42 | 22.403 |  0.802 |            73.535 |
| patchtst730_2yr      |      7 | 23.263 |  0.833 |            75.212 |
| patchtst730_2yr      |   2026 | 22.761 |  0.815 |            77.238 |
| patchtst730_2yr_ens3 |     -1 | 22.25  |  0.797 |            77.135 |

Per-year MAE:

|   period |   ('patchtst730_2yr', 7) |   ('patchtst730_2yr', 42) |   ('patchtst730_2yr', 2026) |   ('patchtst730_2yr_ens3', -1) |
|---------:|-------------------------:|--------------------------:|----------------------------:|-------------------------------:|
|     2024 |                    28.51 |                     28.4  |                       26.57 |                          27.17 |
|     2025 |                    20.82 |                     19.83 |                       20.65 |                          19.72 |
|     2026 |                    23.28 |                     22.02 |                       23.41 |                          22.71 |

## Reference points

- Champion (LGBM+CQR, 365d windows, 2-yr test): MAE 17.87,
  rMAE 0.640. Without load_lags: 17.755.
- Best deep before this campaign: TFT-730 ens-3 MAE 18.31 on
  the 1-yr window only.
- Full numbers: docs/RESULTS.md.
