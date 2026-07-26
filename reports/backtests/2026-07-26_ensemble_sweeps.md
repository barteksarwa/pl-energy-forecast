# Ensemble + CQR sweeps — 2026-07-26

## Weight-window sweep (shipped: 60d)

|   window |    mae |   winkler |   y2024 |   y2025 |   y2026 |
|---------:|-------:|----------:|--------:|--------:|--------:|
|       30 | 17.31  |     85.12 |  18.868 |  16.177 |  18.047 |
|       60 | 17.338 |     85.24 |  18.882 |  16.227 |  18.05  |
|       90 | 17.335 |     85.24 |  18.891 |  16.244 |  18.001 |
|      120 | 17.339 |     85.26 |  18.903 |  16.245 |  18.003 |

Verdict: keep 60d

## 4-member blend (+ TimesFM)

| model   |    mae |   coverage_80_pct |   winkler |
|:--------|-------:|------------------:|----------:|
| blend3  | 17.338 |            79.86  |    85.241 |
| blend4  | 17.907 |            79.945 |    88.345 |

DM p=1.0000. Verdict: keep 3-member

## CQR window sweep (shipped: 90d)

### lgbm

|   window |   coverage |   winkler |   spike_cover |
|---------:|-----------:|----------:|--------------:|
|       30 |      72.96 |     93.14 |          50.8 |
|       60 |      78.52 |     90.26 |          52.1 |
|       90 |      78.63 |     90.32 |          52   |
|      180 |      78.71 |     90.29 |          52   |

Verdict (lgbm): candidate 180d (gate passed)

### blend

|   window |   coverage |   winkler |   spike_cover |
|---------:|-----------:|----------:|--------------:|
|       30 |      78.94 |     85.45 |          54.1 |
|       60 |      79.14 |     85.66 |          54.4 |
|       90 |      79.38 |     85.74 |          54.9 |
|      180 |      79.8  |     85.49 |          54.6 |

Verdict (blend): candidate 180d (gate passed)
