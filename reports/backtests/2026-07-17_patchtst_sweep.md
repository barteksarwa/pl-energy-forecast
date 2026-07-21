# PatchTST screening sweep — 2026-07-17_patchtst_sweep

Seed=42 d_model=64 | 27 configs | 0.3 h

## Top 10 (val pinball, lower = better)

| config              |   patch_len |   stride |   ctx |   val_pinball |
|:--------------------|------------:|---------:|------:|--------------:|
| patch24_s24_ctx1344 |          24 |       24 |  1344 |        0.1236 |
| patch12_s12_ctx1344 |          12 |       12 |  1344 |        0.1253 |
| patch24_s12_ctx1344 |          24 |       12 |  1344 |        0.1255 |
| patch48_s24_ctx1344 |          48 |       24 |  1344 |        0.1261 |
| patch24_s6_ctx1344  |          24 |        6 |  1344 |        0.1262 |
| patch48_s12_ctx1344 |          48 |       12 |  1344 |        0.1291 |
| patch12_s6_ctx1344  |          12 |        6 |  1344 |        0.1295 |
| patch12_s24_ctx1344 |          12 |       24 |  1344 |        0.1308 |
| patch48_s24_ctx2016 |          48 |       24 |  2016 |        0.1316 |
| patch24_s12_ctx2016 |          24 |       12 |  2016 |        0.1317 |

## Interpretation

- Compare best PatchTST val vs best TFT HPO val (0.1184) to gauge
  whether patching adds value beyond the TFT architecture.
- Walk-forward the top-3 with --walkforward flag.
- Quote ONLY walk-forward numbers in results tables.