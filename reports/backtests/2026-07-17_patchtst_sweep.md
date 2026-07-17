# PatchTST screening sweep — 2026-07-17_patchtst_sweep

Seed=42 d_model=64 | 27 configs | 0.2 h

## Top 10 (val pinball, lower = better)

| config              |   patch_len |   stride |   ctx |   val_pinball |
|:--------------------|------------:|---------:|------:|--------------:|
| patch48_s24_ctx2016 |          48 |       24 |  2016 |       793.224 |
| patch48_s12_ctx2016 |          48 |       12 |  2016 |       814.126 |
| patch48_s6_ctx1344  |          48 |        6 |  1344 |       862.22  |
| patch12_s6_ctx672   |          12 |        6 |   672 |       879.748 |
| patch48_s12_ctx672  |          48 |       12 |   672 |       958.038 |
| patch48_s12_ctx1344 |          48 |       12 |  1344 |      1031.41  |
| patch48_s24_ctx1344 |          48 |       24 |  1344 |      1175.36  |
| patch12_s12_ctx1344 |          12 |       12 |  1344 |      1579.59  |
| patch24_s12_ctx2016 |          24 |       12 |  2016 |      1719.72  |
| patch24_s24_ctx2016 |          24 |       24 |  2016 |      1734.27  |

## Interpretation

- Compare best PatchTST val vs best TFT HPO val (0.1184) to gauge
  whether patching adds value beyond the TFT architecture.
- Walk-forward the top-3 with --walkforward flag.
- Quote ONLY walk-forward numbers in results tables.