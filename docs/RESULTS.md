# RESULTS.md — canonical results

One page. Every headline number in the repo lives here.
Other docs link here instead of copying numbers.
Update this file first when a campaign ends.

Updated: 2026-07-21.

## Load — day-ahead national load (MW)

2-year walk-forward. Test 2024-07-16 → 2026-07-14, 17,450 hours.
Source: `reports/backtests/2026-07-16_2yr_summary.md`.

| Model | MAPE | Skill vs naive |
|---|---|---|
| **ridge_tso** (champion) | **2.08%** | 0.63 |
| lgbm_tso | 2.12% | 0.62 |
| TSO forecast (benchmark) | 2.23% | 0.60 |
| ridge (no TSO input) | 4.05% | 0.29 |
| seasonal naive | 5.59% | 0.00 |

- `_tso` models correct the TSO forecast. TSO-free variants were tested too.
- We beat the TSO benchmark by 0.15 pp. Small but consistent.
- 12-month table (incl. lstm_attn 2.43%): `reports/backtests/2026-07-15_overnight_readout.md`.

## Price — day-ahead auction price (EUR/MWh)

2-year walk-forward. Test 2024-07-16 → 2026-07-14, ~17,480 hours.
Source: `reports/figures/backtest_price/metrics_summary.csv`.

| Model | MAE | rMAE | 80% band coverage |
|---|---|---|---|
| **LGBM quantile + CQR** (champion) | **17.87** | **0.640** | 78.9% |
| LEAR + CQR | 18.24 | 0.653 | 79.6% |
| TFT ens-3 (365d windows) | 19.71 | 0.706 | 79.6% |
| PatchTST (365d windows) | 22.98 | 0.823 | 69.5% |
| Naive (1-day) | 27.96 | 1.001 | 53.1% |

- rMAE = MAE relative to naive. Below 1.0 beats naive.
- CQR = conformalized quantile regression. Fixes band coverage honestly.
- LEAR wins interval quality (Winkler 87.96 vs 89.58). LGBM wins MAE.

## Deep-model campaign — final verdict (2026-07-21)

1-year window, test 2025-07-16 →. Same window for all models.
Source: `reports/sensitivity/tft/README.md`.

| Model | MAE | Coverage |
|---|---|---|
| **LGBM quantile** (champion) | **17.66** | — |
| TFT-730 ens-3 (best deep) | 18.31 | 82.8% |
| PatchTST-730 d128 ens-3 | 19.78 | — |
| PatchTST-730 ens-3 | 19.94 | 75.8% |

Loss decomposition, TFT gap to champion:

| Lever | Gap closed / left |
|---|---|
| Training window 365d → 730d | +1.2 recovered |
| 3-seed ensemble | +0.3 recovered |
| Capacity (d64 → d128) | +0.2 recovered |
| Architecture (remaining gap) | +1.5 left |

- Same-window ledger: PatchTST-365 gap was 3.84. Setup levers recovered 1.7.
  Architecture swap recovered 1.5. Residual 0.65.
- So nearly half the deep-model gap was evaluation setup, not architecture.
- Hyperparameters are not the gap. A config sweep proved it.
- Lesson: 1-seed screening picked a mirage three times in this project.

## Feature sensitivity (group ablation)

- Load model: TSO input carries 96% of skill (+1.97 pp when removed).
  Weather +0.08 pp, calendar +0.03 pp, lags +0.00 pp.
- Price champion (1-yr window): RES forecast +4.12, price history +2.02,
  TSO load +0.58, calendar +0.39, load lags −0.08 (dead weight).
- Ablation verdicts depend on the training window. Documented sign flip:
  PatchTST price-history encoder −0.4 → +2.5 EUR/MWh at 365d → 730d.
- Tables: `reports/sensitivity/group_ablation.md` (load),
  `reports/sensitivity/tft/README.md`, `reports/backtests/2026-07-20_price_group_ablation.md`.

## Where the details live

- Model cards: `docs/model_cards/`.
- Comparison figures: `reports/figures/backtest_price/` (15 plots).
- Shadow track record: `docs/shadow_tally.md`.
