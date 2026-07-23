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
Source: `docs/handovers/2026-07-20_cross-model-ablation.md`.

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

## Deep-history campaign (2026-07-22, data extended to 2015+)

Full report: `reports/backtests/2026-07-22_deep_history_campaign.md`.

**Training-window sweep** (LGBM, same 2-yr test, only window changes):

| Train window | MAE | rMAE |
|---|---|---|
| 365d (shipped champion) | 17.87 | 0.640 |
| 730d | 17.48 | 0.626 |
| **1095d (best)** | **17.38** | **0.623** |
| 1460d | 17.52 | 0.628 |

- 1095d + CQR: coverage 78.7% (same as 365d). −0.49 MAE for free.
- RECOMMENDATION: promote 1095d as the champion training window.
  Owner decision pending; daily loop still trains on 365d.
- More history helps up to 3 years, then the 2021-22 crisis regime
  starts hurting (1460d worse than 1095d).

**Crisis-regime test** (5-yr walk-forward incl. 2022 crisis, raw bands):

- LGBM rMAE 0.644 over 43,832 hours — the edge holds across regimes.
- LEAR 0.707. Naive-yesterday MAE 27.2 (crisis years inflate it).
- Per-year: `reports/backtests/2026-07-22_price_crisis5yr_summary.md`.

**Deep re-benchmark, FULL 2-yr test** (730d windows, 3 seeds, was
blocked before the backfill):

| Model | MAE (ens-3) | Gap to champion 17.87 |
|---|---|---|
| TFT-730 | 19.52 | +1.65 |
| PatchTST-730 | 22.25 | +4.38 |

- The 1-yr result (18.31) was flattered by the calm 2025 test year:
  the 2025 slice of this run reproduces it (17.96); 2024 costs 21.26.
- Deep gap WIDENS in harder years. LGBM champion confirmed on the
  full window; the "deep almost caught up" story is window-dependent.

## Foundation models + spike classifier (Phase 5, 2026-07-23)

**Chronos-Bolt zero-shot** (univariate — sees only price history; the
champion also sees RES/TSO/calendar. The gap measures what covariates
+ training buy):

| Model | MAE (2-yr) | rMAE | Coverage (+CQR) |
|---|---|---|---|
| Champion LGBM (365d) | 17.87 | 0.640 | 78.9% |
| TFT-730 ens-3 | 19.52 | 0.699 | 80.9% raw |
| **Chronos-Bolt zero-shot** | **21.98** | **0.787** | 79.9% |
| PatchTST-730 ens-3 (trained!) | 22.25 | 0.797 | 77.1% raw |
| Naive | 27.91 | 1.000 | — |

- A pretrained model with NO training on our data and NO covariates
  beats our trained PatchTST. Sic transit patch attention.
- All gaps DM-significant. Phase 6 fine-tune gate (rMAE < 0.75): closed.
- Source: `reports/backtests/2026-07-23_price_chronos2yr_summary.md`.

**Spike classifier** (2-yr walk-forward, top-5% hours, train-window
labels): AUC 0.966, Brier 0.034, precision@2 0.736. Gate 0.80 passed —
promoted to a daily-report line. Deterministic model; seed sweep
vacuous (42 and 7 bit-identical).

Reliability check (2026-07-23, `src/evaluation/spike_reliability.py`):
- Stable across regimes: ROC-AUC 0.964-0.969 in each of 2024/25/26.
- PR-AUC 0.63 vs 0.05 base rate (12.6x lift over random flagging).
- Probabilities are OVER-confident at the top: predicted >90% -> only
  70% observed. Flag at p>=0.5 hits 63% precision, 53% recall.
- The daily report therefore speaks in historical precision ("about
  6 in 10 flagged hours"), not raw probability. Calibration (isotonic)
  is a possible refinement; not needed for a flag.
Source: `reports/backtests/2026-07-22_spike_screen.md`.

## Statistical significance (Lago et al. 2021 protocol, added 2026-07-22)

Diebold-Mariano on daily loss differentials; Kupiec + Christoffersen on
bands. Source: `reports/backtests/2026-07-22_stats_tests.md`.

| Claim | DM p (one-sided) | Verdict |
|---|---|---|
| 1095d window beats 365d | 0.0009 | significant — promotion evidence solid |
| LGBM beats LEAR (2-yr) | 0.056 | **NOT significant at 5%** |
| LGBM beats TFT-730 ens-3 | 8.7e-09 | significant |
| LGBM beats naive | ~1e-70 | significant |

- Honest correction: the champion's MAE edge over LEAR (17.87 vs 18.24)
  does not clear the 5% significance bar on daily losses. Say "matches
  or slightly beats LEAR", not "beats".
- Bands: LEAR passes Kupiec (unconditional coverage), LGBM marginally
  fails (21.1% violations vs nominal 20%). BOTH fail Christoffersen
  hard — violations cluster on consecutive hours/days. Same lesson as
  the GPD test: the tail problem is conditional, not average-width.

## Band calibration (tail methods, 2-yr stored preds)

Symmetric CQR is the shipped method. Two challengers tested and rejected:

| Method | LGBM spike cover | LEAR spike cover | Verdict |
|---|---|---|---|
| Symmetric CQR (shipped) | 51.3% | 55.6% | stays |
| Asymmetric CQR | 51.3% | 56.3% | rejected (2026-07-17) |
| GPD upper tail (EVT) | 51.3% | 56.2% | rejected (2026-07-22) |

- Spike misses are conditional, not a band-width problem. No
  unconditional calibration moved spike coverage meaningfully.
- Source: `reports/backtests/2026-07-22_gpd_tail_*.csv`.

## Where the details live

- Model cards: `docs/model_cards/`.
- Campaign handovers: `docs/handovers/`.
- Comparison figures: `reports/figures/backtest_price/` (15 plots).
- Shadow track record: `docs/shadow_tally.md`.
