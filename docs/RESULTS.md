# RESULTS.md — canonical results

One page. Every headline number in the repo lives here.
Other docs link here instead of copying numbers.
Update this file first when a campaign ends.

Updated: 2026-07-27.

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

> **Protocol note (2026-07-27, amended same day).** The validation
> review flagged the training mask for including D-1 hours 09:00-23:00
> (finding E2). The scope was later narrowed: DA prices for delivery
> day D-1 clear at auction on D-2, so the full D-1 curve is public at
> the 09:00 D-1 decision moment — no leak for the PRICE target. Price
> tables below stand as-is (the earlier "~0.11 shared bias" caveat is
> withdrawn). The finding stays real for the live-observed LOAD target:
> the engine now cuts load training at 09:00 D-1
> (`target_availability="realtime"`); load tables above predate that
> cutoff and carry a small optimistic bias, shared by every trained
> model, until rerun. Full story: `docs/VALIDATION.md`, amendment.

## Price — day-ahead auction price (EUR/MWh)

2-year walk-forward. Test 2024-07-16 → 2026-07-24, 17,720 hours
(rerun 2026-07-27 under the target-aware cutoff protocol; the deep
rows keep their campaign windows ending 2026-07-14 — noted).
Source: `reports/backtests/2026-07-27_price_conformal_summary.csv`.

| Model | MAE | rMAE | 80% band coverage |
|---|---|---|---|
| **LGBM quantile + CQR** (champion) | **17.83** | **0.640** | 78.5% |
| LEAR + CQR | 18.46 | 0.662 | 79.5% |
| TFT ens-3 (365d windows, to 07-14) | 19.71 | 0.706 | 79.6% |
| PatchTST (365d windows, to 07-14) | 22.98 | 0.823 | 69.5% |
| Naive (1-day) | 27.88 | 1.000 | 53.1% |

- rMAE = MAE relative to naive. Below 1.0 beats naive.
- CQR = conformalized quantile regression. Fixes band coverage honestly.
- LEAR wins interval quality (Winkler 88.9 vs 90.7). LGBM wins MAE —
  and the edge IS significant on this corrected run (DM p=1.9e-03,
  `2026-07-28_stats_tests_ens_dm.csv`); see the significance section
  for the history of this claim.
- Validation note (2026-07-27): this table previously mixed the older
  07-14-window run with newer artifacts across docs. It now cites one
  run; the independent review caught it (`docs/VALIDATION.md`).

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
- Fresh rerun (last 180 days, corrected protocol, 2026-07-28): price
  history +3.46, RES +3.39, calendar +0.59, TSO load +0.47, load lags
  +0.03. Artifact: `2026-07-28_price_group_ablation.csv`.
- The RES ablation also bounds the DISCLOSED look-ahead: the ENTSO-E
  RES series publishes ~18:00 D-1 (after gate; standard EPF proxy —
  DECISIONS 2026-07-16). Everything that proxy is worth is +3.39
  EUR/MWh; the flattery vs a bidder's own 09:00 RES forecast is some
  fraction of that. Known, quantified, on the table.
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
- More history helps up to 3 years, then the 2021-22 crisis regime
  starts hurting (1460d worse than 1095d).
- Robustness (2026-07-23): on the 5-yr crisis test 1095d does not
  degrade (17.37 vs 17.49 — runs differ by 120 test hours, so this is
  a direction check, not a paired test; no DM artifact exists for it;
  2022 slice rMAE 0.678). LEAR also improves at 1095d (18.04 vs
  18.24) — the gain is a data property, not an LGBM quirk.
- Caveat: spike MAE slightly worse at 1095d (63.6 vs 60.7); spike
  handling is the classifier's job, not the point model's.
- RECOMMENDATION (evidence complete): promote 1095d as the champion
  training window. Owner decision pending; daily loop still on 365d.

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
| Champion LGBM (365d) | 17.83 | 0.640 | 78.5% |
| TFT-730 ens-3 (to 07-14) | 19.52 | 0.699 | 80.9% raw |
| **Chronos-Bolt zero-shot** | **21.82** | **0.783** | 79.9% |
| PatchTST-730 ens-3 (trained!, to 07-14) | 22.25 | 0.797 | 77.1% raw |
| TimesFM 2.5 zero-shot | 22.38 | 0.803 | 80.9% raw |
| Naive | 27.88 | 1.000 | — |

- A pretrained model with NO training on our data and NO covariates
  beats our trained PatchTST. Sic transit patch attention.
- Chronos beats TimesFM (DM p=0.012). TimesFM's raw band is the best
  calibrated of any model out of the box (80.9% vs nominal 80%).
- Champion beats both FMs at p<1e-25; DM artifact for all FM pairs on
  the 07-27 rerun preds: `2026-07-28_stats_tests_fm_dm.csv` (an earlier
  "all gaps DM-significant" line predated any artifact; the review
  caught it, and the artifact is now regenerated with each rerun).
- Phase 6 fine-tune gate (rMAE < 0.75): closed.
- Sources: `reports/backtests/2026-07-27_price_chronos2yr_summary.md`,
  `2026-07-27_price_timesfm2yr_summary.md`. FM wrappers align
  forecasts by timestamp since 2026-07-27 (the 09:00-cutoff incident —
  `docs/VALIDATION.md` amendment); the small MAE shifts vs the 07-24
  run (21.93 → 21.82) come from the corrected protocol + window.

**Spike classifier** (2-yr walk-forward, top-5% hours, train-window
labels): AUC 0.967, Brier 0.034, precision@2 0.743 (07-27 rerun,
`2026-07-27_spike_screen.md`; the original 07-23 run scored 0.966 /
0.736). Gate 0.80 passed — promoted to a daily-report line.
Deterministic model; seed sweep vacuous (42 and 7 bit-identical).

Reliability check (2026-07-23, `src/evaluation/spike_reliability.py`):
- Stable across regimes: ROC-AUC 0.964-0.969 in each of 2024/25/26.
- PR-AUC 0.63 vs 0.05 base rate (12.6x lift over random flagging).
- Probabilities are OVER-confident at the top: predicted >90% -> only
  70% observed. Flag at p>=0.5 hits 63% precision, 53% recall.
- The daily report therefore speaks in historical precision ("about
  6 in 10 flagged hours"), not raw probability. Calibration (isotonic)
  is a possible refinement; not needed for a flag.
Source: `reports/backtests/2026-07-22_spike_screen.md`.

## CRPS-weighted ensemble (2026-07-24) — NEW BEST price forecast

Members (pre-declared): champion LGBM+CQR, LEAR+CQR, Chronos+CQR.
Weights: inverse trailing-60d crps3, past-only, equal-weight warm-up.

| Model | MAE (2-yr) | rMAE | Coverage | Winkler |
|---|---|---|---|---|
| **ens_crps_cqr** | **17.33** | **0.621** | **79.9%** | **85.2** |
| ens_crps (raw blend) | 17.33 | 0.621 | 83.9% | 85.5 |
| ens_equal | 17.43 | 0.625 | 83.9% | 86.1 |
| LGBM champion (365d) | 17.83 | 0.640 | 78.5% | 90.7 |
| LGBM (1095d window) | 17.35 | 0.622 | — | — |

- ALL pre-declared gates pass: −0.50 MAE (gate 0.15), DM p=4.1e-04
  (`2026-07-28_stats_tests_ens_dm.csv`), wins every test year
  (2024/25/26), Winkler improves.
- The diversity does the work: a zero-shot univariate FM adds skill to
  two structural models even though it is 4 MAE worse alone.
- Over-coverage FIXED (2026-07-24): a second rolling-CQR pass on the
  blended band tightens it (Q negative when over-covered). Coverage
  83.9% → 79.9%, Winkler 85.5 → 85.2, MAE unchanged (P50 untouched).
  (Validation note 2026-07-27: an earlier draft quoted Winkler 84.7
  from the pre-incident run; the regenerated artifact says 85.2.)
  `ens_crps_cqr` is the promotion candidate.
- Moirai (both variants) excluded by the pre-declared member rule
  (best FM only). See FM section: covariates HURT zero-shot Moirai
  (24.86 vs 23.69, DM p=8.3e-06 —
  `2026-07-28_stats_tests_fm_dm.csv`) — covariate skill needs training.
- Sources: `reports/backtests/2026-07-27_price_ensemble_summary.md`,
  `2026-07-24_price_moirai2yr_summary.csv`.

**4-member blend with TFT (2026-07-27) — NEW BEST price forecast.**
Members: LGBM+CQR, LEAR+CQR, Chronos+CQR, TFT-730 ens-3+CQR. TFT's
full-2yr preds survive under `reports/sensitivity/tft/`; evaluated on
the 17,456h intersection (TFT window ends 2026-07-14).

| Model | MAE | rMAE | Coverage | Winkler | P&L capture |
|---|---|---|---|---|---|
| **ens4_tft (CQR)** | **16.88** | **0.604** | **80.0%** | **82.6** | **0.928** |
| ens3 (same 17,456h window) | 17.36 | 0.621 | 79.9% | 85.2 | 0.926 |
| ens_equal (4 members) | 16.93 | 0.606 | 85.7% raw | 83.8 | — |
| TFT-730 ens-3 alone | 19.53 | 0.699 | 79.4% | 97.5 | — |

- One window for the gate: ens3 rescored on the SAME 17,456h
  intersection (17.36) — an earlier draft compared against ens3's
  17,696h run, mixing windows (review finding). Gate still passes:
  −0.48 MAE (gate 0.10), DM p=2.6e-09
  (`2026-07-28_stats_tests_ens_dm.csv`), wins 2024/25/26, coverage
  nominal, Winkler best ever.
- Honest decomposition: equal weights already give 16.93 on this
  window — CRPS weighting buys the last 0.06; the TFT member does the
  heavy lifting.
- The lesson: TFT lost solo (archived 2026-07-21) but is the best
  diversity donor tested — deep-model errors decorrelate from
  trees/linear/FM. TimesFM (4th member, FM like Chronos) added
  NOTHING (+0.57). Diversity of ERROR STRUCTURE matters, not member
  count or member strength.
- Operational cost is the promotion question: 3-seed TFT in the
  daily loop = MPS inference + monthly refits (~hours/month).
- Source: `reports/backtests/2026-07-27_price_ensemble_tft_summary.md`.

**Blend on 1095d members (2026-07-24) — tested, NOT adopted.**
Pre-declared gates: beat ens-365 (17.34) with DM p<0.05, coverage
78-82%, Winkler not worse.

| Blend | MAE | rMAE | Coverage | P&L capture |
|---|---|---|---|---|
| ens_crps_cqr (365d members, shipped) | 17.34 | 0.622 | 79.9% | 0.926 |
| ens_crps_cqr_1095 | 17.18 | 0.616 | 79.9% | 0.925 |

- MAE gate passes (−0.155) but **DM p=0.0596 — not significant**, and
  the 1095d blend LOSES the 2026 slice (18.36 vs 18.05). P&L identical.
- Finding: window gain and ensemble diversity are partial substitutes.
  Solo, 1095d is worth −0.49 MAE (significant); inside the blend the
  same information is mostly already recovered by diversity — the
  blend's edge over its best member shrinks from −0.50 to −0.17.
- Verdict (same standard as LGBM-vs-LEAR): "matches or slightly
  beats" — no member switch on this evidence. The 1095d promotion for
  the SOLO champion is a separate, still-solid case (DM p=0.0009).
- Spike MAE worse on 1095d blend (63.9 vs 61.5) — the known 1095d
  spike caveat carries through.
- Source: `reports/backtests/2026-07-24_price_ensemble_1095_summary.md`.

## LGBM price HPO (2026-07-25) — defaults survive

14 configs, 1095d windows, screen year 1 / confirm year 2 (top-3 +
control only; confirm never used for selection). Gate: beat control
by ≥0.10 MAE on both years.

| Config | Screen MAE | Confirm MAE | Verdict |
|---|---|---|---|
| control (shipped defaults) | 17.62 | 16.98 | stays |
| lr02_n1500 (best) | 17.55 | 16.91 | gate failed |
| lr03_n1000 | 17.54 | 16.97 | gate failed |
| leaves127 | 17.61 | 17.10 | gate failed |

- NO config clears the gate. The M4 "conservative defaults" were
  already near the optimum for this feature set — the HPO surface is
  flat, the honest and common LGBM outcome on engineered features.
- Side finding: `noloadlags` at 1095d is +0.09 WORSE (17.71 vs 17.62);
  at 365d it was −0.12 better. The load-lags verdict is
  window-conditional — third documented ablation sign-flip.
- Source: `reports/backtests/2026-07-25_lgbm_price_hpo.csv`.

## Battery-arbitrage P&L (2026-07-24) — forecasts in EUR

1 MW / 2 MWh / 0.85 round-trip / 1 cycle/day. Schedule from P50 at
D-1 (per-day LP), settle at actual DA prices. Day-ahead only.
Same 723 days for every model. Capture = P&L / perfect-foresight P&L.
(07-27 corrected-protocol rerun; the review earlier caught a stale
copy of the pre-regeneration run in this table.)

| Model | EUR/day | Capture | Loss days |
|---|---|---|---|
| Perfect foresight | 221 | 1.000 | 0% |
| **ens_crps_cqr** | **206** | **0.928** | 1.4% |
| LGBM champion | 203 | 0.915 | 1.5% |
| LEAR | 202 | 0.912 | 1.5% |
| Chronos zero-shot | 197 | 0.891 | 2.6% |
| TimesFM zero-shot | 195 | 0.881 | 2.6% |
| Naive (yesterday) | 180 | 0.814 | 4.4% |

- MAE rank == capture rank, no flips (PLAN watch item closed).
- Value compresses: naive is 10.6 MAE worse yet captures 81% —
  storage needs hour ORDERING, not price level.
- Ensemble edge over champion: +2.93 EUR/day/MW (~1,070 EUR/yr/MW;
  artifact gives 205.575 vs 202.642).
  Pays for its complexity at portfolio scale, not on one battery.
- Sources: `reports/backtests/2026-07-27_pnl_summary.md`,
  `reports/figures/pnl/cumulative_pnl.png`. Engine:
  `src/evaluation/pnl.py` (9 accounting tests, DST-aware).

## Statistical significance (Lago et al. 2021 protocol, added 2026-07-22)

Diebold-Mariano on daily loss differentials; Kupiec + Christoffersen on
bands. Source: `reports/backtests/2026-07-22_stats_tests.md`.

| Claim | DM p (one-sided) | Verdict |
|---|---|---|
| ens4_tft beats ens3 | 2.6e-09 | significant (`2026-07-28_stats_tests_ens_dm.csv`) |
| ens3 beats champion | 4.1e-04 | significant (same artifact) |
| 1095d window beats 365d | 0.0009 | significant — promotion evidence solid |
| LGBM beats LEAR (2-yr) | 1.9e-03 | significant on the 07-27 corrected run — see history below |
| LGBM beats TFT-730 ens-3 | 8.7e-09 | significant |
| LGBM beats naive | ~1e-70 | significant |

- History of the LEAR claim, kept on purpose: on the 07-22 run
  (17.87 vs 18.24, pre-E1-DST-fix data, window to 07-14) the edge was
  NOT significant (p=0.056) and every doc said "matches, not beats".
  On the 07-27 corrected-protocol run (17.83 vs 18.46, 17,720 h) it
  clears the bar (p=1.9e-03). Three more test weeks + the D-1
  price-vector DST fix moved it. Both numbers have artifacts; the
  0.056 stays quoted here because the discipline — not claiming the
  edge before the data supported it — is the point.
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

- Benchmark writeup (story + master tables): `docs/BENCHMARK.md`.
- Model cards: `docs/model_cards/`.
- Campaign handovers: `docs/handovers/`.
- Comparison figures: `reports/figures/backtest_price/` (15 plots).
- Shadow track record: `docs/shadow_tally.md`.
