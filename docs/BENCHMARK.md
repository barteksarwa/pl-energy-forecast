# Benchmark writeup — Polish day-ahead load & price, 2 years walk-forward

One person, one desk simulation, 20+ models, two years of out-of-sample
hours. This page is the publication-grade summary: protocol, master
tables, findings, honest negatives, and how to reproduce everything.

Canonical numbers live in `docs/RESULTS.md`. This document tells the
story around them. Blog draft; arXiv-style expansion is the stretch goal.

## 1. The task

- **Load:** hourly national load, PL bidding zone, day-ahead.
- **Price:** hourly day-ahead auction price (SDAC), EUR/MWh.
- Forecast decided at 09:00 D-1 — before the 12:00 auction.
- Output: P10 / P50 / P90, every hour, every day.

## 2. Protocol (what makes the numbers trustworthy)

1. **Hard information cutoff.** Only data observable at 09:00 D-1.
   Enforced by asserts and tests, incl. the 25-hour DST day that broke
   a naive "minus 24 hours" lag once. Documented bug, fixed, tested.
   The 2026-07-27 validation review flagged the training mask for
   using D-1 hours 09:00-23:00. Scope was narrowed the same day: DA
   prices for D-1 clear at auction on D-2, so the full D-1 curve is
   public at 09:00 D-1 — no leak for the price target. The engine is
   now target-aware (`target_availability`): price trains through D-1,
   load stops at 09:00 D-1. Both modes regression-tested (see §6 and
   the VALIDATION.md amendment).
2. **Walk-forward only.** Rolling refits (weekly for trained models,
   daily context for zero-shot). No random splits. Test period
   2024-07 → 2026-07, ~17.7k hours.
3. **Baselines always in the table.** Seasonal naive + the external
   benchmark (TSO forecast for load, LEAR for price). Losing models
   stay in the tables.
4. **Significance, not vibes.** Diebold-Mariano on daily losses
   (Lago et al. 2021 protocol), Kupiec + Christoffersen on bands.
   When an edge is not significant, we say "matches", not "beats".
5. **Pre-declared gates.** Challengers state their promotion criteria
   before the run. A model that fails its gate is archived with the
   number, not quietly re-tuned until it passes.
6. **Money as the final metric.** Every price model is also scored in
   EUR through a battery-arbitrage backtest (schedule on P50 at D-1,
   settle at actuals).

## 3. Master table — load (2-yr walk-forward, 17,450 h)

| Model | MAPE | Skill vs naive |
|---|---|---|
| **Ridge + TSO (combiner)** | **2.08%** | 0.63 |
| LightGBM + TSO | 2.12% | 0.62 |
| TSO day-ahead forecast | 2.23% | 0.60 |
| LSTM-attention + TSO (12-mo) | 2.43% | — |
| Ridge (no TSO) | 4.05% | 0.29 |
| Seasonal naive | 5.59% | 0.00 |

The story in one line: once the public TSO forecast is an input, a
ridge combiner beats every one of seven deep architectures we built.

## 4. Master table — price (2-yr walk-forward, ~17.7k h)

MAE in EUR/MWh. Capture = share of perfect-foresight battery-arbitrage
P&L (1 MW / 2 MWh / 0.85 RTE / 1 cycle, day-ahead only).

| Model | MAE | rMAE | Coverage (80% nom.) | P&L capture |
|---|---|---|---|---|
| **4-member ensemble (+ TFT, CQR)** | **16.88** | **0.604** | **80.0%** | **0.931** |
| Ensemble, 3-member variant (CRPS-weighted + CQR) | 17.33 | 0.621 | 79.9% | 0.928 |
| LGBM 1095d window (candidate) | 17.38 | 0.623 | 78.7% | — |
| LGBM 365d + CQR (champion) | 17.83 | 0.640 | 78.5% | 0.915 |
| LEAR + CQR (industry standard) | 18.46 | 0.662 | 79.5% | 0.912 |
| TFT-730 3-seed ensemble | 19.52 | 0.699 | 80.9% raw | — |
| Chronos-Bolt zero-shot + CQR | 21.82 | 0.783 | 79.9% | 0.891 |
| PatchTST-730 3-seed (trained) | 22.25 | 0.797 | 77.1% raw | — |
| TimesFM 2.5 zero-shot | 22.38 | 0.803 | 80.9% raw | 0.881 |
| Moirai univariate zero-shot | 23.69 | 0.849 | 77.0% raw | — |
| Moirai + covariates zero-shot | 24.86 | 0.890 | 75.5% raw | — |
| Naive (same hour yesterday) | 27.88 | 1.000 | 53.1% | 0.814 |

The 4-member row adds an archived TFT (3-seed, 730d) as a diversity
donor. It is scored on the 17,456 h intersection where all members
overlap — the TFT window ends 2026-07-14. On that SAME window the
3-member blend scores 17.35, so the gate delta is −0.48 (gate 0.10),
DM p=2.6e-09 (`reports/backtests/2026-07-28_stats_tests_ens_dm.csv`).
New best on every gate. Its capture (0.931) comes from the same-window
P&L artifact `2026-07-28_ens4_window_metrics.csv` — the main P&L table
is a different (17,720 h) window, so the two capture columns are not
directly comparable row-to-row.

Notes on provenance: ensemble/LGBM/LEAR/FM rows from the 2026-07-27
corrected-protocol rerun (window ends 2026-07-24, 17,720 h;
target-aware cutoff, timestamp-aligned FM wrappers);
TFT/PatchTST/Moirai rows from their documented campaign runs
(TFT/PatchTST windows end 2026-07-15; Moirai regenerated 2026-07-24);
1095d row from
the deep-history campaign — its matched-window comparison is against
365d = 17.87 on the SAME 07-14-ending window (delta 0.49), not against
the 17.83 above. Sources in `docs/RESULTS.md`.

## 5. Findings

1. **Combine, don't replace.** Both products win by correcting a strong
   external signal (TSO forecast; LEAR-style structure), not by
   replacing it.
2. **Ensemble diversity beats member strength.** A zero-shot univariate
   FM that is 4 MAE worse than the champion still improves the blend
   (17.33 vs 17.83, DM p=4.1e-04). Skill-weighting adds only ~0.1 MAE
   over equal weights — the diversity does the work.
3. **Training window is a first-class hyperparameter.** 365→1095 days
   is worth −0.49 MAE for free (DM p=0.0009). Half the "deep models
   lose" gap was window + seeds, not architecture.
4. **One-seed screening lies.** Three times in this project a 1-seed
   winner failed its 3-seed confirmation. Ensembles of 3 seeds are the
   minimum honest unit for deep models.
5. **Zero-shot FMs are strong free baselines, not champions.** Chronos
   beats a trained PatchTST with no training. But covariates + local
   training are worth 4 MAE — and Moirai shows covariates CANNOT be
   exploited zero-shot (they make it worse, DM-significant).
6. **Forecast value compresses in decisions.** The battery captures 81%
   of perfect-foresight value with a naive forecast. 10.6 MAE of model
   skill buys 11 points of capture. MAE ranks survived, but the EUR
   framing is what a desk should use ("~850 EUR/yr per MW per 0.5 MAE").
7. **Spikes are conditional, not a band-width problem.** Three
   unconditional calibration methods failed to move spike coverage.
   The working answer is a conditional spike classifier (AUC 0.967),
   shipped as a daily-report flag in plain words.
8. **Significance is a moving target — claim it only with an artifact.**
   On the 07-22 run the champion's edge over LEAR was NOT significant
   (p=0.056) and we said "matches, not beats". On the 07-27
   corrected-protocol run it clears the bar (p=1.9e-03,
   `2026-07-28_stats_tests_ens_dm.csv`) — three more test weeks plus
   the DST data fix moved it. Both artifacts kept; the discipline of
   not claiming early is the story.
9. **A dead model can still be the best donor.** TFT lost solo and was
   archived. Added to the blend it gives the new best (16.88 vs 17.35
   on one window, DM p=2.6e-09, artifact above), because deep-model
   errors decorrelate from
   trees/linear/FM. A second FM (TimesFM) added nothing (+0.57). What
   the blend wants is a different error STRUCTURE, not another strong
   member.
10. **Selection is not combination.** BOA weights were rejected (18.41
    vs 16.89): BOA minimizes regret against the single best expert, so
    it piles 99.6% of the weight on LGBM and forfeits the diversity
    averaging that drives the blend. Inverse-CRPS combination stays.

## 6. Independent validation (2026-07-27)

An agent-based model-risk review audited the numbers and red-teamed the
code. It is the audit a model-risk function would run.

- **31 confirmed findings** (adversarial: every finding had to survive a
  refuter). 24 were documentation drifts — stale citations and
  hand-transcription errors after regenerated runs. 23 fixed, one (A5) disclosed with a tightened note.
- **2 real code bugs** against the stated cutoff, fixed with regression
  tests: the training mask ignored the 09:00 decision moment, and the
  D-1 price vector picked up the D-2 shape on the day after spring DST.
- **Cutoff finding later scope-narrowed (same day).** DA prices publish
  one day ahead (auction on D-2 for delivery day D-1), so the full D-1
  curve is public at the 09:00 D-1 decision moment — the mask was never
  a leak for the PRICE target. The earlier "~0.11 MAE shared optimism"
  bound (champion 17.95 under the over-strict mask vs 17.84) is
  withdrawn for price; price tables stand as-is. The finding stays real
  for the live-observed LOAD target; load tables carry that small
  shared caveat until rerun. The production daily loop never had the
  defect either way.
- Full review and amendment: `docs/VALIDATION.md`.

No finding overturned a modeling conclusion. The gap was documentation
drift, not concealment.

## 7. Honest negatives (kept, not hidden)

| Attempt | Verdict | Where documented |
|---|---|---|
| PatchTST (365d + 730d) | loses to TFT and champion; encoder premise dead weight at 365d | model card, RESULTS |
| Asymmetric CQR | no spike-coverage gain | model_selection 10 |
| GPD/EVT upper tail | no spike-coverage gain | model_selection 13 |
| Moirai covariate mode | WORSE than univariate (DM-sig.) | learning 24 |
| LGBM "beats" LEAR (07-22 run) | p=0.056 then; p=1.9e-03 on the 07-27 rerun | RESULTS, stats tests |
| load_lags in price model | dead weight (−0.12 MAE without) | DECISIONS 07-22 |
| Chronos fine-tune | gate closed (rMAE 0.787 > 0.75) | DECISIONS 07-23 |
| Outage features | flat | backlog, PLAN |
| LGBM HPO (14 configs) | defaults survived; flat surface | RESULTS, DECISIONS 07-25 |
| BOA ensemble weights | 18.41 vs 16.89; selection, not combination | DECISIONS 07-27 |
| Rolling-90d spike threshold | AUC 0.955 vs 0.966; static stays | DECISIONS 07-27 |
| TimesFM as 4th blend member | added nothing (+0.57 MAE) | RESULTS, DECISIONS 07-27 |

## 8. Reproducibility appendix

- **Data:** ENTSO-E Transparency (token in `.env`), PSE raporty API,
  Open-Meteo archive + previous-runs. `make backfill` rebuilds
  everything from scratch (proven the hard way: the entire `data/`
  store was destroyed and rebuilt on 2026-07-24; all headline numbers
  reproduced — see DECISIONS incident entry).
- **Environment:** `uv sync` (locked versions); `--extra fm` for the
  foundation-model stack.
- **Backtests:**
  - load: `uv run python -m src.evaluation.run_2year_backtest`
  - price: `uv run python -m src.evaluation.run_price_backtest
    --models lear,lgbm_quantile --test-start 2024-07-16 --tag res`
  - calibration: `uv run python -m src.evaluation.run_price_calibration`
  - FMs: same runner, `--models chronos_bolt_zs --refit-days 1`
    (also `timesfm_zs`); Moirai via `scripts/run_moirai_zs.py`
    (scratch venv, numpy pin conflict).
  - ensemble: `uv run python -m src.evaluation.run_price_ensemble`
  - P&L: `uv run python -m src.evaluation.run_pnl`
- **Determinism:** LGBM/LEAR deterministic given data; deep models
  report 3-seed ensembles; zero-shot FMs deterministic single passes.
- **Tests:** `make test` (125 tests; leakage, DST, metrics, conformal,
  ensemble, P&L accounting).

*Written 2026-07-24 (Phase 7). Update alongside RESULTS.md.*
