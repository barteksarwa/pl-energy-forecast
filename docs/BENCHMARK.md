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
   The 2026-07-27 validation review flagged the backtest training mask
   reaching past this cutoff (D-1 09:00-23:00). The fix shipped with a
   regression test, then follow-up analysis narrowed the finding: D-1
   prices are PUBLIC at decision time (fixed at the D-2 auction), so
   the price cutoff is now task-aware (`target_published` vs
   `decision_0900`) and the price tables were never flattered. The
   load-side impact is still to be bounded. The cutoff is tested in
   both modes, not just stated.
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
| **4-member ensemble (+ TFT, CQR)** | **16.89** | **0.605** | **80.0%** | **0.929** |
| Ensemble, 3-member variant (CRPS-weighted + CQR) | 17.34 | 0.622 | 79.9% | 0.926 |
| LGBM 1095d window (candidate) | 17.38 | 0.623 | 78.7% | — |
| LGBM 365d + CQR (champion) | 17.84 | 0.640 | 78.6% | 0.915 |
| LEAR + CQR (industry standard) | 18.46 | 0.662 | 79.5% | 0.911 |
| TFT-730 3-seed ensemble | 19.52 | 0.699 | 80.9% raw | — |
| Chronos-Bolt zero-shot + CQR | 21.93 | 0.787 | 79.9% | 0.891 |
| PatchTST-730 3-seed (trained) | 22.25 | 0.797 | 77.1% raw | — |
| TimesFM 2.5 zero-shot | 22.52 | 0.807 | 80.7% raw | 0.881 |
| Moirai univariate zero-shot | 23.69 | 0.849 | 77.0% raw | — |
| Moirai + covariates zero-shot | 24.86 | 0.890 | 75.5% raw | — |
| Naive (same hour yesterday) | 27.88 | 1.000 | 53.1% | 0.814 |

The 4-member row adds an archived TFT (3-seed, 730d) as a diversity
donor. It is scored on the 17,456 h intersection where all members
overlap — the TFT window ends 2026-07-14. New best on every gate
(DM p=2.3e-09 vs the 3-member blend; source `docs/RESULTS.md`).

Notes on provenance: ensemble/LGBM/LEAR/FM rows from the regenerated
2026-07-24 runs (window ends 2026-07-24); TFT/PatchTST/Moirai rows from
their documented campaign runs (TFT/PatchTST windows end 2026-07-15;
Moirai was regenerated 2026-07-24 on the same window as the FMs above);
1095d row from
the deep-history campaign — its matched-window comparison is against
365d = 17.87 on the SAME 07-14-ending window (delta 0.49), not against
the 17.84 above. Sources in `docs/RESULTS.md`.

## 5. Findings

1. **Combine, don't replace.** Both products win by correcting a strong
   external signal (TSO forecast; LEAR-style structure), not by
   replacing it.
2. **Ensemble diversity beats member strength.** A zero-shot univariate
   FM that is 4 MAE worse than the champion still improves the blend
   (17.34 vs 17.84, DM p=2.5e-04). Skill-weighting adds only 0.12 MAE
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
   The working answer is a conditional spike classifier (AUC 0.966),
   shipped as a daily-report flag in plain words.
8. **The champion's edge over LEAR is not significant** (DM p=0.056).
   We say "matches or slightly beats". The ensemble's edge IS
   significant.
9. **A dead model can still be the best donor.** TFT lost solo and was
   archived. Added to the blend it gives the new best (16.89 vs 17.34,
   DM p=2.3e-09), because deep-model errors decorrelate from
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
  hand-transcription errors after regenerated runs. All 24 fixed.
- **2 real protocol bugs** in the code, fixed with regression tests: the
  backtest training mask reached past the 09:00 cutoff, and the D-1 price
  vector picked up the D-2 shape on the day after spring DST.
- **Impact assessed, then narrowed.** Cutting the flagged hours cost
  the champion 0.11 MAE (17.95 vs 17.84) — but follow-up analysis
  showed those hours are PUBLIC for prices (D-2 auction), so the
  finding is retracted for the price task and the price tables were
  never flattered. It stands for the load task, where the impact is
  not yet bounded. The production daily loop never had the defect —
  it runs after gate closure and cannot see the future.
- The price prediction store is being rebuilt under the task-aware
  cutoff; a load-side bounding run is the open item. Full review:
  `docs/VALIDATION.md`.

No finding overturned a modeling conclusion. The gap was documentation
drift, not concealment.

## 7. Honest negatives (kept, not hidden)

| Attempt | Verdict | Where documented |
|---|---|---|
| PatchTST (365d + 730d) | loses to TFT and champion; encoder premise dead weight at 365d | model card, RESULTS |
| Asymmetric CQR | no spike-coverage gain | model_selection 10 |
| GPD/EVT upper tail | no spike-coverage gain | model_selection 13 |
| Moirai covariate mode | WORSE than univariate (DM-sig.) | learning 24 |
| LGBM "beats" LEAR | p=0.056 — not significant | RESULTS, stats tests |
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
- **Tests:** `make test` (102 tests; leakage, DST, metrics, conformal,
  ensemble, P&L accounting).

*Written 2026-07-24 (Phase 7). Update alongside RESULTS.md.*
