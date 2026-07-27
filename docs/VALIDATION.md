# VALIDATION.md — Independent model-validation review (draft)

**Review date:** 2026-07-26. **Remediation status noted as of:** 2026-07-27.
**Reviewer:** independent agent-based review (MRM style), commissioned by the desk owner.
**Scope owner:** one-person forecasting desk, PL day-ahead load and price.

---

## 1. Scope and method

Three passes, run by separate agents:

1. **Numbers audit.** Every headline number in `docs/RESULTS.md` and `docs/BENCHMARK.md` was traced to a primary artifact in `reports/`. Artifacts were re-opened with pandas, not trusted from prose.
2. **Leakage red-team.** Source review of the price pipeline against the stated rule: forecast for day D is decided at 09:00 on D-1. Modules covered: feature builders, the walk-forward loop, conformal calibration, LEAR bands, ensemble sweeps, P&L engine.
3. **Adversarial confirmation.** Every candidate finding went to a refuter agent. The refuter tried to kill it. Only findings that survived are listed here. Unconfirmed claims were dropped.

Out of scope this pass: load-side feature code review in depth, CI workflows, the 4-member TFT blend (published 2026-07-27, after the review window).

**Headline verdict first.** No finding overturns a modeling conclusion. The champion ordering, the ensemble gates, and the P&L ranking all survive. The defects are: stale citations, hand-transcription errors, three significance claims with no artifact, and two real code bugs against the stated cutoff discipline.

Totals: **31 confirmed findings — 11 major, 20 minor.** 24 documentation, 7 code.

---

## 2. Verification results

### What reproduced cleanly

| Area | Verdict |
|---|---|
| Load master table (RESULTS + BENCHMARK) | Clean. No confirmed discrepancy. |
| Deep-model campaign tables (TFT/PatchTST, loss decomposition) | Clean. |
| Deep-history window sweep (365/730/1095/1460 MAEs) | Clean. Point values match artifacts. |
| Feature-sensitivity / ablation tables | Clean. |
| Spike classifier metrics and reliability numbers | Clean. |
| 2-yr DM stats table (1095d, LEAR, TFT, naive) | Clean. Matches `2026-07-22_stats_tests_dm.csv`. |
| Band-calibration table (CQR / asym / GPD) | Clean. |
| Blend-on-1095d table and verdict | Clean. |
| LGBM HPO table and gate verdicts | Clean except one 0.01 rounding slip (D6). Verdicts unaffected. |

### Confirmed discrepancies

Severity: **M** = major, m = minor. Status: doc fixes shipped 2026-07-27 (see DECISIONS); code fixes open.

**A. Two canonical docs cited two different runs for the same benchmark.**
RESULTS used the 07-14-ending run (17,480 h). BENCHMARK used the regenerated 07-24-ending run (17,696 h). Same tables, different numbers.

| # | Sev | Where | Issue | Status |
|---|---|---|---|---|
| A1 | **M** | RESULTS L34 vs BENCHMARK L62 | LEAR+CQR MAE 18.24 vs 18.46 across docs | fixed 07-27 |
| A2 | m | RESULTS L33 vs BENCHMARK L61 | Champion MAE 17.87 vs 17.84, coverage 78.9 vs 78.6 | fixed 07-27 |
| A3 | m | RESULTS L37 vs BENCHMARK L69 | Naive 27.96 vs 27.88 — rMAE denominators differ | fixed 07-27 |
| A4 | m | RESULTS FM table vs BENCHMARK | Chronos 21.98 (stale run) vs 21.93 | fixed 07-27 |
| A5 | m | BENCHMARK master table | 1095d row (07-14 window) tabled against 365d champion from the 07-24 window; matched delta is 0.49, not 0.46 | disclosed; note tightened |
| A6 | m | BENCHMARK provenance note | Moirai window mis-assigned to the 07-15 campaign; actually the 07-24 regeneration | fixed 07-27 |

**B. Significance claims with no artifact.**

| # | Sev | Where | Issue | Status |
|---|---|---|---|---|
| B1 | **M** | RESULTS FM section | "Chronos beats TimesFM (DM p=0.011)... all gaps DM-significant" — no DM artifact contained any FM pair; p=0.011 existed only in the doc | fixed 07-27: real artifact computed (`2026-07-27_stats_tests_fm_dm.csv`, p=0.0095) |
| B2 | **M** | RESULTS + BENCHMARK | Moirai "covariates hurt, DM-significant" — MAE ordering real, significance unsupported | fixed 07-27: artifact gives p=8e-06 |
| B3 | m | RESULTS deep-history | Crisis check "17.37 vs 17.49, DM p=0.17" — runs differ by 120 test hours (not paired); p-value in no artifact | fixed 07-27: downgraded to direction check |

**C. Ensemble section: transcription errors vs its own cited artifact.**
All against `2026-07-24_price_ensemble_summary.csv`.

| # | Sev | Claimed | Artifact | Status |
|---|---|---|---|---|
| C1 | **M** | ens_crps_cqr Winkler 84.7 ("85.2→84.7") | 85.2 (real transition 85.5→85.2); 84.7 appears in no artifact | fixed 07-27 |
| C2 | **M** | Spike coverage "57.4%→55.3%" | 55.8→55.0; neither quoted value exists in any artifact | fixed 07-27 |
| C3 | m | rMAE 0.620 (ens rows) | 0.622 — every other doc says 0.622 | fixed 07-27 |
| C4 | m | ens_crps coverage 84.2%, Winkler 85.2 | 83.9%, 85.5 | fixed 07-27 |
| C5 | m | ens_equal 17.46 / 0.624 / 84.3% / 85.9 | 17.45 / 0.626 / 84.0% / 86.2 | fixed 07-27 |
| C6 | m | LGBM row (17.87 / 78.9% / 89.5) inside a table sourced from the 07-24 run | 17.84 / 78.6% / 90.3 — table mixed two windows | fixed 07-27 |
| C7 | m | Gate gain "−0.53 MAE" | −0.50 on the cited run | fixed 07-27 |

**D. Battery P&L table: stale copy of an earlier run.**
All against `2026-07-24_pnl_summary.csv` (722 days).

| # | Sev | Issue | Status |
|---|---|---|---|
| D1 | **M** | "Same 713 days" — artifact says 722 for all models | fixed 07-27 |
| D2 | **M** | Every capture rate low: 0.924/0.914/0.908/0.890/0.813 vs 0.926/0.915/0.911/0.891/0.814 | fixed 07-27 |
| D3 | **M** | Loss-day column off for 5 of 7 rows | fixed 07-27 |
| D4 | **M** | Cross-file: BENCHMARK matched the artifact, RESULTS did not | fixed 07-27 |
| D5 | m | RESULTS internally inconsistent (capture 0.924 at L243 vs 0.926 at L197) | fixed 07-27 |
| D6 | m | Ensemble edge "+2.3 EUR/day"; artifact diff 2.353 | fixed 07-27 |
| D7 | m | Chronos/TimesFM capture rounded down (0.890/0.880 vs 0.891/0.881) | fixed 07-27 |
| D8 | m | HPO leaves127 screen MAE 17.62; artifact rounds to 17.61 | fixed 07-27 |

**E. Code findings** — detailed in Section 3.

| # | Sev | Where | Issue | Status |
|---|---|---|---|---|
| E1 | **M** | `src/features/price_lags.py:103` | D-1 price vector built from D-2 on the day after spring DST | FIXED 07-27 (regression test) |
| E2 | **M** | `src/evaluation/backtest.py:54` | Training target includes D-1 hours after the 09:00 cutoff | FIXED 07-27 (test; impact bound ~0.11 MAE; spike-screen loop fixed too) |
| E3 | m | `src/evaluation/conformal.py:69` etc. | Linear-interpolated quantile weakens the finite-sample CQR guarantee | FIXED 07-27 ('higher'; effect ~0 measured) |
| E4 | m | `src/evaluation/conformal.py:137-144` | Asymmetric path sizes the upper tail from the lower-tail count; no upper guard | FIXED 07-27 (per-tail sizing) |
| E5 | m | `src/evaluation/run_ensemble_sweeps.py:117-137` | Blend "CQR window sweep" stacks a second CQR on a fixed 90d first pass; sweep mislabeled | disclosed 07-27 (note on the sweep report; verdict discarded) |
| E6 | m | `src/models/price.py:157-179` | LEAR raw band from in-sample residuals — optimistic width | open |
| E7 | m | `src/features/price_matrix.py:57` | RES forecast post-dates gate closure — disclosed proxy, not a hidden leak | accepted design choice |

---

## 3. Leakage assessment per module

One line each. "Clean" means the red-team found no cutoff violation.

- **`src/features/price_lags.py` (lags):** clean — lags shift by local calendar day via `_shift_local_days`; DST-verified.
- **`src/features/price_lags.py` (`daily_price_vector`):** defect — `Timedelta(days=1)` is 24 absolute hours, so the day after spring DST silently gets the D-2 price shape; stale-window bug, one day per year, wrong direction for leakage but a correctness bug.
- **`src/features/price_matrix.py` (RES covariate):** known look-ahead vs the 12:00 gate; disclosed in docstring, DECISIONS, and model card; standard EPF convention (Lago et al. 2021); accepted.
- **`src/evaluation/backtest.py`:** breach — `dates < day` lets the training target include D-1 09:00–23:00 actuals, which do not exist at the decision moment; no leakage of day D itself; impact small against a 365-day window but it violates the stated hard rule.
- **`src/evaluation/conformal.py`:** temporally clean (trailing-window scores only); defects are statistical — interpolated quantile and asymmetric-path sample sizing can under-deliver the coverage guarantee.
- **`src/models/price.py` (LEAR):** temporally clean — MAD/asinh stats and residuals come only from the fit window; the raw P10/P90 band uses in-sample residuals, so it under-covers (calibration defect, not leakage).
- **`src/evaluation/run_ensemble_sweeps.py`:** no leakage; the blend branch measures a double-CQR band no production path produces, so its sweep verdict is unreliable; the LGBM branch is a genuine single-pass sweep.
- **Ensemble weighting (`run_price_ensemble`):** clean — trailing-60d inverse-CRPS weights, past-only, equal-weight warm-up.
- **`src/evaluation/pnl.py`:** clean — schedules on P50 at D-1, settles at actuals; DST-aware; 9 accounting tests.
- **Load pipeline:** not re-reviewed in depth this pass; existing leakage and DST tests are the current evidence.

---

## 4. Known limitations the desk already documents

Fairness requires listing what the desk disclosed before this review:

- RES day-ahead forecast is a bid-time proxy; caveat repeated in DECISIONS and the model card.
- Champion's edge over LEAR is not DM-significant (p=0.056); docs say "matches or slightly beats".
- Both models fail Christoffersen: band violations cluster. Documented, with the conditional-tail diagnosis.
- Spike coverage ~51–56% is a known conditional problem; three calibration methods were tested, failed, and kept in the record; the spike classifier is the shipped answer.
- Spike classifier over-confidence at high probabilities is measured and worked around in the daily report.
- Ablation verdicts are window-conditional; three sign flips documented.
- One-seed screening picked a mirage three times; 3-seed minimum adopted and written down.
- The 2026-07-24 data-store wipe is logged as an incident; the rebuild reproduced headline numbers.
- The 07-19→21 forecast outage is logged as FAILED days, not backfilled.
- P&L scope is explicit: day-ahead only, no intraday, no fees; comparator, not a business case.
- Losing models stay in the tables ("honest negatives").

This is a genuinely strong disclosure culture. The failures found here are drift and transcription, not concealment.

---

## 5. Recommendations

Prioritized. P1 = fix before quoting the numbers externally.

**P1 — protocol correctness (DONE 2026-07-27 — both fixed, tested, impact bounded):**
1. Fix `backtest.py:54`. Cap the training target at the 09:00 D-1 UTC cutoff, not the day boundary. Add a test. Re-run one champion backtest to bound the impact; expect small, but measure it.
2. Fix `daily_price_vector`. Derive D-1 with the local-day shift helper, not `Timedelta(days=1)`. Add a spring-DST regression test (target day 2024-04-01).

**P2 — claims discipline (mostly done 07-27, make it structural):**
3. Rule: no p-value or "significant" in any doc without a CSV artifact in `reports/`. The FM DM artifact now exists; keep the rule for future claims.
4. One canonical run per table. Never mix windows in a table without a per-row window column.
5. Stop hand-transcribing. Add a small script that regenerates RESULTS tables from the artifact CSVs, and a CI check that diffs doc numbers against artifacts. This single control would have prevented ~20 of the 24 documentation findings.

**P3 — calibration methods (open):**
6. `conformal.py`: use `np.quantile(..., method="higher")` for the finite-sample guarantee; size each tail from its own score count; add a min-length guard to the `latest_offset*` paths.
7. `run_ensemble_sweeps.py`: sweep CQR on an unconformalized blend so the sweep measures the shipped configuration.
8. LEAR raw bands: either wrap in rolling CQR by default or label them "uncalibrated, in-sample" wherever reported. Shipped numbers already use CQR, so this is a labeling fix.

**P4 — cosmetic:**
9. Fix the leaves127 rounding slip (17.61). Re-check all 2-d.p. roundings when the doc-checker from item 5 lands.

**Overall opinion.** The modeling conclusions are sound and unusually well evidenced. The control gap is documentation drift after regenerated runs, plus two code defects against the stated cutoff. Fix P1 and automate P2.5, and this desk's numbers would pass a second independent review without findings.
---

## Addendum (2026-07-27, post-review follow-up)

Finding E2's scope was narrowed by market mechanics after the fix
shipped. D-1 day-ahead prices clear at the D-2 auction (~13:00 local),
so at the 09:00 D-1 decision moment the full D-1 price vector is
public. Training a PRICE model through the end of D-1 is therefore
legitimate; the blanket 09:00 cutoff was over-strict for that task
(and broke context-based zero-shot models when applied). E2 STANDS
for the LOAD task, where the target is a physical actual that does
not exist yet. The training cutoff is now task-aware
(`walk_forward_backtest(train_cutoff=...)`, both modes tested).
The "~0.11 shared flattery" bound is retracted for the price tables
and remains to be measured for the load tables.
