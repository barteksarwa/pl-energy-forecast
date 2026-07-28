# Job-readiness review — 2026-07-27

Critical external-style review. Reviewer persona: hiring manager for
PL/EU energy forecasting roles + senior forecasting engineer. Run as an
independent agent over the full repo (code, docs, reports, public-repo
history). Findings verified against source before reporting.

**Verdict: would interview, would not yet hire on the repo alone.**
Top ~5% of forecasting portfolios. Method survives scrutiny; several
claims do not. ~1 week of cleanup, no new modeling needed.

---

## Bottom line

Protocol discipline (walk-forward, DM tests, pre-declared gates, honest
negatives, conformal calibration, P&L conversion) is at junior-quant
working level. But the repo is oversold in exactly the places a
skeptical reader checks first. Three findings are offer-threatening if
the interviewer finds them before the candidate discloses them. All are
honesty-of-framing and artifact-hygiene problems.

## Blockers (offer-threatening)

1. **README headline vs shipped product.** README leads with "Load:
   beats the Polish TSO — ridge 2.08% vs 2.23%". `daily_run.py:90`
   ships `seasonal_naive_forecast`; daily report prints "Ours (naive,
   incumbent)"; 07-25 report shows naive 7.01% LOSING to TSO 6.04%.
   Ridge stuck in shadow 12 days; load challenger throwing the same
   Open-Meteo timeout since 07-17. Never disclosed in Live status.
   Fix: one disclosure line + fix the timeout so the gate can close.
2. **Flagship p=2.3e-09 has no artifact.** Quoted in README, BENCHMARK,
   RESULTS; `grep -rl "2.3e-09" reports/` returns nothing. Violates the
   repo's own VALIDATION rule ("no p-value without a CSV"). Also
   orphaned: p=2.5e-04, p=0.0596. The 4-member blend was out of the
   validation review's scope — most prominent number, least audited.
3. **ens4 gate measured against out-of-window baseline.** ens4 verified
   on the 17,456 h intersection; the ens3=17.34 comparison row is from
   the 17,696 h run. On the intersection, ens_equal=16.95 — CRPS
   weighting buys 0.06, not the story told. Restate on one window.
4. **README price table + only figure cite the retired 07-14 run.**
   `README.md:59` CSV says LEAR 18.242 / naive 27.961; README says
   18.5 / 27.9. `01_metrics_comparison.png` plots the stale CSV.
5. **"Top 3 drivers" is a hardcoded template.** `report.py:76-82` =
   literal strings; driver #3 admits "(not yet used by the model)".
   Real `top_drivers()` (shap_explain.py:54) is dead code. CLAUDE.md
   rule 3 unmet; HOW doc overstates. Price section has zero drivers.
6. **No CI runs tests or lint.** Only workflow is the daily cron.
   `make lint` fails with 43 ruff errors. 20-minute fix, highest
   effort-to-credibility ratio in the review.
7. **LEAR is ~1/3 of LEAR, labeled "implemented properly".** 27 price
   regressors vs canonical 96 (only D-1 gets the 24-vector; D-2/3/7 are
   same-hour scalars). `day_of_week` fed to LASSO as ordinal int 0-6
   (canonical uses dummies — this is a modeling error). `LassoCV(cv=5)`
   = plain KFold → folds select penalty using future data. Weron is at
   Wrocław; a Polish quant plausibly knows LEAR cold.
8. **15-min MTU absent — actively destroyed at ingest.**
   `pse_client.py:58`, `entsoe_client.py:27,75` resample to 1h. PL
   settles imbalance in 15-min; SDAC moving to 15-min MTU. Blocker for
   trading-desk lanes. Store native MTU, aggregate for models.
9. **No intraday/imbalance modeling.** `price_balancing.parquet` is
   fetched and never read by any code. Zero XBID/RDB/aFRR/mFRR hits.

## Strong findings (fix before applying)

- Today's E2 amendment re-introduced doc drift: withdrawn "~0.11"
  claim still live in BENCHMARK (22-26, 143-146) and README (135-137,
  153); BENCHMARK:145 states the direction backwards.
- HPO ran with `target_availability` default ("realtime") — different
  information set than the reported backtest. Third call site missed.
- Second hand-rolled walk-forward loop in `run_price_ablation.py:43-63`
  never got any cutoff fix; `y.reindex` without dropna.
- RES look-ahead disclosed but never quantified — `--drop-prefix`
  exists; run the ablation, get the number.
- `price_daily.py` reads `config/price_conformal.json` by RELATIVE
  path; any cwd change silently publishes the raw uncalibrated band
  (~72% coverage). Live product bug.
- Five blanket `except Exception` blocks; dead pipeline shows green CI.
- Zero tests on `src/pipeline/`; `redact()` (token-redaction!) untested.
- `test_backtest.py:63` leakage test neutered by `refit_every_days=999`
  — one fit, before any corruption. Flip to 1.
- No golden-value tests — 20% real MAE regression passes green.
- Track record = 7 authentic automated days, not 30 (07-14→17
  batch-committed/backdated on 07-21). Pitch the honest 7.
- Bot report `2026-07-25.md` retroactively overwritten by a human
  re-run (challenger n/a→4.25%, failure line deleted). Never edit a
  published report; append corrections.
- `shadow_tally.md` stale since 07-21 ("Consecutive valid days: 0"
  through seven green bot days).
- Model cards: retired numbers (17,480 h), no production-status field,
  no card for seasonal naive (the actual prod load model) or the spike
  classifier; `lear.md:93` claims "cross-border flows evaluated" —
  no code/data/report exists. Delete today.
- `VALIDATION.md` framing: "independent MRM-style review" reads as
  "I ran another LLM over my repo." Reframe as "adversarial LLM audit
  with refuter pass, 31 real defects found and fixed" — stronger and
  unattackable.
- No prepared interview answer for "how much of this did you write?"
  (CLAUDE.md sits at repo root of the local checkout).

## What holds up (keep saying it)

- Per-day feature rebuild in backtest; local-calendar-day DST lag
  shifts with NaT guards; six strong-form corruption leakage tests.
- DM + Kupiec + Christoffersen, with the honest "both fail
  Christoffersen" and "LGBM vs LEAR p=0.056 — matches, not beats".
- SHAP #1 driver (solar 18.70) independently confirmed by retrain
  ablation (+3.5 MAE) — two methods, same answer, on the merit order.
- P&L engine: DST-aware, 9 accounting tests, converts MAE to EUR.
- 123-test suite runs clean from fresh clone, no network, no data dir.
- career.md job-market research and prepared answers (ridge-vs-LSTM,
  asymmetric-CQR surprise, two hardest bugs) are measured, not
  memorized. Two answers quote stale run numbers (0.638/0.660 → 0.640/
  0.662).

## Fit by lane

- Lanes 3-4 (PL utility analyst, DS forecasting): over-qualified.
- Lanes 1-2 (prop/trading desks): thin without 15-min MTU + one
  imbalance-facing model. Currently a strong DA candidate applying to
  short-term power desks.

## Top 5 actions before applying

1. **Reconcile every claim with what ships** (1 day). Naive-in-prod
   disclosure line; delete cross-border claim; fix TTF provenance
   (DECISIONS:150); strip withdrawn 0.11 from BENCHMARK/README;
   repoint README:59 + regenerate the PNG. All deletions.
2. **Make every number regenerable** (1.5 days). Script that emits
   RESULTS/BENCHMARK tables from reports/*.csv + CI diff check. Run
   the missing DM tests or downgrade wording. Restate ens4 gate on
   one window.
3. **CI + lint + test fixes** (1 day). pytest+ruff workflow; fix 43
   lint errors; `refit_every_days=1` in the leakage test; test
   `redact()`; one end-to-end pipeline test.
4. **Wire `top_drivers()` into the daily report; add price drivers;
   run the RES-drop ablation** (1 day).
5. **Faithful LEAR + keep native 15-min MTU** (2-3 days). One-hot
   DoW/month, D-2/3/7 day-vectors, LassoLarsIC; stop resampling at
   ingest. Expect LEAR to improve and the LGBM edge to shrink —
   publish that.

Then: keep cron green (30 reports arrive by themselves), fix the
Open-Meteo timeout, apply while the record accumulates. In interviews,
volunteer: naive-in-production, LLM-assisted audit, LEAR
simplification — disclosed they are strengths, discovered they cost
the offer.
