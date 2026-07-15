# Decision Log

Three lines per entry: context, decision, why. Newest on top.

---

**2026-07-16 — ENTSO-E merged for deep history; PSE stays canonical in overlap**
Context: token arrived. Cross-check over 18,287 overlap hours: mean |diff| 4.7 MW (0.03%), 1.6% of hours differ >1%.
Decision: canonical load/tso = PSE where present, ENTSO-E fills 2023-01→2024-06. Backup kept as *_pse_only.parquet. Report: reports/backtests/pse_vs_entsoe.csv.
Why: two independent routes agree — data trustworthy; 3.5 years unlock longer backtests and better net training.

**2026-07-15 — Challenger runs in shadow; forecasts tracked in git**
Context: ridge+TSO beat everything on backtest; promotion needs live proof, and CI runners are ephemeral.
Decision: challenger forecasts daily in shadow (scored, not official). Forecast CSVs are committed — the one exception to "no data in git".
Why: shadow days are the UAT evidence for promotion; committed forecasts are timestamped and tamper-evident — a desk-grade audit trail.

**2026-07-15 — TSO forecast admitted as a model feature**
Context: PSE publishes day D's demand forecast ~09:00:12 on D-1; our cutoff is 09:00.
Decision: treat it as known at the cutoff (12 s slack) and feed it to models. Models become forecast combiners.
Why: every desk post-processes the TSO forecast; beating it by combining with it is standard practice, not cheating. Documented in features/matrix.py.

**2026-07-14 — PSE API v2 as primary load source, ENTSO-E for deep history**
Context: ENTSO-E token stuck in email queue; PSE API v2 needs no key and has load + TSO forecast from 2024-06-14.
Decision: backfill and daily ops run on PSE now. ENTSO-E extends history to 2023 and cross-validates once the token arrives.
Why: unblocks the whole pipeline today; two independent sources for the same series is desk-grade hygiene anyway.

**2026-07-14 — Neighbor-country holidays deferred to Phase 2**
Context: PL trades power with DE, CZ, SK, LT, SE, UA; their holidays shift flows.
Decision: Phase 1 load models use PL calendar only. Neighbor holidays join in Phase 2 (price).
Why: PL demand follows the PL calendar; neighbor calendars move prices via cross-border flows, not PL load. Calendar module takes a country list, so adding them later is a config change.

**2026-07-14 — Load first, price second, on shared infrastructure**
Context: job research shows trading desks forecast price; utilities forecast load. Owner wants both markets open.
Decision: Phase 1 = load forecasting daily loop. Phase 2 = PL day-ahead price on the same pipeline. Not optional.
Why: load is the cleanest ops simulation with free data; load forecast then feeds the price model, like a real desk.

**2026-07-14 — Full unattended run deferred, POC automation kept**
Context: owner wants proof the loop can run alone, but not a 30-day commitment yet.
Decision: GitHub Actions cron as free POC for a 7–14 day trial (M9). Full 30-day push after UAT/prod split exists.
Why: proves automation cheaply; track record starts when the process is worth showing.

**2026-07-14 — Forecast cutoff time**
Context: backtests need a fixed "information available" moment.
Decision: forecasts for day D are made at 09:00 CET on day D-1.
Why: mirrors real desk practice before the 12:00 day-ahead auction. Leaves margin for data delays.

**2026-07-14 — Interpretable model is the primary model**
Context: owner knows deep learning; jobs demand explainability.
Decision: LightGBM quantile + SHAP is the "production" model. LSTM/transformer are challengers.
Why: EU energy employers ask "why is the forecast high today?" every single morning.
