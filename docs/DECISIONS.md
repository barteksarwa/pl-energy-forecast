# Decision Log

Three lines per entry: context, decision, why. Newest on top.

---

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
