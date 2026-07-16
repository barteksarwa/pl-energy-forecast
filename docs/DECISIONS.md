# Decision Log

Three lines per entry: context, decision, why. Newest on top.

---

**2026-07-16 — Price series: ENTSO-E EUR/MWh is canonical for modeling**
Context: two price sources exist. PSE csdac-pln (PLN, from 2024-06-14) and ENTSO-E (EUR, from 2023-01-01). Different currencies — cross-check needs an FX series we don't have.
Decision: `price_da_eur.parquet` (ENTSO-E) is the modeling target. PSE PLN stays for display and PLN-denominated portfolio work.
Why: 1.5 extra years of history, and EUR is what SDAC actually clears in. PLN conversion is presentation, not modeling.

**2026-07-16 — Price lags shift by local calendar days, not fixed 24h**
Context: first backtest crashed on 2023-10-29 (25h DST day): minus-24h from the last delivery hour lands inside the target day — real leakage, caught by the cutoff assert.
Decision: price lags = same local clock hour, k local days back. DST-ambiguous/nonexistent hours become NaN and the row drops.
Why: "yesterday's price" means local yesterday to the market. ~2 NaN hours per year per lag is honest; a silent 24h shift is leakage one day a year.

**2026-07-16 — LEAR is per-hour with robust-standardized asinh; pooled/raw variants rejected on evidence**
Context: three LEAR variants measured on the same 2-year walk-forward (17,480 h). Pooled model with same-hour lags: rMAE 1.29. Per-hour + D-1 day vector, asinh on raw prices: rMAE 1.11 (winter months up to 2.64 — sinh-back amplifies ~100x at 100 EUR level). Per-hour + asinh((p−med)/MAD): rMAE 0.744, wins all 25 months.
Decision: ship the third variant as `lear`. Transform per Uniejewski, Weron & Ziel (2018).
Why: matches the literature spec and the literature result. The two failed variants are documented in the model card so nobody re-walks this path.

**2026-07-16 — Strategic direction: Path A (get hired), Phase 2 = price forecasting**
Context: Job market research (Opus agent) + strategic analysis (Fable agent) completed 2026-07-16. Full findings in `docs/notes/job_market.md` and `docs/notes/strategic_direction.md`.
Decision: Priority is getting hired (3-6 months), not building a product. Phase 2 pivot: TGE day-ahead price forecasting before any other extension. Cut: TFT transformer challenger (explain loss is worth more), second EU zone, web UI.
Why: PSE publishes zone-level load forecast free — no paying customer. Trading-quant lane (best pay) wants price forecasts. Adding price doubles reachable roles. A job is the customer-discovery phase for any future product.

**2026-07-16 — Rolling 365-day window is the default; expanding window not adopted**
Context: 2-year ablation tested rolling-365 vs expanding for ridge and ridge_tso.
Decision: rolling 365-day default, no change.
Why: ridge+TSO rolling wins by 0.02pp; ridge ties. The 2022-23 energy-crisis regime biases expanding window. Full writeup in `04_window_ablation.tex`.

**2026-07-16 — TSO ffill for cron-before-publish timing gap**
Context: cron runs at 05:30 UTC (07:30 Warsaw); PSE publishes next-day TSO at ~09:00 Warsaw. Gap = ~90 min. Challenger failed with NaN when trying to use tomorrow's TSO as a feature.
Decision: forward-fill the TSO series before building tomorrow's feature matrix. The last published value (22:00 today) proxies tomorrow's shape until the real forecast lands.
Why: a stale TSO is better than no challenger. Long-term fix: shift cron to 10:00 UTC. Filed as known failure mode in ridge_tso model card.

**2026-07-16 — Shadow promotion tally started; target 14 consecutive valid days**
Context: ridge+TSO passed 12-month walk-forward (2.13% MAPE vs 5.60% naive). UAT rule (PLAN M9): run N shadow days, then decide.
Decision: target = 14 shadow days (two full weeks, covers weekday/weekend/holiday mix). Track in docs/shadow_tally.md. Day 1 = 2026-07-16 (first day with working weather forecast data).
Why: 14 days give the desk a valid week-over-week comparison. 7 days would miss any weekend anomaly.

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
