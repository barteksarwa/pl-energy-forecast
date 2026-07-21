# Decision Log

Three lines per entry: context, decision, why. Newest on top.

---

**2026-07-21 — City weather weights updated to official GUS 2026 data**
Context: config weights were "approximate metro population, rounded" with no source. Owner asked for latest GUS. The 2026 edition was published the same day (2026-07-21).
Decision: weights = city population in millions from GUS "Powierzchnia i ludność w przekroju terytorialnym w 2026 r." (Tabl. 20, as of 2025-12-31), 3 decimals. Lublin (327k) now ranks above Bydgoszcz (321k).
Why: traceable source beats a rounded guess. Effect on the load model is tiny (weather group ablation +0.08 pp) — no re-benchmark needed. Population weighting stays a demand proxy; it is the wrong weighting for RES weather, which is a Phase 4 topic.

**2026-07-21 — Docs consolidated; canonical results page added**
Context: numbers were scattered across README, model cards, handovers, and reports; several docs went stale (PLAN header, README status table, TFT card).
Decision: `docs/RESULTS.md` is the single source for headline numbers; other docs link to it. Superseded backtest summaries and dead specs removed from the tree (git history keeps them).
Why: one page to update means numbers stop drifting. Git history keeps everything anyway.

---

**2026-07-20 — Deep models were window-handicapped; encoder redundancy was an artifact**
Context: overnight robustness runs — PatchTST/TFT ablations at 730d training windows + LGBM 730d ablation + cross-model table.
Decision: quote ablation verdicts WITH their training window. At 730d: PatchTST encoder +2.5 (was −0.4), TFT 19.12 MAE at 79.6% coverage. Deep-model re-benchmark at 730d windows is a candidate future milestone; champion unchanged (LGBM 17.87, extracts most from history +3.95).
Why: ablation conclusions proved conditional on train window — 3 seeds reversed the sign. Honest reporting requires the qualifier; LGBM still wins outright.

**2026-07-19 — PatchTST loss explained: encoder redundant, RES forecast carries skill**
Context: overnight feature analysis (group-ablation walk-forward 3 seeds, permutation, PCA, attention) on best config patch24_s24_ctx1344.
Decision: ablating the 56-day price encoder costs nothing (MAE 23.23 vs full 23.61); ablating RES forecast costs +6.2 EUR/MWh. Verdict note updated; quote ablation (not permutation) for information content.
Why: permutation ranked the encoder first — reliance, not unique information. Retraining ablation is the honest measure; PatchTST's long-context premise adds no skill on this task.

**2026-07-18 — PatchTST attention campaign: negative result, archived**
Context: PatchTST walk-forward (top-3, 25 monthly refits 2024-07-16→2026-07-16). Best config MAE 22.98 EUR/MWh, rMAE 0.823. TFT gate was 19.71 EUR/MWh. TFT gate NOT cleared.
Decision: PatchTST archived. LGBM+conformal stays champion. Attention campaign complete. Next priority: 14-day shadow track record.
Why: patch inductive bias does not compensate for short training windows (365 days) and small model capacity (197k params). Coverage 69.5% vs 80% target — interval quality also worse than all baselines.

**2026-07-18 — Backtesting comparison plots added**
Context: 15 plots generated from 2-year hourly predictions (LGBM, LEAR, TFT, naive). PatchTST hourly preds not saved during walk-forward (only aggregate metrics).
Decision: plots saved to reports/figures/backtest_price/. PatchTST shown in bar charts only (aggregate MAE/rMAE).
Why: hourly predictions needed for time-series plots; PatchTST walk-forward only wrote aggregate CSV. Re-running for hourly preds would take 90 min for diminishing return on a negative result.

---

**2026-07-18 — Zero-variance guard added to standardize_covariates**
Context: PatchTST first sweep showed val pinball 879 vs train 0.26. Baltic offshore wind (wind_off_fcst_mw) came online 2026-07-01 — all-zero in training, non-zero in val. Training std=0, clamped to 1e-6; a 19 MW val value became z-score 19,000,000.
Decision: add zero-variance guard to standardize_covariates. Where raw training std < 1e-4, zero the column in all sets after standardisation (instead of dividing by the clamp). Add test test_deep_data.py.
Why: columns that are constant in training carry no training signal. Any non-zero future value should be treated as "unknown input" → zero-out is safer than a 1e6× amplification. This is a known live-system risk: new RES types enter service every year.

**2026-07-17 — Asymmetric CQR evaluated; symmetric stays in production**
Context: hypothesis that negative-price hours cause lower-tail miscalibration that symmetric CQR cannot fix independently.
Decision: symmetric CQR stays. Measured on 2-year walk-forward: sym cov 79.6%/78.9%, asym cov 79.1%/78.4% for LEAR/LGBM. Upper tail is the bigger problem (q_hi > q_lo for both models). Asymmetric CQR offers 8-12% narrower bands but 0.5pp lower coverage.
Why: coverage guarantee is the primary requirement. Spike forecasting — not negative prices — is the main calibration gap. Asymmetric code kept in conformal.py for future use when spike modelling improves.

**2026-07-17 — TFT walk-forward verdict: trails LEAR, shadow gate not opened**
Context: 60-trial HPO best (val 0.1157, ctx=1344, d128, h8, l2) sent to 3-seed walk-forward over 17,472 test hours (2024-07-16 → 2026-07-18).
Decision: shadow gate NOT opened. TFT ens-3 rMAE 0.706 vs LEAR 0.653 vs LGBM 0.640. Root causes: data ceiling (1.27M params, 300-400 training samples), signal sparsity, quantile training cost. PatchTST sweep next (cheaper architecture, different inductive bias).
Why: 8.1% worse MAE is not a rounding error. Model card and model_selection/08 updated with the honest verdict.

**2026-07-17 — TFT attention campaign: HPO + PatchTST, walk-forward gate**
Context: owner lifted model freeze; screening showed TFT trails tabular by 30% but long context IS real (monotonic improvement). Question: does full HPO close the gap?
Decision: 60-trial Optuna search (ctx + arch jointly), then 3-seed walk-forward to confirm. PatchTST sweep after. Never quote screening numbers as results.
Why: owner hypothesis is legitimate and testable; HPO is far cheaper than premature conclusion. Honest walk-forward result (win or loss) is more valuable than a silent skip.

**2026-07-17 — Outage (UMM) feature: evaluated, rejected**
Context: large unit outages should move price (less supply → higher price). Feature built with ENTSO-E UMM data.
Decision: outage feature NOT adopted. Change vs full model: ±0.05 EUR/MWh (noise). Feature opt-in with --with-outages flag; backfill endpoint returns 503 in CI.
Why: aggregate capacity unavailability is too coarse. Individual outage identity, location, and duration matter more but require unit-level matching we don't have. Documented for a future researcher; research store kept.

**2026-07-17 — Fuel features (TTF/EUA proxy) adopted for LEAR**
Context: winter 2024/25 LEAR monthly bias −15.9 EUR/MWh in January. Gas prices were high; LEAR saw no fuel signal.
Decision: TTF index (LNG import proxy via ENTSO-E) and EUA-tracking ETF added to LEAR feature matrix. LGBM: no improvement (trees already carry the slow level via price lags).
Why: LEAR reduced winter bias to −4.6 EUR/MWh; Jan MAE −2.5 EUR/MWh. Gain concentrated in exactly the months where the mechanism predicts it: high-gas regime. Merit-order mechanism, measured.

**2026-07-17 — Shadow tally started for both load and price**
Context: ridge+TSO challenger and LEAR price model need live proof before promotion.
Decision: both tallies start 2026-07-18 (first valid cron run, after CI data-store fix). Promotion criterion agreed in advance: 14 consecutive valid days + metric check.
Why: 14-day shadow window is desk standard — long enough to cover weekend patterns and holiday anomalies; short enough not to delay a clearly superior model.

---

**2026-07-16 — Phase 2.5: polish before Phase 3; no new models**
Context: Phase 2 build finished ~5 weeks ahead of the get-hired schedule. Remaining gaps were polish, not build: broken bands, stale README, missing market-context docs.
Decision: insert Phase 2.5 (conformal calibration, README overhaul, M8 notes pulled forward) before Phase 3. Freeze on new model architectures.
Why: a recruiter sees the README and the track record, not an eighth LSTM. Owner approved 2026-07-16.

**2026-07-16 — Band calibration: rolling split-conformal (CQR), not tuning**
Context: LGBM band covered 51% vs nominal 80%; LEAR 72%. Options: quantile-parameter tuning, per-hour residual bands, conformal.
Decision: rolling CQR on a 90-day trailing window of out-of-sample errors. Model-agnostic wrapper; P50 untouched; daily loop applies stored offsets (`config/price_conformal.json`).
Why: distribution-free coverage guarantee, walk-forward honest by construction (leakage test proves it), works identically for every current and future model. Result: both bands ~79%.

**2026-07-16 — LEAR stays the daily price publisher despite LGBM's better MAE**
Context: after calibration LGBM+conformal beats LEAR+conformal on MAE (17.8 vs 18.5) with equal coverage.
Decision: LEAR remains the published incumbent; LGBM+conformal is the named challenger for a future M9-style shadow window.
Why: desks do not swap the published model on a backtest — promotion goes through shadow. Swapping day 2 would also reset the just-started price track record.

**2026-07-16 — TSO RES day-ahead forecast accepted as bid-time proxy**
Context: ENTSO-E publishes the TSO wind+solar forecast for day D ~18:00 on D-1 — hours AFTER the 12:00 gate closure. Strictly, bidders could not see this exact series.
Decision: use it as a feature anyway, labeled a proxy. Same convention as the EPF literature (Lago et al. 2021 benchmark uses this exact ENTSO-E series).
Why: bidders run their own RES forecasts at bid time; the TSO series proxies that information set. SHAP says solar forecast is price driver #1 (18.7 EUR mean |SHAP|) — dropping it would cripple the model to protect a technicality. Caveat repeats in the model card.

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
