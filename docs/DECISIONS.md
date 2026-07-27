# Decision Log

Three lines per entry: context, decision, why. Newest on top.

---

**2026-07-27 — Cutoff breach in the backtest loop: fixed, impact bounded at ~0.11 MAE**
Context: validation finding E2 — `dates < day` let walk-forward training see D-1 hours 09:00-23:00, violating the stated decision-moment rule. E1 — the D-1 price vector picked up the D-2 shape on the day after spring DST.
Decision: both fixed with regression tests (118 green). Bounding rerun: champion 17.95 under the corrected protocol vs 17.84 before — ~0.11 MAE of shared flattery. Rankings stand (every trained model had the same extra hours); the production daily loop was never affected (runs 07:30 local, cannot see the future). Published tables keep their numbers with a protocol note; full re-runs under the corrected protocol are the next campaign, owner-scheduled.
Why: the honest fix is the rule going forward plus a measured bound on the past, not silently rewriting every historical table.

**2026-07-27 — TFT joins the blend: 4-member ensemble is the new best (16.89)**
Context: TFT full-2yr preds turned out to live under reports/ (survived the wipe). PLAN's deferred "1-yr with TFT" two-window test became a clean 2-yr 4-member test. Pre-declared: beat 17.34 by >=0.10, DM p<0.05, coverage 78-82.
Decision: ens4 (LGBM+LEAR+Chronos+TFT, all CQR, second CQR on blend) MAE 16.89, DM p=2.3e-09, wins all years, coverage 80.0%, Winkler 82.6, P&L capture 0.929 — every gate smashed. ens4 replaces ens3 as the promotion candidate. Owner call: TFT inference cost in the daily loop (3 seeds, MPS, monthly refits).
Why: archived-solo TFT is the best diversity donor tested (deep errors decorrelate from trees/linear/FM); TimesFM added nothing. Error-structure diversity beats member strength.

**2026-07-27 — Validation review: stale numbers fixed, missing DM artifacts created**
Context: owner-authorized multi-agent MRM-style review (16 finders, adversarial refuters) audited every RESULTS/BENCHMARK number against artifacts and red-teamed leakage.
Decision: confirmed findings fixed — price tables now cite ONE canonical run (the regenerated 07-24 window); Chronos/Moirai DM claims now have a real artifact (`2026-07-27_stats_tests_fm_dm.csv`: Chronos>TimesFM p=0.0095, Moirai-cov worse p=8e-06); Winkler 84.7 corrected to 85.2 (pre-incident value had leaked into the narrative); crisis "DM p=0.17" downgraded to a direction check (no artifact, unpaired windows); P&L captures re-rounded.
Why: exactly the audit a model-risk function would run; the repo's honesty claim has to survive independent review, and now it documents surviving one.

**2026-07-25 — LGBM price HPO: defaults survive; honest negative**
Context: champion ran "conservative untuned defaults" since M4; owner authorized a tuning campaign. 14 configs at 1095d windows, screen on test year 1, confirm top-3+control on year 2 (never used for selection). Pre-declared gate: beat control by >=0.10 MAE on BOTH years.
Decision: NO config passes. Best (slow-learn lr 0.02-0.03, 1000-1500 trees) gains 0.02-0.08/yr. Defaults stay; the card now says "tuned 2026-07-25 — defaults survived". Side finding: noloadlags is +0.09 WORSE at 1095d (was -0.12 better at 365d) — the load-lags-drop flip is window-conditional, cancel it for the 1095d config.
Why: flat HPO surface on well-engineered features is the expected LGBM behavior; chasing 0.05 MAE with tuned configs buys fragility, not skill. Evidence: `reports/backtests/2026-07-25_lgbm_price_hpo.csv`.

**2026-07-24 — Blend on 1095d members: tested, NOT adopted (DM p=0.06)**
Context: natural follow-up to the two pending promotions — rebuild the CRPS blend on 1095d-window LGBM/LEAR. Pre-declared gates: beat ens-365 with DM p<0.05, coverage holds, Winkler not worse.
Decision: MAE improves 17.34→17.18 but NOT significantly (p=0.0596), loses the 2026 slice, P&L capture identical (0.925 vs 0.926). Blend stays on 365d members. Solo-champion 1095d promotion case (p=0.0009) unaffected.
Why: window gain and ensemble diversity are partial substitutes — the blend already recovers most of what the longer window adds. Same honesty standard as LGBM-vs-LEAR: no switch on a non-significant edge.

**2026-07-24 — Benchmark writeup shipped as docs/BENCHMARK.md (Phase 7 S3-S5, blog minimum)**
Context: PLAN Phase 7 asks for a publication-grade summary: protocol, master tables incl. P&L capture, findings, honest negatives, reproducibility appendix.
Decision: one markdown page in docs/, story layer over RESULTS.md (numbers stay canonical there). New ensemble model card added; README refreshed (ensemble headline, P&L column, FM findings, live status). arXiv-style expansion stays a stretch goal.
Why: recruiters and reviewers need one entry point that survives a 3-minute read; the honest-negatives table is the differentiator.

**2026-07-24 — INCIDENT: data/ wiped by a committed worktree symlink; history repaired, refetch running**
Context: a worktree session symlinked `data/` to the main checkout; `git add -A` committed the symlink (`data/**` in .gitignore matches contents, not the path). The merge into main replaced the real directory and git deleted the ignored parquets.
Decision: history rewritten before any push (4 commits rebuilt without data paths), `.gitignore` now ignores the `data` path itself, base-data refetch launched via idempotent `make backfill`. Stored backtest preds regenerate from resume-safe scripts; numbers must reproduce against RESULTS.md.
Why: an honest repo documents its operational failures like a real desk would. Full incident: handover 2026-07-24.

**2026-07-24 — Blend conformalized; ens_crps_cqr is the promotion candidate**
Context: raw CRPS blend over-covered (84.2% vs nominal 80) — the declared blocker before production promotion.
Decision: run rolling CQR a second time on the blended band. Coverage 79.9%, Winkler 85.2→84.7, MAE unchanged. `ens_crps_cqr` replaces `ens_crps` as the candidate. Promotion still awaits owner (interacts with the pending 1095d-window call; adds FM inference to the daily loop).
Why: averaging three calibrated bands over-widens because member errors are imperfectly correlated; CQR's negative offset is the cheapest honest fix.

**2026-07-24 — Battery P&L: MAE rank survives, value compresses**
Context: PLAN Phase 7 asked what forecasts are worth in EUR and flagged possible MAE-vs-capture rank flips.
Decision: per-day LP battery backtest (1 MW/2 MWh/0.85 RTE/1 cycle, DA-only). No rank flips; ensemble captures 92.4%, naive 81.3%. Keep MAE for development, EUR for stakeholder framing.
Why: storage rewards hour ordering, not price level — 10.6 MAE of skill buys 11 capture points. The EUR lens is honest scope: no intraday, no fees, comparator not business case.

**2026-07-24 — CRPS ensemble beats the champion on every gate; Moirai covariates hurt**
Context: Phase 6 close. Ensemble members pre-declared (champion + LEAR + best FM). Moirai zs/cov both ran the full 2-yr window.
Decision: ens_crps 17.34 (champion 17.87): DM p=2.5e-04, wins all three test years, Winkler better. New backtest best. NOT promoted to the daily loop yet — blend over-covers (84%), needs conformal-on-blend + owner call on running FM inference daily. Moirai: covariates degrade zero-shot accuracy (24.86 vs 23.69) — any-variate attention cannot exploit unseen covariates without training.
Why: ensemble gains come from error diversity, not member strength — the weakest member (Chronos, univariate) still adds skill. The Moirai negative closes the "just feed FMs covariates" shortcut honestly.

**2026-07-23 — Chronos-Bolt zero-shot: rMAE 0.787, fine-tune gate closed**
Context: Phase 5 S4-S5. Univariate foundation model, no training, standard 2-yr backtest, daily context refresh. With CQR: coverage 79.9%.
Decision: MAE 21.98 — beats naive (DM p 1e-44) AND our trained PatchTST-730 (22.25); loses to TFT-730 (19.52) and champion (DM p 1e-23). Gate for Phase 6 fine-tuning was rMAE < 0.75: closed at 0.787. Zero-shot row goes in the benchmark table with the univariate fairness footnote.
Why: measures what a covariate-blind foundation model buys on PL prices — a strong baseline for free, not a champion. Fine-tuning judgment deferred to Phase 6 planning with this number in hand.

**2026-07-23 — Spike classifier: gate passed, seed confirm redundant**
Context: Phase 5 S3. Walk-forward 2-yr screen: AUC 0.966, Brier 0.034, precision@2 0.736 (gate was AUC 0.80).
Decision: PROMOTED to a daily-report column (p_spike per hour). Note: the classifier is deterministic (no subsampling in params) — seeds 42/7 produced identical results, so the planned 3-seed confirm is vacuous and was stopped.
Why: three independent tests (asym CQR, GPD, Christoffersen) show tail misses are conditional; this is the conditional signal. Deterministic model → single run IS the result.

**2026-07-22 — GPD (EVT) upper tail rejected; symmetric CQR stays**
Context: Phase 5 S1-S2. Hybrid band (symmetric lower, GPD peaks-over-threshold upper tail) benchmarked against symmetric and asymmetric CQR on stored 2-yr preds. Pre-declared gate: spike coverage +3 pts, pooled coverage 78-82%, Winkler ≤ +2%.
Decision: REJECTED. Spike coverage +0.57 pts (LEAR 56.16 vs 55.59) and +0.0 (LGBM). Third tail-calibration method to lose to plain symmetric CQR. Tables: `reports/backtests/2026-07-22_gpd_tail_*.csv`.
Why: the 90d empirical tail quantile is already stable; spike misses are conditional (the model does not see the spike coming), so no unconditional band width fixes them. Conditional route = spike classifier (S3).

**2026-07-22 — load_lags champion flip deferred to a config-driven feature refactor**
Context: drop confirmed (2-yr MAE 17.755 vs 17.87). Tempting shortcut: drop `load_` columns inside the LGBM model class.
Decision: NOT done at model level — `lgbm_quantile` is shared by the LOAD model, where load lags are core features. Flip waits for per-model feature lists in config (rule 10 refactor).
Why: a price-side tweak must not silently gut the load product. The −0.12 gain keeps; the evidence is stored (`2026-07-21_price_noloadlags2yr`).

---

**2026-07-21 — Cron outage 07-18→07-21: local repo lost its git remote**
Context: no daily reports since 2026-07-17. Diagnosis: local repo had no `origin`; CI cron ran on the old remote, which is gone. Owner created a fresh public repo (`barteksarwa/pl-energy-forecast`) on 2026-07-21 with a curated 7-commit history; no common ancestor with local main (122 commits).
Decision: days 07-19→21 logged as FAILED in both shadow tallies (no forecasts exist). 07-18 forecasts exist and will be scored retroactively. Local work continues; nothing pushed until owner picks a reconciliation path (see PLAN.md Phase 4).
Why: forecasts cannot be produced after the fact — an honest track record shows the hole. Pushing 122 unrelated commits over a hand-curated public repo is destructive; owner's call.

**2026-07-21 — City weather weights updated to official GUS 2026 data**
Context: config weights were "approximate metro population, rounded" with no source. Owner asked for latest GUS. The 2026 edition was published the same day (2026-07-21).
Decision: weights = city population in millions from GUS "Powierzchnia i ludność w przekroju terytorialnym w 2026 r." (Tabl. 20, as of 2025-12-31), 3 decimals. Lublin (327k) now ranks above Bydgoszcz (321k).
Why: traceable source beats a rounded guess. Effect on the load model is tiny (weather group ablation +0.08 pp) — no re-benchmark needed. Population weighting stays a demand proxy; it is the wrong weighting for RES weather, which is a Phase 4 topic.

**2026-07-21 — Docs consolidated; canonical results page added**
Context: numbers were scattered across README, model cards, handovers, and reports; several docs went stale (PLAN header, README status table, TFT card).
Decision: `docs/RESULTS.md` is the single source for headline numbers; other docs link to it. Superseded backtest summaries and dead specs moved to `docs/archive/` and `reports/backtests/archive/`.
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
