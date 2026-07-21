# PLAN.md — Roadmap

Status: **v4, updated 2026-07-21.** Phases 1–3 complete except the track
record (M10) — cron outage 07-18→21, see DECISIONS 2026-07-21.
Current milestone: **Phase 4 / M11–M13**, then Phases 5–7 (owner-approved
2026-07-21).
Canonical numbers: `docs/RESULTS.md`. Job market research: `docs/notes/career.md`.
History: v1/v2 details in git (this file was compressed 2026-07-21).

## Where we stand (2026-07-21)

- Phase 1 (load): DONE. Champion ridge_tso 2.08% MAPE, beats TSO (2.23%).
- Phase 2 + 2.5 (price): DONE. Champion LGBM quantile + CQR, rMAE 0.640.
- Phase 3 M9 (ops): DONE. Shadow/promotion discipline built and documented.
- Attention campaign: DONE, closed 2026-07-21. TFT and PatchTST tested,
  decomposed, archived. See RESULTS.md.
- M10 track record: BROKEN — 4 reports, then cron died with the old
  remote. Restart is Phase 4 work.
- Open owner actions: blog post, learning notes reading, publication call.

## Ambition level (owner call, standing)

Near-production quality, senior-job level. Publication-grade honesty:
never invent numbers; unknown = say so. Clean contracts: new model =
new file, zero edits elsewhere. Free tools only.

## Phases 1–3 — complete (summary)

Full milestone text lives in git history (PLAN v2).

- **M0–M2**: skeleton, clients (ENTSO-E, PSE, Open-Meteo), 3.5y backfill,
  UTC + DST tests, leakage-safe feature matrix, metrics + walk-forward engine.
- **M3–M5 (load)**: baseline campaign → ridge_tso champion; LightGBM
  quantile + SHAP; 7 deep architectures tried, none beat the linear
  combiner once TSO signal is in. All in RESULTS.md + model cards.
- **M6–M7 (price)**: EUR/MWh target, RES/fuel/outage drivers, LEAR
  baseline, LGBM quantile champion, spike problem documented.
- **Phase 2.5**: conformal calibration (coverage 51%→79%), recruiter
  README, market-context notes (balancing, rynek mocy, intraday).
- **M8**: market docs done; CO2/ETS note still pending (M13).
- **M9 (ops)**: cron, UAT/prod via shadow+promotion rules, secret
  redaction, publish-horizon fetching, persist_24h fallback.
- **M10**: HOW_A_FORECAST_IS_MADE.md done; track record interrupted.
- **Attention campaign**: TFT HPO + walk-forward, PatchTST build/sweep,
  730d window + ensemble + capacity decomposition. Verdict in
  RESULTS.md and `notes/model_selection/12_deep_gap_decomposition.tex`.

## Phase 4 — Repo revival + RES geography (proposed 2026-07-21)

### M11 — Repo reconciliation + track record restart

Local repo and the new public repo (`barteksarwa/pl-energy-forecast`,
created by owner 2026-07-21) share no history. Owner picks one:

- (a) Force-push local history (122 commits) to the public repo.
- (b) Keep curated public repo; push squashed story-shaped commits.
- (c) Two remotes: private working (full history, cron) + public showcase.

Then: re-enable Actions cron (`ENTSOE_API_TOKEN` secret), retro-score
the stranded 2026-07-18 forecasts, run the 14-day shadow window to the
end, accumulate 30+ daily reports.

### M12 — RES geography (owner priority)

Where wind and PV sit changes how weather moves the price. Stop
treating renewables as a zone-level blob.

1. **Data hunt** (1 session). Verify before use: ENTSO-E installed
   capacity per unit (≥100 MW); URE/ARE capacity per voivodship; PSE
   capacity statistics; OSM turbine locations as spatial fallback.
   Document in DATA_CATALOG with license terms. Voivodship granularity
   is likely enough.
2. **Capacity-weighted weather** (1–2 sessions). New weight vectors in
   config: wind-capacity-weighted wind speed, PV-capacity-weighted
   irradiance. Reuses city-grid machinery. Zero model-code edits (rule 10).
3. **Honest test.** Group ablation vs the TSO RES forecast. Expected:
   small or none — the TSO forecast already embeds site locations.
   Negative result documented, not hidden.
4. **Stretch: own RES generation forecast.** Wind/solar MW from
   capacity-weighted weather; benchmark vs TSO RES forecast. RES
   forecasting is its own job category in PL/EU.

### M13 — Portfolio close-out

- README track-record section once 30+ reports exist.
- Blog post (owner; draft covers load, needs price/deep chapter).
- CO2/ETS learning note.
- Publication check: PL benchmark table as blog minimum, workshop
  paper stretch. Owner decides with evidence.

## Phases 5–7 — approved 2026-07-21 (after Phase 4)

Standing rules for these phases:

- **Daily commits.** GitHub gets commits every day next week. Work sliced
  into small realistic chunks, one concern per commit, pushed same day.
- **Benchmarks.** The shared 2-yr window stays canonical — every model gets
  that row. On top: per-period breakdowns (per-year, crisis vs calm,
  winter vs summer, spike vs normal hours). A model that wins one regime
  while losing pooled is a finding to report and exploit.
- **Deeper history.** Backfill ENTSO-E load/price/RES to ~2015-2016
  (verify earliest clean dates). Unblocks the 730d deep re-benchmark on
  the full 2-yr test (was "impossible before 2027"). Pre-2021 regime
  documented as evaluation richness, not blindly added to training.

### Phase 5 — Spike tails + Chronos zero-shot (week 1)

Deps: scipy (GPD); `chronos-forecasting` in an optional `fm` group.

- S0: deep-history backfill (overnight) + per-period `summarize_price`.
- S1: close load_lags item (confirm backtest); pyproject rename;
  GPD upper tail in `conformal.py` — POT fit on `y − p90` exceedances,
  rolling 90d past-only, guards (≥30 exceedances, ξ < 1, empirical fallback).
  Lower tail keeps symmetric CQR (asymmetric already tested and rejected).
- S2: `--compare-gpd` in `run_price_calibration.py`, stored preds only.
  Adopt iff spike coverage +3 pts vs both CQR variants AND pooled coverage
  78–82% AND Winkler ≤ +2%. Note ms-13.
- S3: spike classifier screen (`src/models/spike.py`, LGBM binary, top-5%
  label from trailing train window only). AUC ≥ 0.80 → report feature
  (3-seed confirm); else honest negative.
- S4: Chronos-Bolt zero-shot wrapper (`src/models/chronos_zs.py`) through
  the QuantileForecaster Protocol; `--refit-days` flag; MPS with FORCE_CPU
  escape. Deterministic → single run valid.
- S5: Chronos 2-yr backtest + CQR. Fairness footnote: univariate vs
  champion's covariates. Phase 6 fine-tune gate: best FM MAE < 24.
- S6: RESULTS rows; notes learning 21–22, model_selection 14.

### Phase 6 — Foundation-model campaign + ensemble (week 2)

Deps (`fm` group): `timesfm`, `uni2ts` (Moirai). Conflict budget: half a
session, then scratch-venv isolation (outputs are just parquets).

- S1: `fm_common.py` refactor + TimesFM wrapper.
- S2: TimesFM 2-yr backtest (overnight).
- S3: Moirai wrappers — `moirai_zs` (univariate) AND `moirai_cov`
  (RES + TSO as known-future covariates). Their delta = covariate lift
  for FMs, measured cleanly.
- S4: Moirai backtests (overnight).
- S5: CRPS-weighted cross-model ensemble from stored preds
  (`crps3` = mean pinball over 3 quantiles; inverse-score weights from
  trailing 60d, past-only; 60d equal-weight warm-up). Two-window
  discipline: 2-yr without TFT, 1-yr with.
- S6: verdict (beat 17.87 by ≥0.15 AND win both years AND coverage holds);
  notes learning 23–24, model_selection 15–16.
- S6b: deep re-benchmark at 730d on FULL 2-yr test (unblocked by S0).
  3 seeds, per-regime breakdown.
- S7 (only if gate open): Chronos fine-tune, 3 seeds, own loop.

### Phase 7 — Forecast-to-money + publication (week 3)

- S1: battery-arbitrage P&L engine (`src/evaluation/pnl.py` + runner).
  1 MW / 2 MWh / 0.85 round-trip / 1 cycle, schedule from P50 at D−1,
  settle at actual DA prices. Metrics: EUR/day, capture rate vs
  perfect-foresight bound, vs naive strategy. DA-only scope stated.
  Unit tests for the accounting.
- S2: P&L table for all stored models + ensemble; cumulative plot;
  notes learning 25, model_selection 17. Watch for MAE rank ≠ capture rank.
- S3–S5: benchmark writeup (blog minimum, arXiv stretch): protocol,
  master table incl. P&L capture, findings, honest negatives,
  reproducibility appendix. Cards + README + DECISIONS refreshed.

## Backlog (scoped, not scheduled)

- Sub-national/portfolio load POC — ran 2026-07-17, see
  `reports/backtests/2026-07-17_portfolio_poc.md`.
- Outage feature refinements (per-fuel, unplanned-only) — base version
  tested FLAT.
- Deep re-benchmark at 730d on full 2-yr test — possible 2027+ (needs
  730d history before 2024-07; data starts 2023-01).
- Drop load_lags from LGBM price champion (−0.12 dead weight) — config
  change + confirm backtest.

## Learning thread (runs through everything)

Owner must explain every piece in interviews. One short explainer per
concept in `docs/notes/learning/` (20 notes) and verdicts in
`docs/notes/model_selection/` (12 notes). Rule: under one page, short
sentences, one worked example each.

## Token-saving rules

- One config file. Handovers under one page; agents read only latest.
- Canonical numbers only in RESULTS.md.
- Notebooks out of agent context.

## Risks

- ENTSO-E delays/gaps → gap log + oddities section in reports.
- Free gas/CO2 data patchy → proxies, stated openly.
- Price spikes break point metrics → tail evaluation stays in tables.
- Single-seed screening lies → 3-seed minimum for deep verdicts (proven 3x).
