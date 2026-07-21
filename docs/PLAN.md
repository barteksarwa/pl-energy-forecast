# PLAN.md — Roadmap

Status: **v3, updated 2026-07-21.** Phases 1–3 complete except the track
record (M10) — cron outage 07-18→21, see DECISIONS 2026-07-21.
Current milestone: **Phase 4 / M11–M13** (needs owner approval).
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
