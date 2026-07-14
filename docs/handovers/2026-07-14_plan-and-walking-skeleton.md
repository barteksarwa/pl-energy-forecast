# Handover — 2026-07-14 — plan, skeleton, M1 data foundation

## What I did

- Researched PL/EU job market. Wrote `docs/PLAN.md`; owner approved v2.
- Built M0: config, ENTSO-E + Open-Meteo clients, seasonal naive P10/P50/P90,
  metrics, daily_run pipeline, report writer. 21 unit tests green.
- LaTeX notes system: `docs/notes/learning/` + `docs/notes/model_selection/`,
  each with `main.tex` + one note. Both compile. Owner convention in CLAUDE.md.
- Viz module (`src/viz`, `make viz`): weather panels, load vs TSO, forecast fan,
  temperature history. CVD-safe validated palette.
- M1 partial: `docs/DATA_CATALOG.md`; backfill scripts with gap log;
  weather backfilled live — 10 cities × 30,840 h (2023-01-01→2026-07-08), 0 gaps.
- Fixed "no module named src": project is now an installed package (hatchling).
- GitHub connected: private repo `barteksarwa/pl-energy-forecast`, pushed.
- Verified industry usage: ENTSO-E = the EU standard (with known quality quirks
  → our gap log). Open-Meteo = legit for backtests/prototyping; desks buy
  Meteomatics/DTN for production. Documented in DATA_CATALOG.

## State of things

- Works: `make setup/test/smoke/lint/viz/backfill` (weather part).
- Blocked on owner: ENTSO-E token → then `make backfill` (load) + `make dry-run`.
- Weather-leakage trap documented: backtests must use historical weather
  *forecasts*, not ERA5 actuals. Endpoint verification = M2 task. [TBC]

## Decisions made

- Load first, price Phase 2 (DECISIONS.md). LaTeX for owner notes, md for ops.
- Permissions pre-approved in `.claude/settings.json` (uv/make/git/gh).

## Next steps

1. Owner: add ENTSO-E token → `make backfill` → `make dry-run` → `make viz`.
2. M1 close: gap report over load data; DST tests against real ENTSO-E data.
3. M2: features + Open-Meteo historical-forecast endpoint verification.
4. Learning note 02 (day-ahead market + 09:00 cutoff) in LaTeX.

## Watch out for

- `energy-forecast-kickstart/` is leftover; ask owner before deleting.
- Free ENTSO-E rate limits: backfill chunks 90 d + 1 s sleep. Watch for 429s.
- City weights are approximate GUS metro populations.
