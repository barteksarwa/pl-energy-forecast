# Handover — 2026-07-14 — plan approved, walking skeleton built

## What I did

- Researched PL/EU job market. Trading desks forecast price; utilities forecast load.
- Wrote and revised `docs/PLAN.md`. Owner approved v2 (3 phases, 11 milestones).
- Built M0 skeleton: config, ENTSO-E + Open-Meteo clients, seasonal naive with
  P10/P50/P90, metrics (MAE/RMSE/MAPE/pinball), daily_run pipeline, report writer.
- 14 unit tests pass. Open-Meteo live smoke test passes.
- Wrote first learning explainer: seasonality and seasonal naive.

## State of things

- Works: `make setup`, `make test`, `make smoke`, `make lint`.
- Blocked: `make dry-run` needs `ENTSOE_API_TOKEN` in `.env`. Owner must register
  at transparency.entsoe.eu and request RESTful API access.
- Caught and fixed a real DST bug: `+ Timedelta(days=1)` is 24h, not a calendar
  day. Calendar shifts now via `shift_local_day()` in `src/pipeline/daily_run.py`.

## Decisions made

- Load first, price Phase 2, shared pipeline. Logged in DECISIONS.md.
- POC automation (GitHub Actions, 7–14 days) before full 30-day unattended run.
- ENTSO-E data resampled to hourly mean (PL may switch to 15-min resolution).

## Next steps

1. Owner adds ENTSO-E token → run `make dry-run` → verify first report. Closes M0.
2. M1 prep: owner wants online research (papers, docs) on data sources first.
3. M1: `docs/DATA_CATALOG.md`, 3-year backfill, gap log.

## Watch out for

- `energy-forecast-kickstart/` folder is a leftover template. Root README now
  supersedes it. Ask owner before deleting.
- City weights in config are approximate GUS metro populations. Refine in M1.
- ENTSO-E rate limits unknown; backfill in M1 should chunk requests and sleep.
