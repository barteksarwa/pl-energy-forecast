# Day-Ahead Electricity Load Forecasting — Poland

Every morning this repo forecasts tomorrow's hourly electricity demand for Poland.
It scores yesterday's forecast, explains today's, and writes a short report.
It simulates the daily work of a forecasting analyst at a utility or trading desk.

> Status: Milestone 0 (walking skeleton) done. See `docs/PLAN.md` for the roadmap
> and `reports/daily/` for the track record.

## Highlights (filled as milestones land)

- Data: ENTSO-E load + TSO benchmark forecast, Open-Meteo weather, Polish holidays.
- Models: seasonal naive → linear → LASSO-AR → LightGBM quantile (primary) → LSTM → transformer.
- Every forecast ships with P10/P50/P90 and a plain-words explanation.
- Phase 2 adds day-ahead price forecasting on the same pipeline.
- UAT/prod split: new models run in shadow mode before promotion.

## Quickstart

```bash
make setup            # install deps (needs uv)
cp .env.example .env  # add your ENTSO-E token
make test             # unit tests, no network
make smoke            # live API smoke tests
make dry-run          # one full daily cycle: fetch, score, forecast, report
```

## Results

Honest rolling-backtest table lands in Milestone 3. Including the models that lost.

## Repo map

See `CLAUDE.md` → "Target repo structure". Decisions in `docs/DECISIONS.md`.
