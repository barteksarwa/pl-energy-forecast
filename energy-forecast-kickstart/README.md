# Day-Ahead Electricity Load Forecasting — Poland

Every morning this repo forecasts tomorrow's hourly electricity demand for Poland.
It scores yesterday's forecast, explains today's, and writes a short report.
It simulates the daily work of a forecasting analyst at a utility or trading desk.

> Status: kickstart. See `docs/PLAN.md` for the roadmap and `reports/daily/` for the track record.

## Highlights (to be filled as milestones land)

- Data: ENTSO-E load + TSO benchmark forecast, Open-Meteo weather, Polish holidays.
- Models: seasonal naive → linear → LightGBM quantile (primary) → LSTM → transformer.
- Every forecast ships with P10/P50/P90 and a plain-words explanation (SHAP).
- Daily dry run via GitHub Actions. The commit history is the operational log.

## Quickstart

```bash
make setup          # install deps
cp .env.example .env  # add your ENTSO-E token
make backfill       # download history
make dry-run        # run one full daily cycle
```

## Results

Honest rolling-backtest table goes here. Including the models that lost.

## Repo map

See `CLAUDE.md` → "Target repo structure".
