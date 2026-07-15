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

## Results — 12-month walk-forward, honest weather forecasts as inputs

Poland, hourly day-ahead load, decided at 09:00 D-1. 8,762 test hours.
Full campaign details: `reports/backtests/`, `docs/notes/model_selection/`.

| Model | MAPE | MAE (MW) | Skill vs naive |
|---|---|---|---|
| **Ridge + TSO forecast (combiner)** | **2.13%** | **383** | **0.63** |
| LightGBM + TSO forecast | 2.14% | 391 | 0.62 |
| PSE (TSO) day-ahead forecast | 2.31% | 418 | 0.59 |
| LSTM attention + TSO | 2.43% | 448 | 0.56 |
| LightGBM (weather + calendar + lags) | 3.16% | 579 | 0.43 |
| LSTM seq2seq (best of 7 architectures) | 3.67% | 692 | 0.32 |
| Ridge | 4.03% | 718 | 0.30 |
| Seasonal naive (same hour last week) | 5.60% | 1025 | 0.00 |
| Climatology | 8.57% | 1621 | −0.58 |

The models that lost stay in the table. Notable honest findings:
- Combining with the TSO forecast (published at our cutoff) beats the TSO by 8% MAE.
- Once the TSO signal is in, a ridge regression beats every deep net we built.
- Bigger nets lose: accuracy peaks at ~106k parameters on 2 years of data.
- Training-sample augmentation via shifted forecast origins *hurt*. Logged anyway.

## Repo map

See `CLAUDE.md` → "Target repo structure". Decisions in `docs/DECISIONS.md`.
