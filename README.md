# Polish Power Market Forecasting — Day-Ahead Load

Day-ahead hourly electricity load forecast for the PL bidding zone.
The goal: beat the seasonal naive baseline, then challenge the TSO's
own day-ahead forecast.

Work in progress. Current state:

- ENTSO-E, PSE and Open-Meteo API clients with backfill and gap checks
- Seasonal naive baselines (same hour last week)
- Leakage rule: the forecast for day D is made at 09:00 on D-1 —
  only data available at that moment may be used

## Quickstart

```bash
make setup            # install deps (needs uv)
cp .env.example .env  # add your ENTSO-E token (free)
make test             # unit tests, no network
```

Next: calendar/weather/lag features, walk-forward backtest, LightGBM
quantile model with P10/P50/P90 output.
