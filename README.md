# pl-energy-forecast

Day-ahead electricity load and price forecasting for the Polish bidding
zone (PL), with probabilistic forecasts, automated daily operation, and
walk-forward evaluation against the market's own benchmarks.

Every morning a GitHub Actions job fetches the latest market and weather
data, scores yesterday's forecast against what actually happened,
produces tomorrow's forecast, and commits a short report to
[`reports/daily/`](reports/daily/).

## Results

All numbers are out-of-sample from a 2-year walk-forward backtest
(17,450+ hourly observations, weekly model refits). Models never see
data past the day-ahead market deadline: the forecast for day D uses
only information available at 09:00 on day D−1.

**Load (national hourly demand, MW)**

| Model | MAPE |
|---|---|
| Ridge regression + TSO forecast | **2.08%** |
| LightGBM + TSO forecast | 2.12% |
| PSE (TSO) official day-ahead forecast | 2.23% |
| Ridge, market data only | 4.05% |
| Seasonal naive | 5.59% |

The reference points: the *TSO forecast* is the official prediction
published by PSE, the Polish grid operator — the operational standard.
*Seasonal naive* repeats the load from the same hour one week earlier;
any model worth running must beat it. Combining market data with the
TSO's own forecast outperforms the TSO by 0.15 percentage points.

**Day-ahead price (EUR/MWh, SDAC auction)**

| Model | MAE | Relative MAE | 80% interval coverage |
|---|---|---|---|
| LightGBM quantile + conformal calibration | **17.9** | **0.64** | 78.9% |
| LEAR + conformal calibration | 18.2 | 0.65 | 79.6% |
| Naive (same hour yesterday) | 28.0 | 1.00 | 53.1% |

*Relative MAE* is the error divided by the naive model's error — below
1.0 beats naive (Lago et al. 2021 convention). *LEAR* is the standard
econometric baseline in price forecasting research: LASSO-regularized
regression per delivery hour. *80% interval coverage*: each forecast
comes with a P10–P90 band that should contain the realized price 80% of
the time; conformal calibration keeps that promise on rolling data.

## How it works

```
ENTSO-E / PSE / Open-Meteo APIs
        │  fetch + gap checks
        ▼
feature matrix  (calendar, weather, lags — all cutoff-safe)
        ▼
models          (ridge, LightGBM quantile, LEAR, LSTM)
        ▼
conformal calibration of P10/P50/P90 bands
        ▼
daily report    (scores vs actuals, forecast chart, top drivers)
```

- Forecasts are probabilistic: P10/P50/P90 quantiles, evaluated with
  pinball loss and interval coverage, not just point error.
- The evaluation is walk-forward only: the model is refit on a trailing
  window and tested on data it has never seen, week after week.
- Feature attribution (SHAP) explains each day's forecast in the daily
  report.
- Challenger models run in shadow mode and replace the incumbent only
  after beating it over a pre-agreed evaluation window.

## Getting started

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).
A free [ENTSO-E API token](https://transparency.entsoe.eu/) is the only
credential needed.

```bash
git clone https://github.com/barteksarwa/pl-energy-forecast
cd pl-energy-forecast
make setup            # install locked dependencies
cp .env.example .env  # add ENTSOE_API_TOKEN
make test             # unit tests, no network needed
make dry-run          # full daily cycle: fetch, score, forecast, report
```

Reproduce the backtests:

```bash
uv run python -m src.evaluation.run_backtest        # load models
uv run python -m src.evaluation.run_price_backtest  # price models
```

## Data sources

| Source | Data | Cost |
|---|---|---|
| [ENTSO-E Transparency](https://transparency.entsoe.eu/) | load, TSO forecast, day-ahead prices, wind/solar forecasts | free, token |
| [PSE API](https://api.raporty.pse.pl/) | Polish grid operator data (canonical for recent history) | free |
| [Open-Meteo](https://open-meteo.com/) | ERA5 weather + archived weather forecasts, 10 cities | free |

## Repository layout

```
src/
  clients/         API wrappers (ENTSO-E, PSE, Open-Meteo)
  ingestion/       backfill, gap logging, cross-source checks
  features/        calendar, weather, lag features
  models/          baselines, LightGBM, LEAR, LSTM
  evaluation/      metrics, walk-forward backtests, conformal calibration
  interpretability/ SHAP explanations
  pipeline/        the daily run
config/            single YAML config
reports/           daily reports, backtest results, figures
docs/              model cards, methodology
```

## Documentation

- [How a forecast is made](docs/HOW_A_FORECAST_IS_MADE.md) — the daily
  cycle, step by step.
- [Model cards](docs/model_cards/) — assumptions, performance, and
  limitations of each model.
