# Data Catalog

What a real forecasting desk watches, where we get it, what it costs.
Status: verified = we checked the API docs or called it. [TBC] = not yet verified.
Updated: 2026-07-14 (M1).

## Phase 1 — load forecasting

| Data | Source | Access | Cost | Resolution | Status |
|---|---|---|---|---|---|
| Actual load PL | ENTSO-E Transparency | `entsoe-py` `query_load` | free, token | 15 min → we store hourly | verified, live call pending token |
| TSO day-ahead load forecast | ENTSO-E | `query_load_forecast` | free | hourly | verified |
| Weather history (actuals, ERA5) | Open-Meteo Archive API | `archive-api.open-meteo.com/v1/archive` | free non-commercial | hourly, ~5 day delay | verified, called live |
| Weather forecasts (operational) | Open-Meteo Forecast API | `api.open-meteo.com/v1/forecast` | free | hourly, 16 days ahead | verified, called live |
| Historical weather *forecasts* | Open-Meteo Historical Forecast API | `historical-forecast-api.open-meteo.com` [TBC endpoint] | free | hourly | [TBC] — needed for honest backtests, see below |
| Polish holidays | `holidays` package | offline | free | daily | verified |

### The weather leakage trap (important)

At 09:00 on D-1 the desk knows the weather *forecast* for D, not the weather.
A backtest that feeds the model ERA5 actuals overstates accuracy —
it silently removes weather-forecast error.

Rule: train on archive actuals is fine; **evaluate with historical forecasts**
as inputs. Open-Meteo explicitly supports "backtests without look-ahead bias"
via its historical forecast data. Verify endpoint + earliest date in M2. [TBC]

## Phase 2 — price forecasting (day-ahead PL)

| Data | Source | Access | Cost | Status |
|---|---|---|---|---|
| Day-ahead price PL | ENTSO-E (`query_day_ahead_prices`) [TBC method name] | `entsoe-py` | free | [TBC] |
| Wind + solar generation forecast | ENTSO-E | `entsoe-py` | free | [TBC method] |
| Generation per fuel type | ENTSO-E | `entsoe-py` | free | [TBC] |
| Cross-border flows + capacity | ENTSO-E | `entsoe-py` | free | [TBC] |
| Unit/grid outages | ENTSO-E UMM | `entsoe-py` | free | [TBC] |
| PL balancing prices, KSE demand | PSE API v2 | `api.raporty.pse.pl` | free | verified exists; endpoints [TBC] |
| Gas price (TTF) | public proxies (e.g. energy-charts, EIA daily) | HTTP | free proxies | [TBC — desks pay for ICE/EEX feeds] |
| CO2 (EUA) | public settlement data | HTTP | free proxies | [TBC] |
| Coal (API2) | mostly paid | — | paid | likely skip; document impact |

## What real desks have that we will not

Named honestly, for interviews:

- Paid market data: ICE/EEX gas and CO2 ticks, Bloomberg/Refinitiv terminals.
- Commercial weather: Meteomatics, DTN — higher resolution, ensembles, asset-tuned.
  (Axpo uses Meteomatics EURO1k for day-ahead/intraday trading.)
- Intraday order books, proprietary outage intel, customer portfolio data.

Free proxies keep the *methodology* identical. We state the data gap, not hide it.

## Known quality issues (from literature and docs)

- ENTSO-E: missing values and inconsistencies are common; no public flagging
  process. Hence our gap log: every missing interval recorded, never filled
  silently. (Hirth et al. 2018 review, Applied Energy.)
- ENTSO-E PL load: resolution switched to 15 min (EU MTU change). We resample
  to hourly mean in the client.
- ERA5 archive: ~5 day publication delay. Daily ops therefore use the Forecast
  API `past_days` for recent actual-ish weather; archive is for backfills only.
- PSE: old API v1 disabled end of 2025. Use `api.raporty.pse.pl` (v2) only.
