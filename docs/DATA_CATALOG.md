# Data Catalog

What a real forecasting desk watches, where we get it, what it costs.
Every dataset the repo fetches has one row here. All timestamps are UTC.
Updated: 2026-07-21 (Phase 2).

Fetch code: `src/ingestion/backfill.py`. One function per dataset.
Cadence "daily" = refreshed by the daily dry run. All backfills are
idempotent: they resume from the last stored timestamp.

## Datasets on disk

| Dataset | File(s) | Source | Unit | Range | Cadence | Used by |
|---|---|---|---|---|---|---|
| Actual load PL (canonical) | `data/processed/load.parquet` | PSE v2 `kse-load` + ENTSO-E for pre-2024-06 | MW | 2023-01 → now | daily | both |
| TSO day-ahead load forecast | `data/processed/tso_forecast.parquet` | PSE v2 + ENTSO-E history | MW | 2023-01 → D+1 | daily | both (benchmark) |
| Load, ENTSO-E store | `data/processed/entsoe/load.parquet` | ENTSO-E `query_load` | MW | 2023-01 → now | daily | cross-check |
| TSO forecast, ENTSO-E store | `data/processed/entsoe/tso_forecast.parquet` | ENTSO-E | MW | 2023-01 → now | cross-check | cross-check |
| Weather actuals (ERA5) | `data/raw/weather/<city>.parquet`, 10 cities | Open-Meteo Archive API | °C, m/s, %, W/m² | 2023-01 → now (−5 days) | daily | both (training) |
| Archived weather forecasts | `data/raw/weather_forecast/<city>.parquet` | Open-Meteo Previous Runs API | same, at lead 1d and 2d | 2024-01 → now | daily | both (honest backtest input) |
| Day-ahead price (canonical) | `data/processed/price_da_eur.parquet` | ENTSO-E `query_day_ahead_prices` | EUR/MWh | 2023-01 → D+1 | daily | price model |
| Day-ahead price, PLN | `data/processed/price_da.parquet` | PSE v2 `csdac-pln` | PLN/MWh | 2024-06-14 → now | daily | display, cross-check |
| Balancing price | `data/processed/price_balancing.parquet` | PSE v2 `crb-rozl` | PLN/MWh | 2024-06-14 → now | daily | analysis |
| Generation mix (actuals) | `data/processed/generation_mix.parquet` | PSE v2 `his-wlk-cal` | MW (pv, wind, thermal) | 2024-06-14 → now | daily | analysis |
| RES day-ahead forecasts | `data/processed/res_forecast.parquet` | ENTSO-E `fetch_res_forecast` | MW (solar, wind on/offshore) | 2023-01 → D+1 | daily | price model (driver #1) |
| Fuel proxies | `data/processed/fuel_daily.parquet` | yfinance: TTF future + EUA-tracking ETC | EUR/MWh, EUR/t (proxy) | 2023-01 → now | daily close | price model |
| Unit outages | `data/processed/outages.parquet` | ENTSO-E UMM | event-level, MW | ~21k events, 2023 → D+60 | opt-in, manual | research (backtest: flat) |
| Polish holidays | `holidays` package, computed on the fly | offline | flags | any | — | both |

Notes on specific rows:

- **Load merge.** PSE v2 is canonical from 2024-06-14 (DECISIONS 2026-07-14).
  ENTSO-E extends history to 2023 and cross-checks the overlap
  (0.03% mean diff). Merge lives in `crosscheck.py`.
  `*_pse_only.parquet` files are the pre-merge PSE-only backups.
- **Price units.** EUR/MWh is the raw unit of the ENTSO-E feed. It is
  canonical. PLN is display-only; the PSE PLN series doubles as a
  cross-check.
- **Wind offshore.** `wind_off_fcst_mw` is ~0 before 2026. First real
  values: 2026-07-01. PL offshore (Baltic Power) came online 2026-07.
  Models trained on earlier windows never saw it — watch this feature.
- **Fuel proxies.** Free yfinance closes, not ICE/EEX settlement ticks.
  The EUA series is an ETC that tracks EUA futures, not EUA itself.
  Daily resolution only. Good enough for a level signal, stated honestly.
- **Outages.** Deduped on (mrid, revision). Publication time
  (`created_doc_time`) is kept — it is the leakage boundary. Endpoint is
  heavy and throttle-prone, so it is opt-in (`--only entsoe_outages`).
- **Cross-border flows/capacity.** Not fetched yet. Listed as a Phase 2
  candidate; skipped so far. Would come from ENTSO-E if added.
- **Coal (API2).** Paid only. Skipped. We document the impact instead.

## Zone-level vs point data

- RES forecasts, load, prices: aggregated at bidding-zone level (PL).
- Weather: point data at 10 city centers, population-weighted
  (weights in `config/config.yaml`, GUS 2024 population).
- City weather is a load proxy. It is a poor proxy for wind/solar farms,
  which sit far from cities.
- Site-level RES locations (farm coordinates, capacities) are NOT in the
  repo yet. Planned.

## The weather leakage trap (important)

At 09:00 on D-1 the desk knows the weather *forecast* for D, not the weather.
A backtest that feeds the model ERA5 actuals overstates accuracy —
it silently removes weather-forecast error.

Rule: training on archive actuals is fine; **evaluate with archived
forecasts as inputs**. The Previous Runs API serves each variable at
fixed lead-time offsets. We store lead 1d and 2d. Lead 2d mimics what
was known at the 09:00 D-1 cutoff; lead 1d is for forecast-error analysis.

## Other files in data/ (not source data)

- `data/raw/*_latest.parquet` — daily-run scratch, overwritten each run.
- `data/processed/backtest_preds_*` — stored backtest predictions.
- `data/processed/*_ckpts/`, `tft_hpo.db` — model checkpoints and HPO
  state. See model cards, not this catalog.
- `data/processed/gap_log.csv` — every missing interval, never filled
  silently.

## What real desks have that we will not

Named honestly, for interviews:

- Paid market data: ICE/EEX gas and CO2 ticks, Bloomberg/Refinitiv terminals.
- Commercial weather: Meteomatics, DTN — higher resolution, ensembles,
  asset-tuned. (Axpo uses Meteomatics EURO1k for day-ahead/intraday trading.)
- Intraday order books, proprietary outage intel, customer portfolio data.

Free proxies keep the *methodology* identical. We state the data gap, not hide it.

## Known quality issues (from literature and docs)

- ENTSO-E: missing values and inconsistencies are common; no public flagging
  process. Hence our gap log. (Hirth et al. 2018 review, Applied Energy.)
- ENTSO-E PL load: resolution switched to 15 min (EU MTU change). We resample
  to hourly mean in the client.
- ERA5 archive: ~5 day publication delay. Daily ops use the Forecast API
  `past_days` for recent actual-ish weather; archive is for backfills only.
- PSE: old API v1 disabled end of 2025. Use `api.raporty.pse.pl` (v2) only.
- Day-ahead price and RES forecast exist through end of tomorrow only after
  publication (~13:00 / ~18:00 CET on D-1). The backfill retries the same
  window until published — no silent holes.
