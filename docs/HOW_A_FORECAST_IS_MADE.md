# How a forecast is made

The full path from raw data to a published number, for one day.
Readable start to finish in five minutes. Written for a non-expert;
every term is explained the first time.

## The setting

Poland's power market runs on a daily rhythm. Every day at 12:00 CET,
the day-ahead auction fixes electricity prices for every hour of
tomorrow. Everyone who buys or sells power must decide their positions
before that gate closes. Our job: tell them what tomorrow looks like —
demand (load) and price — before they must act.

## 05:30 UTC — the run starts

A scheduled GitHub Actions job wakes up and executes
`src/pipeline/daily_run.py`. No human involved. Everything below is
one unattended run.

## Step 1 — fetch what's new

- Actual electricity consumption (load) up to this morning — PSE API.
- PSE's own load forecast for tomorrow (timing note below).
- Weather forecast for tomorrow — Open-Meteo, 10 Polish cities,
  averaged with population weights.
- Yesterday's day-ahead prices and the wind+solar generation forecast —
  ENTSO-E.

**Timing note.** Our run starts at 05:30 UTC. PSE publishes tomorrow's
forecast around 09:00 local time. So the run often happens first.
Detection is simple. We fetch tomorrow's series from the API. Hours
missing from the response mean "not published yet". The `persist_24h`
function (`src/pipeline/price_daily.py`) fills each missing hour with
the value from 24 hours earlier. The wind+solar forecast (published
~18:00) gets the same treatment. The daily report lists every such
fallback in its "oddities" section.

Every series lands in versioned parquet files, UTC timestamps, with a
gap log. Nothing is silently filled.

## Step 2 — score yesterday

Yesterday's forecasts are compared with what actually happened:

- Load: our error vs the seasonal-naive baseline vs PSE's forecast.
- Price: our error vs "same hour yesterday".

Yesterday's charts are re-drawn with the realized line on top of the
published band — every forecast chart eventually shows how it went.
This is the accountability step; it runs before any new forecast.

## Step 3 — build leakage-safe features

The forecast for tomorrow may only use information that exists right
now. This is enforced, not assumed:

- Load features: consumption lags from 48h back (yesterday's evening
  isn't fully measured yet), calendar (weekday, Polish holidays, bridge
  days), weather forecast, PSE's public forecast.
- Price features: all 24 of yesterday's prices (the auction already
  fixed them — legal), same-hour lags 1/2/3/7 days back, the wind+solar
  forecast (renewables displace expensive plants — the #1 price driver,
  verified two independent ways), the load forecast, calendar.
- Tests prove the cutoff: corrupt every "future" value and the feature
  matrix must not change. A DST test caught a real bug here once.

## Step 4 — the models forecast tomorrow

- **Load, published**: ridge regression that corrects PSE's forecast
  using weather and history. Beats PSE by ~7% MAE over 2 years.
  Not every model leans on PSE. TSO-free variants (plain ridge,
  LightGBM) were tested too. Best TSO-free model: LightGBM at 3.16%
  MAPE. With the PSE signal: ridge at 2.13% (12-month walk-forward).
  The public PSE forecast is the single strongest input.
- **Price, published**: LEAR — 24 small LASSO regressions, one per
  delivery hour (the industry-standard price baseline).
- Each also produces an uncertainty band (P10–P90), widened by a
  calibration factor learned from the last 90 days of real errors
  (conformal calibration) so that "80% band" empirically means ~80%.
- **Champion vs publisher**: for price, LightGBM+conformal is the
  backtest champion (rMAE 0.638 vs LEAR's 0.660). LEAR still
  publishes. A backtest win alone changes nothing on a real desk.
  The challenger runs in shadow: forecast made and scored every day,
  never published. It earns promotion only by beating the incumbent
  over a pre-agreed 14-day window (`docs/shadow_tally*.md`). Each
  product has its own shadow challenger.
- **Deep challengers, tested and archived**: TFT and PatchTST, two
  transformer architectures, got a full tuning campaign on price.
  Best deep result: TFT at 18.31 EUR/MWh MAE vs the LightGBM
  champion's 17.66 (same 1-year window). At this data size, gradient
  boosting extracts more from the price history than deep nets do.
  Full story: `docs/handovers/2026-07-20_cross-model-ablation.md`.

## Step 5 — write the report, commit everything

`reports/daily/YYYY-MM-DD.md`: yesterday's scores, tomorrow's numbers,
the top drivers in plain words, and an "oddities" list that states
every fallback and failure honestly (error text is scrubbed of
secrets first). Charts embedded. The commit lands in git — the
history of commits IS the track record; it cannot be edited
retroactively without leaving a trace.

## What can go wrong, and what happens then

- A data source is down → that section fails alone, the rest of the
  report still publishes, the oddity names the failure.
- A model produces garbage (all-NaN features) → the run refuses to
  publish an empty forecast and says why.
- The forecast is bad → it gets scored badly tomorrow, in public.
  Bad days stay in the record; the worst ones get a post-mortem panel
  (`reports/figures/backtests/`).

## Where to verify any claim

| Claim | Artifact |
|---|---|
| Beats the TSO | `reports/backtests/2026-07-16_2yr_summary.csv` |
| Price beats naive | `reports/backtests/2026-07-16_price_res_summary.csv` |
| Bands calibrated | `reports/backtests/2026-07-17_price_conformal_summary.md` |
| Solar = driver #1 | `reports/backtests/2026-07-20_price_group_ablation.md` + SHAP |
| Deep nets lose to GBM | `docs/handovers/2026-07-20_cross-model-ablation.md` |
| Leakage-safe | `tests/test_features.py`, `tests/test_price_features.py` |
| It runs daily | commit history of `reports/daily/` |
