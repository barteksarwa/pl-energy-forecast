# Polish Power Market Forecasting — Load & Price, Day-Ahead

A production-style forecasting desk for the Polish power market, built
and operated by one person. Every morning it forecasts tomorrow's hourly
**load** and **day-ahead price**, scores yesterday's forecasts against
reality, explains its predictions in plain words, and commits the report.
The git history is the live track record.

**Headline results (2-year walk-forward, leakage-proof):**

- **Load: beats the Polish TSO.** Ridge combiner 2.08% MAPE vs PSE's own
  day-ahead forecast at 2.23%.
- **Price: beats the standard.** LightGBM rMAE 0.638 and LEAR 0.660 vs
  naive (literature range for LEAR: 0.75–0.85). Wins every one of 25
  test months.
- **Calibrated uncertainty.** P10/P90 bands conformally calibrated to
  ~79% empirical coverage (nominal 80%).

## The two products

### 1. Day-ahead load (Phase 1 — complete)

Hourly load for the PL bidding zone, decided at 09:00 D-1, P10/P50/P90.

2-year walk-forward, 17,450 test hours
(`reports/backtests/2026-07-16_2yr_summary.csv`):

| Model | MAPE | MAE (MW) | Skill vs naive |
|---|---|---|---|
| **Ridge + TSO forecast (combiner)** | **2.08%** | **374** | **0.63** |
| LightGBM + TSO forecast | 2.12% | 384 | 0.62 |
| PSE (TSO) day-ahead forecast | 2.23% | 401 | 0.60 |
| Ridge (no TSO) | 4.05% | 710 | 0.29 |
| Seasonal naive (same hour last week) | 5.59% | 1005 | 0.00 |

Deep challengers, evaluated on the 12-month campaign
(`2026-07-15_overnight_readout.md`, `2026-07-14_fcst_summary.csv`,
model cards): LSTM-attention+TSO 2.43%, best plain LSTM of 7
architectures 3.67%, LightGBM-no-TSO 3.16%. None earned a place above
the linear combiner, so none were re-run on 2 years.

Honest findings the table forces:
- The TSO forecast is public at bid time; combining with it beats the
  TSO by ~7% MAE. Once that signal is in, **ridge beats every deep net
  we built** — and we built seven.
- Bigger nets lose. Accuracy peaked at ~106k parameters on 2y of data.
- Cheap screening splits flattered the nets by 0.6–0.9 pp vs honest
  walk-forward. Most tutorials never mention this.

### 2. Day-ahead price (Phase 2 — live since 2026-07)

PL day-ahead auction price (SDAC), EUR/MWh, forecast before gate closure.

| Model | MAE (EUR/MWh) | rMAE | Band coverage (nominal 80%) |
|---|---|---|---|
| **LightGBM quantile + conformal** | **17.8** | **0.638** | 78.7% |
| LEAR + conformal (published daily) | 18.5 | 0.660 | 79.5% |
| Naive (same hour yesterday) | 28.0 | 1.000 | 53.1% |

- LEAR = the industry-standard LASSO price baseline (Ziel & Weron 2018),
  implemented properly: 24 per-hour models, full D-1 price vector,
  variance-stabilized target.
- The **solar forecast is price driver #1** — top SHAP attribution AND
  largest retrain-ablation cost (+3.5 EUR/MWh MAE when dropped). Two
  independent methods, same answer: the merit order, measured.
- Spikes are the open front: all models run ~3x pooled MAE on the top-5%
  priciest hours. Documented, not hidden.

## The daily loop (the actual product)

GitHub Actions cron, 05:30 UTC, unattended:

1. Fetch latest actuals (load, price, weather, wind/solar forecasts).
2. Score yesterday's forecasts against reality; redraw yesterday's
   charts with the realized line ("living figures").
3. Forecast tomorrow: load (incumbent + shadow challenger) and price
   (LEAR + conformal band).
4. Write a report a manager reads in 60 seconds: `reports/daily/`.
5. Commit. The history is the proof of consistent operation.

Model changes go through a promotion gate: challengers run in shadow
for 14 days and replace the incumbent only if they win on metrics agreed

## Why this is credible

- **Leakage paranoia.** The 09:00 D-1 cutoff is enforced by asserts and
  corruption-proof tests. The DST leakage test caught a real bug: on the
  25-hour October day, "minus 24 hours" reaches into the target day.
- **Baselines first.** Nothing ships without beating seasonal naive and
  the external benchmark. Losing models stay in the tables.
- **Walk-forward only.** Every reported number is out-of-sample,
  day-ahead, weekly refits, 2 years of test data (17,480 hours).
- **Desk-style review pack.** Drift, cumulative edge, hourly error
  profile, quantile calibration, worst-day post-mortems, monthly bias:
  `reports/figures/backtests/` (with a how-to-read README).
- **Four bugs found by our own defenses**, each documented with the
  measured impact: DST leakage, asinh blowup, solar-growth extrapolation
  (38,000 EUR/MWh predictions → z-clip guard), gap-permanence.

## Stack

Python · LightGBM · PyTorch · LEAR (LASSO-AR) · scikit-learn · SHAP ·
DuckDB · GitHub Actions · ENTSO-E API · Open-Meteo

## Live status

| Thing | Status |
|---|---|
| Cron (05:30 UTC) | live |
| Load challenger (ridge+TSO) | 14-day shadow run in progress |
| Price incumbent (LEAR) | 14-day shadow run in progress |
| Price challenger (LightGBM) | 14-day shadow run in progress |
| Daily reports | accumulating (target: 30) |

## Quickstart

```bash
make setup            # install deps (needs uv)
cp .env.example .env  # add your ENTSO-E token (free)
make test             # unit tests, no network
make dry-run          # one full daily cycle: fetch, score, forecast, report
```

Backtests: `uv run python -m src.evaluation.run_backtest` (load),
`uv run python -m src.evaluation.run_price_backtest` (price).
SQL analysis of all datasets: `notebooks/01_sql_analysis.ipynb` (DuckDB).

## Map

- `docs/model_cards/` — one honest card per model
- `docs/HOW_A_FORECAST_IS_MADE.md` — the daily cycle, step by step
- `reports/daily/` — the live track record
- `reports/backtests/` — every results table cited above
