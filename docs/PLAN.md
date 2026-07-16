# PLAN.md — Roadmap

Status: **v2 approved by owner 2026-07-14.** Current milestone: M0 (walking skeleton).
Last update: 2026-07-14. v2 after owner feedback on v1.

## What the job market wants (researched 2026-07-14)

Sources: Shell careers, pracuj.pl (Polenergia, Tauron, PGE), EPF literature.

**Load or price?** Both exist. The split:

- Trading desks (Shell, Orlen, Polenergia, Axpo): **price** forecasting.
  Power day-ahead, gas (TTF), CO2 (EUA). Plus fundamentals analysis.
- Utilities, DSOs, balance-responsible parties: **load** forecasting.
  Load errors cost money on the balancing market.
- Load forecast is an *input* to price forecast. The skills stack.

**Models named in ads and industry practice:**

- LASSO-regularized autoregression (LEAR). The standard price baseline. Hard to beat.
- Gradient boosting (XGBoost/LightGBM). Quantile regression for probabilistic output.
- LSTM / deep nets as challengers. Shell explicitly wants ML PhDs.
- Optimization models (stack, dispatch) at fundamentals desks. Out of our scope, but
  we explain the concept in docs.

**Other requirements that repeat:** Python + SQL, model deployment/productionization,
explaining forecasts to non-experts, market knowledge (day-ahead auction, balancing,
CO2, capacity market).

**Conclusion.** Build load first (cleanest daily-ops simulation, data is free and
clean). Then price as Phase 2 on the same infrastructure. Same features pipeline,
same backtest engine, same daily loop. This mirrors a real desk and doubles the
interview surface.

## Ambition level

Owner's call: build system that could be basically level of a senior job.
Since we are doing PhD potentially we could publish so keep sources true, if you dont know something tell that so i can fetch more data, or explain the assumptions.
So we target near-production quality:

- Clean contracts between components. New model = new file, zero edits elsewhere.
- UAT/prod split (see Phase 3). Real deployment discipline, free tools only.
- Reproducible benchmark tables. Publication angle: honest open PL benchmark
  (load + price, quantiles, vs TSO and LEAR) — these are rare for Poland.

## Phase 1 — Load forecasting core

### M0 — Walking skeleton (1 session)

Thinnest end-to-end path. Real data in, naive forecast out, report written.

- Scaffolding: `pyproject.toml`, `Makefile`, `config/config.yaml`, `.env`.
- ENTSO-E client: PL load + TSO day-ahead forecast, few days.
- Open-Meteo client: one city.
- Seasonal naive forecast. One ugly report in `reports/daily/`.
- One smoke test per client.

Done when: `make dry-run` produces a report from live APIs.

### M1 — Data foundation + data-source analysis (2 sessions)

Goal: know what data the job actually uses, then store it cleanly. Before doing this stage i will need to analyze online sources that would allow you to find more information whether that would be science papers or some explanations anywhere online.

- **Data-source analysis** (new, per owner): document in `docs/DATA_CATALOG.md`
  every driver a real desk watches, its source, cost, and when we need it:
  - Load + TSO forecast: ENTSO-E. Free. Phase 1.
  - Weather (temp, wind, radiation, humidity): Open-Meteo. Free. Phase 1.
  - Wind + solar generation forecasts: ENTSO-E. Free. Phase 2 (price driver #1).
  - Cross-border flows and capacity: ENTSO-E. Free. Phase 2.
  - Unit outages: ENTSO-E UMM / PSE. Free. Phase 2.
  - Gas (TTF), coal (API2), CO2 (EUA): free proxies exist (e.g. energy-charts,
    public settlement data); paid feeds are the desk reality. Document both. Phase 2.
  - TGE / ENTSO-E day-ahead prices for PL: target of Phase 2.
- Backfill 3+ years: load, TSO forecast, weather for all config cities.
- Parquet in `data/`, UTC, stable schemas. Gap log, never silent fills.
- Tests: DST days (23h/25h), UTC↔Warsaw.

Done when: backfill idempotent, gap report exists, data catalog reviewed by owner.

### M2 — Features + evaluation engine (1–2 sessions)

- Calendar: hour, weekday, month, PL holidays, bridge days.
- Weather: population-weighted city average.
- Lags respecting the 09:00 D-1 cutoff.
- Metrics: MAE, RMSE, MAPE, pinball. Skill-vs-naive as headline. I need to be able to explain why such metrics is used and what are pros and cons.
- The leakage test: assert no feature sees past the cutoff.
- Full testing campaign

Done when: feature matrix builds for any historical day; cutoff test passes.

### M3 — Baseline campaign (3+ sessions, per owner: full campaign)

Goal: honest, exhaustive baseline table before any fancy model. This is the
credibility core of the repo.

- Baselines: seasonal naive (hour-of-week), 7-day persistence, climatology,
  linear/ridge on lags+calendar, LASSO-AR (LEAR-style, also preps Phase 2).
- TSO day-ahead forecast as the external benchmark.
- Walk-forward backtest over 12+ months. Cutoff respected in every fold.
- Breakdowns: by hour, weekday vs weekend, holidays, seasons, DST days.
- Report: which baseline wins where, and why. Losses documented.
- At this stage i need to be able to explain all seasonality add ons to the models, when to use, where.
- Also which models are good at which use cases (data availability, goals of the task). Which have potential but still need some time/work to be done. And which to scrap and why,
Done when: `make backtest` reproduces the full table; findings written up.

### M4 — LightGBM quantile, the production model (2 sessions)

- LightGBM quantile: P10/P50/P90 per hour.
- SHAP: summary + per-day top-3 drivers in plain words.
- Model card. Swap into daily loop behind config flag.
- Fights the M3 table. If it loses anywhere, we say where.

Done when: report explains its own forecast; P50 beats naive on backtest.

### M5 — Deep challengers (2–3 sessions)

- LSTM with known-future covariates (weather forecast, calendar).
- Small transformer second.
- Permutation importance + saliency. Model cards.
- Into the backtest table. Honest verdict vs LightGBM.

## Phase 2 — Price forecasting (the trading-desk skill)

**Why Phase 2 is the priority.** Job research (2026-07-16, see `docs/notes/job_market.md`)
confirms: trading-quant roles (highest pay, fastest hiring) care about **price**,
not load. Load alone targets utilities. Adding TGE day-ahead price doubles
the number of reachable roles. LEAR is the standard baseline to beat.

### M6 — Price data + fundamentals features (2 sessions)

- [x] **TGE day-ahead prices (RDN):** DONE 2026-07-16. `price_da_eur.parquet`,
  31,022 h from 2023-01-01, zero gaps. ENTSO-E EUR/MWh is the modeling
  target (DECISIONS 2026-07-16); PSE PLN stays for display.
- **Price drivers from M1 catalog:**
  - Wind + solar actual generation (ENTSO-E `query_generation`): free, best driver.
  - Cross-border flow capacity (ENTSO-E `query_crossborder_flows`): interconnection state.
  - Unit outages (ENTSO-E UMM): large unavailabilities move prices.
  - Gas (TTF) proxy: ENTSO-E LNG/gas net imports, or public EEX settlement data.
  - CO2 (EUA) proxy: ICE daily settlement CSV (free, delayed 1 day).
  - Our own load forecast (D-1 produced) as a feature.
- [x] Extend gap log and DST tests to price series. DONE 2026-07-16
  (tests/test_price_features.py — the 25h-day leakage test found a real bug).
- [x] **DuckDB/SQL layer**: DONE 2026-07-16. `notebooks/01_sql_analysis.ipynb`.

### M7 — Price models (3+ sessions)

- [x] **Baseline 1: naive.** DONE 2026-07-16. Yesterday + last-week variants.
- [x] **Baseline 2: LEAR.** DONE 2026-07-16. Per-hour LASSO, D-1 day vector,
  robust-standardized asinh (Uniejewski, Weron & Ziel 2018).
  **Result: rMAE 0.744 vs naive over 2 years, wins all 25 months.**
  Model card: `docs/model_cards/lear.md`. Two failed variants documented there.
- [x] **LightGBM quantile** on fundamentals. DONE 2026-07-16.
  **rMAE 0.638, MAE champion.** SHAP: solar forecast is price driver #1.
  BUT band coverage 51% vs 80% nominal — conformal calibration REQUIRED
  before shipping. Card: `docs/model_cards/lgbm_price.md`.
- [x] Wind+solar day-ahead forecasts backfilled (31,021 h, zero gaps).
  LEAR + RES + extrapolation guard: rMAE 0.660 (z-clip story in card).
- [x] **Price spikes:** DONE — spike MAE + P90 coverage columns in table.
  Both models miss spikes badly (spike MAE ~3x pooled); open problem.
- Same walk-forward engine as M3. Honest table: LGBM wins MAE, LEAR wins
  coverage; neither band is calibrated yet.
- **Next:** conformal band calibration, gas/CO2 proxies, daily-loop
  shadow integration for the price model.
- [x] Learning note: DONE 2026-07-16, `08_price_formation_and_lear.tex`.

### M8 — Market-context docs + portfolio polish

- Learning notes: DAM mechanics, balancing market (RB 2024 reform), rynek mocy, CO2/ETS.
- **Blog post draft:** "I built a day-ahead forecasting desk for the Polish power
  market — and beat the TSO." Public PL benchmark is rare. Top-of-funnel signal.
- README final: results tables up top, 3-minute recruiter read.
- Apply at 30+ daily reports (accumulating since 2026-07-16).

## Phase 3 — Ops maturity (the "senior job transfer" layer)

### M9 — UAT/prod split + POC automation (2 sessions)

Per owner: simulate real deployment discipline, free tools only.

- Environments in config: `dev` (local, any branch), `uat` (shadow run),
  `prod` (the committed track record).
- Promotion rule: a model runs ≥N days in UAT shadow mode; its forecasts are
  scored but not "official". Beats incumbent → promoted to prod. Logged in
  DECISIONS.md. This is how real desks change models.
- POC automation: GitHub Actions cron (free) runs the daily loop for a trial
  window (e.g. 7–14 days) to prove it works unattended. Full 30-day unattended
  push comes after, per owner.

### M10 — Track record + recruiter/publication polish

- 30+ daily reports accumulate in prod mode.
- README final: results tables up top, 3-minute read.
- "How a forecast is made" doc. Model cards complete.
- Publication check: is the PL benchmark table novel enough to write up
  (blog post minimum, workshop paper stretch)? Owner decides with evidence.

## Learning thread (runs through everything)

Owner must explain every piece in interviews. Knows ML, not forecasting
concepts yet. One short explainer per milestone in `docs/notes/learning/`:

- M0: seasonality, seasonal naive.
- M1: day-ahead market timeline, the 09:00 cutoff.
- M2: leakage, walk-forward CV.
- M3: why baselines rule forecasting; skill scores.
- M4: quantile regression, pinball loss, SHAP in plain words.
- M5: why deep models often lose on tabular time series.
- M6–M7: merit order, price drivers, LEAR, spike risk.
- M9: UAT/shadow deployment, why desks promote models slowly.

Rule: under one page, short sentences, one worked example each.

## Token-saving proposals (per CLAUDE.md request)

- One config file: `config/config.yaml`. Environments are keys inside it, not files.
- Handovers under one page. Agents read only the latest.
- Cavecrew subagents for code search in long sessions, preferably by Opus, while Fable is main.
- Notebooks out of agent context. EDA summaries go into docs.

## Risks

- ENTSO-E delays/gaps → gap log from M1, oddities section in reports.
- DST breakage → tests in M1, extended to price in M6.
- Free gas/CO2 data is patchy → documented in data catalog; proxies acceptable,
  stated openly.
- Price spikes break point metrics → tail evaluation in M7.
- Scope creep → Phase 1 must produce a working daily loop before Phase 2 starts.
