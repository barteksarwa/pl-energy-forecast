# CLAUDE.md — Energy Demand Forecasting Project

## What this repo is

A portfolio project that simulates a real job in energy demand forecasting.
The product: a day-ahead electricity load forecast for Poland (PL bidding zone, rynki mocy, ppa and more).
It must look and feel like commercial work. Not like a tutorial.

The owner is a PhD student (ML for solid oxide fuel cells).
He knows LSTMs, transformers, and gradient boosting and other basic models for forecasting but no seasonality concepts yet.
Goal: land a forecasting job in Poland or the EU. Those jobs are mostly in energy companies to predict energy price in next day.
Every choice in this repo should support that goal. Besides making a project we also need to make sure it is interpretable and explainable by the owner.
The repo and project in general should simulate the regular job in this field. You can first fetch or ask me to fetch the some job descriptions.

You need to update Claude.md and other README.md files when relevant. You can also propose ways to run this repo to save tokens - maybe currently we have too many config files.

## Rule 0 — Plan first. Always.

Before you write any code:

1. Read `docs/PROJECT_BRIEF.md`.
2. Create `docs/PLAN.md`. Break the project into milestones. Suggest an order.
3. Show the plan to the owner. Wait for approval.
4. Only then start Milestone 1.

If `docs/PLAN.md` already exists, read it and continue from the current milestone.
If the plan feels stale or wrong, say so. Propose an update. Do not silently drift.

At the start of every session: read the latest handover in `docs/handovers/` first.

## Writing style — this matters a lot

The owner finds long sentences hard to read. So:

- Short sentences. Aim for under 15 words.
- One idea per sentence.
- Plain words. Say "use", not "utilize".
- Prefer bullet points over long paragraphs.
- Explain jargon the first time you use it.
- No filler. No "it is worth noting that".

This applies to ALL text you write: notes, handovers, docstrings, README, commit messages, reports.

## Target repo structure

Build toward this. Create folders as milestones need them, not all at once.

```
.
├── CLAUDE.md
├── README.md
├── Makefile                  # entry points: setup, test, dry-run, backtest
├── pyproject.toml
├── .env.example
├── .github/
│   └── workflows/
│       └── daily-dry-run.yml # the daily "job simulation" (see below)
├── config/
│   └── config.yaml           # zones, cities, model params, paths
├── data/                     # gitignored, only .gitkeep files committed
│   ├── raw/
│   ├── processed/
│   └── forecasts/
├── docs/
│   ├── PROJECT_BRIEF.md      # why this project exists
│   ├── PLAN.md               # you write this first
│   ├── DECISIONS.md          # short decision log
│   ├── model_cards/          # one card per shipped model
│   ├── handovers/            # one file per work session
│   └── notes/                # daily ops notes
├── notebooks/                # EDA only, numbered: 01_..., 02_...
├── reports/
│   └── daily/                # output of the dry run, committed
├── src/
│   ├── clients/              # API connections: ENTSO-E, Open-Meteo
│   ├── ingestion/            # fetch and store data, backfill scripts
│   ├── features/             # calendar, weather, lags
│   ├── models/               # baselines, gbm, lstm, transformer
│   ├── evaluation/           # metrics, rolling backtests
│   ├── interpretability/     # SHAP, importance, forecast explanations
│   └── pipeline/             # daily_run.py orchestrates everything
└── tests/
```

## Data sources

- **ENTSO-E Transparency Platform.** Actual load + the TSO day-ahead load forecast for zone PL.
  Use the `entsoe-py` client. Token comes from `.env` as `ENTSOE_API_TOKEN`. Never hardcode it.
  The TSO forecast is our benchmark to beat, or at least to match.
- **Open-Meteo.** Free weather API, no key needed. Historical archive + forecasts.
  Build a population-weighted average over major Polish cities. Put the city list in config.
- **`holidays` package.** Polish public holidays. Also mark bridge days (Friday after a Thursday holiday).

Verify current API details from official docs before implementing. Do not trust memory.

## Hard rules

1. **No data leakage.** The forecast for day D is made at 09:00 on day D-1.
   Only use data that existed at that moment. Backtests must respect this cutoff. Test it.
2. **Time handling.** Store everything in UTC. Display in Europe/Warsaw.
   DST switches create 23h and 25h days. Handle them. Write a test for them.
3. **Interpretability is a requirement, not a bonus.**
   Every shipped model needs an explanation artifact.
   Gradient boosting → SHAP. Deep models → permutation importance + attention/saliency inspection.
   The daily report must state the top 3 drivers of today's forecast in plain words.
4. **Every model fights the baselines.** Seasonal naive (same hour, last week) and the TSO forecast.
   If a fancy model loses to naive, we say so honestly and document why.
5. **Probabilistic output.** Ship P10 / P50 / P90 quantiles, not just a point forecast.
   Evaluate with pinball loss. Point metrics: MAE, RMSE, MAPE.
6. **No secrets and no data in git.** `.env` and `data/` stay gitignored.
7. **Small, tested, typed.** Type hints everywhere. Tests for pure logic (features, metrics, cutoffs).
   No test theater for API wrappers — one integration smoke test is enough.
8. **Dependencies: keep the stack boring** (pandas, scikit-learn, lightgbm, shap, pytorch).
   Boring/support deps (plotting, IO, testing) are pre-approved — add without asking,
   list them in the session handover. Ask only for heavy or exotic additions.
9. **Keep git and GitHub updated. Don't ask.** Commit after every coherent chunk.
   Push to the remote after every work session (and after milestones at minimum).
   Small conventional commits. The commit history is part of the product.

## The daily dry run — the "job simulation"

This is the heart of the project. It mimics what a forecaster does every morning.

One command: `make dry-run`. Also a GitHub Actions cron (around 05:30 UTC).

Steps, in order:

1. Fetch the newest actual load and weather.
2. Score yesterday's forecast against actuals. Compare with baselines and the TSO forecast.
3. Produce the day-ahead forecast for tomorrow (P10/P50/P90).
4. Write a short report to `reports/daily/YYYY-MM-DD.md`. Include:
   - yesterday's MAPE: ours vs naive vs TSO,
   - one chart (forecast vs actual),
   - top 3 forecast drivers in plain words,
   - anything odd (missing data, holiday effects, weather swings).
5. Commit the report. The commit history becomes proof of consistent operational work.

The report must be readable by a non-technical manager in 60 seconds.

## Dev workflow

- Everything runs through `uv run` (or the `make` targets that wrap it).
  Never bare `python`. Permissions for `uv run`, `make`, `git`, `gh` are
  pre-approved in `.claude/settings.json` — do not ask.
- The project is an installed package (hatchling, `packages = ["src"]`).
  `import src.*` works from any directory. If imports break: `uv sync`.
- After changing scripts, run them once. Never hand over an untested command.

## Owner-facing notes are LaTeX

Markdown is for agents and ops (reports, handovers, decision log).
The owner learns and keeps track from LaTeX notes. Two sets:

- `docs/notes/learning/` — forecasting concepts. One concept = one `NN_topic.tex`.
- `docs/notes/model_selection/` — which model, when, why, honest verdicts.

Rules:
- Each set has a `main.tex`. After creating a new note, add one line to it:
  `\input{NN_topic.tex}`. Nothing else changes in main.
- Number notes in order: `01_`, `02_`, ...
- Same writing style as everything: short sentences, worked examples, interview lines.

## Visualize what we fetch

Data nobody looks at rots. So:

- After a smoke test or backfill fetches data, run `make viz`.
  It plots whatever is in `data/` (weather, load, forecasts) into `reports/figures/`.
- Every new data source gets a plot function in `src/viz/` in the same milestone.
- Plots must make sense down the road: label units, UTC vs local, data source.

## Session rituals

- End every work session with a handover: `docs/handovers/YYYY-MM-DD_topic.md`.
  Use `docs/templates/HANDOVER_TEMPLATE.md`. Keep it under one page.
- Log every non-obvious choice in `docs/DECISIONS.md`. Three lines each: context, decision, why.
- Commits: conventional style (`feat:`, `fix:`, `docs:`, `chore:`). Small commits.

## Definition of done (recruiter-ready)

The project is done when:

- [ ] A recruiter can understand the README in 3 minutes.
- [ ] `make dry-run` works from a fresh clone with only an ENTSO-E token.
- [ ] At least 30 daily reports exist in `reports/daily/`.
- [ ] A rolling backtest over 12+ months compares: naive, TSO, LightGBM, LSTM, transformer.
- [ ] Results table is honest. Losses are explained, not hidden.
- [ ] Each shipped model has a model card.
- [ ] SHAP summary and a "how a forecast is made" doc exist and are readable.
