# Project Brief — Day-Ahead Load Forecasting for Poland

## Why this project exists

I want a job in energy demand forecasting, in Poland or the EU.
Employers there: utilities (PGE, Tauron, Enea), trading firms (Axpo, Statkraft, Respect Energy),
TSOs/DSOs, and forecasting vendors.

Those jobs are not about training one model once.
They are about running a forecast every day, explaining it, and improving it.
This repo simulates that job. That is the whole point.

## The product

Every day, before 09:00 CET, produce an hourly load forecast for Poland for the next day.
Output three numbers per hour: P10, P50, P90.
Explain what drives the forecast, in plain words.
Track accuracy over time and publish it.

## Scope

In scope:
- Zone PL first. One more EU zone later (nice-to-have) to show the code generalizes.
- Weather, calendar, gas prices, and lag features.
- Models: seasonal naive → linear → LightGBM (quantile) → LSTM → small transformer.
- Rolling backtest over at least 12 months of history.
- Daily automated dry run with a human-readable report.

Out of scope:
- Price forecasting. Intraday updates. Trading logic. A web UI.
- Anything that delays the first working daily loop.

## Order of value (what matters most, first)

1. A daily loop that runs and reports. Even with the naive model.
2. Honest evaluation against the TSO's own day-ahead forecast.
3. Interpretability: SHAP, driver summaries, clear model cards.
4. Deep models (LSTM, transformer) as challengers, only after 1–3 work.

I already know deep learning. The job market wants proof of
operational discipline and interpretability. So the project leads with those.

## Success criteria

- The daily loop runs unattended for 30+ days.
- Our P50 beats seasonal naive clearly. Getting close to the TSO forecast is a win.
- Every number in the results table can be reproduced with one command.
- A hiring manager can skim the repo and think: "this person has done the job already."

## Known risks

- ENTSO-E API quirks: publication delays, resolution changes, gaps. Log gaps, never silently fill.
- DST days break naive time math. Test them.
- Polish holidays and bridge days shift load a lot. Feature them early.
- Weather forecast errors leak into load errors. Separate the two in evaluation if possible.
