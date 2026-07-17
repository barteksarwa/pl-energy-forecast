# I beat the Polish TSO's day-ahead load forecast with ridge regression

_Status: DRAFT for owner to personalise and publish. Numbers are real and
traceable to `reports/backtests/2026-07-16_2yr_summary.csv`. Short
sentences throughout. Every claim verifiable in the repo._

---

Every morning at 09:00, Poland's grid operator publishes tomorrow's
electricity demand forecast. It is very good. Over two years of hourly
data, PSE achieves 2.23% MAPE. My model beats it — 2.08% — using ridge
regression and 25 features. That is a 7% relative improvement, which
sounds small until you remember that load forecast errors cost money on
the balancing market every hour they occur.

This post explains how I built it, what I tried that didn't work, and why
the winning model turned out to be the boring one.

---

## The task

Day-ahead load forecasting means: at 09:00 on day D−1, produce an hourly
forecast for all 24 hours of day D. Not a day in advance — exactly the
morning before. Everything after 09:00 is future information. Using it is
called leakage, and it makes your model look better than it will ever be
in production.

The consumer for this forecast is a balance-responsible party. If their
portfolio consumes more or less than they declared, they pay the balancing
market for the deviation. Smaller forecast error = smaller balancing cost.
PSE, the Polish transmission system operator, publishes their own forecast
for free. My job was to do better.

---

## The baseline ladder (the credibility section)

Before I trained anything, I needed to know what "good" looks like. I ran
every baseline I could think of in a proper walk-forward backtest:
17,450 test hours, 2024 to 2026, no look-ahead.

| Model | MAPE | vs naive |
|---|---|---|
| Seasonal naive (same hour last week) | 5.59% | baseline |
| Ridge (no external signal) | 4.05% | −1.54 pp |
| LightGBM | 3.16% | −2.43 pp |
| PSE (TSO) day-ahead forecast | 2.23% | −3.36 pp |
| LightGBM + TSO forecast | 2.12% | −3.47 pp |
| **Ridge + TSO forecast** | **2.08%** | **−3.51 pp** |

The table has a twist: the TSO forecast is public information at 09:00.
Using it as a model feature is legal and standard. The task shifts from
"beat the expert" to "correct the expert." Once that signal is inside the
model, everything changes.

---

## Why the winning model is ridge regression

I trained seven LSTM architectures, an LSTM with temporal attention, a
Temporal Fusion Transformer (TFT), and LightGBM. Every single one lost to
ridge regression with the TSO feature.

Here is why that happened.

**The signal is linear.** The TSO forecast explains 96% of the skill in our
sensitivity analysis. Once that feature is in the model, the remaining
residual is driven by weather, weekday pattern, and holidays — all of which
are near-linear at the level of precision that matters. Ridge handles this
efficiently. Deep nets try to learn the same relationship with 10–100x
more parameters, from the same data, and overfit.

**Cheap screening splits lie.** I caught this by accident. On a single 3-month
validation set, my best LSTM looked 0.7 percentage points better than ridge.
On the honest walk-forward over two years, ridge won. The gap is called
the "flattery trap": neural nets are better at memorising the validation
period than at generalising to the future. The fix is walk-forward
evaluation with monthly refits over as many test years as you can afford.
Most tutorials never mention this.

**This is the right result.** Knowing when NOT to use deep learning is a
signal of seniority. On a 2-year, hourly, tabular dataset with 25
features and a strong linear external signal, ridge beats transformers.
A real desk would draw the same conclusion and move on.

---

## The pipeline (what actually runs every morning)

At 05:30 UTC, a GitHub Actions job runs `daily_run.py` without any human
involvement:

1. Fetch latest actual load and weather. ENTSO-E for deep history, PSE
   API for fresh data, Open-Meteo for weather over 10 Polish cities
   (population-weighted average).
2. Score yesterday's forecast. My error vs naive vs PSE. If I was worse
   than PSE, it says so. Publicly. In the commit.
3. Forecast tomorrow (P10 / P50 / P90).
4. Write a 60-second manager report. Top 3 drivers in plain words.
5. Commit to git. The history is the track record.

The shadow deployment works like this: before a new model earns the
"official" label, it runs for 14 days in shadow mode — producing forecasts
that get scored but never published. Only if it beats the incumbent on
pre-agreed metrics does it get promoted. This is how real desks change
models. It is also how you avoid discovering in production that your
backtest was optimistic.

---

## What caught bugs before they reached production

- **DST leakage test.** On the 25-hour October day, computing "minus 24
  hours" reaches into the target day. Fixed and tested. One working test is
  worth ten careful comments.
- **Asinh target transformation.** Stabilises variance before fitting.
  Fine — until the inverse transform back to MW occasionally produced
  `nan`. Root cause: `mean/std` computed on 24-hour windows; the std
  was zero on flat holiday nights. Fixed with a floor.
- **Solar-growth extrapolation.** In the price model (Phase 2), the
  solar forecast column had grown 40% year-on-year in the training data.
  LEAR extrapolated linearly and produced 38,000 EUR/MWh predictions in
  2026. Fixed with a z-clip guard: if a feature value is more than 4
  standard deviations outside the training distribution, clip it.

None of these were hypothetical. All three were caught by tests or by
bad-number alerts during backtest runs. The repo has a "bugs found by our
own defences" section in the README.

---

## What's next

I am extending the same infrastructure to day-ahead price forecasting
(TGE SDAC, EUR/MWh). The standard baseline is LEAR — LASSO-Estimated
AutoRegression (Ziel & Weron 2018). On a 2-year walk-forward, my LGBM
model achieves rMAE 0.638 vs LEAR's 0.660. Both beat naive (1.0) by 36%.

I am also testing whether TFT's attention mechanism over 12 weeks of
price history closes the 30% gap with the tabular models. Early results:
long context is real but not sufficient. Results will be in the repo.

If your desk forecasts load or price in the European power market,
I would like to talk.

→ [GitHub repo](https://github.com/barteksarwa/pl-energy-forecast)

---

_All numbers from `reports/backtests/2026-07-16_2yr_summary.csv`
(load) and `reports/backtests/2026-07-17_price_res_out_fuel_summary.csv`
(price). Walk-forward, leakage-proof, out-of-sample._
