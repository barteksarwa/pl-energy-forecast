# Price backtest review pack — how to read it

Six figures. Each answers one question a desk's model review asks.
Numbers below come from `reports/backtests/2026-07-16_price_res_summary.csv`
and `2026-07-16_price_diagnostics.csv`. Test: 2024-07-16 → 2026-07-14,
17,480 hours, walk-forward with weekly refits.

Color code everywhere: **blue = LightGBM** (champion), **orange = LEAR**
(baseline), grays = naive benchmarks.

## 1. `price_bt_rolling_mae.png` — is the model drifting?

30-day rolling MAE. A desk checks this weekly.

**"Rolling" is not "recursive".** Every forecast in the backtest is an
independent day-ahead forecast fed only realized history — no model
output is ever fed back, so no error accumulates. The rolling window is
a plain moving AVERAGE over those independent daily errors. Raw daily
MAE is too noisy to read (market volatility dominates); the 30-day mean
exposes the only thing this chart is for: drift.

What to look for:
- Models hugging the bottom, gap to naive stable → skill holds across regimes.
- Model line creeping toward naive → drift; retrain or investigate.

Our read: both models keep a ~10 EUR/MWh gap under naive through the
whole 2 years. Winter volatility lifts everyone; the gap survives it.
**Verdict: no drift.**

## 2. `price_bt_cumulative_edge.png` — is the edge real or one lucky month?

Cumulative daily MAE saved vs naive-1d. Read it like a P&L curve:
- Steady climb → skill is earned every week.
- Staircase (one jump, then flat) → the "edge" was one event. Suspicious.

Our read: both lines climb near-linearly to ~7,000 EUR/MWh cumulative.
**Verdict: real, persistent edge.**

## 3. `price_bt_mae_by_hour.png` — where in the day does it fail?

MAE per delivery hour, local time. Prices have hour-specific physics:
night = flat and easy, morning ramp, solar dip, evening ramp = scarcity.

Our read: hour 19 (evening ramp) is the worst hour for every model
(~29 EUR/MWh vs ~9 at night). This is where spikes live and where a
trader's position hurts most. **Verdict: evening ramp is the open front.**

## 4. `price_bt_calibration.png` — can risk trust the quantiles?

The share of actuals falling below each predicted quantile. Honest band
= points on the diagonal. Risk management reads this chart FIRST: a P90
that gets pierced 26% of the time is not a P90.

Our read: LEAR's P10 is honest (11% vs 10 nominal); its P90 leaks (17%
above). LightGBM leaks both tails symmetrically (23% below P10, 26%
above P90 — nominal is 10/10). **Verdict (updated): raw bands failed here, so Phase 2.5 added rolling
conformal calibration — LGBM 51→79%, LEAR 72→79.5% coverage. The chart
shows the RAW bands; the calibrated table is
`reports/backtests/2026-07-16_price_conformal_summary.md`.**

## How example days are selected (worst / median / best)

Method, fixed and mechanical — no cherry-picking:

1. For every local calendar day in the test period, compute the
   CHAMPION model's (LightGBM) mean absolute P50 error over the day's
   24 hours.
2. Rank all ~730 test days by that number.
3. `worst_days` = top 4. `best_days` = bottom 4. `median_days` = the 4
   days straddling the 50th percentile — the "typical day" picture.

Selection is always by the champion's error, even though all models are
drawn in each panel: picking per-model days would make panels
incomparable, and picking by a non-champion would let the champion's
misses hide. Code: `_pick_days` in `src/viz/backtest_plots.py`.

Why all three matter: worst = risk (what a bad day costs), median =
the honest everyday expectation (what the MAE number actually looks
like), best = the ceiling (calm days where lags carry everything).

## 5. `price_bt_worst_days.png` — what did the miss look like?

Post-mortem panel, desk culture: every bad day gets looked at, not
averaged away. Champion's four worst days, all models overlaid.

Our read: 3 of 4 are winter evening spikes (Nov 2025 – Jan 2026) where
actuals blew 200+ EUR above every model's band — classic scarcity
pricing, no fundamentals in our feature set flag it (no outages, no
fuel prices). The 4th is a −400 EUR solar day where LEAR overshot
downward. **Verdict: misses are explainable and share one cause —
missing scarcity/fuel signals, not model brokenness.**

## 6. `price_bt_monthly_bias.png` — systematic over/under-forecasting?

Mean signed error per month. A model can have great MAE and still lean
one way — a desk cares because bias = systematically mispriced position.

Our read: both models slightly under-forecast (−2 to −4 EUR/MWh on
average), worst in winter 2024/25 (up to −15). Under-forecasting in
winter = the models don't fully believe scarcity premiums.
**Verdict: small persistent negative bias; candidate for a simple
bias-correction term; revisit after fuel/outage features.**

## The workflow this mimics

Real desks run this loop (see `docs/notes/learning/09_desk_model_review.tex`):

1. **Daily**: yesterday's error vs benchmark, automated (our daily cron).
2. **Weekly/monthly**: this pack — drift, calibration, post-mortems.
3. **Promotion gate**: challenger beats incumbent over a fixed shadow
   window on pre-agreed metrics (our shadow tally does this for load).
4. Model changes logged with evidence (our DECISIONS.md).

Regenerate: `uv run python -m src.viz.backtest_plots`
