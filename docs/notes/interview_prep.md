# Interview prep — energy forecasting roles

Every answer is specific, backed by a number from this repo, and short.
"I know this because I measured it" beats "I know this from a course."

---

## Project overview questions

**"Walk me through your project."**

I built a day-ahead forecasting desk for the Polish power market. It
forecasts load and price, scores itself every morning, and commits the
report to git. The load model beats PSE's own forecast: 2.08% vs 2.23%
MAPE, 2-year walk-forward. The price model beats the industry standard
LEAR baseline: rMAE 0.638 vs 0.660. Everything runs unattended on GitHub
Actions.

**"Why load AND price?"**

Load forecasting is cleaner: public data, no negative values, strong
seasonal signal. It is the core skill. But the highest-paying roles are
on trading desks, and trading desks care about price. I added price on
the same infrastructure — same feature pipeline, same backtest engine,
same daily loop. That mirrors what a real desk does.

**"How long did it take?"**

Built over a few weeks in parallel with PhD work. The key constraint was
making it look like real work, not a tutorial — shadow deployment,
walk-forward backtests, model cards with honest verdicts.

---

## Technical questions — load

**"Why did ridge beat your deep learning models?"**

On this task, the TSO forecast explains 96% of skill (measured by group
ablation: removing it costs 1.97 pp MAPE). Once that linear external
signal is in the model, the residual is near-linear too. Ridge handles
it with 25 parameters. LSTM uses 100,000+ and overfits the validation
period. The screening split flatters nets by 0.6–0.9 pp vs honest
walk-forward — a trap I measured explicitly.

**"What features did you use?"**

25 features: load lags (24h, 48h, 168h, 7-day mean), calendar (hour,
day-of-week, month, PL holidays, bridge days), weather (temperature,
solar radiation, wind — 10 cities, population-weighted), and the PSE
TSO day-ahead forecast. Features live in `src/features/`, composable
from config.

**"How did you prevent data leakage?"**

I enforce a cutoff: the forecast for day D uses only data available at
09:00 on D−1. This is checked by a corruption test: I zero out all
"future" values and assert the feature matrix doesn't change. A DST
test also checks the 25-hour October day — "minus 24 hours" reaches
into the target day on the fall changeover. That bug was real and the
test caught it.

**"What is walk-forward cross-validation?"**

Instead of a random train/test split, I move a window forward in time.
I train on 365 days, test on the next 30, then shift everything forward.
Repeat until I have covered 2 years of test data (17,450 hours). This
respects time ordering, measures real deployment performance, and
prevents the validation period from "leaking" into training. A random
split on time-series data would let the model see future patterns;
walk-forward prevents that.

**"What metrics do you use and why?"**

MAPE (mean absolute percentage error) for load — easy to explain to
non-technical stakeholders and widely used by TSOs. Also MAE and skill
score vs naive baseline. For price I use rMAE (MAE relative to naive-1d)
because price can be zero or negative — MAPE becomes undefined. I also
report pinball loss for the quantile evaluation and conformal coverage
for the uncertainty band.

---

## Technical questions — price

**"What is LEAR?"**

LASSO-Estimated AutoRegression (Ziel & Weron 2018). Standard price
baseline: 24 separate LASSO regressions, one per delivery hour. Inputs:
lagged prices at 24h, 48h, 168h, load forecast, calendar. The LASSO
selects which lags matter for each hour. It is the benchmark every price
model must beat. Ours achieves rMAE 0.660; our LightGBM challenger is
at 0.638.

**"What drives day-ahead electricity prices in Poland?"**

Three main drivers. First, the merit order: plants dispatch in cost
order; the marginal plant sets the price. More wind/solar pushes cheap
units to the margin → price falls (we measured: wind >4 GW cuts price
by 21%). Second, demand: high load + low renewables → gas marginal →
high price. Third, fundamentals: TTF gas price and EUA CO2 allowances
set the variable cost of the marginal unit. We added fuel proxies to
LEAR and cut winter bias from −15.9 to −4.6 EUR/MWh.

**"What is conformal calibration and why do you need it?"**

A quantile model outputs P10/P90, but raw quantile regression achieved
only 51% empirical coverage on a nominal 80% band. Conformal prediction
fixes this: compute nonconformity scores on a calibration set, find the
(1−α) quantile, add that offset to future predictions. I use a rolling
90-day calibration window, updated daily, so the correction adapts to
regime shifts. After calibration: LGBM 78.7%, LEAR 79.5%. The fix is
in `config/price_conformal.json`, recomputed every morning.

**"How do you handle price spikes?"**

Imperfectly, and I say so. Spike MAE (top 5% hours) is 60 EUR/MWh for
LGBM vs 77 EUR/MWh for naive — a 22% improvement, but 3× the pooled
MAE. We evaluated the outage feature (ENTSO-E UMM) and it was FLAT:
the aggregate capacity unavailability signal is too coarse. Real spike
forecasting needs unit-level outage data and real-time gas prices.
Both are known gaps in the model card.

**"Why do you use rMAE for price instead of MAPE?"**

798 hours in our 3-year sample had negative prices (solar oversupply).
MAPE divides by price; negative or near-zero prices make it undefined
or misleading. rMAE (MAE / MAE of naive) is sign-agnostic and gives
the same "how much better than guessing?" interpretation.

---

## Market structure questions

**"How does the day-ahead market work in Poland?"**

Poland uses the Single Day-Ahead Coupling (SDAC) model via TGE (Towarowa
Giełda Energii). Sellers and buyers submit bids and offers for each of
the 24 hours of tomorrow. Gate closure is 12:00 CET. Results publish at
12:55. Our forecast is for 09:00 D−1 — before gate closure, when a
trader must decide positions.

**"What is the balancing market?"**

After gate closure, actual consumption or generation deviates from the
day-ahead schedule. PSE runs the balancing mechanism (rynek bilansujący)
in real time. Since June 2024, Poland reformed to 15-minute settlement
with scarcity pricing: when the system is short, balancing prices spike
to a defined cap. A good load forecast reduces imbalances and the fees
they trigger.

**"What is rynek mocy?"**

Poland's capacity market. Generators and large consumers sign contracts
to guarantee availability N years ahead (currently 5). The market runs
annual auctions. Revenue from capacity contracts is separate from the
energy price. A desk forecasting revenue for a generation portfolio
needs to include capacity payments; they can be 20–30% of total revenue.

**"What is PSE, TGE, and URE?"**

PSE (Polskie Sieci Elektroenergetyczne) — the transmission system
operator and balancing authority. TGE (Towarowa Giełda Energii) — the
power exchange running spot and forward markets. URE (Urząd Regulacji
Energetyki) — the energy regulator approving tariffs and market rules.

---

## Behavioural / process questions

**"How do you know the model is ready for production?"**

Fourteen shadow days. The challenger runs every morning, produces
forecasts that get scored but never published. After 14 consecutive
valid days, I compare its mean daily metric against the incumbent on
the pre-agreed criterion. If it wins, I promote it and log the decision
in DECISIONS.md. If it loses or ties, the incumbent stays. This mirrors
how real desks change models: the burden of proof is on the challenger.

**"What was the hardest bug you found?"**

The solar-growth extrapolation bug. The LEAR price model extrapolated
linearly from training data where solar installed capacity was 40% lower.
In 2026, the solar forecast values were far outside the training range.
LEAR produced 38,000 EUR/MWh predictions. Fix: z-clip input features
at ±4 standard deviations of the training distribution. One line of code;
three days of debugging to find it.

**"You're a PhD student in a different field. Why energy?"**

The skill set transfers directly: time series, gradient boosting, deep
learning, evaluation discipline. Energy markets are interesting because
the physics (generation, transmission, storage) constrains the economics
in hard ways — it is not pure statistical arbitrage. And the societal
impact is clear: better forecasts mean lower balancing costs, which flows
to consumers. I built this project to prove the transfer, not just claim it.

---

## Questions to ask the interviewer

1. "What is your current MAPE on the day-ahead load / price task?"
2. "What features does your production model use that you wish you had
   more of?" (Answer will tell you what they want to improve.)
3. "How long does a model change typically take from backtest to
   production? What is your shadow/UAT process?"
4. "What is your biggest source of forecast error — weather uncertainty,
   market events, outages?"
5. "Do you forecast P10/P90 or just P50? How do you calibrate intervals?"
