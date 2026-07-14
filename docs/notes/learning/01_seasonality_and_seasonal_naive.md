# Seasonality and the seasonal naive model

## What seasonality is

A repeating pattern with a fixed period. Electricity load has three at once:

- **Daily.** Night valley, morning ramp, evening peak. Period: 24 hours.
- **Weekly.** Weekdays high, weekends low. Monday morning differs from Saturday morning. Period: 7 days.
- **Yearly.** Winter heating and dark hours push load up. Summer cooling adds a smaller bump. Period: 365 days.

Key insight: the strongest predictor of load at hour H is load at hour H in a past period.

## The seasonal naive model

Forecast = the value one season ago. Nothing else.

Our version: tomorrow at 14:00 = last week's same weekday at 14:00.
Why 7 days and not 1? A 1-day lag would predict Monday from Sunday. Wrong regime.
The 7-day lag keeps the weekday pattern. It handles daily + weekly seasonality for free.

## Worked example

Last Tuesday 18:00 the load was 21,400 MW. Tomorrow is Tuesday.
Seasonal naive forecast for 18:00: 21,400 MW. Done.

For uncertainty we look at the last 4 same-weekday values at 18:00:
21,400 / 20,900 / 22,100 / 21,000 MW. P10 ≈ 20,930. P90 ≈ 21,890.
The spread of recent history becomes the uncertainty band.

## When it fails

- Holidays. Next Tuesday is a holiday, last Tuesday was not.
- Weather swings. A cold snap raises load; last week was mild.
- Trends. Slow economic or efficiency changes leak in over months.

Every fancy model must beat this first. In forecasting, many do not.
That is why it is the mandatory baseline (see `docs/PLAN.md`, M3).

## Interview line

"My headline metric is skill over seasonal naive — how much error the model
removes relative to just copying last week. If a deep net has zero skill,
it is not learning anything the calendar did not already know."
