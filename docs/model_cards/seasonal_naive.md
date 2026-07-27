# Model card — seasonal_naive (load)

**Production status: PUBLISHED daily** — the load incumbent. Every
model card should say what ships; this one ships.

## What it is

P50 copies the load of the same hour 7 days ago. The band is the spread
of the last 4 same-weekday observations. No training, no parameters.
File: `src/models/naive.py`.

## Why a naive is in production

Discipline, not accident. The daily loop started with the strongest
zero-cost baseline; trained challengers must beat it LIVE for 14
consecutive valid days before promotion (PLAN M9, DECISIONS
2026-07-14). The ridge+TSO challenger wins in backtest (2.08% vs naive
5.59% MAPE) but its shadow run stalled on a fetch outage 07-17 → 07-27
(fixed with retries) — so the gate has not closed and the naive still
ships. That is the promotion system working as designed.

## Performance

2-yr walk-forward: MAPE 5.59%, the denominator of every load skill
score. Loses to the TSO forecast (2.23%) daily — the report says so
in its own table every morning.

## Honest limitations

- Blind to weather, holidays, and trend; holiday weeks are its worst.
- Band is a crude spread, not calibrated.
- It exists to be beaten. The interesting number is how long a
  challenger takes to EARN beating it live.
