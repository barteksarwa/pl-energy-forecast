# Shadow tally — ridge_tso challenger

Target: 14 consecutive valid shadow days.
Promotion criterion: each valid day = challenger produces a forecast (no
failure) and we have a score for yesterday. MAPE comparison informs but
does not gate: we promote on operational reliability, not single-day wins.

## Tally

| Date | Status | Challenger MAPE | Incumbent MAPE | TSO MAPE | Note |
|---|---|---|---|---|---|
| 2026-07-15 | FAILED | n/a | 3.98% | 2.12% | No weather forecast data (backfilled after) |
| 2026-07-16 | FAILED | n/a | 5.78% | 1.20% | TSO NaN — ffill fix deployed same day |
| 2026-07-17 | FAILED | n/a | 5.06% | 1.83% | CI runner had no data store (weather_forecast missing) — root cause of ALL failures so far; fixed same day (PR #3/#4: rolling cache + backfill step) |
| 2026-07-18 | unscored | – | – | – | Forecast produced by 2026-07-17 cron ✓ (data/forecasts/2026-07-18_challenger.csv committed). Never scored: cron died 2026-07-18. Retroactive scoring is legitimate (forecast predates actuals) — pending |
| 2026-07-19 | FAILED | n/a | n/a | n/a | No cron run. Local repo lost its git remote; CI stopped. No forecast exists for this day |
| 2026-07-20 | FAILED | n/a | n/a | n/a | Same outage. No forecast exists |
| 2026-07-21 | FAILED | n/a | n/a | n/a | Same outage. Outage found and diagnosed this day. Restart plan: PLAN.md Phase 4 |

**Consecutive valid days: 0. Cron outage 2026-07-18 → 2026-07-21 (remote lost, CI dead). Track record restarts after repo reconciliation.**

## What counts as "valid"

- Challenger forecast was produced (no exception).
- The forecast was stored in `data/forecasts/` and committed.
- Score (next day) was computed.

A day where challenger fails does NOT count toward the 14, but does NOT
reset a streak of valid days that came after. We count the streak from
the last failure.

## Next checkpoint

First valid day expected: 2026-07-18 — the 2026-07-17 fixes gave the CI
runner a persistent data store (the challenger could never train before).

Promotion decision: logged in DECISIONS.md, model card status updated to
"prod", config flag flipped.
