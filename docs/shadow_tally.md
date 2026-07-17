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
| 2026-07-18 | pending | – | – | – | Forecast produced by 2026-07-17 cron ✓ (data/forecasts/2026-07-18_challenger.csv committed). Scored 2026-07-19. 2026-07-17 cron also scored 2026-07-16: challenger 1.72%, TSO 1.88% |

**Consecutive valid days: 0 (pending scoring of 2026-07-18 on 2026-07-19).**

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
