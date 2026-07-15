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

**Consecutive valid days: 0**

## What counts as "valid"

- Challenger forecast was produced (no exception).
- The forecast was stored in `data/forecasts/` and committed.
- Score (next day) was computed.

A day where challenger fails does NOT count toward the 14, but does NOT
reset a streak of valid days that came after. We count the streak from
the last failure.

## Next checkpoint

First valid day expected: 2026-07-17 (cron will run after the ffill fix is
merged and pushed to main).

Promotion decision: logged in DECISIONS.md, model card status updated to
"prod", config flag flipped.
