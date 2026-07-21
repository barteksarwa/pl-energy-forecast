# Shadow tally — all tracks

One file, three tracks. Promotion decisions are separate.
Valid day = forecast produced + committed + scored next morning.
A failed day does not count and does not reset the streak.
Promotion goes to DECISIONS.md and flips the relevant config/publisher.

## Outage log

Cron DOWN 2026-07-18 → 2026-07-21 (CI outage).
CI outage. Days 07-19 → 21: no forecasts exist, permanently FAILED.
2026-07-18 forecasts exist (produced pre-outage); retroactive scoring
pending.

## Track 1: load — ridge_tso challenger

Target: 14 consecutive valid shadow days. Promotion on operational
reliability; MAPE informs but does not gate.

| Date | Status | Challenger MAPE | Incumbent MAPE | TSO MAPE | Note |
|---|---|---|---|---|---|
| 2026-07-15 | FAILED | n/a | 3.98% | 2.12% | No weather forecast data (backfilled after) |
| 2026-07-16 | FAILED | n/a | 5.78% | 1.20% | TSO NaN — ffill fix deployed same day |
| 2026-07-17 | FAILED | n/a | 5.06% | 1.83% | CI runner had no data store — root cause of all failures; fixed same day (PR #3/#4). Also scored 07-16: challenger 1.72%, TSO 1.88% |
| 2026-07-18 | valid (retro) | 2.47% | 2.80% | 2.66% | Retro-scored 2026-07-21 (`score_stored_forecasts`). Challenger beat TSO and incumbent. Band cover 79.2% |
| 2026-07-19 → 21 | FAILED | n/a | n/a | n/a | Outage (see log above) |

**Consecutive valid days: 0.**

## Track 2: price — LEAR (incumbent) reliability

Target: 14 consecutive valid days before the price forecast is "live"
in the README.

| Date (target day) | Status | LEAR MAE | naive-1d MAE | Note |
|---|---|---|---|---|
| 2026-07-17 | INVALID | – | – | Local run only; official track counts cron runs |
| 2026-07-18 | valid (retro) | 19.26 | 41.44 | Retro-scored 2026-07-21. Band cover 79.2% |
| 2026-07-19 → 21 | FAILED | – | – | Outage (see log above) |

**Consecutive valid days: 0.**

## Track 3: price — LGBM+conformal challenger vs LEAR

In shadow since 2026-07-17. Promotion criteria agreed IN ADVANCE (M9):
14+ valid days; challenger promotes if mean daily MAE beats LEAR AND
band coverage is not worse by more than 5 pp; ties → incumbent stays.

| Date (target day) | LGBM MAE | LEAR MAE | LGBM wins? |
|---|---|---|---|
| 2026-07-18 | 22.62 (cover 62.5%) | 19.26 (cover 79.2%) | NO — LEAR day |

**Valid shadow days: 0.**
