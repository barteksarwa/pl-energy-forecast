# Shadow tally — all tracks

One file, three tracks. Promotion decisions are separate.
Valid day = forecast produced + committed + scored next morning.
A failed day does not count and does not reset the streak.
Promotion goes to DECISIONS.md and flips the relevant config/publisher.

## Resumed 2026-07-28

Backfilled target days 2026-07-22 → 07-27. Source: the bot-committed
daily reports on `origin/main`, files `reports/daily/2026-07-23.md`
through `2026-07-28.md`. Each report scores the previous day, so rows
are keyed by target day.

Source caveats:

- Most of those reports exist on `origin/main` only, not in the local
  worktree. Row notes say which.
- The local `reports/daily/2026-07-23.md` is a stale local run, not the
  bot version. Numbers here come from origin.
- The 2026-07-24 row uses the corrected local `2026-07-25.md`, not the
  bot version. See the correction note at the bottom of that file.

## Outage log

Cron DOWN 2026-07-18 → 2026-07-21. Local repo lost its git remote; CI
stopped. Days 07-19 → 21: no forecasts exist, permanently FAILED.
2026-07-18 forecasts exist (produced pre-outage); retroactive scoring
pending. Restart: PLAN.md M11.

## Track 1: load — ridge_tso challenger

Target: 14 consecutive valid shadow days. Promotion on operational
reliability; MAPE informs but does not gate.

| Date | Status | Challenger MAPE | Incumbent MAPE | TSO MAPE | Note |
|---|---|---|---|---|---|
| 2026-07-15 | FAILED | n/a | 3.98% | 2.12% | No weather forecast data (backfilled after) |
| 2026-07-16 | FAILED | n/a | 5.78% | 1.20% | TSO NaN — ffill fix deployed same day |
| 2026-07-17 | FAILED | n/a | 5.06% | 1.83% | CI runner had no data store — root cause of all failures; fixed same day (PR #3/#4). Also scored 07-16: challenger 1.72%, TSO 1.88% |
| 2026-07-18 | valid (retro) | 2.47% | 2.80% | 2.66% | Retro-scored 2026-07-21 (`score_stored_forecasts`). Challenger beat TSO and incumbent. Band cover 79.2% |
| 2026-07-19 → 21 | FAILED | n/a | n/a | n/a | Outage (see log above). Reports 07-21/07-22 do print incumbent MAPE for 07-20 (4.38%) and 07-21 (5.96%); challenger n/a both days. No committed day-ahead run during the outage, so the hole stands |
| 2026-07-22 | FAILED | n/a | 6.89% | 4.94% | Challenger down: Open-Meteo read timeout. Scored in report 2026-07-23 (origin; the local copy is a stale local run) |
| 2026-07-23 | FAILED | n/a | 5.00% | 4.19% | Same Open-Meteo timeout. Report 2026-07-24 is on origin/main only |
| 2026-07-24 | FAILED | 4.25% | 7.01% | 6.04% | The bot version had challenger n/a. The 4.25% comes from a manual re-run, not the cron; the official track counts cron runs. Correction note at the bottom of `reports/daily/2026-07-25.md` |
| 2026-07-25 | FAILED | n/a | 4.39% | 3.66% | Same Open-Meteo timeout. Report 2026-07-26 is on origin/main only |
| 2026-07-26 | FAILED | n/a | 4.01% | 5.01% | Same Open-Meteo timeout. Report 2026-07-27 is on origin/main only |
| 2026-07-27 | FAILED | n/a | 1.34% | 2.72% | Same Open-Meteo timeout. Report 2026-07-28 is on origin/main only |

**Consecutive valid days: 0.**

The challenger has not produced a cron forecast since 2026-07-18. Six
straight days lost to the same Open-Meteo read timeout. The load track
is blocked on that timeout, not on model quality.

## Track 2: price — LEAR (incumbent) reliability

Target: 14 consecutive valid days before the price forecast is "live"
in the README.

| Date (target day) | Status | LEAR MAE | naive-1d MAE | Note |
|---|---|---|---|---|
| 2026-07-17 | INVALID | – | – | Local run only; official track counts cron runs |
| 2026-07-18 | valid (retro) | 19.26 | 41.44 | Retro-scored 2026-07-21. Band cover 79.2% |
| 2026-07-19 → 21 | FAILED | – | – | Outage (see log above). Reports 07-21/07-22 print MAE as nan (no saved price forecast) |
| 2026-07-22 | valid | 16.18 | 12.88 | naive-1d beat LEAR. Report 2026-07-23 (origin; the local copy is a stale local run showing nan) |
| 2026-07-23 | valid | 18.53 | 22.63 | Report 2026-07-24, origin/main only |
| 2026-07-24 | valid | 22.75 | 25.75 | The bot version said 22.67; this row uses the corrected re-run. Correction note at the bottom of `reports/daily/2026-07-25.md` |
| 2026-07-25 | valid | 21.93 | 34.09 | Report 2026-07-26, origin/main only |
| 2026-07-26 | valid | 16.65 | 13.24 | naive-1d beat LEAR. Report 2026-07-27, origin/main only |
| 2026-07-27 | valid | 39.95 | 33.55 | naive-1d beat LEAR. Worst LEAR day so far. Report 2026-07-28, origin/main only |

**Consecutive valid days: 6** (2026-07-22 → 07-27). Target 14.

LEAR lost to naive-1d on 3 of those 6 days. Band coverage is not printed
in the daily reports, so it is not tracked here yet.

## Track 3: price — LGBM+conformal challenger vs LEAR

In shadow since 2026-07-17. Promotion criteria agreed IN ADVANCE (M9):
14+ valid days; challenger promotes if mean daily MAE beats LEAR AND
band coverage is not worse by more than 5 pp; ties → incumbent stays.

| Date (target day) | LGBM MAE | LEAR MAE | LGBM wins? |
|---|---|---|---|
| 2026-07-18 | 22.62 (cover 62.5%) | 19.26 (cover 79.2%) | NO — LEAR day |
| 2026-07-19 → 21 | – | – | FAILED — outage |
| 2026-07-22 | 19.28 | 16.18 | NO — LEAR day |
| 2026-07-23 | 20.48 | 18.53 | NO — LEAR day |
| 2026-07-24 | 25.16 | 22.75 | NO — LEAR day. Corrected re-run values; bot version was 24.99 / 22.67, same verdict |
| 2026-07-25 | 17.23 | 21.93 | YES |
| 2026-07-26 | 10.46 | 16.65 | YES |
| 2026-07-27 | 34.60 | 39.95 | YES |

**Valid shadow days: 6** (2026-07-22 → 07-27). Target 14.

Running mean over those 6 days: LGBM 21.20, LEAR 22.67. LGBM leads by
1.47 EUR/MWh. Not a decision. Coverage is not printed in the daily
reports, so the coverage half of the promotion rule is untested.
