# Shadow tally — price models

Two tracks. Promotion decisions are separate.

## Track 1: LEAR (incumbent) — operational reliability

LEAR publishes the daily price forecast (conformal band). Target:
14 consecutive valid days before the price forecast is called "live"
in the README.

Valid day = forecast produced + committed + scored the next morning.

| Date (target day) | Status | LEAR MAE | naive-1d MAE | Note |
|---|---|---|---|---|
| 2026-07-17 | INVALID | – | – | forecast produced locally, but the CRON price step failed (no data store on runner); official track counts cron runs only |
| 2026-07-18 | pending | – | – | cron fix merged 2026-07-17; first cron-produced price forecast expected 2026-07-18 |

**Consecutive valid days: 0 (scoring starts 2026-07-17 cron)**

## Track 2: LightGBM+conformal (challenger) — beats incumbent?

Runs in shadow since 2026-07-17 (target day 2026-07-18). Scored daily,
never published. Promotion criterion, agreed IN ADVANCE (M9 rule):

- window: 14 valid days minimum,
- challenger promotes if its mean daily MAE over the window is lower
  than LEAR's AND its band coverage is not worse by more than 5 pp,
- ties or mixed results → incumbent stays (change has a cost).

| Date (target day) | LGBM MAE | LEAR MAE | LGBM wins? |
|---|---|---|---|
| 2026-07-18 | – | – | pending |

**Valid shadow days: 0**

## Bookkeeping rules

Same as the load tally (`docs/shadow_tally.md`): a failed day does not
count and does not reset the streak; count from the last failure.
Update this file from the daily report scores. Promotion decision goes
to DECISIONS.md and flips the publisher in `src/pipeline/price_daily.py`.
