# Handover — 2026-07-15 — overnight campaign (read this first in the morning)

## Where the answers are

- **`reports/backtests/2026-07-15_overnight_readout.md`** — everything collated.
- Raw: `outputs/deep_campaign_v2/v3/v4_*.csv`, `outputs/logs/overnight.log`.
- If a stage says FAIL in overnight.log, its block is missing from the readout;
  the rest of the night continued.

## What ran tonight (in order)

1. v3 LSTM ladder: vanilla, BiLSTM, Luong attention (screening).
2. Walk-forward with **TSO forecast as feature**: lgbm + ridge. The big bet.
3. Nets + TSO covariate (screening): enc_dec h64, futmlp h256.
4. Origin augmentation 4x (screening): enc_dec h64/h128.
5. Capacity close-out: enc_dec h512.
6. **Deep walk-forward** (README-grade): enc_dec h64, with and without TSO.
7. LightGBM mini-tune (2 configs, both with TSO feature).
8. Readout generated, committed, pushed automatically.

## Fixed reference numbers (walk-forward, honest weather, as of last night)

TSO 2.31% | lgbm 3.16% | ridge 4.03% | naive 5.60%.
Anything tonight must beat these ON THE SAME EXAM to matter.
Screening numbers (v2/v3/v4 CSVs) rank ideas only.

## Decisions taken while you slept

- TSO forecast admitted as feature (published 09:00 D-1 = our cutoff).
  DECISIONS.md entry; models are now forecast combiners.
- Bigger models: evidence against (enc_dec peaks at 106k params); h512 run
  closes it.

## Phase 2 prepared

- PSE serves day-ahead price (`csdac-pln`), balancing prices (`crb-rozl`),
  RCE, generation mix — keyless, verified by live calls.
- `docs/PHASE2_KICKSTART.md`: data table, the portfolio POC design
  (synthetic retailer: load + PV + wind from our weather, valued at real
  prices), proposed milestone order, 2 questions for you.

## Morning session (done before owner returned)

- Attention walk-forward closed the ladder: 3.74% plain / 2.43% +TSO.
  Nets confirmed behind the linear combiner on the real exam.
- Phase 2 data backfilled keyless: day-ahead price, balancing CEN,
  generation mix — 18k hours each, 0 gaps. Price history figure in viz.
- README: full honest results table published.
- **Shadow challenger live**: ridge+TSO trains each morning, forecasts in
  shadow, scored daily in the report. First score lands tomorrow.
- **GitHub Actions cron live** (05:30 UTC, keyless): the daily loop now
  runs unattended and commits its own reports. M9 POC done early.
- Forecast CSVs now committed (audit trail; DECISIONS entry).

## Morning checklist for the owner

1. Read `reports/backtests/2026-07-15_overnight_readout.md` + note 03.
2. Approve/adjust `docs/PHASE2_KICKSTART.md` (2 open questions in it).
3. ENTSO-E token when it arrives: `make backfill` → 2023+ history
   + PSE cross-check.
4. Decide: how many shadow days before promoting the challenger? (Suggest 14.)
