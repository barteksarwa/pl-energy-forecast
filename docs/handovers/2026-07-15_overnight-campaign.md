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

## Morning checklist

1. Read the readout. 2. Approve/adjust Phase 2 kickstart. 3. If ENTSO-E
token arrived: `make backfill` extends history to 2023 + cross-check vs PSE.
