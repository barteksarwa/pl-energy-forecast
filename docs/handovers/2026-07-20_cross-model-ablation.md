# Handover — 2026-07-20 cross-model ablation night (overnight 2)

## What ran (all complete, no crashes)

1. **PatchTST training-window test** (3 seeds, test 2025-07-16→):
   365d MAE 21.50/cov 67.0% → 730d MAE 20.27/cov 74.9%. Root cause of
   the walk-forward loss confirmed.
2. **PatchTST full ablation at 730d** (8 groups × 3 seeds):
   **encoder REVERSED** −0.4 → +2.5 EUR/MWh. Redundancy was a window
   artifact. New ranking: RES +5.8 > wind +4.3 > solar +2.9 >
   encoder +2.5 > TSO +1.3 > calendar +0.4 > anchor +0.1.
3. **TFT ablation** (365d: 6 groups; 730d: full/encoder/res, 3 seeds):
   365d: full 20.65, encoder +1.2, RES +2.6.
   **730d: full 19.12 ± 0.46, coverage 79.6%** (at target without
   conformal), encoder +2.0, RES +3.2.
4. **LGBM 730-day ablation** (CPU, weekly refits, test 2024-07-16→):
   full 17.87 | price_lags +3.95 | RES +3.60 | TSO +0.54 |
   calendar +0.43 | load_lags −0.12 (dead weight, removal candidate).

## The story (see reports/sensitivity/tft/README.md for the table)

- LGBM extracts the MOST from price history and has the best MAE.
  Champion wins by using history harder, not by ignoring it.
- Weaker use of history → heavier lean on RES forecast
  (PatchTST +5.8 vs LGBM +3.6).
- ALL deep results this campaign used 365d windows — a handicap worth
  ~1.5-2 EUR/MWh + coverage. TFT-730 hits 19.12 @ 79.6% coverage.
- Interview line: ablation verdicts are conditional on training config;
  3 seeds flipped the encoder sign when the window doubled.

## Engineering

- `patchtst_feature_analysis.py`: `window` stage + `--train-days`
  (suffixed CSVs `_w730`); `walk_forward_ablate` generalized
  (net_factory/lr/batch/name) — one walk-forward for both architectures.
- `run_tft_ablation.py`: TFT ablation runner, `--train-days`, `--groups`.
- Worktree data/ is symlinked per-file to the main checkout
  (`data/` is partially git-tracked — don't symlink the whole dir).
- Known trap: `ps aux | grep` self-match killed a queue launcher whose
  payload contained the pattern; bracket the pattern *everywhere*.

## Addendum (same session, after night 2)

Same-window follow-up (test 2025-07-16 →, 8,760 h):

- LGBM re-ablated on the deep test window: full 17.66 | RES +4.12 |
  history +2.02 | TSO +0.58 | calendar +0.39 | load_lags −0.08.
  Ordering FLIPS vs the 2-yr window: RES > history on the calm year.
  LGBM history value (+2.02) ≈ TFT-730 encoder (+2.00).
- **TFT-730 ens-3 (median of 3 seeds): MAE 18.31, rMAE 0.668,
  coverage 82.8%.** Gap to champion now 0.65 EUR/MWh (3.7%), down
  from ~3. LGBM stays champion; verdicts window-qualified.
- Rerun determinism verified: 18.63/19.56/19.18 bit-exact.

## Possible next steps

1. **Deep re-benchmark at 730d windows** on the full 2-year test is NOT
   possible before 2027 (needs 730d history before 2024-07-16; data
   starts 2023-01). A 1-year-test comparison table exists instead.
2. Drop load_lags from the LGBM champion (−0.12) — config change + one
   backtest to confirm.
3. Shadow track record continues (`docs/shadow_tally.md`).
4. Portfolio README once track record established.

## Where everything is

- `reports/sensitivity/patchtst/` — README (full story), ablation CSVs
  (365 + w730), window_walkforward.csv, plots.
- `reports/sensitivity/tft/` — README (cross-model table), CSVs, plots.
- `reports/backtests/2026-07-19_price_group_ablation.(csv|md)` — LGBM.
- PR #10 (draft) has the whole campaign.
