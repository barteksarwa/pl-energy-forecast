# Handover — 2026-07-17 — TFT verdict + PatchTST sweep

## What happened this session

**TFT HPO** (60 trials) completed. Final best: trial 56, val=0.1157
(ctx=1344h, d128, h8, l2, dropout=0.183, lr=0.00174, batch=32).

**TFT walk-forward** (3 seeds, 17,472 test hours) completed in 3.8h.
**Verdict: TFT trails LEAR** (MAE 19.71 vs 18.23, rMAE 0.706 vs 0.653).
Root causes: data ceiling, signal sparsity, quantile training cost.
Shadow gate NOT opened. See `docs/model_cards/tft_price.md`.

**PatchTST sweep** launched at ~17:45 UTC. 27 configs (patch×stride×ctx).
Log: `/Users/bartlomiejsarwa/.claude/jobs/7035aac8/tmp/patchtst_sweep.log`
PID: 31676. Estimated 2-4h on MPS.

## New files committed (branch: worktree-phase2-price-lear)

### Code
- `src/evaluation/conformal.py`: asymmetric CQR (rolling_conformal_asymmetric,
  latest_offset_asymmetric) — separate per-tail corrections for P10/P90
- `src/evaluation/run_price_calibration.py`: --compare-asymmetric flag
- `src/evaluation/metrics.py`: winkler_score() — sharpness + coverage in one number
- `src/evaluation/run_price_backtest.py`: winkler column in summary table

### Tests
- `tests/test_conformal.py`: 4 new asymmetric CQR tests (74 total passing)
- `tests/test_metrics.py`: 3 Winkler score tests (hand-verified)

### Reports
- `reports/backtests/2026-07-17_tft_hpo.md` + `_trials.csv`: 60-trial HPO study
- `reports/backtests/2026-07-17_tft_hpo_walkforward.(csv|md)`: walk-forward verdict
- `reports/backtests/2026-07-17_asym_cqr_comparison.csv`: sym vs asym coverage table
- `reports/sensitivity/tft_vsn_weights.csv`: final VSN weights

### Docs
- `docs/model_cards/tft_price.md`: full verdict (was pending) + why-TFT-lost
- `docs/model_cards/tft_price.md`: HPO best updated (trial 56 → val 0.1157)
- `docs/notes/model_selection/08_price_attention_campaign.tex`: filled result table
- `docs/notes/model_selection/09_price_lear_vs_lgbm.tex`: LEAR vs LGBM verdict
- `docs/notes/model_selection/10_cqr_vs_asymmetric.tex`: sym vs asym CQR finding
- `docs/notes/learning/17_probabilistic_evaluation.tex`: pinball, CQR, Winkler
- `docs/notes/learning/18_forecast_to_trading.tex`: forecast → trading decision
- `docs/notes/learning/19_cross_border_and_coupling.tex`: SDAC, FBC, PL-DE spread
- `docs/notes/interview_prep.md`: 4 new Q&A + TFT loss story, CQR measurement

## Key findings from this session

1. **TFT loses to LEAR by 8.1% MAE**. Not a rounding error. Data ceiling is
   the primary explanation (300-400 training samples vs 1.27M params).

2. **Asymmetric CQR measured**: symmetric 79.6% coverage, asym 79.1%. Upper
   tail (price spikes) is the bigger calibration gap, not negative prices.
   Symmetric stays in production.

3. **Winkler score**: LEAR+conformal = 87.964 (best). Better than
   LGBM+conformal (89.576). LEAR wins on the combined sharpness+coverage metric.

4. **VSN weights flipped**: in early screening solar was #1 (0.217); in the
   final 60-trial model TSO is #1 (0.235). With 56-day context, the encoder
   absorbs solar lags; VSN then emphasises demand (TSO) as the complementary signal.

## PatchTST sweep — what to do when it finishes

Log file: `/Users/bartlomiejsarwa/.claude/jobs/7035aac8/tmp/patchtst_sweep.log`

Expected output:
- `reports/backtests/2026-07-17_patchtst_screening.md` with ranked table
- Top-3 configs filtered: patch_len, stride, ctx

Next steps after sweep:
1. Read the screening report. Identify top-3 configs by val pinball.
2. If any config has val < 0.120 (TFT's screening val): consider walk-forward.
   Use `--walkforward` flag: `uv run python -m src.models.deep.run_patchtst_sweep --walkforward`
3. If all configs trail TFT: document that patching does not help on this data.
4. Write model_selection/11_patchtst_verdict.tex with findings.

## Shadow tally status

- Load shadow: 2026-07-18 pending (scored 2026-07-19 by cron)
- Price shadow Track 1 (LEAR): 2026-07-18 pending
- Price shadow Track 2 (LGBM): 2026-07-18 pending

Update tallies once 2026-07-19 cron runs.

## Do-not-touch while PatchTST runs

- `data/processed/` — PatchTST writes checkpoints here
- `reports/backtests/2026-07-17_patchtst_*` — will be written when sweep ends
- MPS GPU — occupied by PatchTST sweep (PID 31676)

## Next session entry point

1. Check PatchTST results: `cat /Users/bartlomiejsarwa/.claude/jobs/7035aac8/tmp/patchtst_sweep.log | tail -50`
2. Update shadow tallies for 2026-07-18
3. Open PR for branch `worktree-phase2-price-lear` (all work committed and pushed)
