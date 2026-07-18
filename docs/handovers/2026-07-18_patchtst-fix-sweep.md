# Handover — 2026-07-18 PatchTST fix + sweep

## What happened this session

### 1. PatchTST val explosion — root cause found and fixed

**Bug:** val pinball 879 (expected ~0.14). ALL 27 configs showed this.

**Root cause:** `wind_off_fcst_mw` (Baltic offshore wind) column.
- First non-zero: 2026-07-01 (Baltic Phase 1 commissioning).
- Training period (2023-2025): all zeros. Training std = 0.
- Z-score step: std clamped to 1e-6. Val value 19 MW → z-score 19,000,000.
- PatchTST (linear+GELU, no saturation) amplified this to val pinball 879.
- TFT survived: LSTM + GRN sigmoid gates saturate at extreme inputs.

**Fix:** `src/models/deep/data.py` — `standardize_covariates`:
- Compute raw std before clamp.
- If `f_sd_raw < 1e-4` (constant in training): zero that column in all sets.
- 4 tests added: `tests/test_deep_data.py`.

**Verification:** After fix, patch12_s6_ctx672 shows val=0.1428 at epoch 0 (was 879).

### 2. Git history cleanup

Per user request: stripped all `Co-Authored-By: Claude` lines from:
- `worktree-phase2-price-lear` (96 commits) — force-pushed
- `worktree-precious-soaring-crescent` (42 commits) — force-pushed
- `main` was already clean (0 Claude commits)

**Side effect:** filter-branch orphaned `worktree-phase2-price-lear` from `main` (no common ancestor).
**Fix:** created `phase2-price-lear-v2` branch from `origin/main`, cherry-picked 26 unique commits.
**PR #8 DRAFT:** https://github.com/barteksarwa/pl-energy-forecast/pull/8 (from `phase2-price-lear-v2`)

### 3. PatchTST sweep v2 — COMPLETE (0.3h)

24 configs ran (3 skipped: n_patches > 256). Full results: `reports/backtests/2026-07-17_patchtst_sweep.csv`

Top-10:

| rank | patch_len | stride | ctx     | val_pinball |
|------|-----------|--------|---------|-------------|
| 1    | 24        | 24     | 1344 h  | 0.1236      |
| 2    | 12        | 12     | 1344 h  | 0.1253      |
| 3    | 24        | 12     | 1344 h  | 0.1255      |
| 4    | 48        | 24     | 1344 h  | 0.1261      |
| 5    | 24        | 6      | 1344 h  | 0.1262      |
| 6    | 48        | 12     | 1344 h  | 0.1291      |
| 7    | 12        | 6      | 1344 h  | 0.1295      |
| 8    | 12        | 24     | 1344 h  | 0.1308      |
| 9    | 48        | 24     | 2016 h  | 0.1316      |
| 10   | 24        | 12     | 2016 h  | 0.1317      |

Key findings:
- **ctx=1344h (56 days) dominates.** Top-8 all use ctx=1344. Ctx=672 consistently in bottom half.
- **24h patches win** for top rank. One day = one token: calendar rhythm is the natural signal level.
- **PatchTST best val (0.1236) trails TFT HPO val (0.1157)** on this split.

### 4. PatchTST walk-forward — COMPLETE (negative result)

Walk-forward ran on top-3: patch24_s24_ctx1344, patch12_s12_ctx1344, patch24_s12_ctx1344.
Results: `reports/backtests/2026-07-17_patchtst_sweep_walkforward.csv`

| config | MAE (EUR/MWh) | rMAE | coverage 80% | spike MAE |
|--------|---------------|------|--------------|-----------|
| patch24_s24_ctx1344 | **22.98** | **0.823** | 69.5% | 79.3 |
| patch12_s12_ctx1344 | 23.62 | 0.846 | 68.3% | 83.0 |
| patch24_s12_ctx1344 | 24.49 | 0.877 | 70.0% | 80.2 |

**Verdict: TFT gate NOT cleared.** Best MAE 22.98 > TFT 19.71 EUR/MWh. Coverage collapsed to 69.5% vs 80% target.
Root cause: short training windows (365 days, 25 refits) + small capacity (197k params) → faster overfitting than TFT.

### 5. Documentation

- `docs/notes/learning/20_patchtst_architecture.tex`: added zero-variance bug section
- `docs/notes/model_selection/11_patchtst_verdict.tex`: full verdict table filled in
- `docs/notes/interview_prep.md`: added offshore wind bug story
- `docs/DECISIONS.md`: zero-variance guard + PatchTST verdict logged

### 6. Backtesting plots — COMPLETE

15 plots generated: `reports/figures/backtest_price/`
- Metrics comparison (MAE, RMSE, rMAE bar charts with PatchTST included)
- 2-year overview and 6 seasonal zoom windows
- Error distribution, coverage, MAPE, rolling MAE, scatter, spike MAE

Script: `src/viz/backtest_price_plots.py`

## What's pending

1. **Merge PR #8** and push all branches to origin/main. (done at end of this session)
2. **Overnight PatchTST feature analysis** (12h run): ablation, PCA, permutation importance.
3. **Shadow track record**: 14-day shadow window for load and price models continues.
   See `docs/shadow_tally.md` for current day count.

## Key numbers to remember

| model          | val pinball | walk-forward MAE | walk-forward rMAE |
|----------------|-------------|------------------|-------------------|
| LGBM+conformal | —           | 17.8 EUR/MWh     | 0.640 (champion)  |
| LEAR+conformal | —           | 18.23 EUR/MWh    | 0.653             |
| TFT ens-3      | 0.1157      | 19.71 EUR/MWh    | 0.706             |
| PatchTST best  | 0.1236      | 22.98 EUR/MWh    | 0.823 (❌ TFT gate not cleared) |
| naive 1-day    | —           | 28.0 EUR/MWh     | 1.000             |

PatchTST attention campaign: **negative result**. LGBM+conformal stays champion.

## Branch status

- `phase2-price-lear-v2`: PR #8 branch. Merged to main at end of session.
- `worktree-phase2-price-lear`: same content, local only.
