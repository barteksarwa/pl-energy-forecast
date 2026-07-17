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

Method: `git filter-branch -f --msg-filter 'sed "/^Co-Authored-By:/d"'`

### 3. PatchTST sweep v2 — running

PID 38070. Log: `/Users/bartlomiejsarwa/.claude/jobs/7035aac8/tmp/patchtst_sweep_v2.log`

Results so far (4 configs, TFT HPO best val=0.1157):

| patch_len | stride | ctx  | n_patches | val     |
|-----------|--------|------|-----------|---------|
| 12        | 6      | 672  | 111       | 0.1428  |
| 12        | 6      | 1344 | 223       | 0.1356  |
| 12        | 6      | 2016 | 335       | SKIP    |
| 12        | 12     | 672  | 56        | 0.1390  |
| 12        | 12     | 1344 | 112       | 0.1272  |

Key insight: longer context (1344h = 56 days) is clearly better.
Next: ctx=2016 (168 patches) and patch24/patch48 configs.

### 4. Documentation added

- `docs/notes/learning/20_patchtst_architecture.tex`: added zero-variance bug section
- `docs/notes/interview_prep.md`: added offshore wind bug story
- `docs/DECISIONS.md`: logged zero-variance guard decision

## What's pending

1. **Sweep completion**: 23 configs still to run (~2-3h). Log above.
2. **Walk-forward top-3**: After screening, launch with `--walkforward` flag.
   Best 3 configs by val pinball → monthly-refit walk-forward (2024-07-16 → present).
3. **Model selection note 11**: Fill in results table in `docs/notes/model_selection/11_patchtst_verdict.tex`.
4. **PatchTST model card**: If any config beats TFT walk-forward (MAE 19.71 / rMAE 0.706),
   create `docs/model_cards/patchtst_price.md` and open shadow gate.

## Launch commands when sweep finishes

```bash
# Walk-forward on top-3 (get top-3 from CSV first):
uv run python -m src.models.deep.run_patchtst_sweep --walkforward

# Or target specific configs manually:
uv run python -m src.models.deep.run_patchtst_sweep \
    --seed 42 --d_model 64 --walkforward
```

Walk-forward output: `reports/backtests/2026-07-18_patchtst_sweep_walkforward.csv`

## Key numbers to remember

| model         | val pinball | walk-forward MAE | walk-forward rMAE |
|---------------|-------------|------------------|-------------------|
| LEAR+conformal| —           | 18.23 EUR/MWh    | 0.653             |
| LGBM+conformal| —           | 17.8 EUR/MWh     | 0.640 (champion)  |
| TFT ens-3     | 0.1157      | 19.71 EUR/MWh    | 0.706             |
| PatchTST best | 0.1272+     | TBD              | TBD               |

PatchTST needs walk-forward MAE < 19.71 EUR/MWh to beat TFT (a low bar).
If it reaches ≤ 18.23 EUR/MWh, it beats LEAR (would open shadow gate).
