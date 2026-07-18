# Handover — 2026-07-18 PatchTST feature analysis

## What happened this session

Continued from `2026-07-18_patchtst-fix-sweep.md`.

Two bugs fixed in `patchtst_feature_analysis.py`, overnight run launched.

### Bugs fixed

**1. Permutation used numpy `.copy()` on torch tensor.**
`te.fut.copy()` fails silently — tensor has no `.copy()`. Fixed with `te.fut.clone()` and a `permute_cols()` helper:
```python
def permute_cols(te_samples, cols):
    te_perm = copy.copy(te_samples)
    fut_clone = te_samples.fut.clone()
    perm_idx = torch.randperm(fut_clone.shape[0])
    fut_clone[:, :, cols] = fut_clone[perm_idx][:, :, cols]
    te_perm.fut = fut_clone
    return te_perm
```

**2. Attention section called `net.patch_embed(enc)` with wrong shape.**
`enc` is `(1, 1344, 1)`. `patch_embed` expects `(B*C, n_patches, patch_len)`.
Fixed by extracting patches with `net._patch()` first:
```python
B, T, C = enc_i.shape
chans = enc_i.permute(0, 2, 1).reshape(B * C, T)
patches = net._patch(chans)
tokens = net.patch_embed(patches)
```

### Tests before launch

- Section `--section permutation`: PASS. Baseline MAE 19.627, all groups ran.
- Section `--section attention`: PASS. 60 days collected, saved `attention_analysis.png`.
- Section `--section pca`: PASS (already worked from prior session).
- Section `--section ablation`: NOT tested separately (6h+ run, launched as part of full run).

### Permutation importance results (preview — full checkpoint, 291 test days)

| feature group | ΔMAE (EUR/MWh) |
|---------------|---------------|
| res_forecast  | +16.4 ← dominant |
| tso_forecast  | +8.5 |
| calendar      | +1.2 |
| price_lag168  | +0.4 |

Calendar breakdown: `is_weekend` (+0.56), `is_holiday` (+0.48) most important.
`hour_sin`/`hour_cos`: +0.000 — PatchTST does NOT use hour-of-day signal from calendar in fut.
(Makes sense: 24h patches already embed intraday position implicitly.)

### Overnight run (PID 56005)

Launched:
```
nohup uv run python -m src.models.deep.patchtst_feature_analysis \
  > /Users/bartlomiejsarwa/.claude/jobs/7035aac8/tmp/patchtst_feature_analysis.log 2>&1 &
```

Sections in order (total ~12h):
1. permutation — ~30 min (already completed in-process before launch, will rerun)
2. attention — ~20 min
3. pca — ~10 min
4. ablation — ~7.5 h (5 feature-group walk-forwards × ~90 min each)

Outputs will appear in:
- `reports/figures/patchtst_features/` (PNG plots)
- `reports/backtests/patchtst_permutation_importance.csv`
- `reports/backtests/patchtst_ablation_results.csv`
- `reports/backtests/patchtst_pca_variance.csv`

Monitor: `tail -f /Users/bartlomiejsarwa/.claude/jobs/7035aac8/tmp/patchtst_feature_analysis.log`

## What's pending (next session)

1. **Read ablation results** — which feature group matters most for walk-forward MAE?
   Hypothesis from permutation: `res_forecast` >> `tso_forecast` > `calendar` >> `lag168`.
2. **Write LaTeX note** `docs/notes/model_selection/12_patchtst_features.tex`.
   Document permutation importance, ablation table, attention pattern interpretation.
3. **Portfolio README** — once feature analysis is done, attention campaign is fully documented.
4. **14-day shadow track record** — continues automatically via cron.

## Files changed

- `src/models/deep/patchtst_feature_analysis.py` — two bug fixes (permutation + attention)
- `docs/handovers/2026-07-18_patchtst-feature-analysis.md` — this file
- `reports/figures/patchtst_features/` — perm_importance.png, attention_analysis.png,
  pca_features.png (created during testing)
- `reports/backtests/patchtst_permutation_importance.csv` — preview results
- `reports/backtests/patchtst_pca_variance.csv` / `patchtst_pca_loadings.csv`
