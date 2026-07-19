# Handover — 2026-07-19 PatchTST feature analysis (overnight)

## What happened this session

### 1. New harness: `src/models/deep/patchtst_feature_analysis.py`

Four stages on the best sweep config (patch24_s24_ctx1344, 197k params):

- `ablation`: zero one input group, retrain, full 2-year walk-forward. 3 seeds.
- `perm`: permutation importance, screening checkpoint, val 2026+, 10 shuffles.
- `pca`: raw 24h price patches + learned pooled representations.
- `attention`: mean attention maps. patch=stride=24h → one patch = one day.

Engineering notes:
- Walk-forward slices ONE prebuilt master sample set instead of rebuilding
  per refit. ~10x faster, bit-identical outputs (verified).
- Net init is now seeded, not only training. Unseeded init made the first
  run per process irreproducible. Bit-exact reruns verified.
- Ablation CSV is incremental and idempotent (finished (group, seed)
  pairs skip on restart).

Full run: 18 walk-forwards + cheap stages ≈ 8 h wall on MPS. No crashes.

### 2. Results — the negative result is now explained

All numbers: `reports/sensitivity/patchtst/` (README has the full story).

**Group ablation (walk-forward MAE, 3 seeds, EUR/MWh):**

| ablated | MAE | Δ vs full 23.61 |
|---------|-----|-----------------|
| encoder (56d price history) | 23.23 | **-0.38 (better without!)** |
| calendar | 23.68 | +0.08 |
| tso_load | 24.08 | +0.47 |
| anchor168 | 24.47 | +0.86 |
| solar only | 26.67 | +3.07 |
| wind_on only | 27.72 | +4.11 |
| res_fcst (all) | 29.84 | **+6.23** |

- The price-history encoder adds NOTHING. Zeroing it even improves
  coverage (68.5% → 73.2%). PatchTST's core premise is dead weight here.
- RES forecast carries the skill. Without it: worse than naive (rMAE 1.07).

**Permutation vs ablation:** permutation ranks the encoder FIRST
(Δpinball +0.18). Both correct: permutation = reliance of a fixed model,
ablation = unique information. Price history is fully redundant with
covariates. Good interview story.

**Attention:** top attended day ages 0, 7, 12, 1 (recency + weekly) but
the map is nearly flat (0.014-0.022 vs uniform 0.018).

**PCA:** patches low-rank (PC1-4 = 93%), learned reps ~2-dimensional
(PC1-2 = 97%).

Perm caveats (in README): hour_sin/cos identical across samples → shuffle
is a no-op; wind_off zeroed by the zero-variance guard.

### 3. Docs updated

- `docs/notes/model_selection/11_patchtst_verdict.tex`: new section
  "Feature analysis: why it lost" + interview line.
- `docs/DECISIONS.md`: entry 2026-07-19.
- Also committed 13 load-model sensitivity artifacts left uncommitted by
  the 2026-07-18 session (`reports/sensitivity/*`).

## What's pending

1. Shadow track record continues (`docs/shadow_tally.md`).
2. Portfolio README once track record established.
3. ~~Per-feature ablation inside res_fcst~~ DONE same session:
   wind_on +4.11 > solar +3.07; sum 7.2 > joint 6.2 (overlap).

## Key numbers to remember

Champion unchanged: LGBM+conformal 17.8 EUR/MWh, rMAE 0.640.
PatchTST archived WITH explanation: encoder redundant, RES forecast
is the signal, +6.2 EUR/MWh when removed.
