# Model card — PatchTST — price task

## What it is

- PatchTST (Nie et al. 2023): a transformer over "patches".
- A patch = a fixed slice of the series turned into one token.
  Our best config uses 24h patches: one token = one day.
- Channel-independent: each input variable runs through its own copy
  of shared weights. No cross-channel attention.
- Quantile heads p10/p50/p90, pinball loss. 197k params (d_model=64).
- File: `src/models/deep/patchtst.py`.
- Target: PL day-ahead price, EUR/MWh (SDAC auction, clears ~12:00
  CET on D-1; tomorrow's price is the target, never an input).

**Verdict: archived. Loses to both TFT and the LGBM champion.**
The loss is fully explained below — that explanation is the value.

## Inputs

- Same feature groups as TFT, from config: 56-day (1344h) price
  history encoder, RES forecast (solar + onshore + offshore wind),
  TSO day-ahead load forecast, calendar, anchor price lag-168h.
- These known-future covariates help ALL models, not one architecture
  (cross-model table in `reports/sensitivity/tft/README.md`).
- Channel independence was a defensible bet before training: only
  1 of 21 channel pairs has |corr| > 0.5 at hourly resolution
  (`reports/sensitivity/channels_verdict.txt`).
- Does NOT see: fuel/CO2 prices, cross-border flows, outages.

## Training

- Sweep: patch_len {12,24,48} × stride {6,12,24} × ctx {672,1344,2016}.
  24 configs ran. Winner: patch24_s24_ctx1344 (val pinball 0.1236).
- ctx=1344h (56 days) dominated: top-8 all use it.
- Original walk-forward: 365-day rolling windows, monthly refits.
- Follow-ups: 730-day windows, 3-seed ensembles, capacity sweep
  (d_model 64/96/128/192).

## Performance

**Original 2-year walk-forward** (365d windows, 2024-07-16 →):

| model | MAE (EUR/MWh) | rMAE | coverage 80% |
|---|---|---|---|
| LightGBM + conformal | 17.87 | 0.640 | 78.7% |
| TFT HPO ens-3 | 19.71 | 0.706 | 79.6% |
| **PatchTST best** | **22.98** | **0.823** | **69.5%** |
| naive-1d | 27.98 | 1.002 | 52.9% |

TFT gate NOT cleared. Coverage collapsed 10.5pp below target.

**Root-cause fixes, 1-yr test window** (2025-07-16 →, 8,760 h):

| variant | MAE | coverage 80% |
|---|---:|---:|
| 365d windows, 3 seeds | 21.50 ± 0.55 | 67.0% |
| 730d windows, 3 seeds | 20.27 ± 0.23 | 74.9% |
| 730d ens-3 (d64) | 19.94 | 75.8% |
| 730d ens-3 (d128, best) | 19.78 | — |
| TFT-730 ens-3 (same window) | 18.31 | 82.8% |
| LGBM champion (same window) | 17.66 | ~80% (conformal) |

**Loss decomposition vs champion 17.66** (1-yr window):
window +1.2 | ensemble +0.3 | capacity +0.2 | architecture +1.5
(PatchTST → TFT at the same window and ensemble). Every root-cause
candidate was isolated and measured. Capacity is marginal; the big
terms are the training window and the architecture.

## Why it lost — the encoder story

- **At 365d windows the price-history encoder added nothing.**
  Zeroing all 56 days of history and retraining did not hurt
  (−0.4 EUR/MWh, within seed noise). Coverage even improved.
- **At 730d windows the sign FLIPPED: encoder worth +2.5** (3 seeds).
  The redundancy was a window artifact — 365 days was too little data
  to learn how to use history, not proof history is useless.
- **Lesson: ablation verdicts are conditional on training config.**
- The skill lived in the known-future covariates. RES forecast
  ablation: +6.2 EUR/MWh at 365d, +5.8 at 730d. Without it the model
  is worse than naive (rMAE 1.07). Wind (+4.1) beats solar (+3.1):
  wind is less predictable, so its forecast carries more unique info.
- Permutation importance ranked the encoder FIRST while ablation
  ranked it last (365d). Both correct: permutation measures what a
  fixed model relies on; retrain-ablation measures unique information.
  Classic redundancy signature.
- Attention maps were nearly flat (weights 0.014–0.022 vs uniform
  0.018). Faint recency and weekly stripes, nothing more.

## Known failure modes

- Coverage collapse on short training windows (69.5% vs 80% target).
- No saturating gates: the all-zero offshore-wind column produced
  z-scores of 19,000,000 and val pinball 879. TFT's sigmoid gates
  survived the same bug. Fixed by the zero-variance guard in
  `src/models/deep/data.py` (constant training columns are zeroed).
- Channel independence discards the load × solar interaction — cheap
  by the correlation check, but the champion gets it for free.
- Single-seed screening is unreliable. Three times in this project a
  1-seed winner failed to replicate. Always confirm on 3 seeds.

## What would change the verdict

- **More data.** 730d windows on the full 2-year test are only
  possible 2027+ (needs 730d of history before 2024-07-16; data
  starts 2023-01).
- Even then, PatchTST must first beat TFT (18.31 vs 19.78 at the same
  window and ensemble). The +1.5 architecture term says the missing
  piece is LSTM + variable selection, not patch attention.

## Status

- Archived 2026-07-21 with a full explanation. Shadow gate never
  reached. DECISIONS.md entries: 2026-07-18 (verdict + zero-variance
  guard), 2026-07-19 (feature analysis).
- Interview line: "My encoder ablation changed sign from −0.4 to
  +2.5 EUR/MWh when I doubled the training window — I now treat every
  ablation verdict as conditional on the training config."

## Files

- Implementation: `src/models/deep/patchtst.py`
- Feature analysis harness: `src/models/deep/patchtst_feature_analysis.py`
- Zero-variance guard: `src/models/deep/data.py` (tests: `tests/test_deep_data.py`)
- Sweep: `reports/backtests/2026-07-17_patchtst_sweep.csv`
- Walk-forward: `reports/backtests/2026-07-17_patchtst_sweep_walkforward.csv`
- Full story + ablation CSVs + plots: `reports/sensitivity/patchtst/`
- Channel coupling check: `reports/sensitivity/channels_verdict.txt`
