# Model card — TFT (Temporal Fusion Transformer) — price task

## What it is

- TFT = Temporal Fusion Transformer (Lim et al. 2021).
- Deep quantile model. Outputs p10/p50/p90 for all 24 delivery hours.
- Trained with pinball loss (the quantile-regression loss).
- File: `src/models/deep/tft.py`.
- Target: PL day-ahead price, EUR/MWh (ENTSO-E, SDAC auction).
- Role: challenger to the champion (LGBM quantile + conformal).

**Verdict: best deep model in this repo. Still loses to the champion
by 0.65 EUR/MWh on the shared 1-year test.** Champion stands. The full
loss autopsy is below — most of the original 3 EUR/MWh gap was our own
training config, not the architecture.

## Architecture

- Encoder: 56 days (1344 h) of past hourly price. Instance-normalised.
- Variable Selection Network (VSN): learned soft weights over the
  known-future covariates. "Known-future" = tomorrow's values are
  already published today (forecasts, calendar).
- LSTM encoder-decoder compresses local patterns before attention.
- Temporal self-attention over the encoder window.
- Three quantile heads: p10, p50, p90. 1.27M parameters.

## Inputs

- Past: hourly price history (the encoder).
- Known-future: TSO day-ahead load forecast, wind + solar (RES)
  forecast, calendar features, anchor price lag-168h.
- These covariates help ALL models, not TFT specifically
  (cross-model table below).
- Timeline: the SDAC auction for day D clears ~12:00 CET on D-1.
  Tomorrow's price is the target, never an input.
- Does NOT see: fuel/CO2 prices, cross-border flows, outages.

## Training

- HPO: 60-trial Optuna search (`data/processed/tft_hpo.db`).
  Winner: ctx=1344h, d_model=128, 8 heads, 2 LSTM layers,
  dropout 0.183, lr 0.00174, batch 32. Trial 56 of 60.
- Original campaign: rolling 365-day windows, monthly refits.
  That window was a self-inflicted handicap (see below).
- Final config: 730-day windows + median-of-3-seeds ensemble.
- A 730d config sweep (8 configs) confirmed the HPO config is fine.
  Hyperparameters are NOT the remaining gap.

## Performance

**Original 2-year walk-forward** (365d windows, test
2024-07-16 → 2026-07-18, 17,472 h, monthly refits):

| model | MAE (EUR/MWh) | rMAE | coverage 80% | spike MAE |
|---|---|---|---|---|
| LightGBM + conformal | 17.87 | 0.640 | 78.7% | 60.7 |
| LEAR + conformal | 18.23 | 0.653 | 79.4% | 70.0 |
| TFT HPO ens-3 | 19.71 | 0.706 | 79.6% | 74.7 |
| naive-1d | 27.98 | 1.002 | 52.9% | 78.2 |

**Final same-window comparison** (test 2025-07-16 →, 8,760 h;
`reports/sensitivity/tft/README.md`):

| model | MAE (EUR/MWh) | rMAE | coverage 80% |
|---|---|---|---|
| LGBM champion | 17.66 | — | ~80% (conformal) |
| **TFT-730 ens-3** | **18.31** | **0.668** | **82.8%** |
| TFT-730 single-seed | 19.12 | 0.699 | 79.6% |
| PatchTST-730 ens-3 | 19.94 | — | 75.8% |

- Gap to champion: **0.65 EUR/MWh (3.7%)**, down from ~3.
- TFT hits the 80% coverage target natively — no conformal wrap.
  Best band calibration of any model in the repo.
- 730d windows cannot be tested on the full 2-year window yet
  (needs 730d of history before 2024-07-16; data starts 2023-01).

## Why TFT lost — the final decomposition

Three causes, each isolated and measured:

1. **365-day training windows: ~+1.2–1.5 EUR/MWh.** Doubling the
   window took the single-seed 1-yr MAE from 20.65 to 19.12 and fixed
   coverage. Every deep model in the campaign ran on 365d windows.
   The handicap was ours, not the model's.
2. **No seed ensemble: +0.8.** Median of 3 seeds: 19.12 → 18.31.
3. **The remaining 0.65 is architectural.** LGBM extracts more from
   price history on ~30k hourly rows and wins on tabular data density.
   The config sweep ruled out hyperparameters: the seed-42 "winner"
   (dropout 0.30, MAE 17.97) did not replicate on seeds 7/2026
   (20.17 / 19.43). Its ensemble (18.36) matched the baseline (18.31).

Campaign-wide decomposition vs champion 17.66 (1-yr window):
window +1.2 | ensemble +0.3 | capacity +0.2 | architecture +1.5
(that last term is PatchTST → TFT at the same window and ensemble).

**Lesson: 1-seed screening picks mirages.** It happened three times
in this project. Never promote a config from a single seed.

## Cross-model value of information

Retrain ablation: zero one input group, retrain, walk forward.
ΔMAE in EUR/MWh (`reports/sensitivity/tft/README.md` for caveats):

| input group | LGBM (17.87) | TFT-730 (19.12) | PatchTST-730 (20.27) |
|---|---:|---:|---:|
| price history | +3.95 | +2.00 | +2.46 |
| RES forecast | +3.60 | +3.20 | +5.76 |
| TSO load fcst | +0.54 | +0.26 (365d) | +1.27 |
| calendar | +0.43 | +0.21 (365d) | +0.38 |

- The champion wins by using price history HARDER, not by ignoring it.
- The weaker a model's use of history, the more it leans on RES.
- Known-future covariates carry skill for every model.

## VSN feature importance

From the HPO best model (`reports/sensitivity/tft_vsn_weights.csv`).

| feature | vsn_weight | interpretation |
|---|---|---|
| tso_forecast_mw | 0.235 | #1 — demand level sets price tier |
| solar_fcst_mw | 0.179 | #2 — merit order (PV suppresses midday price) |
| wind_on_fcst_mw | 0.122 | #3 — dispatches before coal; price discount |
| anchor_price_lag168 | 0.104 | week-ago regime anchor |
| hour_sin | 0.101 | intraday seasonality |
| is_weekend | 0.081 | demand suppressor |
| doy_sin | 0.073 | seasonal level (gas vs coal marginal) |

Note: TSO forecast ranks #1 here (vs solar in SHAP for LGBM). With 56
days of context, the encoder absorbs the solar merit-order signal
through price autocorrelation; VSN then emphasises the complementary
demand signal. Both methods identify the same market physics.

## Known failure modes

- Spike MAE 74.7 vs LGBM 60.7 — misses price spikes worse than
  the champion. Tails are weak across the whole table.
- 3.8h walk-forward runtime on MPS vs minutes for LGBM.
- Single seeds vary by ~1 EUR/MWh — never ship one seed.
- Verdicts are window-qualified: the 730d results only exist on the
  1-year test window.

## What would change the verdict

- **More data.** A 730d-window benchmark on the full 2-year test is
  only possible 2027+. If TFT-730 keeps its 82.8% coverage and closes
  MAE there, the shadow gate reopens.
- If native band calibration ever outranks MAE as the shipping
  criterion, TFT already wins that column.

## Status

- Archived as best deep challenger, 2026-07-21. Shadow gate NOT opened.
- Campaign closed: every root-cause candidate isolated and measured.
- Interview line: "Ablation verdicts are conditional on training
  config — doubling the window flipped an encoder ablation sign and
  closed most of a 3 EUR/MWh gap I had blamed on the architecture."

## Files

- Implementation: `src/models/deep/tft.py`
- HPO: `src/models/deep/run_tft_hpo.py`; study `data/processed/tft_hpo.db`
- Walk-forward: `src/models/deep/run_tft_hpo_walkforward.py`
- Ablation: `src/models/deep/run_tft_ablation.py`
- 730d config sweep: `src/models/deep/run_tft730_sweep.py`
- Results + story: `reports/sensitivity/tft/README.md`
- VSN weights: `reports/sensitivity/tft_vsn_weights.csv`
- 2-yr walk-forward: `reports/backtests/2026-07-17_tft_hpo_walkforward.(csv|md)`
