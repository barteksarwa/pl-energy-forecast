# Model card — TFT (Temporal Fusion Transformer) — price task

## Purpose

Day-ahead price forecasting (EUR/MWh, 24 hourly outputs) for the Polish
SDAC market. Architecture from Lim et al. (2021). Tested as a challenger
to LEAR + LightGBM to answer the owner's hypothesis:

> "Attention over months of price history should catch regime information
> that 7-day lags miss."

**Verdict: TFT trails LEAR on MAE. Shadow gate not opened.**
The honest loss is documented below.

## Architecture

- Encoder: past price series (instance-normalised, learned by LSTM)
- Variable Selection Network (VSN): soft per-timestep feature importance
  over known-future covariates (calendar, wind/solar forecast, TSO load
  forecast, anchor price lag-168h)
- LSTM encoder-decoder: compresses local patterns before attention
- Temporal self-attention: captures long-range dependencies over the
  encoder window
- Quantile output heads: p10, p50, p90

Best HPO config (60-trial Optuna search, `data/processed/tft_hpo.db`):

| param | value |
|---|---|
| encoder_hours | 1344 (56 days) |
| d_model | 128 |
| n_heads | 8 |
| lstm_layers | 2 |
| dropout | 0.183 |
| lr | 0.00174 |
| batch | 32 |
| val pinball (screening split) | 0.1157 |

Final best: trial 56 of 60. Optuna converged on ctx=1344h + d128 + h8.
Screening val was 0.6–0.9pp optimistic vs walk-forward — as predicted.

## Walk-forward results (3-seed confirmation)

Test: 2024-07-16 → 2026-07-18 (17,472 hours). Monthly refits.
Same window as LEAR/LGBM. Runtime: 3.8h on MPS.

| model | MAE (EUR/MWh) | rMAE | coverage 80% | spike MAE |
|---|---|---|---|---|
| LightGBM + conformal | 17.87 | 0.640 | 78.7% | 60.7 |
| LEAR + conformal | 18.23 | 0.653 | 79.4% | 70.0 |
| **TFT HPO ens-3** | **19.71** | **0.706** | **79.6%** | **74.7** |
| TFT seed 7 | 20.69 | 0.741 | 78.3% | 77.5 |
| TFT seed 42 | 20.77 | 0.744 | 75.1% | 71.5 |
| TFT seed 99 | 20.79 | 0.745 | 76.6% | 78.3 |
| naive-1d | 27.98 | 1.002 | 52.9% | 78.2 |

**TFT trails LEAR by 1.48 EUR/MWh MAE (8.1%) and by 0.053 rMAE.**
Ensemble coverage (79.6%) is the best of any model, but coverage is
not the limiting factor — MAE is. Shadow gate NOT opened.

## Why TFT lost

Three candidate explanations:

**1. Data ceiling.** With only 3 years of training data and monthly refits
using 365-day windows, TFT's 1.27M parameters overfit the early-in-window
data. Ridge regression wins on load for the same reason: the signal is
linear after the TSO forecast is in the model, and a small model generalises
better from limited data.

**2. Architecture mismatch.** TFT was designed for multivariate forecasting
with many known-future series (retail demand, promotions, etc.). Price
forecasting is dominated by one autocorrelation signal (yesterday's price)
and one covariate (solar forecast). The VSN and LSTM encoder add complexity
without proportionate signal.

**3. Quantile training at low data.** TFT trains all three quantile heads
jointly. With 308 training samples per refit (early in the walk-forward),
the tails are undersampled. LGBM trains separate trees per quantile with
40,000+ samples — the tabular data density advantage is decisive.

**What long context IS good for.** Screening showed rMAE improves monotonically
with encoder length up to 1344h. The 56-day context captures regime memory
(gas crises, solar growth seasons). The problem is that capturing this regime
memory costs 4h of compute and does not close the MAE gap.

## VSN feature importance

From the 60-trial best model (export: `reports/sensitivity/tft_vsn_weights.csv`).

| feature | vsn_weight | interpretation |
|---|---|---|
| tso_forecast_mw | 0.235 | #1 — demand level sets price tier |
| solar_fcst_mw | 0.179 | #2 — merit order (PV suppresses midday price) |
| wind_on_fcst_mw | 0.122 | #3 — dispatches before coal; price discount |
| anchor_price_lag168 | 0.104 | week-ago regime anchor |
| hour_sin | 0.101 | intraday seasonality |
| is_weekend | 0.081 | demand suppressor |
| doy_sin | 0.073 | seasonal level (gas vs coal marginal) |

Note: TSO forecast ranks #1 in the final model (vs solar in the early
screening). With 56 days of context, the encoder absorbs the solar merit-order
signal through price autocorrelation; VSN then emphasises the demand signal
(TSO) which is complementary. Both approaches identify the same physics.

## Comparison: three importance methods

| rank | SHAP (LGBM, global) | Group ablation (LGBM) | VSN (TFT, future covs only) |
|---|---|---|---|
| 1 | solar_fcst_mw 18.7 | res_forecast +3.5 MAE | tso_forecast_mw 0.235 |
| 2 | price_lag_1d 14.1 | price_lags +2.8 MAE | solar_fcst_mw 0.179 |
| 3 | wind_on_fcst_mw 8.4 | calendar +0.3 MAE | wind_on_fcst_mw 0.122 |

SHAP and VSN agree that solar + demand + wind are the top future covariates.
They differ on rank because VSN has 56 days of price history in the encoder;
the solar effect is already embedded in lags.

## Status

- [x] Screening: TFT trails tabular by 30% (d64 sweep, 3 contexts)
- [x] Bug fixed: prediction-time covariate standardisation (apply_covariate_stats)
- [x] HPO: 60-trial Optuna search. Best val 0.1157 (ctx=1344, d128, h8, l2)
- [x] VSN weights exported (reports/sensitivity/tft_vsn_weights.csv)
- [x] Walk-forward 3-seed confirmation completed (3.8h on MPS)
- [x] Verdict: TFT trails LEAR (rMAE 0.706 vs 0.653). Shadow gate NOT opened.
- [x] Model selection note 08 updated with final numbers
- [ ] PatchTST sweep: next test (27 configs, cheaper architecture)

## Files

- Implementation: `src/models/deep/tft.py`
- HPO: `src/models/deep/run_tft_hpo.py`
- Walk-forward: `src/models/deep/run_tft_hpo_walkforward.py`
- Screening: `src/models/deep/run_tft_price.py`
- Data builder: `src/models/deep/price_data.py`
- HPO study: `data/processed/tft_hpo.db`
- Best checkpoint: `data/processed/tft_hpo_ckpts/best.pt`
- VSN weights: `reports/sensitivity/tft_vsn_weights.csv`
- Walk-forward results: `reports/backtests/2026-07-17_tft_hpo_walkforward.(csv|md)`
