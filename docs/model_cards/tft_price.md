# Model card — TFT (Temporal Fusion Transformer) — price task

## Purpose

Day-ahead price forecasting (EUR/MWh, 24 hourly outputs) for the Polish
SDAC market. Architecture from Lim et al. (2021). Tested as a challenger
to LEAR + LightGBM to answer the owner's hypothesis:

> "Attention over months of price history should catch regime information
> that 7-day lags miss."

This card records the honest outcome of that test.

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

Final best (trial 56 of 60). Previous session best was trial 49 at 0.1176 (l=1);
Optuna found lstm_layers=2 generalises better on this screening split.

**Note on screening vs walk-forward**: single-split val flatters nets by
0.6–0.9 pp vs walk-forward (measured on the load task). Walk-forward
numbers are the only ones that go in the results table.

## Walk-forward results (3-seed confirmation)

_Fill in after `run_tft_hpo_walkforward.py` completes._

Test: 2024-07-16 → present. Monthly refits. Same window as LEAR/LGBM.

| model | MAE (EUR/MWh) | rMAE | coverage 80% | spike MAE |
|---|---|---|---|---|
| **TFT HPO ens-3** | _pending_ | _pending_ | _pending_ | _pending_ |
| LEAR + conformal | 18.24 | 0.653 | 79.5% | 71.0 |
| LightGBM + conformal | 17.87 | 0.640 | 78.7% | 60.6 |
| naive-1d | 27.94 | 1.000 | 53.1% | 77.6 |

Verdict: _pending walk-forward run_.

## VSN feature importance

From the final HPO training (export: `reports/sensitivity/tft_vsn_weights.csv`).
VSN weights apply only to known-future covariates; autoregressive price
information enters through the encoder LSTM path, not the selection network.

| feature | vsn_weight | interpretation |
|---|---|---|
| solar_fcst_mw | 0.217 | #1 — merit order (PV suppresses midday price) |
| tso_forecast_mw | 0.170 | #2 — demand level sets price tier |
| wind_on_fcst_mw | 0.126 | #3 — dispatches before coal; price discount |
| doy_sin | 0.098 | seasonal level (gas vs coal marginal) |
| anchor_price_lag168 | 0.097 | week-ago regime anchor |
| hour_sin/cos | 0.067/0.067 | intraday seasonality |
| is_bridge_day | 0.055 | demand suppressor |

VSN agrees with SHAP on the top feature (solar #1). The disagreement on
price lags is by design: the encoder absorbs lag information; the VSN
only sees future covariates.

## Comparison: three importance methods

| rank | SHAP (LGBM, global) | Group ablation (LGBM) | VSN (TFT, future covs only) |
|---|---|---|---|
| 1 | solar_fcst_mw 18.7 | res_forecast +3.5 MAE | solar_fcst_mw 0.217 |
| 2 | price_lag_1d 14.1 | price_lags +2.8 MAE | tso_forecast_mw 0.170 |
| 3 | wind_on_fcst_mw 8.4 | calendar +0.3 MAE | wind_on_fcst_mw 0.126 |

All three methods agree: solar is the top driver among future covariates.
Price lags rank high in SHAP/ablation but not VSN because VSN only
measures future-covariate weights (by architecture design).

## Honest limitations

- Walk-forward numbers are pending. Screening val (0.1176) is for config
  selection only; it will be 0.6–0.9 pp optimistic.
- Screening showed d32 and d128 can achieve similar val (within 0.0008).
  Walk-forward with 3 seeds will confirm which generalises.
- Spikes: no known mechanism exists to materially improve spike MAE with
  architecture alone. The outage feature was flat.
- Computational cost: 3-seed × 2yr walk-forward ≈ 4–6h on MPS. Inference
  at score time is fast (one forward pass per day).

## Status

- [x] Screening: TFT trails tabular by 30% (d64 sweep, 3 contexts)
- [x] Bug fixed: prediction-time covariate standardisation (apply_covariate_stats)
- [x] HPO: 60-trial Optuna search completed. Best val 0.1176 (ctx=1344, d128, h8)
- [x] VSN weights exported (reports/sensitivity/tft_vsn_weights.csv)
- [ ] Walk-forward 3-seed confirmation (run_tft_hpo_walkforward.py)
- [ ] Verdict: TFT vs LEAR vs LGBM on identical window
- [ ] If TFT wins: shadow gate (14 days), promotion decision
- [ ] If TFT loses: model_selection/08 updated with final verdict

## Files

- Implementation: `src/models/deep/tft.py`
- HPO: `src/models/deep/run_tft_hpo.py`
- Walk-forward: `src/models/deep/run_tft_hpo_walkforward.py`
- Screening: `src/models/deep/run_tft_price.py`
- Data builder: `src/models/deep/price_data.py`
- HPO study: `data/processed/tft_hpo.db`
- Best checkpoint: `data/processed/tft_hpo_ckpts/best.pt`
- VSN weights: `reports/sensitivity/tft_vsn_weights.csv`
