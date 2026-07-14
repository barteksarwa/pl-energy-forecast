# Reproduction spec — DeepAR (Salinas et al. 2020)

Paper: "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent
Networks", Int. J. Forecasting 36(3), arXiv:1704.04110.

## What the paper does

- Target: probabilistic multi-step forecasts; electricity benchmark =
  hourly consumption of 370 customers (not national load).
- Model: LSTM, autoregressive — input at step t includes the observed
  (or self-fed) previous target value plus covariates.
- Verified config: 3 LSTM layers × 40 cells, dropout 0.1, Adam.
- Output: Gaussian likelihood head; quantiles via sampling.
- Trains ONE model across all 370 series (cross-learning is a key claim).

## Our adaptation (src/models/deep/nets.py :: DeepARStyle)

Kept: 3×40 LSTM, dropout 0.1, Adam, autoregressive previous-target input,
teacher forcing in training, self-feeding at inference.

Deviations (all deliberate, logged here):
1. Quantile heads on pinball loss instead of Gaussian likelihood — matches
   how every other model in our table is scored.
2. Single series (PL national load). No cross-series learning — we have one
   country. This removes DeepAR's main advantage; expectations calibrated.
3. Our covariates (weather forecasts, PL calendar) instead of the paper's.
4. Encoder window 336 h with instance normalization (paper scales by
   series average).

## Comparison rule

deepar_style competes in the v2 sweep on the same split, seeds and metric
as all variants. It is a *style* recreation on a different dataset —
we cannot and do not claim to reproduce the paper's numbers.

## Papers we did NOT recreate (and why)

- Kong et al. 2019 (LSTM STLF): hyperparameters not fully published;
  residential-level target. Context only.
- Marino et al. 2016 (seq2seq): single-building target, config not
  verifiable from accessible sources. The enc_dec variant covers the idea.
- TFT (Lim et al. 2021): the right next challenger, but heavier; queued
  for M5 after the LSTM readout.
