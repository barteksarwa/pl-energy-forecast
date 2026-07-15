# Model card — lstm_attn_tso (LSTM Luong attention + TSO)

## What it is

A seq2seq LSTM with Luong (dot-product) attention, trained with the TSO
day-ahead forecast as a known-future covariate. The encoder reads 14 days
of hourly history; the decoder autoregressively generates the 24-hour horizon.
At each decoder step, attention over all 336 encoder hidden states builds a
context vector — removing the fixed-vector bottleneck of plain seq2seq.

File: `src/models/deep/nets.py` (`LstmLuongAttn`, name `lstm_attn`).
Trained via `src/models/deep/train.py` (pinball loss, teacher forcing,
MPS/CUDA/CPU adaptive).

## Inputs

Two input streams to the encoder/decoder:

- **Past (encoder, 336 steps × enc_feat):** load lags, calendar features,
  weather actuals (ERA5 in training; archived lead-2 forecast in test).
- **Future (decoder, 24 steps × fut_feat):** calendar features (hour, day,
  month, holiday), weather forecast (lead-2), **TSO day-ahead forecast** (MW).
  All are known at the 09:00 D-1 cutoff.

Model size: 106,627 parameters at hidden=64 (the production-tested size).

## Training

- Sliding windows of 14 encoder days → 24-h decoder horizon.
- Pinball loss (α=0.1, 0.5, 0.9) — 3 quantiles from a single forward pass.
- Teacher forcing during training; autoregressive at inference.
- Early stopping on validation pinball (patience 5).
- Walk-forward test: 12 months, retrained at each weekly refit.
- NOT tuned beyond architecture selection; tuning logged in M3–M5 notes.

## Performance (walk-forward, honest lead-2 weather)

### 12-month test (8762 h)

| model | MAPE | MAE (MW) | pinball P10 / P50 / P90 |
|---|---|---|---|
| ridge_tso (production candidate) | 2.13% | 383 | 92 / 192 / 92 |
| **lstm_attn + TSO** | **2.43%** | **448** | 101 / 224 / 113 |
| lstm_enc_dec + TSO | 2.49% | 460 | 109 / 230 / 109 |
| TSO alone | 2.31% | 418 | – / 209 / – |
| lstm_attn (no TSO) | 3.74% | 699 | 170 / 349 / 167 |

Source: `reports/backtests/2026-07-15_overnight_readout.md`.

**Screening flattery warning:** the attention model scored 2.88% MAPE in
single-split screening, 3.74% in walk-forward (no TSO). Gap = +0.86 pp.
This is the largest flattery gap in the ladder and is fully explained:
screening used a fixed hold-out that happened to sit in a stable load regime.
Walk-forward is the authoritative number.

Why the attention model still loses to ridge+TSO despite more capacity:
the remaining signal after conditioning on the TSO forecast is nearly linear
(recency bias + seasonal correction). Attention is optimized for long-range
temporal dependencies, but after the TSO already captures load shape, the
encoder output over 336 steps adds noise rather than signal — overfitting
to training-window idiosyncrasies.

## Interpretability

No analytic coefficients. Interpretability tools:
- **Permutation importance** (on the validation set): shows that `tso_forecast_mw`
  has the highest permutation importance by a large margin, confirming the same
  pattern as ridge.
- **Attention weight visualization**: `decoder attention[:,·,·]` shows which
  encoder steps each prediction attends to — typically peaking at 168 h ago
  (same weekday, same hour), consistent with seasonal naive intuition.
- Saliency analysis and attention plots: `src/models/deep/make_readout.py`.

## Known failure modes

- Slower at inference and refit than ridge (100× training time).
- Cannot extrapolate beyond training range — same as any neural net.
- Screening-to-walk-forward gap proves the 365-day window is tight for a
  seq2seq; longer history would likely narrow the gap.
- With TSO not yet published (cron at 05:30 UTC), the future decoder input
  carries a stale TSO — same vulnerability as ridge_tso, more damaging here
  because the decoder has 24 explicit TSO slots rather than a single feature.
- DST days (23h/25h): skipped in training; LSTM expects a fixed 24-step horizon.

## Status

**dev.** Kept as the "best deep net" benchmark for the model table and for
future re-evaluation if history grows past 5 years. Not recommended for UAT
ahead of ridge_tso or lgbm_tso. Revisit if load structure changes enough
that nonlinearity matters again (e.g. large penetration of behind-the-meter PV).
