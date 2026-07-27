# Model card — spike_classifier (price)

**Production status: PUBLISHED daily** — a report flag, not a band
modifier. Prints spike risk for tomorrow's priciest hours; never moves
the published quantiles.

## What it is

LightGBM binary classifier: probability that an hour lands in the top
5% of prices. File: `src/models/spike.py`. Promoted 2026-07-23
(DECISIONS): walk-forward AUC 0.966 under the corrected day-ahead
cutoff protocol.

## Why a classifier and not a wider band

Three unconditional band fixes (asymmetric CQR, GPD tail, rolling
spike threshold) all failed their pre-declared spike-coverage gates.
Spikes are conditional events; the honest route is a dedicated
probability shown next to the band in plain words.

## Leakage rule

The "top 5%" threshold comes from each refit's TRAINING window, never
from the test period. A pooled evaluation-side threshold would score
better and be a lie.

## Inputs / training

Same feature matrix as the price models (D-1 price curve, lags, RES +
TSO load forecasts, calendar, fuel proxies). 365-day rolling window,
weekly refits, walk-forward.

## Honest limitations

- AUC 0.966 is ranking skill; calibrated probabilities are decent
  (Brier scored in the screen) but not perfect — treat the flag as
  risk triage, not a price forecast.
- Trained on PL 2023+ only; a regime never seen (new interconnector,
  fuel shock) degrades it silently. The daily report's scoring loop is
  the detection mechanism.
