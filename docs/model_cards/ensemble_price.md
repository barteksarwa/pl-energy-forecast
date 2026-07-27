# Model card — ens4_tft (price ensemble)

**Production status: BACKTEST CANDIDATE** — not in the daily loop.
Best price forecast on every backtest gate; promotion pending owner
(TFT inference cost).

## What it is

A CRPS-weighted blend of four price models, re-conformalized.
Current best price forecast on every gate. Promotion to the daily
loop pending owner decision (the open question is TFT's cost).

Members (pre-declared before the run):
- `lgbm_quantile` + CQR — the trained champion (covariates).
- `lear` + CQR — the industry-standard linear baseline (covariates).
- `chronos_bolt_zs` + CQR — best zero-shot foundation model
  (univariate; a diversity donor).
- `tft_730` 3-seed ensemble + CQR — the archived deep challenger
  (the biggest diversity donor; see below).

Code: `src/evaluation/ensemble.py`, runner
`src/evaluation/run_price_ensemble.py`.

## How it works

Mechanics are unchanged from the 3-member blend. TFT is just a
fourth member.

1. Each member's bands are conformally calibrated first.
2. Score each member per hour with `crps3` = mean pinball loss over
   the three stored quantiles (honest CRPS proxy — the name says so).
3. Weight for day D = inverse of the member's mean crps3 over the
   trailing 60 days before D. Past-only. Equal weights during the
   60-day warm-up.
4. Blend quantile-wise (weighted average per quantile), re-sort so
   P10 ≤ P50 ≤ P90.
5. Run rolling CQR AGAIN on the blended band. Averaging calibrated
   bands over-covers; the second pass tightens it back to nominal
   (Q goes negative). MAE untouched — only bands move.

## Performance (2-yr walk-forward, 17,456 h intersection)

TFT's stored preds end 2026-07-14, so the blend is scored on the
17,456 h all four members share. Not the full 17,696 h window.

| metric | ens4_tft | ens3 (fallback) | LGBM champion | TFT alone |
|---|---|---|---|---|
| MAE (EUR/MWh) | **16.89** | 17.34 | 17.84 | 19.53 |
| rMAE | **0.605** | 0.622 | 0.640 | 0.699 |
| coverage [P10,P90] | **80.0%** | 79.9% | 78.6% | 79.4% |
| Winkler | **82.6** | 85.2 | 90.3 | 97.5 |
| P&L capture (battery) | **0.929** | 0.926 | 0.915 | — |

Pre-declared gates, all passed: beat ens3 by ≥0.10 MAE (got 0.45),
DM p < 0.05 (got 2.3e-09), win every test year (won 2024/25/26),
coverage nominal (80.0%), Winkler not worse (best ever).

## Honest caveats

- TFT's operational cost is the promotion question. Three seeds in
  the daily loop means MPS inference plus monthly refits
  (~hours/month). The other three members are cheap by comparison.
- The blend is scored on the 17,456 h intersection, not the full
  2-yr window. TFT's preds end 2026-07-14.
- BOA weighting was tested and rejected (18.41 vs 16.89). BOA piles
  99.6% weight on LGBM — it minimizes regret vs the best single
  expert, so it collapses to selection and forfeits the diversity
  averaging that drives the blend. Inverse-CRPS stays.
- TimesFM as a 4th member was tested and rejected (+0.57, i.e. worse).
  It is a foundation model like Chronos, so it adds no new error
  structure. Diversity of ERROR STRUCTURE matters, not member count.
- Spike MAE is still worse than the solo champion. Blending smooths
  the tail — spike handling is the classifier's job, not this model's.
- Fallback: the 3-member blend (17.34) remains the promotion
  candidate if the desk cannot run deep-model infrastructure. It
  needs no MPS and no TFT refits. Only 0.45 MAE separates them.

## Interpretability

Weights are the explanation. The daily report can state the current
lean, e.g. "this week the blend leans mostly LGBM, with TFT and LEAR
next", and why (trailing CRPS). Member-level SHAP still applies to the
structural members; TFT has its own VSN importances.

## Verdicts and notes

- model_selection note 16 (ensemble verdict), 17 (P&L verdict).
- learning note 23 (mechanics), 25 (EUR framing).
- Canonical numbers: `docs/RESULTS.md` (4-member blend block).
- TFT's diversity-donor role: `docs/model_cards/tft_price.md`.
