# Model card — ens_crps_cqr (price ensemble)

## What it is

A CRPS-weighted blend of three price models, re-conformalized.
Current best price forecast on every gate. Promotion to the daily
loop pending owner decision.

Members (pre-declared before the run):
- `lgbm_quantile` + CQR — the trained champion (covariates).
- `lear` + CQR — the industry-standard linear baseline (covariates).
- `chronos_bolt_zs` + CQR — best zero-shot foundation model
  (univariate; the diversity donor).

Code: `src/evaluation/ensemble.py`, runner
`src/evaluation/run_price_ensemble.py`.

## How it works

1. Each member's bands are conformally calibrated first.
2. Score each member per hour with `crps3` = mean pinball loss over
   the three stored quantiles (honest CRPS proxy — the name says so).
3. Weight for day D = inverse of the member's mean crps3 over the
   trailing 60 days before D. Past-only. Equal weights during the
   60-day warm-up.
4. Blend quantile-wise (weighted average per quantile), re-sort so
   P10 ≤ P50 ≤ P90.
5. Run rolling CQR AGAIN on the blended band. Averaging three
   calibrated bands over-covers (84.2%); the second pass tightens it
   back to nominal (Q goes negative). MAE untouched — only bands move.

## Performance (2-yr walk-forward, 17,696 h, regenerated 2026-07-24)

| metric | ens_crps_cqr | LGBM champion | LEAR |
|---|---|---|---|
| MAE (EUR/MWh) | **17.34** | 17.84 | 18.46 |
| rMAE | **0.622** | 0.640 | 0.662 |
| coverage [P10,P90] | **79.9%** | 78.6% | 79.5% |
| Winkler | **85.2** | 90.3 | 88.8 |
| spike MAE (top 5%) | 61.5 | **60.4** | 70.8 |
| P&L capture (battery) | **0.926** | 0.915 | 0.911 |

Pre-declared gates, all passed: beat champion by ≥0.15 MAE (got 0.50),
DM p < 0.05 (got 2.5e-04), win every test year (won 2024/25/26),
Winkler not worse (improved).

## Honest caveats

- Skill weighting is worth only ~0.12 MAE over equal weights.
  The diversity does the work, not the weighting.
- Spike MAE marginally worse than the champion — blending smooths.
- Operational cost: three models in the daily loop; Chronos brings
  the transformers stack (`uv run --extra fm`).
- Interacts with the pending 1095d-window promotion: rebuilding the
  blend on 1095d members is the natural follow-up.

## Interpretability

Weights are the explanation: the daily report can state "this week the
blend leans 45% LGBM / 35% LEAR / 20% Chronos" and why (trailing CRPS).
Member-level SHAP still applies to the structural members.

## Verdicts and notes

- model_selection note 16 (ensemble verdict), 17 (P&L verdict).
- learning note 23 (mechanics), 25 (EUR framing).
