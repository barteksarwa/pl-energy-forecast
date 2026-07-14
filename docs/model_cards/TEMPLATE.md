# Model card — <name>

One page. A colleague must know in 2 minutes what this model is,
what it eats, how good it is, and when not to trust it.

## What it is

- Model type, quantile mechanism, file in `src/models/`.

## Inputs

- Feature groups + count. Data sources. What it does NOT see.

## Training

- Window, refit cadence, hyperparameters (and whether tuned).

## Performance (walk-forward, honest weather)

- Table row vs baselines + TSO. Test period. Link to reports/backtests/.
- Where it is weak: hours, day types, tail behavior.

## Interpretability

- SHAP / importance artifact and its top 3 global drivers, in plain words.

## Known failure modes

- Bullet list. Honest.

## Status

- dev / uat / prod. Date promoted, by which DECISIONS.md entry.
