# Price LGBM feature-group ablation — 2026-07-19_price_group_ablation

Walk-forward, weekly refits, last 730 days, P50 MAE (EUR/MWh).
Retrain ablation = value of information. Compare with SHAP rank
(reports/sensitivity/shap_importance_price.csv) — they answer
different questions; the gap between them measures redundancy.

| config                      |    mae |   delta_vs_full |
|:----------------------------|-------:|----------------:|
| full                        | 17.869 |           0     |
| drop price_lags (29 cols)   | 21.816 |           3.947 |
| drop res_forecast (3 cols)  | 21.465 |           3.596 |
| drop tso_load_fcst (1 cols) | 18.411 |           0.542 |
| drop load_lags (7 cols)     | 17.749 |          -0.12  |
| drop calendar (10 cols)     | 18.303 |           0.434 |
