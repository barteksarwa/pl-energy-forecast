# Price LGBM feature-group ablation — 2026-07-28_price_group_ablation

Walk-forward, weekly refits, last 180 days, P50 MAE (EUR/MWh).
Retrain ablation = value of information. Compare with SHAP rank
(reports/sensitivity/shap_importance_price.csv) — they answer
different questions; the gap between them measures redundancy.

| config                      |    mae |   delta_vs_full |
|:----------------------------|-------:|----------------:|
| full                        | 18.697 |           0     |
| drop price_lags (29 cols)   | 22.156 |           3.459 |
| drop res_forecast (3 cols)  | 22.087 |           3.39  |
| drop tso_load_fcst (1 cols) | 19.163 |           0.466 |
| drop load_lags (7 cols)     | 18.723 |           0.026 |
| drop calendar (10 cols)     | 19.287 |           0.59  |
