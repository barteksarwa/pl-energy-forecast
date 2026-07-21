# Price LGBM feature-group ablation — 2026-07-17_price_group_ablation

Walk-forward, weekly refits, last 180 days, P50 MAE (EUR/MWh).
Retrain ablation = value of information. Compare with SHAP rank
(reports/sensitivity/shap_importance_price.csv) — they answer
different questions; the gap between them measures redundancy.

| config                      |    mae |   delta_vs_full |
|:----------------------------|-------:|----------------:|
| full                        | 18.827 |           0     |
| drop price_lags (29 cols)   | 21.63  |           2.803 |
| drop res_forecast (3 cols)  | 22.37  |           3.543 |
| drop tso_load_fcst (1 cols) | 19.079 |           0.252 |
| drop load_lags (7 cols)     | 18.789 |          -0.038 |
| drop calendar (10 cols)     | 19.161 |           0.334 |
