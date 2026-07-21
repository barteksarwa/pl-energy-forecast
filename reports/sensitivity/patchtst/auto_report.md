# PatchTST feature analysis

Config: patch24_s24_ctx1344, d_model=64.
Context: PatchTST lost the 2-year walk-forward (MAE 22.98 vs TFT 19.71 vs LGBM 17.8 EUR/MWh).
This analysis shows where its signal comes from.

## Group ablation (walk-forward, 3 seeds)

Zero one input group, retrain, rerun 2-year walk-forward.
ΔMAE vs full = importance of that group.

| group     |   mae_mean |   mae_std |   rmae_mean |   cov_mean |   seeds |   delta_mae |
|:----------|-----------:|----------:|------------:|-----------:|--------:|------------:|
| encoder   |     23.231 |     0.102 |       0.832 |     73.188 |       3 |      -0.376 |
| full      |     23.607 |     0.529 |       0.845 |     68.546 |       3 |       0     |
| calendar  |     23.683 |     0.434 |       0.848 |     71.35  |       3 |       0.076 |
| tso_load  |     24.079 |     0.549 |       0.862 |     70.662 |       3 |       0.473 |
| anchor168 |     24.466 |     0.296 |       0.876 |     67.151 |       3 |       0.86  |
| solar     |     26.673 |     0.217 |       0.955 |     69.027 |       3 |       3.066 |
| wind_on   |     27.721 |     0.212 |       0.993 |     73.331 |       3 |       4.114 |
| res_fcst  |     29.838 |     0.36  |       1.068 |     73.958 |       3 |       6.231 |

![ablation](ablation_delta_mae.png)

## Permutation importance (screening split, val 2026+)

| feature             |   delta_pinball |   delta_pinball_std |   delta_mae_eur |   delta_mae_std |
|:--------------------|----------------:|--------------------:|----------------:|----------------:|
| enc_price_history   |          0.1791 |              0.0105 |         26.1462 |          1.496  |
| solar_fcst_mw       |          0.1121 |              0.0033 |         15.1829 |          0.4918 |
| tso_load_fcst       |          0.0947 |              0.0066 |         13.8607 |          0.9628 |
| wind_on_fcst_mw     |          0.0577 |              0.0061 |          8.2773 |          0.8911 |
| price_anchor_lag168 |          0.0037 |              0.0006 |          0.548  |          0.1227 |
| is_weekend          |          0.0033 |              0.0008 |          0.7972 |          0.1671 |
| is_holiday          |          0.001  |              0.0009 |          0.0712 |          0.0784 |
| doy_sin             |          0.0005 |              0.0002 |          0.0987 |          0.0319 |
| is_bridge_day       |          0      |              0.0004 |          0.0055 |          0.0583 |
| hour_sin            |          0      |              0      |          0      |          0      |
| hour_cos            |          0      |              0      |          0      |          0      |
| wind_off_fcst_mw    |          0      |              0      |          0      |          0      |
| doy_cos             |         -0.0003 |              0.0005 |         -0.0968 |          0.0761 |

![perm](permutation_importance.png)

## Attention

Top-5 most-attended past days (last layer):

|   day_age |   mean_attention_last_layer |
|----------:|----------------------------:|
|         0 |                      0.0211 |
|         7 |                      0.0204 |
|        12 |                      0.0202 |
|         1 |                      0.02   |
|        10 |                      0.0199 |

![attention](attention_patterns.png)

## PCA

![patches](pca_patches.png)

![reps](pca_representations.png)
