# Overnight readout — 2026-07-15

## 2026-07-15_fcst_tso_summary

# Backtest summary — 2026-07-15_fcst_tso

Test period: 2025-07-13 → 2026-07-13. Weather input: archived forecasts, lead 2 days (honest).

| model          |     mae |    rmse |   mape_pct |   pinball_p10 |   pinball_p50 |   pinball_p90 |   n_hours |   skill_vs_naive |
|:---------------|--------:|--------:|-----------:|--------------:|--------------:|--------------:|----------:|-----------------:|
| ridge          |  383.19 |  509.33 |       2.13 |         92.14 |        191.6  |         92.48 |      8762 |             0.63 |
| lgbm_quantile  |  394.01 |  521.63 |       2.16 |        110.81 |        197    |        120.1  |      8762 |             0.62 |
| tso_forecast   |  417.86 |  556.78 |       2.31 |        nan    |        208.93 |        nan    |      8762 |             0.59 |
| seasonal_naive | 1025.14 | 1537.35 |       5.6  |        353.79 |        512.57 |        281.75 |      8762 |             0    |


## deep walk-forward: lstm_enc_dec_h64

MAPE 3.67% | MAE 692 MW | pinball 162.2/345.8/169.0

## deep walk-forward: lstm_enc_dec_h64_tso

MAPE 2.49% | MAE 460 MW | pinball 108.5/230.2/109.2

## deep walk-forward: lstm_lstm_attn_h64

MAPE 3.74% | MAE 699 MW | pinball 170.3/349.4/166.8

## deep walk-forward: lstm_lstm_attn_h64_tso

MAPE 2.43% | MAE 448 MW | pinball 101.2/223.9/113.3

## LSTM ladder (v3, screening)

| variant      |   hidden |   test_mape |   test_pinball_p50 |         n_params |
|:-------------|---------:|------------:|-------------------:|-----------------:|
| bilstm       |       50 |       5.753 |            487.295 |  31272           |
| bilstm       |      100 |       4.578 |            393.76  | 102472           |
| lstm_attn    |       32 |       3.104 |            266.5   |  28739           |
| lstm_attn    |       64 |       2.882 |            249.015 | 106627           |
| lstm_attn    |      128 |       2.876 |            248.16  | 409859           |
| lstm_attn    |      256 |       2.9   |            250.955 |      1.60615e+06 |
| vanilla_lstm |       50 |       5.616 |            479.455 |  15672           |
| vanilla_lstm |      100 |       4.694 |            403.865 |  51272           |
| vanilla_lstm |      200 |       4.854 |            419.195 | 182472           |

## Nets + TSO covariate (screening)

| variant    |   hidden |   test_mape |   test_pinball_p50 |   n_params |
|:-----------|---------:|------------:|-------------------:|-----------:|
| enc_dec    |       64 |       2.393 |            206.95  |     106691 |
| enc_futmlp |      256 |       2.311 |            199.445 |     869379 |

## Origin augmentation (screening)

| variant   |   hidden |   test_mape |   test_pinball_p50 |   n_params |
|:----------|---------:|------------:|-------------------:|-----------:|
| enc_dec   |       64 |       3.73  |             316.16 |     106435 |
| enc_dec   |      128 |       3.655 |             310.77 |     409475 |

## Capacity axis incl. h512 (screening)

| variant      |   hidden |   test_mape |   test_pinball_p50 |         n_params |
|:-------------|---------:|------------:|-------------------:|-----------------:|
| deepar_style |       40 |       3.686 |            317.42  |  69883           |
| enc_dec      |       32 |       3.212 |            276.32  |  28643           |
| enc_dec      |       64 |       3     |            258.055 | 106435           |
| enc_dec      |      128 |       3.144 |            269.33  | 409475           |
| enc_dec      |      256 |       3.151 |            270.4   |      1.60538e+06 |
| enc_dec      |      512 |       3.275 |            277.6   |      6.35648e+06 |
| enc_direct   |       32 |       5.876 |            492.395 |  20616           |
| enc_direct   |       64 |       5.822 |            494.545 |  69832           |
| enc_direct   |      128 |       5.242 |            446.555 | 254280           |
| enc_direct   |      256 |       5.146 |            437.475 | 967240           |
| enc_futmlp   |       32 |       3.44  |            291.345 |  15459           |
| enc_futmlp   |       64 |       3.558 |            301.13  |  57539           |
| enc_futmlp   |      128 |       3.184 |            270.31  | 221571           |
| enc_futmlp   |      256 |       3.13  |            265.465 | 869123           |
| residual     |       32 |       3.682 |            312.09  |  28643           |
| residual     |       64 |       3.856 |            324.95  | 106435           |
| residual     |      128 |       3.614 |            306.385 | 409475           |
| residual     |      256 |       3.557 |            299.985 |      1.60538e+06 |
