# TFT HPO walk-forward — 2026-07-17_tft_hpo_walkforward

**Verdict: trails LEAR** (TFT ens MAE 19.71 vs LEAR 18.23 EUR/MWh)

Best HPO config: `{'encoder_hours': 1344, 'd_model': 128, 'n_heads': 8, 'lstm_layers': 2, 'dropout': 0.1829168556254403, 'lr': 0.0017425197272885047, 'batch': 32}`
Test: 2024-07-16 → 2026-07-18 (17472 hours)
Seeds: [42, 7, 99] | Runtime: 3.8 h

## Results (same-window comparison)

| model                   |    mae |   rmse |   rmae |   coverage_80_pct |   spike_mae |   n_hours |
|:------------------------|-------:|-------:|-------:|------------------:|------------:|----------:|
| lgbm_quantile_conformal | 17.867 | 28.776 |  0.64  |            78.692 |      60.704 |     17472 |
| lear_conformal          | 18.232 | 32.777 |  0.653 |            79.373 |      69.951 |     17472 |
| tft_hpo_ens3            | 19.713 | 32.239 |  0.706 |            79.556 |      74.725 |     17472 |
| tft_hpo_seed7           | 20.69  | 33.545 |  0.741 |            78.308 |      77.455 |     17472 |
| tft_hpo_seed42          | 20.772 | 33.332 |  0.744 |            75.092 |      71.505 |     17472 |
| tft_hpo_seed99          | 20.794 | 33.599 |  0.745 |            76.643 |      78.315 |     17472 |
| price_naive_yesterday   | 27.982 | 44.255 |  1.002 |            52.85  |      78.162 |     17472 |

## What to do with this

- **If TFT beats LEAR**: 3-seed confirmed, open shadow gate, write model card.
  Promotion criterion: mean daily MAE over 14 shadow days < LEAR MAE.
- **If TFT trails**: document WHY (architecture ceiling, data ceiling, or both).
  The honest verdict is as valuable as a win for the portfolio.

Per-seed predictions: `data/processed/backtest_preds_price_res/tft_hpo_ens.parquet`