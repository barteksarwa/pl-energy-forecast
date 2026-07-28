# Battery-arbitrage P&L — 2026-07-27_pnl

Battery: 1 MW / 2 MWh / 0.85 round-trip / 1 cycle per day.
Schedule from each model's P50 at D-1, settled at actual DA
prices. Day-ahead market only. Same days for every model.
Capture rate = total P&L / perfect-foresight P&L.

|              |   eur_per_day |   total_eur |   capture_rate |   loss_days_pct |   n_days |
|:-------------|--------------:|------------:|---------------:|----------------:|---------:|
| perfect      |       221.466 |      160120 |          1     |           0     |      723 |
| ens_crps_cqr |       205.575 |      148630 |          0.928 |           1.383 |      723 |
| lgbm         |       202.642 |      146510 |          0.915 |           1.521 |      723 |
| lear         |       201.895 |      145970 |          0.912 |           1.521 |      723 |
| chronos      |       197.341 |      142677 |          0.891 |           2.628 |      723 |
| timesfm      |       195.051 |      141022 |          0.881 |           2.628 |      723 |
| naive        |       180.316 |      130368 |          0.814 |           4.426 |      723 |

![cumulative](../figures/pnl/cumulative_pnl.png)
