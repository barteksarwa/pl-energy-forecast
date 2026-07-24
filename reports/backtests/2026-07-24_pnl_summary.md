# Battery-arbitrage P&L — 2026-07-24_pnl

Battery: 1 MW / 2 MWh / 0.85 round-trip / 1 cycle per day.
Schedule from each model's P50 at D-1, settled at actual DA
prices. Day-ahead market only. Same days for every model.
Capture rate = total P&L / perfect-foresight P&L.

|              |   eur_per_day |   total_eur |   capture_rate |   loss_days_pct |   n_days |
|:-------------|--------------:|------------:|---------------:|----------------:|---------:|
| perfect      |       221.284 |      157776 |          1     |           0     |      713 |
| ens_crps_cqr |       204.506 |      145813 |          0.924 |           1.683 |      713 |
| lgbm         |       202.178 |      144153 |          0.914 |           2.104 |      713 |
| lear         |       200.967 |      143290 |          0.908 |           1.543 |      713 |
| chronos      |       196.963 |      140435 |          0.89  |           2.665 |      713 |
| timesfm      |       194.641 |      138779 |          0.88  |           2.665 |      713 |
| naive        |       179.834 |      128221 |          0.813 |           4.488 |      713 |

![cumulative](../figures/pnl/cumulative_pnl.png)
