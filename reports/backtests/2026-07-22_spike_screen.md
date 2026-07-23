# Spike classifier screen — 2026-07-22_spike_screen

Top-5% price hours. Walk-forward, weekly refits, 365d train window, test from 2024-07-16.
Label threshold from each training window (leakage-safe).
Gate: AUC >= 0.80 for a daily-report column.

|   seed |   auc |   brier |   precision_at2 |   spike_days |   n_hours |   spike_rate |
|-------:|------:|--------:|----------------:|-------------:|----------:|-------------:|
|     42 | 0.966 |  0.0338 |          0.7359 |          301 |     17552 |         0.05 |
