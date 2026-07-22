# Statistical tests — 2026-07-22_stats_tests

Diebold-Mariano on daily L1 loss differentials (multivariate
version, Lago et al. 2021). One-sided H1: model A more accurate.

| comparison                  |   dm_stat |   p_one_sided |   n_days | verdict                   |
|:----------------------------|----------:|--------------:|---------:|:--------------------------|
| lgbm_win1095 vs lgbm_win365 |    -3.118 |      0.000911 |      732 | A significantly better    |
| lgbm vs lear (2-yr)         |    -1.591 |      0.0558   |      730 | no significant difference |
| lgbm vs tft730_ens3 (2-yr)  |    -5.635 |      8.73e-09 |      726 | A significantly better    |
| lgbm vs naive (2-yr)        |   -17.641 |      6.02e-70 |      729 | A significantly better    |

Band validation, shipped conformal bands (nominal 20% violations):
Kupiec = unconditional coverage; Christoffersen = violations
independent (low p = violations CLUSTER).

| band                    |   violation_rate |   kupiec_p |   christoffersen_p |   n_hours |
|:------------------------|-----------------:|-----------:|-------------------:|----------:|
| lgbm_quantile_conformal |           0.2108 |       0    |                  0 |     17504 |
| lear_conformal          |           0.2041 |       0.18 |                  0 |     17504 |
