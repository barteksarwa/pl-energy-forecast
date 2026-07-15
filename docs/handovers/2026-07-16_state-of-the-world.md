# Handover — 2026-07-16 — state of the world (new-session entry point)

Read this + docs/PLAN.md; older handovers are history.

## What exists and works

- **Daily loop, unattended.** GitHub Actions cron 05:30 UTC runs
  fetch → score → forecast → report → commits itself. Keyless (PSE).
  Incumbent: seasonal naive (labeled BASELINE). **Shadow challenger:
  ridge+TSO** trains each morning, scored daily in the report.
- **Data, all zero-gap, hourly UTC parquet in data/processed/:**
  load + TSO forecast (PSE 2024-06-14+; ENTSO-E 2023+ in `entsoe/`,
  merged into canonical — see pse_vs_entsoe.csv), weather actuals
  (ERA5 2023+, 10 cities), archived weather forecasts (lead 1–2 d,
  2024+), day-ahead price, balancing price, generation mix (2024-06+).
  ENTSO-E token lives in .env (never commit).
- **Models** (contract: fit/predict p10/p50/p90; REGISTRY in
  src/models/base.py): naive, climatology, ridge, LASSO-AR,
  LightGBM quantile, 7 LSTM architectures (src/models/deep/).
- **Evaluation:** walk-forward engine (weekly refits, 09:00 D-1 cutoff
  enforced + corruption-proof tests), screening splits for cheap ranking,
  diagnostics plots, SHAP.

## The results table (walk-forward, 12 mo, honest weather)

ridge+TSO 2.13% | lgbm+TSO 2.14% | **TSO 2.31%** | attn+TSO 2.43% |
lgbm 3.16% | LSTM 3.67% | ridge 4.03% | naive 5.60%.
Full context: README table, docs/notes/model_selection/03, overnight readout.

Closed questions (do not re-derive): bigger nets lose (peak 106k params);
origin augmentation hurts; screening flatters nets 0.6–0.9 pp;
lgbm tuning marginal; TSO-combination is the winning pattern.

## Owner decisions pending

1. Phase 2 kickstart approval: docs/PHASE2_KICKSTART.md
   (portfolio composition? hourly vs 15-min price target?).
2. Shadow days before challenger promotion (suggest 14) → then
   DECISIONS entry + config flip + model card update.

## New since the token arrived (2026-07-16)

- ENTSO-E backfilled 2023+ into data/processed/entsoe/; cross-check vs PSE:
  mean |diff| 4.7 MW over 18,287 overlap hours. Canonical load/tso now
  2023-01-01 → today (31,006 h). Sources switchable via config data_source.
- Unlocked by 3.5 y of history: 2-year-test-period backtests, richer net
  training, and the rolling-vs-expanding ablation is now meaningful.

## Next work, in plan order (Phase 2 after owner approves)

1. M6a price viz done; price baselines next: naive-yesterday, LEAR on
   fundamentals (gen mix + load forecast + calendar). NO MAPE for prices.
2. Portfolio POC (retailer P&L in PLN) per kickstart doc.
3. Deferred Phase 1 polish: model cards for ridge+TSO and LSTM,
   rolling-vs-expanding ablation, seasonality explainer tex (M3 promise),
   PatchTST/TFT only if owner still wants them after reading verdicts.

## Traps for the next session

- Weather forecasts (lead-2) exist only 2024+; pre-2024 training rows use
  ERA5 actuals — fine for training, never for evaluation claims.
- DST days: nets skip them in training; naive/report handle 23/25 h days.
- PSE publishes day-D TSO forecast ~09:00:12 D-1 — 12 s after our cutoff;
  accepted as available (DECISIONS 2026-07-15).
- One MPS trainer at a time. FORCE_CPU=1 for smoke tests during runs.
- Don't trigger GH workflows manually while pushing (push race) —
  the cron handles itself.
