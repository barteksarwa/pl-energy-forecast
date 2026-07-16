# Handover — 2026-07-16 — models, backtest, TFT, sensitivity

Read this + docs/PLAN.md. Ignore older handovers.

## What happened this session

### Bug fixed: challenger TSO NaN
Cron runs 05:30 UTC. PSE publishes tomorrow's TSO ~07:00 UTC. Gap = 90 min.
Ridge+TSO failed daily (NaN). Fix: `tso.ffill()` in `challenger.py` before
building tomorrow's feature matrix. Deployed in PR #1.

### Shadow tally started
See `docs/shadow_tally.md`. Days 1 (2026-07-15) and 2 (2026-07-16) failed
before the fix. Day 3 (2026-07-17 cron) is first expected valid day —
**only if PR #1 is merged before 05:30 UTC**.

### Code on branch `worktree-precious-soaring-crescent` (PR #1)
- `src/pipeline/challenger.py` — TSO ffill fix
- `src/evaluation/run_2year_backtest.py` — 2yr walk-forward + rolling/expanding ablation
- `src/evaluation/run_sensitivity.py` — PCA, group ablation, SHAP, LASSO, PCA-feature sweep
- `src/models/deep/tft.py` — TFT with Variable Selection Network (VSN)
- `src/models/deep/run_tft_campaign.py` — d32/d64/d128 × ±TSO, 2yr walk-forward
- `run_heavy_queue.sh` — orchestrator: waits for 2yr backtest → sensitivity → TFT
- Model cards: `ridge_tso.md`, `lstm_attn_tso.md`, updated `lgbm_quantile.md`
- `docs/shadow_tally.md`, `docs/notes/model_selection/04_window_ablation.tex`

### Heavy compute running in background
`run_heavy_queue.sh` launched locally at ~22:52 UTC. Sequence:
1. Wait for 2yr backtest (`*_2yr_summary.csv` in reports/backtests/)
2. Sensitivity/PCA/SHAP/ablation/PCA-feature-sweep (~3 h)
3. TFT campaign d32/d64/d128 × ±TSO on MPS (~8–12 h)
Expected outputs: `reports/backtests/*_2yr_(summary|ablation).csv`,
`reports/sensitivity/`, `reports/backtests/*_tft_campaign.csv`

The 2yr backtest itself is also running in background (launched separately,
should finish before the queue script finds its result file).

## Current results table (12-month walk-forward, unchanged)

ridge+TSO 2.13% | lgbm+TSO 2.16% | TSO 2.31% | lstm_attn+TSO 2.43% |
lgbm 3.16% | LSTM 3.67% | ridge 4.03% | naive 5.60%

2-year results: pending (in `reports/backtests/*_2yr_summary.csv`).

## Outstanding work for next session

1. **After backtest finishes**: fill in numbers in `04_window_ablation.tex`
   (rolling vs expanding verdict). Commit the backtest CSVs.
2. **After sensitivity finishes**: read `pca_feature_backtest.md` — if PCA
   features beat raw, add it as a model variant; if not, document why.
3. **After TFT finishes**: update `04_window_ablation.tex` (or new `05_tft.tex`),
   update README table, write TFT model card.
4. **Merge PR #1** before 05:30 UTC (07:30 Warsaw) for challenger fix.
5. **Phase 2**: owner confirmed price forecasting (hourly, 500 MW peak portfolio).
   Start M6 (price data + fundamentals) after all Phase 1 polish above.
6. **shadow_tally.md**: update daily as reports come in.

## Traps for the next session

- `run_heavy_queue.sh` writes to main repo paths (data/, reports/) not worktree.
  If you need to restart a job, use the main repo's venv:
  `/Users/bartlomiejsarwa/Documents/forecasting/.venv/bin/python`.
- PR #1 must be merged into main for the cron to use the challenger fix.
  Don't push directly to main — merge via PR.
- Rolling vs expanding ablation is INSIDE `run_2year_backtest.py` (separate step).
  It runs automatically after the main 4-model section.
- TFT VSN weights: after training, `net.enc_vsn.weights` and `net.fut_vsn.weights`
  hold per-sample feature importance. Load a checkpoint and inspect them.
- Weather for TFT: same hybrid (ERA5 pre-2024, lead-2 2024+). TFT training
  drops any day with NaN weather automatically (in `build_samples`).
