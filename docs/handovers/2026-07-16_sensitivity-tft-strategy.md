# Handover — 2026-07-16 — Sensitivity, strategy, Phase 2 direction

## What I did

- 2-year walk-forward backtest (2024-07-16 → 2026-07-14, 17,450 h). Results committed.
- Rolling-vs-expanding window ablation. Rolling 365d wins. Verdict in `04_window_ablation.tex`.
- Model cards: `ridge_tso.md`, `lstm_attn_tso.md`, `lgbm_quantile.md` (updated).
- Fixed challenger TSO NaN bug (`tso.ffill()` in `src/pipeline/challenger.py`).
- Launched and fixed sensitivity analysis (`run_sensitivity.py`):
  - SHAP bug fixed: `ax=ax` removed from `shap.summary_plot()`.
  - Added skip-if-done guards (idempotent re-runs).
  - All 8 analyses queued; steps 1-6 done, PCA feature backtest still running.
- Wrote `05_sensitivity.tex` with group ablation + permutation importance + LASSO + SHAP.
- Opus agent: full job market research → `docs/notes/job_market.md`.
- Fable agent: strategic direction → `docs/notes/strategic_direction.md`.

## State of things

**Works:**
- ridge_tso production model: 2.08% MAPE (2-yr), beats TSO (2.23%), runs daily.
- Daily cron at 05:30 UTC (GitHub Actions). Shadow tally started 2026-07-15. 0 valid days so far (TSO NaN bug fixed 2026-07-16; expect first valid run 2026-07-17).
- Sensitivity outputs in `reports/sensitivity/`: PCA, correlation, group ablation, permutation importance, LASSO path, SHAP all done.

**Still running:**
- `run_sensitivity.py` PID 89816 (PCA feature backtest, step 7/7). ~5h elapsed, ~149% CPU. KernelPCA RBF bottleneck. Expected to finish within next 1-2h.
- TFT campaign NOT yet started. Will launch automatically once PCA backtest completes (wakeup scheduled).

**Key results:**
- Group ablation: TSO removal +1.971pp (96% of model skill). Weather +0.077pp. Calendar +0.032pp. Lags +0.003pp.
- SHAP top-5 (lgbm_tso): tso_forecast_mw 2292 MW, hour_sin 67, hour_local 55, doy_sin 49, load_mean_7d 46.
- LASSO: all 25 features survive at alpha=0.1.

## Job market research findings

Source: `docs/notes/job_market.md` (Opus agent, 2026-07-16).

**Two hiring lanes:**
1. **Quant/trading** (InCommodities, Danske Commodities, Vattenfall trading, Shell): highest pay, fastest hiring. Wants Python + power-market mechanics + price forecasting. No deep learning required. "Prior knowledge of the energy sector is not required" (Danske) — PhD quant enough.
2. **Data-science forecasting** (ENGIE, talcom NL, Envelio): production ML for load/generation. Closer to what we built now.
3. **Polish utilities** (Polenergia, PGE, Tauron): easiest entry lane, Excel/SQL first, lower ML bar.

**What they model:**
- LEAR (LASSO-regularized autoregression) is the standard price baseline. Hard to beat.
- LightGBM/XGBoost are the de-facto standard for day-ahead price/load (not deep learning).
- Probabilistic/quantile output is the direction; P10/P50/P90 puts us ahead of median candidates.
- Inputs: weather (NWP/ERA5/ECMWF), day-ahead prices, wind+solar generation forecasts, fundamentals (gas/CO2/cross-border flows), load forecasts.

**What they want to hear:** Python (production-level), SQL, backtesting discipline, probabilistic forecasts, explain to non-technical stakeholders, market mechanics (DAM/IDM/balancing/rynek mocy/TGE).

**Critical gap:** No price forecasting. Trading desks live on price. Adding TGE day-ahead price (M6-M7) doubles the number of reachable roles.

**Domain knowledge checklist (for interview prep):**
- Day-Ahead Market (DAM/RDN Poland) — auction, gate closure, hourly→15-min products
- Intraday (IDM/XBID/RDB) — continuous trading, forecast updates
- Balancing market (RB) — Poland reformed June 2024: 15-min settlement, scarcity pricing
- PSE (TSO), TGE (exchange), URE (regulator), rynek mocy (capacity market)
- Merit-order stack (coal/gas/nuclear/wind dispatch order determines price)
- CO2 (EUA) and gas (TTF) as price drivers

## Strategic direction

Source: `docs/notes/strategic_direction.md` (Fable agent, 2026-07-16).

**Verdict: Path A (get hired), now.**

PSE publishes zone-level load forecast for free → no paying customer for our current product. A job is the customer-discovery phase for any future business.

**Fastest sequence to first interview:**
1. **Now → +2 weeks:** keep cron alive (track record accrues), start M6 price data + LEAR baseline
2. **+2 → +6 weeks:** LightGBM price model (TGE RDN fixing), SHAP drivers
3. **+6 → +8 weeks:** DuckDB/SQL layer (half-day), README polish, blog post: "I beat the Polish TSO"
4. **+8 weeks:** start applying. 30 daily reports exist by then.

**Cut from current plan:**
- TFT/transformer challenger. LSTM already lost; explaining *why* is worth more.
- Second EU zone. Zero interview delta.
- Web UI.

**Non-compete flag:** Before signing any employment contract, negotiate an IP carve-out for this repo. Polish contracts often assign pre-existing work to employer. Git history proves pre-employment origin.

## Decisions made

- Rolling 365-day window is default. Expanding costs +0.02pp with no benefit.
- Path A (get hired) before Path B (product). See `docs/DECISIONS.md`.
- Phase 2 priority: TGE price forecasting before any other extension.

## Next steps

1. **Merge PR (worktree-precious-soaring-crescent → main)** before next cron (05:30 UTC July 17).
2. **Start M6:** fetch TGE day-ahead prices via ENTSO-E (endpoint: `query_day_ahead_prices`). Backfill 3 years.
3. **Add LEAR baseline** for price: LASSO-AR on lagged prices + calendar. The benchmark to beat.
4. **DuckDB layer**: one notebook, query load/weather/forecast parquets via SQL. Half-day.
5. **Draft blog post outline**: "I beat the Polish TSO's day-ahead load forecast".
6. PCA backtest + TFT run autonomously (wakeup scheduled). `05_sensitivity.tex` will be updated with PCA numbers when done. TFT results → `06_tft.tex` + `docs/model_cards/tft.md`.

## Watch out for

- Shadow tally is at 0 valid days. TSO fix deployed 2026-07-16. First valid run expected 2026-07-17. Check `docs/shadow_tally.md`.
- TFT campaign will run 8-12h when it starts. Writes to `reports/backtests/*_tft_campaign.csv`.
- KernelPCA in sensitivity was brutally slow (O(n²) kernel on full dataset). If repeating: subsample to 5,000 rows for KPCA fitting in `_pca_transform_x`.
- PR currently at branch `worktree-precious-soaring-crescent`. Check GitHub for current PR status before merging.
- TGE price data: use ENTSO-E `query_day_ahead_prices(country_code='PL', ...)`. Returns EUR/MWh. Convert to PLN/MWh only for display; store EUR.
