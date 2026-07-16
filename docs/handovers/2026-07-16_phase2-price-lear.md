# Handover — 2026-07-16 (evening) — Phase 2: price data + LEAR

## What I did

- Merged PR #1 (was already merged on GitHub; pulled main, cleared
  duplicate untracked files after byte-identical diff check).
- **M6 price data:** `fetch_day_ahead_price` in entsoe_client,
  `backfill_entsoe_prices` in backfill.py. 31,022 h backfilled
  (2023-01-01 → today), zero gaps, zero NaN. EUR/MWh canonical
  (DECISIONS). 15-min SDAC data resampled to hourly mean.
- **Price feature pipeline:** `price_lags.py` (local-day shifts),
  `price_matrix.py` (calendar + price lags + D-1 vector + load lags + TSO).
- **LEAR shipped and it wins:** rMAE 0.744 vs naive-yesterday over
  2 years (17,480 h), wins/ties all 25 months. Table:
  `reports/backtests/2026-07-16_price_summary.md`. Card: `model_cards/lear.md`.
- **DuckDB notebook:** `notebooks/01_sql_analysis.ipynb` (SQL evidence).
- **Learning note:** `08_price_formation_and_lear.tex` (merit order, LEAR).
- **Blog outline:** `docs/notes/blog_post_outline.md`.
- 7 new leakage/DST tests. Full suite: 55 passed.

## Three bugs this session (all caught by our own defenses)

1. Sonnet's try/except in backfill would have made data gaps permanent
   (resume skips past failed chunks). Removed — fail loudly is correct.
2. The 25-hour DST day makes minus-24h leak INTO the target day.
   The cutoff assert caught it. Price lags now shift by local days.
3. asinh on raw prices exploded winter forecasts ~100x (rMAE 2.64 in
   Dec 2025). Robust standardization first (median/MAD) fixed it.
   All three variants measured and documented in the model card.

## State of things

- **Running:** sensitivity PCA backtest (PID 89816, main checkout,
  ~6h CPU). TFT campaign queued behind it (wakeup handles it).
  Owner also runs an unrelated Optuna HPO (PID 5742) — not ours.
- **Shadow tally:** day 3 expected clean tomorrow (TSO ffill fix live).
- **Cron:** next run 05:30 UTC 2026-07-17 — PR #1 fix is on main. ✓
- Worktree `phase2-price-lear`, branch `worktree-phase2-price-lear`.
  data/processed + data/raw symlinked to main checkout (data is
  gitignored; .gitkeep deletions NOT committed).

## Next steps (M7 continues)

1. Wind/solar generation FORECASTS from ENTSO-E (day-ahead, not actuals)
   — price driver #1, legal known-future covariate.
2. LightGBM quantile price model on fundamentals + SHAP.
   Should also fix LEAR's band under-coverage (73.4% vs 80% nominal).
3. Spike-tail evaluation (MAE on top-5% hours, P90 coverage there).
4. Daily price forecast into the ops loop (config flag, shadow mode
   like the load challenger).
5. Blog post: owner writes from `blog_post_outline.md`.

## Watch out for

- LEAR coverage 73.4% is below nominal — the static residual band is
  too narrow in spike months. Known, documented, LGBM-quantile is the fix.
- `price_da.parquet` (PSE, PLN) vs `price_da_eur.parquet` (ENTSO-E, EUR):
  do NOT cross-check without an FX series. EUR is the modeling target.
- Parquet index column is `__index_level_0__` in DuckDB — the notebook
  aliases it; new queries must too.
- Full-history feature assembly takes ~2 min; LEAR 2-yr backtest ~6 min.
