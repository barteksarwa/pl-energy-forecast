# Handover — 2026-07-14 — plan → skeleton → real data → first results

One huge day. Keep this page; details live in git log and DECISIONS.md.

## What works now

- `make dry-run` — real daily report from PSE data (report + fan chart done today).
- `make backtest` — walk-forward, 4 baselines + TSO row, writes reports/backtests/.
- `make backfill` — weather actuals (0 gaps), weather forecasts (lead 1–2 d,
  running), PSE load + TSO forecast 2024-06-14→now (0 gaps), ENTSO-E pending token.
- `make viz` — 5 figures. `make test` — 39 green. All pushed to
  github.com/barteksarwa/pl-energy-forecast (private).

## First honest numbers (12-month walk-forward, ERA5-actuals weather)

ridge MAPE 4.00% (skill 0.31) > lasso 4.13% > naive 5.62% > climatology 8.59%.
TSO ≈ 2.6%. Gap to TSO = M4 LightGBM's job. Full table: reports/backtests/.

## Key decisions today (all in DECISIONS.md)

- PSE API v2 = primary load source (keyless). ENTSO-E = deep history + cross-check.
- Weather leakage: backtests use archived lead-2 forecasts (Previous Runs API).
- Load first, price Phase 2. Neighbor holidays Phase 2. LaTeX notes, owner compiles.

## Next steps

1. Rerun backtest with `--weather forecast` once forecast backfill finishes
   (was 6/10 cities at handover time). Compare vs actuals run — quantifies
   the weather-leakage effect. Good DECISIONS/note material.
2. M4: LightGBM quantile + SHAP + model card. Then swap into daily loop.
3. Daily report: embed the fan chart (report says "chart lands in M3").
4. ENTSO-E token arrives → backfill 2023+, cross-check PSE vs ENTSO-E.
5. Learning notes: 06 baselines-reading (what the table says), model card 01.

## Watch out for

- PSE v2 history starts 2024-06-14. Nothing earlier exists on v2.
- sklearn 1.9: LassoCV uses `alphas=int`, not `n_alphas`.
- Previous Runs API: 120-day chunks max, else timeouts.
- `energy-forecast-kickstart/` still awaiting owner delete decision.
