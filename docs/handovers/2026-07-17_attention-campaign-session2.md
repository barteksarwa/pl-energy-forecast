# Handover — 2026-07-17 — Attention campaign, session 2

## What happened this session

- Read prior handover (2026-07-17_attention-campaign.md) from the branch.
- Read HPO progress: 14 trials done, best val 0.1184, best config below.
- Prepared walk-forward harness for HPO winner: `src/models/deep/run_tft_hpo_walkforward.py`.
- Prepared PatchTST sweep script: `src/models/deep/run_patchtst_sweep.py`.
- Wrote learning note `14_attention_and_patching.tex` (attention, VSN, patching, interview line).

## HPO state at handover

```
Trials complete: 14 / 60 (1 running, ~4 hours remaining on MPS)
Best trial: 8 | val pinball 0.1184
Best params: encoder_hours=2016, d_model=64, n_heads=8, lstm_layers=2,
             dropout=0.052, lr=0.000462, batch=16
Emerging pattern: 2016h context wins; n_heads=8 dominates top-5.
Log: ~/.claude/jobs/f9ff997b/tmp/logs/tft_hpo.log
DB:  data/processed/tft_hpo.db (resume-safe)
```

## When HPO finishes

1. HPO script auto-exports: VSN weights → `reports/sensitivity/tft_vsn_weights.csv`
   and final trial table → `reports/backtests/YYYY-MM-DD_tft_hpo.md`.
2. Run the walk-forward (needs MPS — wait for HPO to complete):
   ```
   uv run python -m src.models.deep.run_tft_hpo_walkforward
   ```
   Runtime: ~3-6 h (3 seeds × 2yr test, monthly refits).
   Output: `reports/backtests/YYYY-MM-DD_tft_hpo_walkforward.md` with verdict.
3. Read the verdict. Two paths:
   - **TFT beats LEAR** (MAE < 18.24): write model card, open shadow gate.
   - **TFT trails LEAR**: write `08_tft_hpo_verdict.tex` (model selection).
     Honest loss beats a hidden win for the portfolio.
4. Run PatchTST sweep (MPS, after walk-forward finishes):
   ```
   uv run python -m src.models.deep.run_patchtst_sweep [--walkforward]
   ```
   Runtime: ~2-4 h (27 configs screening, optionally walk-forward top-3).
5. VSN analysis: load `reports/sensitivity/tft_vsn_weights.csv`, compare
   vs SHAP feature order from `reports/sensitivity/` and group ablation.
   Write three-column comparison table for the portfolio.

## shadow tallies

- Load: 0 consecutive valid days. First expected 2026-07-18 (cron fix live).
- Price: 0 consecutive valid days. First expected 2026-07-18 (same fix).
- Check `docs/shadow_tally.md` and `docs/shadow_tally_price.md`.
- After 2026-07-18 cron runs: update tallies from `reports/daily/2026-07-18.md`.

## Do NOT touch while HPO runs

- One MPS job at a time. HPO holds MPS until it finishes.
- walk-forward and PatchTST wait for HPO.

## Files created this session

- `src/models/deep/run_tft_hpo_walkforward.py` — reads HPO best, runs 3-seed walk-forward
- `src/models/deep/run_patchtst_sweep.py` — 27-config PatchTST screening + optional walk-forward
- `docs/notes/learning/14_attention_and_patching.tex` — attention/VSN/patching explainer
