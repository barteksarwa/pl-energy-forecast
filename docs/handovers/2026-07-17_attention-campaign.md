# Handover — 2026-07-17 — attention campaign kickoff (NEW-SESSION ENTRY POINT)

Read this + docs/PLAN.md (Backlog + Phase 2.5/3 sections). Prior
handover (2026-07-16_phase2-price-lear.md) is history.

## State of the world, short

- Load: ridge+TSO 2.08% beats TSO 2.23% (2yr). Price: LGBM 17.8 / LEAR
  18.5 vs naive 28.0 MAE; conformal bands ~79%. LEAR publishes daily,
  LGBM+conformal is the shadow challenger (tally: shadow_tally_price.md).
- Daily cron: data-store cache + ENTSOE token fixed 2026-07-17
  (PR #3/#4/#5). First fully-complete cron report expected 2026-07-18.
  A validation dispatch was warming the cache at handover — check
  `gh run list` first thing.
- TFT long-context screening: helps monotonically, loses to tabular by
  ~30% same-window. Note: model_selection/07. Big bug fixed on the way:
  prediction-time covariate standardization (apply_covariate_stats).

## Running at handover (check before starting anything on MPS)

1. **TFT HPO** (MPS, caffeinate, PID in logs/hpo_batch.log):
   60 Optuna trials, price task, context length in the search space.
   Log: ~/.claude/jobs/f9ff997b/tmp/logs/tft_hpo.log
   Study: data/processed/tft_hpo.db (resume-safe — rerun the same
   command to add trials). Ends with VSN weight export to
   reports/sensitivity/tft_vsn_weights.csv + trials CSV in
   reports/backtests/. NOTE: 2 CPU smoke trials are in the study —
   ignore trial 0-1 in readouts (marked by tiny runtimes).
2. **Price backtest, outages+fuel** (CPU): tag res_out_fuel.
   Log: ~/.claude/jobs/f9ff997b/tmp/price_backtest_fuel.log
   Read: does fuel move MAE / winter bias / spike MAE vs the
   res_out table (outages alone were FLAT).

## The attention campaign (owner priority, model freeze lifted)

Owner hypothesis: attention models have the biggest potential here.
Test it properly, stage by stage; each verdict honest:

1. Read HPO results. Confirm the winner with WALK-FORWARD (screening
   flatters nets 0.6-0.9pp — never quote the split number).
   `run_tft_price.py` walk-forward harness works; parameterize it with
   the best config.
2. VSN weights vs SHAP vs ablation — three importance methods, one
   comparison table. Learning note material (13 covers SHAP-vs-ablation;
   VSN is the in-model third angle).
3. PatchTST screening: src/models/deep/patchtst.py is built and
   smoke-tested (46k params, trains, sane preds). Sweep patch_len
   {12,24,48} x stride {6,12,24} x ctx {672,1344,2016}, single split,
   then walk-forward the winner. Channel independence is justified by
   the data: reports/sensitivity/channels_verdict.txt (1/21 pairs
   |corr|>0.5).
4. Same-window comparison vs LEAR/LGBM (pattern:
   2026-07-17_tft_price_same_window.csv). Spike MAE separately.
5. If any net wins: 3-seed confirm, model card, shadow gate. If none
   wins: the verdict note explains WHY (that is worth as much).

## Also open

- Fuel verdict -> if useful, wire into daily price step + conformal
  offsets refresh (run_price_calibration).
- Shadow tallies: first valid cron day expected 2026-07-18 (both load
  challenger and price). Update tallies from the morning report.
- Blog post: owner writes (outline in docs/notes/blog_post_outline.md).

## Watch out

- One MPS trainer at a time (HPO is holding it).
- uv run in the worktree works (worktree has its own synced .venv).
- data/processed + data/raw in the worktree are SYMLINKS to the main
  checkout — do not git-checkout the .gitkeep files (breaks symlinks;
  skip-worktree is set).
- Never quote screening numbers as results. Same-window or nothing.
