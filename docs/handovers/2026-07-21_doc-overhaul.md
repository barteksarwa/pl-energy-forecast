# Handover — 2026-07-21 doc overhaul + repo revival plan

## What happened

Owner asked: analyze repo, answer open questions, clean the markdown,
push the plan to the next phase.

## Found (important)

1. **Local repo had no git remote.** Cron died with it: no daily runs
   2026-07-18 → 07-21. Days 07-19→21 have no forecasts — permanent
   track-record hole, logged honestly in both tallies.
2. **Owner created a new public repo the same day**
   (`barteksarwa/pl-energy-forecast`), curated 7-commit history,
   no common ancestor with local main. NOTHING PUSHED — owner must
   pick a reconciliation path (PLAN.md M11, three options).
3. 2026-07-18 forecasts exist (produced pre-outage). Retroactive
   scoring is legitimate and pending.
4. Pre-existing bug: full `pytest` segfaulted (torch + LightGBM OpenMP
   clash on macOS). Fixed via split test run in Makefile.

## Done

- `docs/RESULTS.md` created — canonical numbers page. Docs link to it.
- PLAN v3: honest status header + Phase 4 (M11 reconcile/restart,
  M12 RES geography — owner priority, M13 close-out).
- Model cards: `tft_price.md` rewritten (730d verdict, decomposition),
  `patchtst_price.md` created.
- `HOW_A_FORECAST_IS_MADE.md`: publication-fallback mechanism explained
  (persist_24h), champion-vs-publisher section, deep challengers added.
- README: full 5-model price table, comparison figure embedded, honest
  outage status, TFT/PatchTST closure.
- New LaTeX note `model_selection/12_deep_gap_decomposition.tex`.
- DATA_CATALOG refreshed (14 datasets, verified on disk).
- Archived: PHASE2_KICKSTART, blog outline, deepar spec, 6 superseded
  backtest summaries → `docs/archive/`, `reports/backtests/archive/`.
- City weather weights → official GUS 2025 (Tabl. 22, 31.12.2024).
  Effect tiny (weather ablation +0.08 pp); no re-benchmark.
- `.env.example` restored. Stale worktree merged + removed.

## Next session

1. Owner picks repo reconciliation (a/b/c in PLAN M11). Then push,
   re-enable cron + `ENTSOE_API_TOKEN` secret.
2. Score 2026-07-18 retroactively; restart shadow window.
3. M12 data hunt: RES capacity by location (URE/ARE, ENTSO-E per-unit,
   OSM turbines).
4. Owner: compile both `main.tex` (PDFs stale since 07-14; note 12 new).

## Numbers stated here

All traced: `docs/RESULTS.md` (sources listed per table).
