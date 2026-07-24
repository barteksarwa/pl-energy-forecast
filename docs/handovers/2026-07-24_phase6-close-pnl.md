# Handover — 2026-07-24: Phase 6 closed + P&L layer (Phase 7 S1-S2)

## Done this session

- **Blend conformalized** (Phase 6 loose end): second CQR pass on the
  CRPS blend. Coverage 84.2% → 79.9%, Winkler 85.2 → 84.7, MAE
  unchanged (17.34). `ens_crps_cqr` is now the promotion candidate.
  Stored: `data/processed/backtest_preds_price_res/ens_crps_cqr.parquet`.
- **Battery P&L engine** (Phase 7 S1): `src/evaluation/pnl.py`.
  Per-day LP (scipy HiGHS), 1 MW / 2 MWh / 0.85 RTE / 1 cycle,
  DST-aware day lengths, 9 accounting tests. DA-only scope stated.
- **P&L table** (Phase 7 S2): `uv run python -m src.evaluation.run_pnl`.
  Ensemble 205 EUR/day (92.4% capture) > LGBM 202 > LEAR 201 >
  Chronos 197 > TimesFM 195 > naive 180 (81.3%). No MAE-rank flips.
  Key story: value compresses — storage needs hour ordering, not level.
- **Notes**: learning 23 (ensembles/CRPS), 24 (FM covariates),
  25 (forecast value/P&L); model_selection 15 (FM verdict),
  16 (ensemble verdict), 17 (P&L verdict).
- RESULTS.md: ensemble table updated + new P&L section.
  DECISIONS.md: two entries.

## Owner decisions pending (unchanged + one new)

1. Promote 1095d training window (evidence complete since 07-22).
2. Soften public README "beats the standard" (LEAR not significant).
3. NEW: promote `ens_crps_cqr` to the daily price loop? Interacts
   with (1) — natural follow-up is rebuilding the blend on 1095d
   members. Cost: Chronos inference daily (`--extra fm`).

## What remains in PLAN Phase 7

- S3-S5: benchmark writeup (blog minimum): protocol, master table
  incl. P&L capture, honest negatives, reproducibility appendix.
  Cards + README + DECISIONS refresh.

## INCIDENT — data/ wiped by a committed symlink (2026-07-24)

What happened, in order:
1. Session ran in a worktree; `data/` was symlinked to the main
   checkout to reach the parquets.
2. `git add -A` committed that symlink. `.gitignore` had `data/**`,
   which matches CONTENTS, not the `data` path itself.
3. Merging into local main made git replace the real `data/`
   directory with the symlink (now self-referencing). Git treats
   gitignored files as disposable — it deleted the parquets.

Damage: all of `data/raw` + `data/processed` (base series AND stored
hourly backtest predictions). NOT lost: reports/, RESULTS.md numbers,
docs, outputs/ (checkpoints, campaign CSVs), tracked forecast CSVs,
git history, the public repo.

Repair done this session:
- History rewritten BEFORE any push: rebuilt the 4 commits without
  data paths; main reset and fast-forwarded to the clean branch.
- `.gitignore` now also ignores the `data` path itself.
- Base-data refetch launched (`make backfill`, idempotent,
  log: `logs/backfill_recovery_2026-07-24.log`).

Regeneration status (END OF SESSION — mostly DONE):
- Base data: RESTORED, incl. deep history. price_da_eur 100,960 h,
  res_forecast 101,279 h, canonical load 2015→2026 (PSE + ENTSO-E via
  crosscheck), weather 10 cities 2015→now, fuel. PSE 500 on one
  csdac chunk was transient; retry passed.
- Backtest preds RESTORED and VERIFIED against RESULTS.md (test
  window now ends 8 days later, so tiny drifts are expected and
  explained): LGBM 17.84 (was 17.87), LEAR 18.46 (18.24),
  Chronos 21.93 (21.98), TimesFM 22.52 (22.52),
  ens_crps_cqr 17.34 / coverage 79.9% (identical),
  P&L ensemble capture 92.6% (was 92.4). All claims hold.
- STILL MISSING (owner call): Moirai preds (scratch venv also gone —
  rebuild venv + overnight; negative result already documented) and
  TFT/deep stored preds (`tft_hpo_ens` etc. — overnight MPS runs;
  only needed for the 1-yr two-window table).

Lesson (added to CLAUDE.md candidates): never symlink into a git
checkout; mount data via config paths instead.

## Gotchas

- Branch `worktree-phase6-close-pnl` is LOCAL ONLY — origin is the
  public curated repo; never push local history there (standing owner
  rule). Republish curated via `~/Documents/repo-reset/`.
- P&L LP has a 1e-6 activity penalty — breaks lossless charge/discharge
  ties toward no-trade. Don't remove it.
- `daily_pnl` skips any local day with missing hours (expected count
  computed from DST calendar, 23/24/25h).
