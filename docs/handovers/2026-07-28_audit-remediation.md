# Handover — 2026-07-28 — cutoff incident + audit remediation

## What happened

1. **Incident.** The E2 fix (09:00 cutoff) broke the FM wrappers.
   Context ended 08:00 D-1; forecasts landed 15 hours off. Chronos
   scored MAE 55.9 vs 21.9 true. Caught before publication. Chain
   killed twice (another session relaunched it once).
2. **Scope correction.** DA prices for D-1 clear at auction on D-2.
   Full D-1 curve is public at 09:00 D-1. E2 applies to LOAD only.
   Engine now takes `target_availability` ("realtime" | "day_ahead").
   Wrappers align by timestamp (`forecast_span`). All tested.
3. **Full corrected rerun.** All numbers reproduce. New canonical
   (17,720 h): ens4 16.88 / champion 17.83 / LEAR 18.46 / chronos
   21.82 / spike AUC 0.967 / P&L capture 0.928. RESULTS, BENCHMARK,
   README updated. Figures regenerated from stored preds.
4. **Job-readiness review** (Opus, hiring-manager persona). Verdict:
   would interview, not yet hire on repo alone. Full text:
   `docs/notes/2026-07-27_job_readiness_review.md`. Most findings
   fixed same session (below).

## Audit findings fixed

- Claims reconciled: naive-in-prod disclosed in README; LEAR labeled
  simplified with a deviations list; cross-border claim deleted;
  TTF/EUA provenance fixed; audit reframed as adversarial LLM audit.
- Every significance claim now has a CSV artifact
  (`2026-07-28_stats_tests_ens_dm.csv`, `_fm_dm.csv`). The orphaned
  p=2.3e-09 reproduced at 2.6e-09. ens4 gate restated on one window.
- LGBM-vs-LEAR: NOW significant (p=1.9e-03) on the corrected run.
  History (p=0.056, "matches not beats") kept on purpose.
- CI runs pytest + ruff on every push. `make lint` passes (43 fixed).
- Leakage test hardened (daily refits, mid-test corruption).
  `redact()` tested. Conformal config repo-anchored (cwd bug).
- Daily report drivers are real now: ridge challenger |coef×z|, LEAR
  per-hour attribution, shared plain-words vocab. Template deleted.
- Open-Meteo retries (challenger stalled 07-17→27 on one timeout).
- Model cards: production-status lines; new cards for seasonal naive
  and spike classifier (the two models that actually ship).
- `lear_full` variant: D-2/3/7 day-vectors, one-hot calendar,
  LassoLarsIC. Benchmark running at handover time.
- 15-min MTU stored at ingest (`*_15min.parquet`) — accrues from
  today; historical refetch is a one-off backlog item.
- Cron turns red if the report is missing/hollow (commit-first).
- Engine logs skipped days + NaN row loss. Ablation script reuses the
  engine (fork deleted). One canonical fut-tensor column list.

## Open / owner decisions

- **Branch `worktree-price-cutoff-revert` holds everything.** No PR
  possible (public main shares no history — repo split). Merge path
  is the owner's reconciliation decision (PLAN M11).
- lear_full + RES-drop ablation results: check
  `logs/lear_full_ablation_2026-07-28.log`, then write verdicts.
- Load tables still carry the E2 caveat until a load rerun.
- shadow_tally stale since 07-21 — score the bot days, resume streak.
- 15-min historical refetch; imbalance model (top job-market gap).
- Doc-tables-from-artifacts script + CI diff (review action 2) — not
  built yet; highest remaining control gap.
- PLAN v4 fully consumed. New PLAN needs owner.

## Where things are

- Corrected preds + conformal offsets copied to main checkout
  (`data/processed/backtest_preds_*`, `config/price_conformal.json`).
- Backups of pre-incident artifacts: `data/backup_20260727_prechain/`.
- Full review + remediation trail: VALIDATION.md amendment, DECISIONS
  2026-07-27, this handover.

## Addendum — merge + agent round (later 2026-07-28)

- Branch MERGED into local main (owner call). Both sessions had built
  the same E2 fix; branch side kept, API unified to
  `target_availability`, parallel session's day-D guarantee ported as
  a corruption test. Public main untouched (no common history).
- Doc-number control shipped: scripts/check_doc_numbers.py, 135
  checks, CI-enforced. Caught 5 drifts on arrival + 2 more of ours
  within minutes. The audit's top control gap is closed.
- ens4 window artifact (`2026-07-28_ens4_window_metrics.csv`): real
  capture 0.931; the quoted 0.928 was copied across rows. ens3 on the
  intersection: 17.35 / Winkler 85.5 / capture 0.927.
- lear_full: faithful LEAR LOSES (19.18 vs 18.51). Simplification now
  defended by a measurement.
- Group ablation rerun: RES look-ahead bounded at +3.39 MAE.
- Imbalance v1: |bal−DA| ≈ 151 PLN/MWh, solar-hour peaked, sign a
  coin flip, our errors uncorrelated (0.04). Next model must predict
  spread SIGN.
- Shadow tally resumed from origin/main bot reports. Load challenger
  dead since 07-18 on the same Open-Meteo timeout (retry fix in this
  branch unblocks it). Price shadows: 6 valid days, LGBM ahead.
- Load rerun done: 2.16% vs TSO 2.25% under the corrected cutoff.
  E2 closed on BOTH sides; protocol note in RESULTS closed.
- NEW follow-up: coverage accounting flagged up to 4,502 training
  rows/refit dropped by NaN filtering in the load matrix
  (weather-archive holes). Quantify and patch the archive.
- Public-repo exposure: pushing this branch put the full local
  history on the public remote as a branch. Owner: delete the remote
  branch or fold it into the reconciliation decision.
