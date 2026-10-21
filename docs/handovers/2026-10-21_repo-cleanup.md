# Handover — 2026-10-21 — repo cleanup

## What I did

- Checked every tracked file for duplicates and orphans.
- Checked by content hash, by name, and with grep for references.
- Removed 7 files. Nothing else changed in code or numbers.

| Removed file | Why |
|:--|:--|
| `campaign_status.sh` | Polled a hard-coded PID from 2026-07-22. Dead. |
| `run_heavy_queue.sh` | One-shot queue. Waited for the July 2-year backtest. Done long ago. |
| `docs/notes/model_selection/src/viz/__init__.py` | Empty stray file. Nothing imports it. |
| `reports/backtests/2026-07-22_price_conformal_summary.csv` | Byte-identical to the cited `2026-07-17_` file. |
| `reports/backtests/2026-07-22_price_conformal_summary.md` | Same as the `2026-07-17_` md. Only the title differed. |
| `reports/backtests/2026-07-17_price_diagnostics.csv` | Byte-identical to the cited `2026-07-16_` file. |
| `reports/sensitivity/patchtst/preds_cap730_d64_s42.parquet` | Byte-identical to `preds_full_w730_s42.parquet`. Same config and seed. Score stays in `capacity730.csv`. |

- Added a DECISIONS entry for the cleanup rule.

## State of things

- Tests: 197 + 4 passed. Lint clean. Doc-number check: 135/135.
- Branch was reset to local `main` first. The old base was a stale July commit.

## Decisions made

- Delete only exact duplicates and dead scripts. Dated reruns stay.
- Reason: docs cite older runs as history ("the 07-22 run", "the 07-24 run").

## Next steps (owner decision)

1. Superseded dated reruns in `reports/backtests/` (07-23/07-24 Chronos, TimesFM, conformal, res, spike). Keep as audit trail, or delete and rely on git?
2. `run_deep_history_campaign.sh` at repo root. Move to `scripts/`? Its `cd "$(dirname "$0")"` must change if moved.
3. `docs/templates/DAILY_NOTE_TEMPLATE.md` is unused. No daily notes exist.
4. One-off campaign runners in `src/models/deep/run_*.py`. They are the provenance of committed results. I kept them all.
5. Config is already lean: one `config.yaml` plus a generated `price_conformal.json`.

## Watch out for

- Local `main` and `origin/main` have diverged. Origin has ~100 bot report commits. Local has ~230 work commits. Reconcile before the next push to main.
