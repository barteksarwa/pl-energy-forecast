# Handover — 2026-07-26/27: night campaign (owner-authorized)

## Headline

**New best price forecast: 4-member blend (LGBM+LEAR+Chronos+TFT),
MAE 16.89, rMAE 0.605, coverage 80.0%, Winkler 82.6, P&L capture
0.929.** DM p=2.3e-09 vs the 3-member blend; wins every test year.
TFT's full-2yr preds had survived the wipe under `reports/` all along
— the archived-solo model is the best diversity donor tested.

## What ran

1. **HPO campaign** (previous session): defaults survived. Done.
2. **Ensemble/CQR sweeps** (stored preds, pre-declared gates):
   weight window 60d stays; TimesFM 4th member REJECTED (+0.57 —
   FM errors correlate); CQR window 90d stays (blend-branch sweep
   later flagged methodologically unreliable — see report note).
3. **Spike screen regenerated**: AUC 0.966 reproduces. (Fixed a
   missing-mkdir crash first.)
4. **TFT "regen"**: no-op — preds were never lost. Incident recovery
   was already 100%.
5. **Validation workflow** (multi-agent MRM review, 85 agents):
   31 confirmed findings — full report `docs/VALIDATION.md`.
   - 24 documentation defects (stale cross-doc numbers after the
     regeneration, transcription slips, 3 significance claims with
     no artifact) — ALL FIXED; FM DM artifact now exists
     (`2026-07-27_stats_tests_fm_dm.csv`).
   - 2 real protocol bugs — FIXED with regression tests:
     E1 D-1 price vector took the D-2 shape after spring DST;
     E2 training mask saw D-1 hours past the 09:00 decision moment.
     Impact bound: corrected champion 17.95 vs 17.84 (~0.11 shared
     flattery; rankings stand; production loop never affected).
   - Open (P3, minor): conformal interpolation method, asymmetric
     tail sizing, LEAR raw-band labeling.
6. **Web-research workflow**: curated resource note
   `docs/notes/resources_epf_2026-07.md` — videos, courses, papers
   with one adoptable idea each, PL/EU syllabus, skills gap vs
   live postings. Top ideas: BOA online weights, interval methods
   scored by battery P&L, JSU head, Chronos-2 ARX rerun.

## Owner decisions pending (updated)

1. **Promotion target changed:** ens4 (with TFT) is now the
   candidate — cost question is TFT inference in the daily loop
   (3 seeds, MPS, monthly refits).
2. 1095d window for the solo champion (unchanged, solid).
3. Re-run all price tables under the corrected cutoff protocol
   (compute-days; the ~0.11 bound is measured and documented).
4. Public README softening (unchanged).
5. New PLAN: validation P3 items, BOA weights, resource-note ideas.

## Gotchas

- Corrected cutoff = `x.index < D-1 09:00 local` in
  `walk_forward_backtest` — any new backtest is automatically on the
  new protocol; numbers will read ~0.1 worse than old tables. That
  is correct, not a regression.
- TFT blend evaluated on the 17,456h intersection (TFT window ends
  07-14).
- Moirai scratch venv: `~/Documents/moirai_scratch/moirai_env`.
