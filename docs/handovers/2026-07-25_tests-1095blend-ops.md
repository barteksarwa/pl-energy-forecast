# Handover — 2026-07-25: coverage tests, 1095d-blend verdict, ops verify

## Done

- **Test-coverage audit + gap closure.** Three untested critical
  modules now covered (14 new tests, suite at 116):
  - LEAR: z-clip regression (the 38k EUR blowup), transform
    round-trip, DST unseen-hour fallback, column guard, naive bands.
  - Spike classifier: train-only threshold (leakage property),
    determinism, probability sanity.
  - crosscheck: `merge_canonical()` extracted; PSE-wins precedence
    pinned by tests.
  Untested by design (rule 7): run_* orchestration, API wrappers.
- **1095d-member blend: honest negative.** 17.18 vs 17.34 but DM
  p=0.0596 (gate was p<0.05), loses the 2026 slice, P&L capture tie.
  Blend STAYS on 365d members. Finding: window gain and ensemble
  diversity are partial substitutes (details in RESULTS.md).
  This decouples the two pending owner decisions.
- **Ops verified post-incident.** `make dry-run` clean on the rebuilt
  store: fetch, scoring, load+price forecasts, spike line, report
  2026-07-25 committed. Challenger beat TSO on load yesterday.
- **Moirai regenerated and VERIFIED** (scratch venv rebuilt at
  `~/Documents/moirai_scratch/moirai_env`): zs 23.70 / cov 24.87 vs
  documented 23.69 / 24.86 — covariate negative reproduces. The only
  preds still missing are the deep/TFT ones (owner-scheduled).

## Owner decisions pending (all evidence now in)

1. 1095d window for the SOLO champion (DM p=0.0009 — solid).
2. `ens_crps_cqr` promotion to the daily loop (365d members settled).
3. Public README "beats the standard" softening.
4. TFT stored-pred regeneration scheduling (overnight MPS; only
   feeds the 1-yr two-window table).
5. New PLAN — v4 is fully consumed. arXiv writeup is the open stretch.

## Gotchas

- Ensemble runner: `--suffix` keeps variant runs from clobbering the
  shipped blend parquets; 1095d members are conformalized inline —
  the shared calibration script (and its daily-loop offsets file)
  never sees them. Keep it that way.
- `make viz` now skips `*_spike.csv` (they have no band columns).
