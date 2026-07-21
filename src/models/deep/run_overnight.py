"""Overnight queue 2026-07-15. Runs unattended ~11h; answers three questions:

Q1  Does the TSO forecast as a feature beat everything? (forecast combination)
Q2  Does origin augmentation (more samples from same data) help the nets?
Q3  Confirm the capacity ceiling (h512) and tune LightGBM a little.

Stages run sequentially; each is guarded — one failure skips, night continues.
Everything appends to CSVs; readout written at the end + committed to git.

Run: nohup caffeinate -i uv run python -u -m src.models.deep.run_overnight \
       > outputs/logs/overnight.log 2>&1 &
"""

from __future__ import annotations

import subprocess
import sys
import time

import pandas as pd

from src.config import REPO_ROOT

LOG = lambda m: print(f"[{pd.Timestamp.now()}] {m}", flush=True)  # noqa: E731


def run(cmd: list[str], desc: str) -> bool:
    LOG(f"START {desc}: {' '.join(cmd)}")
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT, timeout=4 * 3600)
        LOG(f"DONE  {desc} in {(time.time() - t0) / 60:.1f} min")
        return True
    except Exception as e:  # noqa: BLE001 — the night must go on
        LOG(f"FAIL  {desc}: {e}")
        return False


def wait_for_ladder() -> None:
    while subprocess.run(["pgrep", "-f", "run_ladder"], capture_output=True).returncode == 0:
        LOG("waiting for v3 ladder to finish ...")
        time.sleep(120)


def main() -> int:
    uv = ["uv", "run", "python", "-u", "-m"]
    wait_for_ladder()

    # Q1 — TSO forecast as feature. The big bet.
    run(uv + ["src.evaluation.run_backtest", "--weather", "forecast", "--with-tso",
              "--models", "seasonal_naive,ridge,lgbm_quantile"],
        "walk-forward with TSO feature (lgbm + ridge)")

    # Q1 for nets — screening with TSO covariate.
    run(uv + ["src.models.deep.run_screen_tso"], "nets + TSO screening")

    # Q2 — origin augmentation.
    run(uv + ["src.models.deep.run_screen_aug"], "origin augmentation screening")

    # Q3a — capacity close-out at h512.
    run(uv + ["src.models.deep.run_h512"], "enc_dec h512 close-out")

    # README-grade: deep walk-forward, with and without TSO.
    run(uv + ["src.models.deep.run_walkforward", "--variant", "enc_dec",
              "--hidden", "64"], "deep walk-forward enc_dec h64")
    run(uv + ["src.models.deep.run_walkforward", "--variant", "enc_dec",
              "--hidden", "64", "--with-tso"], "deep walk-forward enc_dec h64 + TSO")

    # Q3b — LightGBM mini-tuning (2 configs beyond default).
    run(uv + ["src.evaluation.run_lgbm_tune"], "lgbm mini-tuning")

    # Readout + commit.
    run(uv + ["src.models.deep.make_readout"], "overnight readout")
    subprocess.run(["git", "add", "-A"], cwd=REPO_ROOT)
    subprocess.run(["git", "commit", "-q", "-m",
                    "results: overnight campaign — TSO feature, augmentation, "
                    "capacity close-out, deep walk-forward, lgbm tuning"],
                   cwd=REPO_ROOT)
    subprocess.run(["git", "push", "-q"], cwd=REPO_ROOT)
    LOG("NIGHT COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
