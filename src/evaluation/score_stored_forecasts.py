"""Retro-score stored daily forecasts against actuals.

Use after an ops outage: forecasts were produced and committed on time,
but the next-morning scoring run never happened. Scoring later is
leakage-free — the forecast files predate the actuals.

Scores whatever exists in data/forecasts/ for the given date:
- load:  {date}.csv (incumbent), {date}_challenger.csv
- price: price_{date}.csv (LEAR), price_{date}_challenger.csv (LGBM)

Prints MAE (+ MAPE for load) vs actuals, plus naive and TSO references.
Run: python -m src.evaluation.score_stored_forecasts --date 2026-07-18
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.config import load_config
from src.evaluation.metrics import mae, mape


def _read_forecast(path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["time_utc"], index_col="time_utc")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="target day, YYYY-MM-DD")
    args = parser.parse_args()

    cfg = load_config()
    proc = cfg.paths["data_processed"]
    fdir = cfg.paths["forecasts"]
    day = args.date

    load_act = pd.read_parquet(proc / "load.parquet").iloc[:, 0]
    price_act = pd.read_parquet(proc / "price_da_eur.parquet").iloc[:, 0]
    tso = pd.read_parquet(proc / "tso_forecast.parquet").iloc[:, 0]

    print(f"=== Retro-score for {day} ===")
    any_scored = False

    for label, fname, actual in [
        ("load incumbent", f"{day}.csv", load_act),
        ("load challenger", f"{day}_challenger.csv", load_act),
        ("price LEAR", f"price_{day}.csv", price_act),
        ("price LGBM (shadow)", f"price_{day}_challenger.csv", price_act),
    ]:
        fc = _read_forecast(fdir / fname)
        if fc is None:
            print(f"{label:20s} no file ({fname})")
            continue
        y = actual.reindex(fc.index)
        if y.isna().all():
            print(f"{label:20s} actuals not available yet for these hours")
            continue
        ok = y.notna()
        m = mae(y[ok], fc.loc[ok, "p50"])
        line = f"{label:20s} MAE {m:8.2f}"
        if label.startswith("load"):
            line += f"  MAPE {mape(y[ok], fc.loc[ok, 'p50']):5.2f}%"
            t = tso.reindex(fc.index)[ok]
            if t.notna().any():
                line += f"  (TSO MAPE {mape(y[ok][t.notna()], t[t.notna()]):5.2f}%)"
        else:
            naive = actual.reindex(fc.index - pd.Timedelta(hours=24))
            naive.index = fc.index
            if naive[ok].notna().any():
                line += f"  (naive-1d MAE {mae(y[ok], naive[ok]):8.2f})"
        cover = ((y[ok] >= fc.loc[ok, "p10"]) & (y[ok] <= fc.loc[ok, "p90"])).mean()
        line += f"  cover80 {100 * cover:5.1f}%  n={int(ok.sum())}"
        print(line)
        any_scored = True

    if not any_scored:
        print("Nothing scored. Refresh actuals first (backfill) or check date.")
        return 1
    print("Paste results into docs/shadow_tally.md (mark as retro-scored).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
