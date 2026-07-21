"""Cross-check PSE vs ENTSO-E load, then extend canonical history.

Two independent routes to the same physical series. If they disagree beyond
rounding, something is wrong with a pipeline — ours or theirs — and we want
to know before a model does.

Merge rule (DECISIONS 2026-07-14/16): PSE canonical where both exist;
ENTSO-E fills the pre-2024-06-14 history. Backup written before overwrite.

Run: python -m src.ingestion.crosscheck
"""

from __future__ import annotations

import sys

import pandas as pd

from src.config import load_config
from src.ingestion.gaps import log_gaps


def compare(a: pd.Series, b: pd.Series, name: str) -> dict:
    df = pd.concat([a.rename("pse"), b.rename("entsoe")], axis=1).dropna()
    diff = (df["pse"] - df["entsoe"]).abs()
    rel = diff / df["pse"].abs().clip(lower=1.0)
    out = {
        "series": name,
        "overlap_hours": len(df),
        "mean_abs_diff_mw": round(float(diff.mean()), 1),
        "p99_abs_diff_mw": round(float(diff.quantile(0.99)), 1),
        "hours_over_1pct": int((rel > 0.01).sum()),
    }
    print(out, flush=True)
    return out


def main() -> int:
    cfg = load_config()
    proc = cfg.paths["data_processed"]
    entsoe_dir = proc / "entsoe"
    if not (entsoe_dir / "load.parquet").exists():
        print("No ENTSO-E store. Run: python -m src.ingestion.backfill --only entsoe")
        return 1

    pse_load = pd.read_parquet(proc / "load.parquet").iloc[:, 0]
    ent_load = pd.read_parquet(entsoe_dir / "load.parquet").iloc[:, 0]
    pse_tso = pd.read_parquet(proc / "tso_forecast.parquet").iloc[:, 0]
    ent_tso = pd.read_parquet(entsoe_dir / "tso_forecast.parquet").iloc[:, 0]

    rows = [compare(pse_load, ent_load, "load"),
            compare(pse_tso, ent_tso, "tso_forecast")]
    report = pd.DataFrame(rows)
    out_csv = proc.parent.parent / "reports" / "backtests" / "pse_vs_entsoe.csv"
    report.to_csv(out_csv, index=False)

    # Extend canonical: ENTSO-E where PSE has nothing (the deep history).
    for stem, pse_s, ent_s in [("load", pse_load, ent_load),
                               ("tso_forecast", pse_tso, ent_tso)]:
        path = proc / f"{stem}.parquet"
        backup = proc / f"{stem}_pse_only.parquet"
        if not backup.exists():
            pse_s.to_frame().to_parquet(backup)
        merged = pse_s.combine_first(ent_s).sort_index()
        added = len(merged) - len(pse_s)
        merged.to_frame().to_parquet(path)
        log_gaps(merged, f"merged_{stem}", proc / "gap_log.csv")
        print(f"{stem}: canonical extended by {added} hours "
              f"({merged.index.min().date()} → {merged.index.max().date()})",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
