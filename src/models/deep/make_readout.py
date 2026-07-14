"""Collate every overnight result into one morning readout."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.config import REPO_ROOT, load_config
from src.evaluation.metrics import mae, mape, pinball_loss

OUT = REPO_ROOT / "reports" / "backtests"


def csv_block(path: Path, title: str, cols: list[str]) -> list[str]:
    if not path.exists():
        return [f"## {title}", "", "_missing — stage failed, see overnight.log_", ""]
    df = pd.read_csv(path)
    keep = [c for c in cols if c in df.columns]
    g = df.groupby([c for c in ["variant", "hidden"] if c in df.columns])[
        [c for c in ["test_mape", "test_pinball_p50", "n_params"] if c in df.columns]
    ].mean().round(3).reset_index()
    return [f"## {title}", "", g.to_markdown(index=False), ""]


def main() -> int:
    cfg = load_config()
    o = REPO_ROOT / "outputs"
    lines = ["# Overnight readout — 2026-07-15", ""]

    # Walk-forward table(s) produced tonight.
    for f in sorted(OUT.glob("2026-07-15_*_summary.md")):
        lines += [f"## {f.stem}", "", f.read_text(), ""]

    # Deep walk-forward preds → score directly.
    load = pd.read_parquet(cfg.paths["data_processed"] / "load.parquet").iloc[:, 0]
    for p in sorted((cfg.paths["data_processed"] / "backtest_preds_fcst").glob("lstm_*.parquet")):
        preds = pd.read_parquet(p)
        y = load.reindex(preds.index)
        lines += [
            f"## deep walk-forward: {p.stem}", "",
            f"MAPE {mape(y, preds['p50']):.2f}% | MAE {mae(y, preds['p50']):.0f} MW | "
            f"pinball {pinball_loss(y, preds['p10'], .1):.1f}/"
            f"{pinball_loss(y, preds['p50'], .5):.1f}/"
            f"{pinball_loss(y, preds['p90'], .9):.1f}", "",
        ]

    lines += csv_block(o / "deep_campaign_v3.csv", "LSTM ladder (v3, screening)", [])
    lines += csv_block(o / "deep_campaign_v4_tso.csv", "Nets + TSO covariate (screening)", [])
    lines += csv_block(o / "deep_campaign_v4_aug.csv", "Origin augmentation (screening)", [])
    lines += csv_block(o / "deep_campaign_v2.csv", "Capacity axis incl. h512 (screening)", [])

    out = OUT / "2026-07-15_overnight_readout.md"
    out.write_text("\n".join(lines))
    print(f"readout: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
