"""Conformal calibration runner: table + the daily-loop offset artifact.

1. Applies rolling conformal calibration to the stored walk-forward
   predictions of both price models; writes the comparison table.
2. Writes config/price_conformal.json — the band-widening offsets the
   daily loop applies to fresh forecasts. Offsets come from the trailing
   90 days of out-of-sample errors, so they are exactly what the desk
   knew at the end of the backtest.

Rerun after every new price backtest, or monthly, whichever comes first
(the offset drifts with volatility regimes).

Run: python -m src.evaluation.run_price_calibration
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from src.config import load_config
from src.evaluation.backtest import BacktestResult
from src.evaluation.conformal import latest_offset, rolling_conformal
from src.evaluation.run_price_backtest import summarize_price

CALIBRATED = ("lear", "lgbm_quantile")
ALL_MODELS = ("price_naive_yesterday", "price_naive_week") + CALIBRATED


def main() -> int:
    cfg = load_config()
    proc = cfg.paths["data_processed"]
    preds_dir = proc / "backtest_preds_price_res"
    y_full = pd.read_parquet(proc / "price_da_eur.parquet").iloc[:, 0]

    results, offsets = [], {}
    for name in ALL_MODELS:
        p = pd.read_parquet(preds_dir / f"{name}.parquet")
        results.append(BacktestResult(model_name=name, predictions=p))
        if name in CALIBRATED:
            adj = rolling_conformal(p, y_full)
            adj.to_parquet(preds_dir / f"{name}_conformal.parquet")
            results.append(
                BacktestResult(model_name=f"{name}_conformal", predictions=adj)
            )
            offsets[name] = round(latest_offset(p, y_full), 3)

    y = y_full.reindex(results[0].predictions.index)
    table = summarize_price(results, y)

    out_dir = proc.parent.parent / "reports" / "backtests"
    tz = cfg.timezone_local
    stamp = f"{pd.Timestamp.now(tz).date()}_price_conformal"
    table.to_csv(out_dir / f"{stamp}_summary.csv")
    md = [
        f"# Price backtest — conformal band calibration — {stamp}",
        "",
        "Rolling split-conformal (CQR, 90-day trailing window of",
        "out-of-sample errors, walk-forward honest). P50 untouched — only",
        "the band moves. First 30 days keep the raw band.",
        "",
        table.round(3).to_markdown(),
        "",
    ]
    (out_dir / f"{stamp}_summary.md").write_text("\n".join(md))

    offsets["_meta"] = {
        "method": "rolling split-conformal (CQR), 90d trailing window, coverage 0.8",
        "source": "backtest_preds_price_res, walk-forward out-of-sample errors",
        "window_end": str(y_full.index.max().date()),
        "refresh": "rerun src.evaluation.run_price_calibration after each backtest / monthly",
    }
    with open("config/price_conformal.json", "w") as f:
        json.dump(offsets, f, indent=2)

    print(table.round(3).to_string())
    print(f"\noffsets: { {k: v for k, v in offsets.items() if k != '_meta'} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
