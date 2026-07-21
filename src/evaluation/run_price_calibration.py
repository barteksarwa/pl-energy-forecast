"""Conformal calibration runner: table + the daily-loop offset artifact.

1. Applies rolling conformal calibration to the stored walk-forward
   predictions of both price models; writes the comparison table.
2. Writes config/price_conformal.json — the band-widening offsets the
   daily loop applies to fresh forecasts. Offsets come from the trailing
   90 days of out-of-sample errors, so they are exactly what the desk
   knew at the end of the backtest.

Flags:
  --compare-asymmetric   Also run asymmetric CQR and print a comparison
                         table (symmetric vs asymmetric per model). Does
                         not update price_conformal.json.

Rerun after every new price backtest, or monthly, whichever comes first
(the offset drifts with volatility regimes).

Run: python -m src.evaluation.run_price_calibration
     python -m src.evaluation.run_price_calibration --compare-asymmetric
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from src.config import load_config
from src.evaluation.backtest import BacktestResult
from src.evaluation.conformal import (
    latest_offset,
    latest_offset_asymmetric,
    rolling_conformal,
    rolling_conformal_asymmetric,
)
from src.evaluation.run_price_backtest import summarize_price

CALIBRATED = ("lear", "lgbm_quantile")
ALL_MODELS = ("price_naive_yesterday", "price_naive_week") + CALIBRATED


def _coverage(preds: pd.DataFrame, y: pd.Series) -> float:
    y_al = y.reindex(preds.index)
    return float(((y_al >= preds["p10"]) & (y_al <= preds["p90"])).mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-asymmetric", action="store_true")
    args = parser.parse_args()

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

    if args.compare_asymmetric:
        print("\n--- Asymmetric CQR comparison ---")
        rows = []
        for name in CALIBRATED:
            p = pd.read_parquet(preds_dir / f"{name}.parquet")
            sym = rolling_conformal(p, y_full)
            asym = rolling_conformal_asymmetric(p, y_full)
            changed = sym["p90"] != p["p90"]
            q_lo, q_hi = latest_offset_asymmetric(p, y_full)
            rows.append({
                "model": name,
                "raw_cov": round(_coverage(p, y_full), 3),
                "sym_cov": round(_coverage(sym, y_full), 3),
                "asym_cov": round(_coverage(asym, y_full), 3),
                "sym_offset": round(offsets[name], 3),
                "asym_q_lo": round(q_lo, 3),
                "asym_q_hi": round(q_hi, 3),
            })
        cmp = pd.DataFrame(rows).set_index("model")
        print(cmp.to_string())
        cmp_stamp = f"{pd.Timestamp.now(tz).date()}_asym_cqr_comparison"
        cmp.to_csv(out_dir / f"{cmp_stamp}.csv")
        print(f"\nSaved: reports/backtests/{cmp_stamp}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
