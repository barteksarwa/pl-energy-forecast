"""Score ens4 and ens3 on the SAME window — the ens4 intersection.

The 4-member blend only exists where all members overlap (the TFT
window ends 2026-07-14, ~17.5k hours). Its gate comparisons and P&L
capture must come from that one window, scored against ens3 on the
same hours. The main P&L runner cannot host this: it intersects every
model's index globally, so adding ens4 there would silently shrink
every published number (two-window discipline).

This runner emits the artifact behind the ens4 rows in RESULTS /
BENCHMARK / README: point metrics, band metrics, and P&L capture for
ens4, ens3 and naive on the identical hour set.

Run: python -m src.evaluation.run_ens4_window_metrics
"""

from __future__ import annotations

import sys

import pandas as pd

from src.config import load_config
from src.evaluation.backtest import BacktestResult
from src.evaluation.pnl import Battery, daily_pnl, summarize_pnl
from src.evaluation.run_price_backtest import summarize_price

PREDS = {
    "ens4_tft": "backtest_preds_price_res/ens_crps_cqr_tft.parquet",
    "ens3": "backtest_preds_price_res/ens_crps_cqr.parquet",
    "price_naive_yesterday": "backtest_preds_price_res/price_naive_yesterday.parquet",
}


def main() -> int:
    cfg = load_config()
    proc = cfg.paths["data_processed"]
    y = pd.read_parquet(proc / "price_da_eur.parquet").iloc[:, 0]

    preds = {k: pd.read_parquet(proc / p) for k, p in PREDS.items()}
    window = preds["ens4_tft"].dropna().index
    preds = {k: p.reindex(window).dropna() for k, p in preds.items()}

    results = [BacktestResult(k, p) for k, p in preds.items()]
    table = summarize_price(results, y)

    battery = Battery()
    captures = {}
    for k, p in preds.items():
        daily = daily_pnl(p["p50"], y.reindex(p.index), battery)
        row = summarize_pnl(daily)
        perfect = summarize_pnl(daily.assign(pnl_eur=daily["perfect_eur"]))
        captures[k] = {
            "eur_per_day": row["eur_per_day"],
            "capture_rate": row["capture_rate"],
            "perfect_eur_per_day": perfect["eur_per_day"],
            "n_days": row["n_days"],
        }
    table = table.join(pd.DataFrame(captures).T)

    out_dir = proc.parent.parent / "reports" / "backtests"
    stamp = f"{pd.Timestamp.now(cfg.timezone_local).date()}_ens4_window_metrics"
    table.to_csv(out_dir / f"{stamp}.csv")
    md = [
        f"# ens4 window metrics — {stamp}",
        "",
        "ens4 and ens3 scored on the identical hour set (the 4-member",
        "intersection; the TFT member's window ends 2026-07-14). This is",
        "the artifact behind the ens4 gate rows and its P&L capture.",
        "coverage_80_pct: nominal 80. Capture = P&L / perfect foresight,",
        "same days for every row.",
        "",
        table.round(3).to_markdown(),
        "",
    ]
    (out_dir / f"{stamp}.md").write_text("\n".join(md))
    print(table.round(3).to_string())
    print(f"Written to {out_dir}/{stamp}.(csv|md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
