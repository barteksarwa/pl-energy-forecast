"""Battery-arbitrage P&L for every stored price model.

What a forecast is worth in EUR: schedule the PLAN Phase 7 battery
(1 MW / 2 MWh / 0.85 RTE / 1 cycle) on each model's P50, settle at
actual DA prices. Perfect foresight (same LP on actuals) is the
upper bound; the naive-forecast schedule is the floor strategy.

All models are scored on the SAME days (index intersection), so the
table is directly comparable. TFT is excluded (its stored preds
cover a different window — two-window discipline).

Run: uv run python -m src.evaluation.run_pnl
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.pnl import Battery, daily_pnl, summarize_pnl

PROC = Path("data/processed")
OUT_DIR = Path("reports/backtests")
FIG_DIR = Path("reports/figures/pnl")

# (label, path, color, linestyle) — colors follow the entity across
# all repo figures; linestyles are the CVD-safe secondary encoding
MODELS = [
    ("ens_crps_cqr", PROC / "backtest_preds_price_res/ens_crps_cqr.parquet",
     "#9b59b6", "-"),
    ("lgbm", PROC / "backtest_preds_price_res/lgbm_quantile_conformal.parquet",
     "#2ecc71", "-"),
    ("lear", PROC / "backtest_preds_price_res/lear_conformal.parquet",
     "#3498db", "--"),
    ("chronos", PROC / "backtest_preds_price_chronos2yr/chronos_bolt_zs.parquet",
     "#e74c3c", "-."),
    ("timesfm", PROC / "backtest_preds_price_timesfm2yr/timesfm_zs.parquet",
     "#e67e22", "-."),
    ("naive", PROC / "backtest_preds_price_res/price_naive_yesterday.parquet",
     "#95a5a6", ":"),
]


def main() -> int:
    y = pd.read_parquet(PROC / "price_da_eur.parquet").iloc[:, 0]
    battery = Battery()

    preds = {}
    for label, path, _, _ in MODELS:
        if path.exists():
            preds[label] = pd.read_parquet(path)["p50"]
        else:
            print(f"skip {label} (no preds at {path})")

    idx = None
    for p in preds.values():
        idx = p.index if idx is None else idx.intersection(p.index)

    rows, curves = {}, {}
    for label, p in preds.items():
        daily = daily_pnl(p.reindex(idx), y.reindex(idx), battery)
        rows[label] = summarize_pnl(daily)
        curves[label] = daily["pnl_eur"].cumsum()
        if "perfect" not in curves:  # identical for every model (same days)
            curves["perfect"] = daily["perfect_eur"].cumsum()
            rows["perfect"] = summarize_pnl(
                daily.assign(pnl_eur=daily["perfect_eur"]))

    table = pd.DataFrame(rows).T.sort_values("eur_per_day", ascending=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"{pd.Timestamp.now('Europe/Warsaw').date()}_pnl"
    table.to_csv(OUT_DIR / f"{stamp}_summary.csv")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(curves["perfect"].index, curves["perfect"], color="#2c3e50",
            linestyle=(0, (1, 1)), linewidth=2, label="perfect foresight")
    for label, path_, color, ls in MODELS:
        if label in curves:
            c = curves[label]
            ax.plot(c.index, c, color=color, linestyle=ls,
                    linewidth=2, label=label)
    ax.set_title("Cumulative battery-arbitrage P&L "
                 "(1 MW / 2 MWh / 0.85 RTE, day-ahead only)")
    ax.set_ylabel("Cumulative P&L (EUR)")
    ax.set_xlabel("Local date (Europe/Warsaw)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cumulative_pnl.png", dpi=150)

    md = [
        f"# Battery-arbitrage P&L — {stamp}", "",
        "Battery: 1 MW / 2 MWh / 0.85 round-trip / 1 cycle per day.",
        "Schedule from each model's P50 at D-1, settled at actual DA",
        "prices. Day-ahead market only. Same days for every model.",
        "Capture rate = total P&L / perfect-foresight P&L.", "",
        table.round(3).to_markdown(), "",
        "![cumulative](../figures/pnl/cumulative_pnl.png)", "",
    ]
    (OUT_DIR / f"{stamp}_summary.md").write_text("\n".join(md))
    print(table.round(3).to_string())
    print(f"\nWritten {OUT_DIR}/{stamp}_summary.(csv|md) and "
          f"{FIG_DIR}/cumulative_pnl.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
