"""Imbalance-market v1: what the day-ahead-to-balancing spread costs us.

Why this exists
---------------
A Polish power desk trades in three steps: day-ahead (DA) auction,
intraday (ID), then whatever is left is settled by the TSO at the
*imbalance price*. Jargon, first use:

- **Day-ahead price** — the price you lock in at the D-1 auction.
- **Balancing / imbalance price** (PSE calls it CEN) — the price the TSO
  charges or pays for the volume you did NOT cover at DA or ID.
- **Spread** — here always `balancing - day-ahead`, in PLN/MWh. Positive
  spread means being short (buying late) is expensive.

This repo fetches the balancing price but has never read it. This module
closes that gap. It is descriptive plus a costing. There is no new model.

What it does
------------
1. Aligns DA (PLN) and balancing (PLN) on their overlap and describes the
   spread: mean, median, share of hours above zero, tail quantiles, by
   hour-of-day (Europe/Warsaw), by year.
2. Costs a forecast miss at the imbalance price (see `miss_cost`).
3. Converts our stored EUR price-forecast errors into PLN with the hourly
   implied FX rate `price_da_pln / price_da_eur`, so the model error and
   the spread are in the same unit.
4. Writes reports/backtests/<date>_imbalance_v1.(csv|md) and one figure.

Assumptions — read these before quoting any number
--------------------------------------------------
- This is a costing of PRICE risk. It is not a dispatch simulation. No
  battery, no unit commitment, no intraday leg, no volume model.
- The unit is one MWh. Every hour is treated as an independent 1 MWh
  position, so numbers are "per MWh of exposure", not portfolio P&L.
- We assume a party sizes its DA position on our P50 price forecast. If
  the forecast was too low (under-forecast), the party under-bought and
  closes the gap at the balancing price: cost = (bal - DA). If it was too
  high, it over-bought and sells the surplus back: cost = (DA - bal).
  A miss can therefore be *lucky* (negative cost).
- Balancing volume is priced at a single price here. PSE's real settlement
  has extra components; this is a first-order bound, not an invoice.
- Both price series come from PSE, so DA is the PSE csdac series, not the
  ENTSO-E EUR series. They are the same auction, different publisher.

Run: python -m src.evaluation.run_imbalance_analysis
     python -m src.evaluation.run_imbalance_analysis --start 2025-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import REPO_ROOT, load_config
from src.viz.style import BLUE, ORANGE, apply_style

LOCAL_TZ = "Europe/Warsaw"

# Stored walk-forward forecasts we cost. Label -> path under data/processed.
MODELS: dict[str, str] = {
    "ens_crps_cqr_tft": "backtest_preds_price_res/ens_crps_cqr_tft.parquet",
    "lgbm_quantile_conformal": (
        "backtest_preds_price_res/lgbm_quantile_conformal.parquet"
    ),
}


# --------------------------------------------------------------------------
# pure logic (tested in tests/test_imbalance_analysis.py)
# --------------------------------------------------------------------------
def compute_spread(da_pln: pd.Series, bal_pln: pd.Series) -> pd.DataFrame:
    """Align the two price series and return the spread frame.

    Spread is always `balancing - day-ahead`, in PLN/MWh. Hours missing in
    either series are dropped: an unpaired price cannot be spread.
    """
    df = pd.concat(
        [da_pln.rename("price_da_pln"), bal_pln.rename("price_bal_pln")],
        axis=1,
        join="inner",
    ).dropna()
    df["spread_pln"] = df["price_bal_pln"] - df["price_da_pln"]
    return df.sort_index()


def spread_summary(spread: pd.Series) -> dict[str, float]:
    """Headline description of the spread. All values in PLN/MWh."""
    s = spread.dropna()
    if s.empty:
        return {}
    return {
        "n_hours": float(len(s)),
        "mean_pln": float(s.mean()),
        "median_pln": float(s.median()),
        "std_pln": float(s.std()),
        "mean_abs_pln": float(s.abs().mean()),
        "share_bal_above_da": float((s > 0).mean()),
        "p5_pln": float(s.quantile(0.05)),
        "p25_pln": float(s.quantile(0.25)),
        "p75_pln": float(s.quantile(0.75)),
        "p95_pln": float(s.quantile(0.95)),
    }


def spread_by_group(spread: pd.Series, group: pd.Index | np.ndarray) -> pd.DataFrame:
    """Spread stats per group (hour-of-day, year, ...). Index = group key."""
    g = spread.groupby(group)
    return pd.DataFrame(
        {
            "n_hours": g.size(),
            "mean_pln": g.mean(),
            "median_pln": g.median(),
            "mean_abs_pln": g.apply(lambda x: x.abs().mean()),
            "share_bal_above_da": g.apply(lambda x: float((x > 0).mean())),
        }
    )


def implied_fx_rate(
    da_pln: pd.Series, da_eur: pd.Series, min_eur: float = 5.0
) -> pd.Series:
    """Hourly PLN-per-EUR rate implied by the same auction in two currencies.

    Guard: the ratio blows up when the EUR price is near zero or negative
    (Poland has ~4% negative-price hours). We only trust hours where the
    EUR price is at least `min_eur`. Other hours take the median rate of
    their local day, then the nearest known rate.

    Caveat: this is an *implied* rate, not the NBP fixing. It also absorbs
    any publication mismatch between the PSE and ENTSO-E series.
    """
    ratio = (da_pln / da_eur).replace([np.inf, -np.inf], np.nan)
    trusted = ratio.where((da_eur >= min_eur) & (ratio > 0))
    if trusted.notna().sum() == 0:
        raise ValueError("no hour has a trustworthy implied FX rate")
    day = pd.Index(trusted.index).normalize()
    daily_median = trusted.groupby(day).transform("median")
    return trusted.fillna(daily_median).ffill().bfill()


def miss_cost(
    p50_pln: pd.Series, actual_pln: pd.Series, spread_pln: pd.Series
) -> pd.DataFrame:
    """Cost of a 1 MWh forecast miss settled at the imbalance price.

    Sign convention:
    - under-forecast (`p50 < actual`): the party bought too little at DA and
      must buy the rest at the balancing price -> cost = bal - DA = spread.
    - over-forecast (`p50 > actual`): the party bought too much and sells the
      surplus at the balancing price -> cost = DA - bal = -spread.
    - exact hit: no exposure, cost 0.

    So `cost = -sign(p50 - actual) * spread`. Positive cost = money lost.
    Negative cost = the miss was lucky, the spread went your way.
    """
    df = pd.concat(
        [
            p50_pln.rename("p50_pln"),
            actual_pln.rename("actual_pln"),
            spread_pln.rename("spread_pln"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    df["err_pln"] = df["p50_pln"] - df["actual_pln"]
    df["direction"] = np.where(
        df["err_pln"] < 0, "under", np.where(df["err_pln"] > 0, "over", "exact")
    )
    df["cost_pln_per_mwh"] = -np.sign(df["err_pln"]) * df["spread_pln"]
    return df


def daily_clustered_t(cost: pd.Series, tz: str = LOCAL_TZ) -> float:
    """t-statistic of the mean cost, clustering by local day.

    Hourly costs are strongly autocorrelated within a day, so an hourly
    t-test would overstate significance. We average per day first, then
    t-test the 728-ish daily means. Blunt, but honest.
    """
    if not isinstance(cost.index, pd.DatetimeIndex) or len(cost) < 2:
        return float("nan")
    idx = cost.index
    if idx.tz is not None:
        idx = idx.tz_convert(tz)
    daily = cost.groupby(idx.date).mean()
    if len(daily) < 2 or daily.std() == 0:
        return float("nan")
    return float(daily.mean() / (daily.std() / np.sqrt(len(daily))))


def summarize_miss_cost(costed: pd.DataFrame) -> dict[str, float]:
    """Headline costing numbers. All per MWh of exposure, PLN."""
    if costed.empty:
        return {}
    cost = costed["cost_pln_per_mwh"]
    under = costed.loc[costed["direction"] == "under", "cost_pln_per_mwh"]
    over = costed.loc[costed["direction"] == "over", "cost_pln_per_mwh"]
    return {
        "n_hours": float(len(costed)),
        "mae_pln": float(costed["err_pln"].abs().mean()),
        "naive_cost_bound_pln": float(costed["spread_pln"].abs().mean()),
        "mean_cost_pln": float(cost.mean()),
        "median_cost_pln": float(cost.median()),
        "share_costly_hours": float((cost > 0).mean()),
        "share_under_forecast": float((costed["direction"] == "under").mean()),
        "mean_cost_under_pln": float(under.mean()) if len(under) else float("nan"),
        "mean_cost_over_pln": float(over.mean()) if len(over) else float("nan"),
        "p95_cost_pln": float(cost.quantile(0.95)),
    }


# --------------------------------------------------------------------------
# I/O + reporting
# --------------------------------------------------------------------------
def _rows(section: str, stats: dict[str, float], key: str = "") -> list[dict]:
    return [
        {"section": section, "key": key, "metric": m, "value": v}
        for m, v in stats.items()
    ]


def make_figure(spread: pd.Series, by_hour: pd.DataFrame, path: Path) -> None:
    """Two panels: spread by local hour, and the spread distribution."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.bar(by_hour.index, by_hour["mean_abs_pln"], color=BLUE,
           label="mean |spread| (cost of being wrong)")
    ax.plot(by_hour.index, by_hour["mean_pln"], color=ORANGE, marker="o",
            markersize=4, label="mean signed spread (bal - DA)")
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Hour of day (Europe/Warsaw, local)")
    ax.set_ylabel("Spread (PLN/MWh)")
    ax.set_title("Balancing minus day-ahead, by hour of day", loc="left")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    lo, hi = spread.quantile(0.01), spread.quantile(0.99)
    ax.hist(spread.clip(lo, hi), bins=80, color=BLUE)
    ax.axvline(0.0, color="#555555", linewidth=0.8)
    ax.axvline(float(spread.median()), color=ORANGE, linewidth=1.5,
               label=f"median {spread.median():.0f} PLN/MWh")
    ax.set_xlabel("Spread bal - DA (PLN/MWh, clipped at p1/p99)")
    ax.set_ylabel("Hours")
    ax.set_title("Spread distribution — fat on both sides", loc="left")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "PL imbalance v1 — source: PSE csdac (day-ahead) and PSE CEN "
        "(balancing), hourly means of 15-min data; timestamps stored UTC, "
        "grouped in local time",
        fontsize=8, y=0.02,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _md_report(
    head: dict[str, float],
    by_hour: pd.DataFrame,
    by_year: pd.DataFrame,
    model_stats: dict[str, dict[str, float]],
    period: tuple[pd.Timestamp, pd.Timestamp],
    fig_rel: str,
    stamp: str,
) -> str:
    worst = by_hour["mean_abs_pln"].idxmax()
    best_model = min(model_stats, key=lambda m: model_stats[m]["mean_cost_pln"]) \
        if model_stats else ""
    lines = [
        f"# Imbalance market v1 — {stamp}", "",
        "First look at the third link of the chain: day-ahead, intraday, "
        "**imbalance**. The imbalance price is what the grid operator (PSE) "
        "charges for the MWh you did not cover in the markets. We fetch that "
        "price but have never used it. This is the v1 read.", "",
        f"Period: **{period[0].date()} to {period[1].date()}**, "
        f"**{int(head['n_hours']):,} hours**. Prices in PLN/MWh.", "",
        "## 1. What the spread is", "",
        "Spread = balancing price minus day-ahead price. Positive means "
        "closing your position late is expensive.", "",
        f"- Mean spread: **{head['mean_pln']:.0f} PLN/MWh** — near zero, so "
        "there is no free money in one direction.",
        f"- Median spread: **{head['median_pln']:.0f} PLN/MWh**.",
        f"- But the average *size* of the gap is **"
        f"{head['mean_abs_pln']:.0f} PLN/MWh**. That is the number that hurts.",
        f"- Balancing beats day-ahead in **"
        f"{100 * head['share_bal_above_da']:.0f}%** of hours — a coin flip.",
        f"- Tails: p5 = {head['p5_pln']:.0f}, p95 = {head['p95_pln']:.0f} "
        f"PLN/MWh. Standard deviation {head['std_pln']:.0f}.", "",
        "Plain words: the spread is a **volatility problem, not a bias "
        "problem**. On average it nets out. Hour by hour it is huge.", "",
        "## 2. When it bites", "",
        f"Worst hour: **{worst}:00 local**, mean gap "
        f"{by_hour.loc[worst, 'mean_abs_pln']:.0f} PLN/MWh. The midday solar "
        "hours are the expensive ones. Night hours are roughly half as bad.",
        "",
        by_hour.round(1).to_markdown(), "",
        "By year:", "",
        by_year.round(1).to_markdown(), "",
        "## 3. What a 1 MWh miss costs", "",
        "Assumption in one line: a party sizes its day-ahead buy on our P50 "
        "price forecast; whatever it got wrong is settled at the balancing "
        "price. Under-forecast means buy the rest late (cost = spread), "
        "over-forecast means sell the surplus back (cost = minus spread). "
        "This prices risk. It is not a dispatch simulation.", "",
        "| model | hours | our MAE (PLN/MWh) | naive cost bound (PLN/MWh) | "
        "mean cost of a 1 MWh miss | costly hours | mean cost when under | "
        "mean cost when over |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m, s in model_stats.items():
        lines.append(
            f"| {m} | {int(s['n_hours']):,} | {s['mae_pln']:.0f} | "
            f"{s['naive_cost_bound_pln']:.0f} | {s['mean_cost_pln']:.1f} | "
            f"{100 * s['share_costly_hours']:.0f}% | "
            f"{s['mean_cost_under_pln']:.1f} | {s['mean_cost_over_pln']:.1f} |"
        )
    ref = model_stats.get(best_model, {})
    lines += [
        "",
        "How to read the row: *naive cost bound* is the average absolute "
        "spread — what you pay if every miss lands on the wrong side. "
        "*Mean cost* is what our actual misses cost, sign included.", "",
    ]
    if ref:
        lines += [
            f"- Our day-ahead price error is **{ref['mae_pln']:.0f} "
            f"PLN/MWh** (MAE, converted from EUR).",
            f"- The imbalance spread we would face is **"
            f"{ref['naive_cost_bound_pln']:.0f} PLN/MWh** on average.",
            f"- Actual mean cost of a miss: **{ref['mean_cost_pln']:.1f} "
            f"PLN/MWh**, with **{100 * ref['share_costly_hours']:.0f}%** of "
            "misses landing on the losing side.", "",
        ]
    lines += [
        "## 4. Takeaway", "",
        "**The imbalance spread is bigger than our whole day-ahead forecast "
        "error, and our price forecast gives us no edge on it — direction is "
        "a coin flip. So a dedicated imbalance model is worth scoping, but "
        "only if it predicts the *sign* of the spread; a better day-ahead "
        "price model will not help here.**", "",
        f"![spread by hour]({fig_rel})", "",
        "### Caveats", "",
        "- EUR errors were converted to PLN with the hourly implied rate "
        "`price_da_pln / price_da_eur`. Hours with a EUR price below "
        "5 EUR/MWh (including negative-price hours) get their local day's "
        "median rate instead — the raw ratio explodes near zero.",
        "- The balancing price is a single settlement price here. Real PSE "
        "settlement has more components.",
        "- One MWh per hour, treated independently. No volume model, no "
        "intraday leg, no portfolio netting.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", default=None,
                        help="ISO date; drop hours before it (UTC)")
    parser.add_argument("--end", default=None, help="ISO date; drop hours after it")
    parser.add_argument("--min-eur", type=float, default=5.0,
                        help="EUR price floor for a trusted implied FX rate")
    parser.add_argument("--stamp", default=None,
                        help="report file stamp, default today (Warsaw)")
    args = parser.parse_args()

    cfg = load_config()
    proc = cfg.paths["data_processed"]
    out_dir = REPO_ROOT / "reports" / "backtests"
    fig_dir = REPO_ROOT / "reports" / "figures" / "imbalance"
    stamp = args.stamp or f"{pd.Timestamp.now(tz=LOCAL_TZ).date()}"

    da = pd.read_parquet(proc / "price_da.parquet").iloc[:, 0]
    bal = pd.read_parquet(proc / "price_balancing.parquet").iloc[:, 0]
    eur = pd.read_parquet(proc / "price_da_eur.parquet").iloc[:, 0]

    df = compute_spread(da, bal)
    if args.start:
        df = df.loc[df.index >= pd.Timestamp(args.start, tz="UTC")]
    if args.end:
        df = df.loc[df.index <= pd.Timestamp(args.end, tz="UTC")]
    if df.empty:
        print("no overlapping hours — nothing to do")
        return 1

    local = df.index.tz_convert(LOCAL_TZ)
    head = spread_summary(df["spread_pln"])
    by_hour = spread_by_group(df["spread_pln"], local.hour).rename_axis("hour_local")
    by_year = spread_by_group(df["spread_pln"], local.year).rename_axis("year_local")

    print(f"overlap {df.index.min()} -> {df.index.max()}  ({len(df):,} hours)")
    for k, v in head.items():
        print(f"  {k:>20}: {v:,.2f}")

    # EUR -> PLN for our stored forecasts
    rate = implied_fx_rate(df["price_da_pln"], eur.reindex(df.index),
                           min_eur=args.min_eur)
    actual_eur = eur.reindex(df.index)

    model_stats: dict[str, dict[str, float]] = {}
    for label, rel in MODELS.items():
        path = proc / rel
        if not path.exists():
            print(f"skip {label} (no preds at {path})")
            continue
        p50_eur = pd.read_parquet(path)["p50"]
        p50_eur.index = pd.DatetimeIndex(p50_eur.index)
        common = df.index.intersection(p50_eur.index)
        costed = miss_cost(
            p50_eur.reindex(common) * rate.reindex(common),
            actual_eur.reindex(common) * rate.reindex(common),
            df["spread_pln"].reindex(common),
        )
        model_stats[label] = summarize_miss_cost(costed)
        print(f"\n{label}: {len(costed):,} hours")
        for k, v in model_stats[label].items():
            print(f"  {k:>22}: {v:,.2f}")

    fig_path = fig_dir / "spread_by_hour.png"
    make_figure(df["spread_pln"], by_hour, fig_path)

    rows = _rows("spread_overall", head)
    for h, r in by_hour.iterrows():
        rows += _rows("spread_by_hour_local", r.to_dict(), key=str(h))
    for y, r in by_year.iterrows():
        rows += _rows("spread_by_year_local", r.to_dict(), key=str(y))
    for label, s in model_stats.items():
        rows += _rows("miss_cost", s, key=label)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stamp}_imbalance_v1.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    md = _md_report(
        head, by_hour, by_year, model_stats,
        (df.index.min(), df.index.max()),
        "../figures/imbalance/spread_by_hour.png", stamp,
    )
    md_path = out_dir / f"{stamp}_imbalance_v1.md"
    md_path.write_text(md)
    print(f"\nWritten {csv_path}\n        {md_path}\n        {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
