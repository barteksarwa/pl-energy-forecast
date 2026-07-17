"""Desk-style backtest review pack for the price models.

Mimics what a trading-desk model review actually looks at, one question
per figure:

1. rolling MAE        — is the model drifting? which regime hurts?
2. cumulative edge    — is the edge vs benchmark stable or one lucky month?
3. MAE by hour        — where in the day does it fail? (evening ramp)
4. calibration        — can risk trust the quantiles?
5. worst days         — post-mortem panel: what did the miss look like?
6. monthly bias       — systematic over/under-forecasting?

Color contract (validated, see style.py): models carry hue
(lgbm=blue, lear=orange), benchmarks are neutral gray reference marks.
Every line is direct-labeled; a CSV table companion covers the
low-contrast relief obligation.

Run: python -m src.viz.backtest_plots
Reads data/processed/backtest_preds_price_res/, writes
reports/figures/backtests/ and reports/backtests/*_price_diagnostics.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import load_config
from src.viz.style import BAND_ALPHA, BLUE, ORANGE, apply_style

LOCAL_TZ = "Europe/Warsaw"

GRAY_DARK = "#6e6e6e"
GRAY_LIGHT = "#a8a8a8"

# Fixed order and color per entity. Never repainted by filtering.
MODELS = {
    "lgbm_quantile": ("LightGBM", BLUE),
    "lear": ("LEAR", ORANGE),
    "price_naive_yesterday": ("naive-1d", GRAY_DARK),
    "price_naive_week": ("naive-7d", GRAY_LIGHT),
}
CHAMPION = "lgbm_quantile"
BENCHMARK = "price_naive_yesterday"


def _load(preds_dir: Path, price_path: Path) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    preds = {m: pd.read_parquet(preds_dir / f"{m}.parquet") for m in MODELS}
    idx = preds[CHAMPION].index
    y = pd.read_parquet(price_path).iloc[:, 0].reindex(idx)
    return preds, y


def _label_end(ax: plt.Axes, x, y, text: str, color: str) -> None:
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                color=color, fontsize=9, fontweight="bold", va="center")


def _label_ends(ax: plt.Axes, items: list[tuple[float, float, str, str]]) -> None:
    """Direct labels at line ends, staggered so converging lines stay legible.

    items: (x, y, text, color). Labels keep their own x; only the label y
    is nudged apart (min gap = 4.5% of the y-range), never the data.
    """
    lo, hi = ax.get_ylim()
    min_gap = (hi - lo) * 0.045
    ordered = sorted(items, key=lambda t: t[1])
    ys = [t[1] for t in ordered]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    for (x, _, text, color), y_lab in zip(ordered, ys):
        _label_end(ax, x, y_lab, text, color)


def _finish(fig: plt.Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_rolling_mae(preds: dict, y: pd.Series, out: Path) -> Path:
    """30-day rolling MAE. The drift monitor a desk checks weekly."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ends = []
    for m, (label, color) in MODELS.items():
        err = (preds[m]["p50"] - y).abs()
        roll = err.resample("1D").mean().rolling(30, min_periods=20).mean()
        ax.plot(roll.index, roll.values, color=color, linewidth=2.0)
        last = roll.dropna()
        ends.append((last.index[-1], last.iloc[-1], label, color))
    _label_ends(ax, ends)
    ax.set_ylabel("30-day rolling MAE (EUR/MWh)")
    ax.set_xlabel("Time (UTC)")
    ax.set_title("Price P50 error over time — walk-forward, weekly refits", loc="left")
    ax.margins(x=0.08)
    return _finish(fig, out)


def plot_cumulative_edge(preds: dict, y: pd.Series, out: Path) -> Path:
    """Cumulative MAE saved vs naive-1d. Flat stretch = no edge there.

    The desk's 'is the edge real or one lucky month' view: a healthy
    model climbs steadily; steps and plateaus localize regime problems.
    """
    apply_style()
    naive_err = (preds[BENCHMARK]["p50"] - y).abs().resample("1D").mean()
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    for m in ("lgbm_quantile", "lear"):
        label, color = MODELS[m]
        err = (preds[m]["p50"] - y).abs().resample("1D").mean()
        edge = (naive_err - err).cumsum()
        ax.plot(edge.index, edge.values, color=color, linewidth=2.0)
        _label_end(ax, edge.index[-1], edge.iloc[-1], label, color)
    ax.axhline(0, color=GRAY_DARK, linewidth=1.0)
    ax.annotate("naive-1d", (y.index[0], 0), xytext=(0, 5),
                textcoords="offset points", color=GRAY_DARK, fontsize=9)
    ax.set_ylabel("Cumulative daily MAE saved (EUR/MWh)")
    ax.set_xlabel("Time (UTC)")
    ax.set_title("Cumulative edge vs naive-1d — steady climb = real skill", loc="left")
    ax.margins(x=0.08)
    return _finish(fig, out)


def plot_mae_by_hour(preds: dict, y: pd.Series, out: Path) -> Path:
    """Error by delivery hour (local). Peaks and solar dip stress models."""
    apply_style()
    hours = y.index.tz_convert(LOCAL_TZ).hour
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ends = []
    for m, (label, color) in MODELS.items():
        err = (preds[m]["p50"] - y).abs()
        prof = err.groupby(hours).mean()
        ax.plot(prof.index, prof.values, color=color, linewidth=2.0, marker="o",
                markersize=3.5)
        ends.append((23, prof.loc[23], label, color))
    _label_ends(ax, ends)
    ax.set_xticks(range(0, 24, 3))
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_xlabel(f"Delivery hour ({LOCAL_TZ})")
    ax.set_title("Where in the day the error lives", loc="left")
    ax.margins(x=0.10)
    return _finish(fig, out)


def plot_calibration(preds: dict, y: pd.Series, out: Path) -> Path:
    """Quantile reliability: share of actuals below each predicted quantile.

    On the diagonal = trustable band. Below the diagonal at P90 means the
    band is too narrow — risk management reads this chart first.
    """
    apply_style()
    nominal = [10, 50, 90]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 100], [0, 100], color=GRAY_LIGHT, linewidth=1.0, zorder=1)
    ends = []
    for m in ("lgbm_quantile", "lear", BENCHMARK):
        label, color = MODELS[m]
        emp = [100.0 * (y <= preds[m][q]).mean() for q in ("p10", "p50", "p90")]
        ax.plot(nominal, emp, color=color, linewidth=2.0, marker="o", markersize=6,
                zorder=2)
        ends.append((nominal[-1], emp[-1], label, color))
    _label_ends(ax, ends)
    ax.set_xticks(nominal)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Nominal quantile (%)")
    ax.set_ylabel("Empirical: share of actuals below (%)")
    ax.set_title("Quantile calibration — diagonal = honest band", loc="left")
    ax.margins(x=0.15)
    return _finish(fig, out)


def _pick_days(err_day: pd.Series, pick: str) -> list:
    """Select 4 local days by the champion's daily mean |error|.

    worst  = top 4, best = bottom 4, median = the 4 straddling the 50th
    percentile of the ranked days. Selection is ALWAYS by the champion
    model so panels are comparable across models; see the README next to
    the figures for why.
    """
    ranked = err_day.sort_values(ascending=False)
    if pick == "worst":
        return list(ranked.head(4).index)
    if pick == "best":
        return list(ranked.tail(4).index)
    if pick == "median":
        mid = len(ranked) // 2
        return list(ranked.iloc[mid - 2 : mid + 2].index)
    raise ValueError(f"unknown pick: {pick}")


def plot_day_panels(preds: dict, y: pd.Series, out: Path, pick: str = "worst") -> Path:
    """Panel of 4 days (worst / median / best by champion daily MAE)."""
    apply_style()
    err_day = (preds[CHAMPION]["p50"] - y).abs().groupby(
        y.index.tz_convert(LOCAL_TZ).date
    ).mean()
    worst = _pick_days(err_day, pick)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharey=False)
    for ax, day in zip(axes.flat, worst):
        local = y.index.tz_convert(LOCAL_TZ)
        mask = pd.Index(local.date) == day
        hrs = local[mask].hour + local[mask].minute / 60
        p = preds[CHAMPION][mask]
        ax.fill_between(hrs, p["p10"], p["p90"], color=BLUE, alpha=BAND_ALPHA,
                        linewidth=0)
        ax.plot(hrs, p["p50"], color=BLUE, linewidth=2.0, label="LightGBM P50")
        ax.plot(hrs, preds["lear"][mask]["p50"], color=ORANGE, linewidth=2.0,
                label="LEAR P50")
        ax.plot(hrs, preds[BENCHMARK][mask]["p50"], color=GRAY_DARK, linewidth=1.4,
                linestyle="--", label="naive-1d")
        ax.plot(hrs, y[mask], color="black", linewidth=2.0, label="actual")
        ax.set_title(str(day), loc="left", fontsize=10)
        ax.set_xticks(range(0, 24, 6))
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axes[1, :]:
        ax.set_xlabel(f"Hour ({LOCAL_TZ})")
    for ax in axes[:, 0]:
        ax.set_ylabel("Price (EUR/MWh)")
    titles = {
        "worst": "Champion's four worst days — what the miss looked like",
        "median": "Four typical days (median error) — the everyday picture",
        "best": "Champion's four best days — when everything lines up",
    }
    fig.suptitle(titles[pick], x=0.01, ha="left", fontsize=11)
    return _finish(fig, out)


def plot_monthly_bias(preds: dict, y: pd.Series, out: Path) -> Path:
    """Mean signed error per month. Persistent sign = systematic bias."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    width = 12.0  # days, for month-wide grouped bars
    for i, m in enumerate(("lgbm_quantile", "lear")):
        label, color = MODELS[m]
        bias = (preds[m]["p50"] - y).resample("ME").mean()
        ax.bar(bias.index + pd.Timedelta(days=i * 13 - 13), bias.values,
               width=width, color=color, label=label)
    ax.axhline(0, color=GRAY_DARK, linewidth=1.0)
    ax.set_ylabel("Mean error, P50 − actual (EUR/MWh)")
    ax.set_xlabel("Month (UTC)")
    ax.set_title("Monthly bias — above zero = over-forecasting", loc="left")
    ax.legend(frameon=False)
    return _finish(fig, out)


def diagnostics_table(preds: dict, y: pd.Series) -> pd.DataFrame:
    """The numbers behind the pack. Table-view companion for the figures."""
    hours = y.index.tz_convert(LOCAL_TZ).hour
    rows = []
    for m, (label, _) in MODELS.items():
        p = preds[m]
        err = p["p50"] - y
        by_hour = err.abs().groupby(hours).mean()
        rows.append(
            {
                "model": label,
                "mae_eur": err.abs().mean(),
                "bias_eur": err.mean(),
                "below_p10_pct": 100.0 * (y < p["p10"]).mean(),
                "above_p90_pct": 100.0 * (y > p["p90"]).mean(),
                "worst_hour_local": int(by_hour.idxmax()),
                "worst_hour_mae": by_hour.max(),
                "worst_day": str(
                    err.abs().groupby(y.index.tz_convert(LOCAL_TZ).date)
                    .mean().idxmax()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("model")


def main() -> int:
    cfg = load_config()
    proc = cfg.paths["data_processed"]
    preds_dir = proc / "backtest_preds_price_res"
    if not preds_dir.exists():
        print(f"Missing {preds_dir} — run the price backtest first.")
        return 1
    preds, y = _load(preds_dir, proc / "price_da_eur.parquet")

    fig_dir = proc.parent.parent / "reports" / "figures" / "backtests"
    made = [
        plot_rolling_mae(preds, y, fig_dir / "price_bt_rolling_mae.png"),
        plot_cumulative_edge(preds, y, fig_dir / "price_bt_cumulative_edge.png"),
        plot_mae_by_hour(preds, y, fig_dir / "price_bt_mae_by_hour.png"),
        plot_calibration(preds, y, fig_dir / "price_bt_calibration.png"),
        plot_day_panels(preds, y, fig_dir / "price_bt_worst_days.png", "worst"),
        plot_day_panels(preds, y, fig_dir / "price_bt_median_days.png", "median"),
        plot_day_panels(preds, y, fig_dir / "price_bt_best_days.png", "best"),
        plot_monthly_bias(preds, y, fig_dir / "price_bt_monthly_bias.png"),
    ]
    table = diagnostics_table(preds, y)
    out_csv = proc.parent.parent / "reports" / "backtests" / (
        f"{pd.Timestamp.now(LOCAL_TZ).date()}_price_diagnostics.csv"
    )
    table.to_csv(out_csv)
    for p in made:
        print("made   ", p)
    print(table.round(2).to_string())
    print("table  ", out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
