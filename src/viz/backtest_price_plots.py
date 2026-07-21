"""Backtesting comparison plots for price models.

Reads hourly predictions from data/processed/backtest_preds_price_res/
and actual prices from data/processed/price_da_eur.parquet.

Outputs 10+ figures to reports/figures/backtest_price/.
Run: uv run python -m src.viz.backtest_price_plots
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

OUT = Path("reports/figures/backtest_price")
OUT.mkdir(parents=True, exist_ok=True)

PREDS_DIR = Path("data/processed/backtest_preds_price_res")
PRICE_PATH = Path("data/processed/price_da_eur.parquet")

# Models to include — (label, parquet stem, color, linestyle)
MODELS = [
    ("LGBM+CQR", "lgbm_quantile_conformal", "#2ecc71", "-"),
    ("LEAR+CQR", "lear_conformal", "#3498db", "--"),
    ("TFT ens-3", "tft_hpo_ens", "#e67e22", "-."),
    ("Naive 1-day", "price_naive_yesterday", "#95a5a6", ":"),
]

# PatchTST aggregate only (no hourly preds saved)
PATCHTST_AGG = {
    "mae": 22.98,
    "rmse": None,
    "rmae": 0.823,
    "coverage_80_pct": 69.5,
}

# Aggregate stats from 2yr backtest CSVs
AGG_STATS = {
    "LGBM+CQR": {"mae": 17.87, "rmse": 28.78, "rmae": 0.640, "coverage_80_pct": 78.7},
    "LEAR+CQR": {"mae": 18.23, "rmse": 32.78, "rmae": 0.653, "coverage_80_pct": 79.4},
    "TFT ens-3": {"mae": 19.71, "rmse": 32.24, "rmae": 0.706, "coverage_80_pct": 79.6},
    "Naive 1-day": {"mae": 27.93, "rmse": 39.54, "rmae": 1.000, "coverage_80_pct": None},
    "PatchTST": {"mae": 22.98, "rmse": None, "rmae": 0.823, "coverage_80_pct": 69.5},
}

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_data() -> tuple[pd.Series, dict[str, pd.DataFrame]]:
    actual = pd.read_parquet(PRICE_PATH)["price_da_eur"]
    preds = {}
    for label, stem, _, _ in MODELS:
        path = PREDS_DIR / f"{stem}.parquet"
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
        preds[label] = df
    return actual, preds


def align(actual: pd.Series, pred: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    a = actual.copy()
    p = pred.copy()
    a.index = a.index.tz_convert("UTC")
    p.index = p.index.tz_convert("UTC")
    idx = a.index.intersection(p.index)
    return a.loc[idx], p.loc[idx]


def mape(actual: pd.Series, pred: pd.Series) -> float:
    mask = actual.abs() > 1e-3
    return float(100.0 * (pred[mask] - actual[mask]).abs().div(actual[mask].abs()).mean())


def compute_metrics(actual: pd.Series, preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for label, pred in preds.items():
        y, p = align(actual, pred)
        p50 = p["p50"]
        err = p50 - y
        mae = float(err.abs().mean())
        rmse = float(np.sqrt((err ** 2).mean()))
        rm = float(mae / 27.93)  # vs naive 1-day
        cov = float(100.0 * ((y >= p["p10"]) & (y <= p["p90"])).mean())
        rows.append({
            "model": label,
            "mae": mae,
            "rmse": rmse,
            "rmae": rm,
            "coverage_80_pct": cov,
        })
    # PatchTST aggregate only
    rows.append({
        "model": "PatchTST",
        "mae": 22.98,
        "rmse": None,
        "rmae": 0.823,
        "coverage_80_pct": 69.5,
    })
    return pd.DataFrame(rows)


# ── Figure 1: Aggregate MAE comparison ──────────────────────────────────────

def plot_mae_comparison(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors_map = {
        "LGBM+CQR": "#2ecc71",
        "LEAR+CQR": "#3498db",
        "TFT ens-3": "#e67e22",
        "Naive 1-day": "#95a5a6",
        "PatchTST": "#9b59b6",
    }
    for ax, col, ylabel in zip(
        axes,
        ["mae", "rmse", "rmae"],
        ["MAE (EUR/MWh)", "RMSE (EUR/MWh)", "rMAE (vs naive)"],
    ):
        sub = metrics.dropna(subset=[col])
        bars = ax.bar(
            sub["model"], sub[col],
            color=[colors_map.get(m, "#888") for m in sub["model"]],
            edgecolor="white", linewidth=0.5,
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{col.upper()} — 2-year walk-forward")
        ax.tick_params(axis="x", rotation=30)
        if col == "rmae":
            ax.axhline(1.0, color="#e74c3c", ls="--", lw=1, label="naive baseline")
            ax.legend(fontsize=8)
    fig.suptitle("Price model comparison — PL Day-Ahead 2024-07-16 → 2026-07-15", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "01_metrics_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved 01_metrics_comparison.png")


# ── Figure 2: 2-year overview (monthly medians) ──────────────────────────────

def plot_overview(actual: pd.Series, preds: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    y_monthly = actual.resample("ME").median()
    ax.plot(y_monthly.index, y_monthly.values, "k-", lw=2, label="Actual (monthly median)", zorder=5)

    for label, color, ls in [
        ("LGBM+CQR", "#2ecc71", "-"),
        ("LEAR+CQR", "#3498db", "--"),
        ("TFT ens-3", "#e67e22", "-."),
    ]:
        p = preds[label]
        ya, p = align(actual, p)
        p50m = p["p50"].resample("ME").median()
        ax.plot(p50m.index, p50m.values, color=color, ls=ls, lw=1.5, alpha=0.85, label=label)

    # LGBM band
    p = preds["LGBM+CQR"]
    ya, p = align(actual, p)
    p10m = p["p10"].resample("ME").median()
    p90m = p["p90"].resample("ME").median()
    ax.fill_between(p10m.index, p10m.values, p90m.values, alpha=0.12, color="#2ecc71", label="LGBM 80% interval")

    ax.set_ylabel("Price (EUR/MWh)")
    ax.set_title("2-year backtest — monthly median price: actual vs models")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "02_overview_2yr.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved 02_overview_2yr.png")


# ── Figures 3-8: Zoom windows (7 days each, 6 periods) ──────────────────────

ZOOM_WINDOWS = [
    ("Summer 2024", "2024-07-22", "2024-07-29", "03_zoom_summer2024.png"),
    ("Autumn 2024", "2024-10-14", "2024-10-21", "04_zoom_autumn2024.png"),
    ("Winter peak 2024-25", "2024-12-09", "2024-12-16", "05_zoom_winter2024.png"),
    ("Spring 2025", "2025-04-07", "2025-04-14", "06_zoom_spring2025.png"),
    ("Summer 2025", "2025-07-07", "2025-07-14", "07_zoom_summer2025.png"),
    ("Winter 2025-26", "2025-12-08", "2025-12-15", "08_zoom_winter2025.png"),
]


def plot_zoom(
    actual: pd.Series,
    preds: dict[str, pd.DataFrame],
    title: str,
    start: str,
    end: str,
    fname: str,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 4))
    t0, t1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    y_win = actual.loc[t0:t1]
    ax.plot(y_win.index, y_win.values, "k-", lw=2.5, label="Actual", zorder=5)

    colors = {"LGBM+CQR": "#2ecc71", "LEAR+CQR": "#3498db", "TFT ens-3": "#e67e22", "Naive 1-day": "#95a5a6"}
    ls_map = {"LGBM+CQR": "-", "LEAR+CQR": "--", "TFT ens-3": "-.", "Naive 1-day": ":"}

    for label in ["LGBM+CQR", "LEAR+CQR", "TFT ens-3", "Naive 1-day"]:
        p = preds[label].loc[t0:t1]
        if len(p) == 0:
            continue
        ax.plot(p.index, p["p50"].values, color=colors[label], ls=ls_map[label], lw=1.5, alpha=0.9, label=f"{label} p50")

    # LGBM band
    p_lgbm = preds["LGBM+CQR"].loc[t0:t1]
    if len(p_lgbm) > 0:
        ax.fill_between(p_lgbm.index, p_lgbm["p10"].values, p_lgbm["p90"].values,
                        alpha=0.15, color="#2ecc71", label="LGBM 80%")

    ax.set_ylabel("Price (EUR/MWh)")
    ax.set_title(f"{title} — hourly predictions vs actual")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:00"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.legend(loc="upper right", fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


# ── Figure 9: Error distribution ────────────────────────────────────────────

def plot_error_distribution(actual: pd.Series, preds: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"LGBM+CQR": "#2ecc71", "LEAR+CQR": "#3498db", "TFT ens-3": "#e67e22", "Naive 1-day": "#95a5a6"}

    # Error (p50 - actual)
    errors = {}
    for label, pred in preds.items():
        y, p = align(actual, pred)
        errors[label] = (p["p50"] - y).values

    labels = list(errors.keys())
    data = [errors[l] for l in labels]
    bps = axes[0].violinplot(data, positions=range(len(labels)), showmedians=True, showextrema=False)
    for i, (body, label) in enumerate(zip(bps["bodies"], labels)):
        body.set_facecolor(colors.get(label, "#888"))
        body.set_alpha(0.6)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=20)
    axes[0].axhline(0, color="black", lw=0.8, ls="--")
    axes[0].set_ylabel("Error: forecast − actual (EUR/MWh)")
    axes[0].set_title("Error distribution (p50 − actual)")

    # Absolute error
    abs_errors = {l: np.abs(e) for l, e in errors.items()}
    bp2 = axes[1].boxplot(
        [abs_errors[l] for l in labels],
        tick_labels=labels, patch_artist=True,
        flierprops=dict(marker=".", markersize=1, alpha=0.3),
        medianprops=dict(color="black", lw=1.5),
    )
    for patch, label in zip(bp2["boxes"], labels):
        patch.set_facecolor(colors.get(label, "#888"))
        patch.set_alpha(0.6)
    axes[1].set_xticklabels(labels, rotation=20)
    axes[1].set_ylabel("Absolute error (EUR/MWh)")
    axes[1].set_title("Absolute error distribution")

    fig.suptitle("Forecast error distributions — 2-year walk-forward")
    fig.tight_layout()
    fig.savefig(OUT / "09_error_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved 09_error_distribution.png")


# ── Figure 10: Error by hour of day ─────────────────────────────────────────

def plot_mae_by_hour(actual: pd.Series, preds: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = {"LGBM+CQR": "#2ecc71", "LEAR+CQR": "#3498db", "TFT ens-3": "#e67e22", "Naive 1-day": "#95a5a6"}
    ls_map = {"LGBM+CQR": "-", "LEAR+CQR": "--", "TFT ens-3": "-.", "Naive 1-day": ":"}

    tz = "Europe/Warsaw"
    for label, pred in preds.items():
        y, p = align(actual, pred)
        err = (p["p50"] - y).abs()
        hour = y.index.tz_convert(tz).hour
        mae_by_hour = err.groupby(hour).mean()
        ax.plot(mae_by_hour.index, mae_by_hour.values,
                color=colors.get(label, "#888"), ls=ls_map.get(label, "-"),
                lw=2, alpha=0.85, label=label, marker="o", markersize=3)

    ax.set_xlabel("Hour of day (Warsaw time)")
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("MAE by hour of day — price spike times visible at 07-09h and 18-21h")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "10_mae_by_hour.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved 10_mae_by_hour.png")


# ── Figure 11: Rolling 30-day MAE ───────────────────────────────────────────

def plot_rolling_mae(actual: pd.Series, preds: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = {"LGBM+CQR": "#2ecc71", "LEAR+CQR": "#3498db", "TFT ens-3": "#e67e22", "Naive 1-day": "#95a5a6"}
    ls_map = {"LGBM+CQR": "-", "LEAR+CQR": "--", "TFT ens-3": "-.", "Naive 1-day": ":"}

    for label, pred in preds.items():
        y, p = align(actual, pred)
        mae_roll = (p["p50"] - y).abs().rolling(30 * 24).mean()
        ax.plot(mae_roll.index, mae_roll.values,
                color=colors.get(label, "#888"), ls=ls_map.get(label, "-"),
                lw=1.5, alpha=0.9, label=label)

    ax.set_ylabel("Rolling 30-day MAE (EUR/MWh)")
    ax.set_title("Rolling 30-day MAE over time — model accuracy vs market regime")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "11_rolling_mae.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved 11_rolling_mae.png")


# ── Figure 12: Monthly 80% coverage ─────────────────────────────────────────

def plot_coverage(actual: pd.Series, preds: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = {"LGBM+CQR": "#2ecc71", "LEAR+CQR": "#3498db", "TFT ens-3": "#e67e22"}
    ls_map = {"LGBM+CQR": "-", "LEAR+CQR": "--", "TFT ens-3": "-."}

    for label in ["LGBM+CQR", "LEAR+CQR", "TFT ens-3"]:
        pred = preds[label]
        y, p = align(actual, pred)
        in_band = ((y >= p["p10"]) & (y <= p["p90"])).astype(float)
        cov_monthly = 100.0 * in_band.resample("ME").mean()
        ax.plot(cov_monthly.index, cov_monthly.values,
                color=colors[label], ls=ls_map[label], lw=1.5, alpha=0.9, label=label, marker="o", markersize=3)

    ax.axhline(80, color="#e74c3c", ls="--", lw=1.5, label="Target 80%")
    ax.set_ylabel("80% interval coverage (%)")
    ax.set_ylim(50, 100)
    ax.set_title("Monthly 80% prediction interval coverage (target: 80%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "12_coverage_monthly.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved 12_coverage_monthly.png")


# ── Figure 13: MAPE by month ─────────────────────────────────────────────────

def plot_mape_monthly(actual: pd.Series, preds: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = {"LGBM+CQR": "#2ecc71", "LEAR+CQR": "#3498db", "TFT ens-3": "#e67e22", "Naive 1-day": "#95a5a6"}
    ls_map = {"LGBM+CQR": "-", "LEAR+CQR": "--", "TFT ens-3": "-.", "Naive 1-day": ":"}

    for label, pred in preds.items():
        y, p = align(actual, pred)
        mask = y.abs() > 1e-3
        pct_err = ((p["p50"] - y).abs() / y.abs() * 100).where(mask)
        mape_monthly = pct_err.resample("ME").mean()
        ax.plot(mape_monthly.index, mape_monthly.values,
                color=colors.get(label, "#888"), ls=ls_map.get(label, "-"),
                lw=1.5, alpha=0.9, label=label)

    ax.set_ylabel("MAPE (%)")
    ax.set_title("Monthly MAPE — high MAPE months signal price spikes or near-zero prices")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "13_mape_monthly.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved 13_mape_monthly.png")


# ── Figure 14: Scatter — predicted vs actual ─────────────────────────────────

def plot_scatter(actual: pd.Series, preds: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True, sharex=True)
    titles = ["LGBM+CQR", "LEAR+CQR", "TFT ens-3"]
    colors_map = {"LGBM+CQR": "#2ecc71", "LEAR+CQR": "#3498db", "TFT ens-3": "#e67e22"}

    lim_min, lim_max = -50, 450
    for ax, label in zip(axes, titles):
        y, p = align(actual, preds[label])
        p50 = p["p50"]
        mask = y.abs() > 0.5
        ax.scatter(y[mask], p50[mask], alpha=0.03, s=1, color=colors_map[label], rasterized=True)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=0.8, alpha=0.5, label="Perfect")
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel("Actual (EUR/MWh)")
        ax.set_ylabel("Forecast p50 (EUR/MWh)" if ax == axes[0] else "")
        mae_v = float((p50[mask] - y[mask]).abs().mean())
        ax.set_title(f"{label}\nMAE={mae_v:.1f}")
        ax.set_aspect("equal")

    fig.suptitle("Forecast p50 vs actual — 2-year walk-forward", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "14_scatter_pred_vs_actual.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved 14_scatter_pred_vs_actual.png")


# ── Figure 15: Price spike analysis ──────────────────────────────────────────

def plot_spike_analysis(actual: pd.Series, preds: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    # Spike = price above 95th percentile
    spike_cut = actual.quantile(0.95)
    mask = actual >= spike_cut
    spike_actual = actual[mask]

    colors = {"LGBM+CQR": "#2ecc71", "LEAR+CQR": "#3498db", "TFT ens-3": "#e67e22"}
    positions = np.array([0, 1, 2])
    width = 0.25

    spike_maes = []
    for label in ["LGBM+CQR", "LEAR+CQR", "TFT ens-3"]:
        y, p = align(actual, preds[label])
        spike_mae = float((p.loc[mask, "p50"] - y[mask]).abs().mean())
        spike_maes.append(spike_mae)

    bars = ax.bar(["LGBM+CQR", "LEAR+CQR", "TFT ens-3"], spike_maes,
                  color=["#2ecc71", "#3498db", "#e67e22"], edgecolor="white")
    ax.bar_label(bars, fmt="%.1f EUR/MWh", padding=2, fontsize=9)
    ax.set_ylabel("MAE on spike hours (EUR/MWh)")
    ax.set_title(f"Spike MAE (hours with price ≥ {spike_cut:.0f} EUR/MWh — top 5%)")
    fig.tight_layout()
    fig.savefig(OUT / "15_spike_mae.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved 15_spike_mae.png")


def main() -> None:
    print("Loading data…")
    actual, preds = load_data()

    # Restrict to walk-forward test period
    t0 = pd.Timestamp("2024-07-16", tz="UTC")
    actual.index = pd.to_datetime(actual.index, utc=True)
    actual = actual.loc[t0:]
    preds = {k: v.loc[t0:] for k, v in preds.items()}

    print("Computing metrics…")
    metrics = compute_metrics(actual, preds)
    metrics_path = OUT / "metrics_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")
    print(metrics.to_string(index=False))

    print("\nGenerating plots…")
    plot_mae_comparison(metrics)
    plot_overview(actual, preds)
    for title, start, end, fname in ZOOM_WINDOWS:
        plot_zoom(actual, preds, title, start, end, fname)
    plot_error_distribution(actual, preds)
    plot_mae_by_hour(actual, preds)
    plot_rolling_mae(actual, preds)
    plot_coverage(actual, preds)
    plot_mape_monthly(actual, preds)
    plot_scatter(actual, preds)
    plot_spike_analysis(actual, preds)

    print(f"\nAll plots saved to {OUT}/")


if __name__ == "__main__":
    main()
