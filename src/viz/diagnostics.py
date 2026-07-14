"""Backtest diagnostics: WHERE do models fail, not just how much on average.

Reads saved predictions from data/processed/backtest_preds/.
Run: python -m src.viz.diagnostics  (or make viz-diag)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import REPO_ROOT, load_config
from src.viz.style import CATEGORICAL, apply_style

FIGURES = REPO_ROOT / "reports" / "figures"
LOCAL_TZ = "Europe/Warsaw"
MODEL_ORDER = ["ridge", "lasso_ar", "seasonal_naive", "climatology", "tso_forecast"]


def _abs_pct_err(y: pd.Series, p50: pd.Series) -> pd.Series:
    df = pd.concat([y.rename("y"), p50.rename("f")], axis=1).dropna()
    return ((df["y"] - df["f"]).abs() / df["y"] * 100).rename("ape")


def plot_mape_by_hour(errs: dict[str, pd.Series], out: Path) -> Path:
    """The money plot: which hours hurt. Peaks cost most on the balancing market."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for (name, ape), color in zip(errs.items(), CATEGORICAL + ["#555555"]):
        by_hour = ape.groupby(ape.index.tz_convert(LOCAL_TZ).hour).mean()
        ax.plot(by_hour.index, by_hour.values, label=name, color=color)
    ax.set_xlabel(f"Hour of day ({LOCAL_TZ})")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Backtest error by hour of day", loc="left")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_mape_by_weekday(errs: dict[str, pd.Series], out: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for (name, ape), color in zip(errs.items(), CATEGORICAL + ["#555555"]):
        by_dow = ape.groupby(ape.index.tz_convert(LOCAL_TZ).dayofweek).mean()
        ax.plot(by_dow.index, by_dow.values, label=name, color=color, marker="o")
    ax.set_xticks(range(7), days)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Backtest error by weekday", loc="left")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_daily_mape_distribution(errs: dict[str, pd.Series], out: Path) -> Path:
    """Distribution of per-day MAPE. The tail is the operational risk."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for (name, ape), color in zip(errs.items(), CATEGORICAL + ["#555555"]):
        daily = ape.groupby(ape.index.tz_convert(LOCAL_TZ).date).mean()
        daily = daily.sort_values().reset_index(drop=True)
        ax.plot(
            daily.index / max(len(daily) - 1, 1) * 100, daily.values,
            label=name, color=color,
        )
    ax.set_xlabel("Share of days (%) — sorted best to worst")
    ax.set_ylabel("Daily MAPE (%)")
    ax.set_title("Daily error distribution: the right tail is the risk", loc="left")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    cfg = load_config()
    # Prefer the honest run (forecast weather) when it exists.
    candidates = ["backtest_preds_fcst", "backtest_preds_actuals"]
    preds_dir = next(
        (cfg.paths["data_processed"] / c for c in candidates
         if (cfg.paths["data_processed"] / c).exists()),
        None,
    )
    if preds_dir is None:
        print("No saved predictions. Run: make backtest")
        return 1
    print(f"Using {preds_dir.name}")
    y = pd.read_parquet(cfg.paths["data_processed"] / "load.parquet").iloc[:, 0]

    errs: dict[str, pd.Series] = {}
    for name in MODEL_ORDER:
        path = preds_dir / f"{name}.parquet"
        if path.exists():
            errs[name] = _abs_pct_err(y, pd.read_parquet(path)["p50"])
    if not errs:
        print("No prediction files found.")
        return 1

    made = [
        plot_mape_by_hour(errs, FIGURES / "backtest_mape_by_hour.png"),
        plot_mape_by_weekday(errs, FIGURES / "backtest_mape_by_weekday.png"),
        plot_daily_mape_distribution(errs, FIGURES / "backtest_daily_mape_dist.png"),
    ]
    for p in made:
        print(f"made    {p.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
