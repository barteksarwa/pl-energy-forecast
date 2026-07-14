"""Plot whatever data exists in data/. Run: make viz.

Skips missing files. Prints what it made and what it skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.config import REPO_ROOT, load_config
from src.viz.plots import (
    plot_forecast_band,
    plot_load_history,
    plot_load_vs_tso,
    plot_weather_overview,
)

FIGURES = REPO_ROOT / "reports" / "figures"


def main() -> int:
    cfg = load_config()
    raw = cfg.paths["data_raw"]
    made: list[Path] = []
    skipped: list[str] = []

    # Latest dry-run pulls.
    load_p = raw / "load_latest.parquet"
    tso_p = raw / "tso_forecast_latest.parquet"
    if load_p.exists():
        load = pd.read_parquet(load_p).iloc[:, 0]
        tso = pd.read_parquet(tso_p).iloc[:, 0] if tso_p.exists() else None
        made.append(plot_load_vs_tso(load, tso, FIGURES / "load_vs_tso_latest.png"))
    else:
        skipped.append("load_latest.parquet (run dry-run or smoke first)")

    for city in cfg.cities:
        wp = raw / f"weather_{city.name}_latest.parquet"
        if wp.exists():
            made.append(
                plot_weather_overview(
                    pd.read_parquet(wp), city.name, FIGURES / f"weather_{city.name}_latest.png"
                )
            )

    # Newest saved forecast.
    forecasts = sorted(cfg.paths["forecasts"].glob("*.csv"))
    if forecasts:
        newest = forecasts[-1]
        fc = pd.read_csv(newest, index_col="time_utc", parse_dates=True)
        made.append(plot_forecast_band(fc, newest.stem, FIGURES / "forecast_latest.png"))
    else:
        skipped.append("data/forecasts/*.csv (no forecast yet)")

    # Full backfilled history, if present.
    hist_p = cfg.paths["data_processed"] / "load.parquet"
    if hist_p.exists():
        made.append(
            plot_load_history(pd.read_parquet(hist_p).iloc[:, 0], FIGURES / "load_history.png")
        )

    for p in made:
        print(f"made    {p.relative_to(REPO_ROOT)}")
    for s in skipped:
        print(f"skipped {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
