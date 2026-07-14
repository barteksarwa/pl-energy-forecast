"""The baseline campaign runner: `make backtest`.

Loads real data, builds a cutoff-safe feature matrix day by day, walks every
registered model forward, writes an honest summary table.

Needs: data/processed/load.parquet (ENTSO-E backfill) + backfilled weather.
Weather input: archived forecasts at lead 2 when available (honest), else
ERA5 actuals with a loud warning in the report.

Run: python -m src.evaluation.run_backtest [--models seasonal_naive,ridge]
     [--test-start 2025-07-01]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

import src.models.baselines  # noqa: F401  (populates REGISTRY)
import src.models.gbm  # noqa: F401
from src.config import Config, load_config
from src.evaluation.backtest import BacktestResult, summarize, walk_forward_backtest
from src.features.matrix import build_features
from src.features.weather import load_weather_forecast_history, load_weather_history
from src.models.base import REGISTRY
from src.pipeline.daily_run import local_day_hours_utc, shift_local_day


def assemble_features(
    load: pd.Series, weather: pd.DataFrame, tz: str, start: pd.Timestamp,
    end: pd.Timestamp, tso: pd.Series | None = None,
) -> pd.DataFrame:
    """Cutoff-safe X for every local day in [start, end]. One build per day."""
    frames = []
    day = start
    while day <= end:
        hours = local_day_hours_utc(day, tz)
        cutoff = shift_local_day(day, -1, tz) + pd.Timedelta(hours=9)
        frames.append(build_features(hours, load, weather, cutoff, tso=tso))
        day = shift_local_day(day, 1, tz)
    return pd.concat(frames)


def main() -> int:
    cfg: Config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(REGISTRY))
    parser.add_argument("--test-start", default=None, help="YYYY-MM-DD local")
    parser.add_argument(
        "--weather", choices=["auto", "forecast", "actuals"], default="auto",
        help="forecast = archived lead-2 (honest); actuals = ERA5 (optimistic)",
    )
    parser.add_argument("--with-tso", action="store_true",
                        help="add the TSO day-ahead forecast as a feature")
    parser.add_argument("--tag", default=None, help="suffix for output names")
    args = parser.parse_args()

    load_path = cfg.paths["data_processed"] / "load.parquet"
    if not load_path.exists():
        print("Missing data/processed/load.parquet — run `make backfill` with an "
              "ENTSO-E token first.")
        return 1
    load = pd.read_parquet(load_path).iloc[:, 0]

    if args.weather == "actuals":
        weather = load_weather_history(cfg)
        weather_source = "ERA5 ACTUALS — optimistic, see DATA_CATALOG leakage note"
    else:
        try:
            weather = load_weather_forecast_history(cfg)
            weather_source = "archived forecasts, lead 2 days (honest)"
        except (FileNotFoundError, KeyError):
            if args.weather == "forecast":
                raise
            weather = load_weather_history(cfg)
            weather_source = "ERA5 ACTUALS — optimistic, see DATA_CATALOG leakage note"

    tz = cfg.timezone_local
    first = load.index[0].tz_convert(tz) + pd.Timedelta(days=30)
    last = load.index[-1].tz_convert(tz) - pd.Timedelta(days=1)
    test_start = (
        pd.Timestamp(args.test_start, tz=tz)
        if args.test_start
        else shift_local_day(last, -365, tz)
    )

    tso_feature = None
    if args.with_tso:
        tso_feature = pd.read_parquet(
            cfg.paths["data_processed"] / "tso_forecast.parquet"
        ).iloc[:, 0]
    print(f"Assembling features {first.date()} → {last.date()} ...")
    x = assemble_features(load, weather, tz, pd.Timestamp(first.date(), tz=tz),
                          pd.Timestamp(last.date(), tz=tz), tso=tso_feature)
    y = load.reindex(x.index)

    results: list[BacktestResult] = []
    for name in args.models.split(","):
        print(f"Backtesting {name} ...")
        results.append(
            walk_forward_backtest(REGISTRY[name], x, y, test_start.tz_convert("UTC"))
        )

    # The external benchmark: the TSO's own day-ahead forecast (point only).
    tso_path = cfg.paths["data_processed"] / "tso_forecast.parquet"
    if tso_path.exists():
        tso = pd.read_parquet(tso_path).iloc[:, 0]
        test_hours = results[0].predictions.index
        tso_pred = pd.DataFrame(
            {"p10": float("nan"), "p50": tso.reindex(test_hours), "p90": float("nan")},
            index=test_hours,
        )
        results.append(BacktestResult(model_name="tso_forecast", predictions=tso_pred))

    weather_tag = "fcst" if "forecasts" in weather_source else "actuals"
    if args.with_tso:
        weather_tag += "_tso"
    if args.tag:
        weather_tag += f"_{args.tag}"

    # Persist hourly predictions — diagnostics plots read these.
    preds_dir = cfg.paths["data_processed"] / f"backtest_preds_{weather_tag}"
    preds_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        r.predictions.to_parquet(preds_dir / f"{r.model_name}.parquet")

    table = summarize(results, y)
    out_dir = cfg.paths["data_processed"].parent.parent / "reports" / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{pd.Timestamp.now(tz).date()}_{weather_tag}"
    table.to_csv(out_dir / f"{stamp}_summary.csv")
    md = [
        f"# Backtest summary — {stamp}",
        "",
        f"Test period: {test_start.date()} → {last.date()}. "
        f"Weather input: {weather_source}.",
        "",
        table.round(2).to_markdown(),
        "",
    ]
    (out_dir / f"{stamp}_summary.md").write_text("\n".join(md))
    print(table.round(2).to_string())
    print(f"\nWritten to {out_dir}/{stamp}_summary.(csv|md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
