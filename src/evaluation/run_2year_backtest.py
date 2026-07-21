"""2-year walk-forward backtest (2024-07-16 → present) + rolling-vs-expanding ablation.

Models: seasonal_naive, ridge, ridge_tso, lgbm_tso.

Weather:
  Test days (2024-07-16+): lead-2 forecast archive (honest, no leakage).
  Training days pre-2024: ERA5 actuals (fine for training — model learns
  the weather→load relationship; ERA5 is the "ground truth").
  Combined into one hybrid series before assembling features.

TSO feature:
  PSE canonical series starts 2023-01-01. All test AND training days have
  valid TSO. No gap to handle.

Rolling-vs-expanding ablation:
  Same ridge and ridge_tso models, same 2-year test window.
  Rolling: 365-day window (current prod default).
  Expanding: all history from 2023-01-01 (grows with time).
  Verdict written to docs/notes/model_selection/.

Run: uv run python -m src.evaluation.run_2year_backtest
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import src.models.baselines  # noqa: F401
import src.models.gbm  # noqa: F401
from src.config import load_config
from src.evaluation.backtest import BacktestResult, summarize, walk_forward_backtest
from src.features.weather import load_weather_forecast_history, load_weather_history
from src.models.baselines import RidgeForecaster
from src.models.gbm import LightGBMQuantile
from src.pipeline.daily_run import local_day_hours_utc, shift_local_day
from src.features.matrix import build_features

TEST_START_LOCAL = "2024-07-16"
TZ = "Europe/Warsaw"


# ---------------------------------------------------------------------------
# Named wrappers — same model code, distinct .name for the output table
# ---------------------------------------------------------------------------

class RidgeTSO(RidgeForecaster):
    name = "ridge_tso"


class LGBMQuantileTSO(LightGBMQuantile):
    name = "lgbm_tso"


# ---------------------------------------------------------------------------
# Hybrid weather: ERA5 before 2024-01-01, lead-2 forecasts from 2024-01-01
# ---------------------------------------------------------------------------

def make_hybrid_weather(cfg) -> pd.DataFrame:
    era5 = load_weather_history(cfg)
    try:
        fcst = load_weather_forecast_history(cfg)
    except (FileNotFoundError, KeyError):
        print("WARNING: no lead-2 forecast archive found — using ERA5 only (optimistic)")
        return era5
    cutoff = pd.Timestamp("2024-01-01", tz="UTC")
    pre = era5[era5.index < cutoff]
    post = fcst[fcst.index >= cutoff]
    return pd.concat([pre, post]).sort_index()


# ---------------------------------------------------------------------------
# Feature assembly (day-by-day, cutoff-safe)
# ---------------------------------------------------------------------------

def assemble(
    load: pd.Series,
    weather: pd.DataFrame,
    tz: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    tso: pd.Series | None = None,
) -> pd.DataFrame:
    frames = []
    day = start
    while day <= end:
        hours = local_day_hours_utc(day, tz)
        cutoff = shift_local_day(day, -1, tz) + pd.Timedelta(hours=9)
        frames.append(build_features(hours, load, weather, cutoff, tso=tso))
        day = shift_local_day(day, 1, tz)
    return pd.concat(frames)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    cfg = load_config()
    tz = cfg.timezone_local

    load_path = cfg.paths["data_processed"] / "load.parquet"
    tso_path = cfg.paths["data_processed"] / "tso_forecast.parquet"
    if not load_path.exists():
        print("Missing data/processed/load.parquet — run `make backfill` first.")
        return 1

    load = pd.read_parquet(load_path).iloc[:, 0]
    tso = pd.read_parquet(tso_path).iloc[:, 0] if tso_path.exists() else None
    weather = make_hybrid_weather(cfg)

    test_start_local = pd.Timestamp(TEST_START_LOCAL, tz=tz)
    data_start = load.index[0].tz_convert(tz) + pd.Timedelta(days=30)
    data_end = load.index[-1].tz_convert(tz) - pd.Timedelta(days=1)

    print(f"Assembling features {data_start.date()} → {data_end.date()} ...")
    x_no_tso = assemble(load, weather, tz,
                        pd.Timestamp(data_start.date(), tz=tz),
                        pd.Timestamp(data_end.date(), tz=tz))
    x_with_tso = assemble(load, weather, tz,
                          pd.Timestamp(data_start.date(), tz=tz),
                          pd.Timestamp(data_end.date(), tz=tz),
                          tso=tso)

    y = load.reindex(x_no_tso.index)
    test_start_utc = test_start_local.tz_convert("UTC")
    last_date = data_end.date()

    # ------------------------------------------------------------------
    # Task 1: 2-year backtest, four models
    # ------------------------------------------------------------------
    print("\n=== 2-year walk-forward backtest ===")
    print(f"Test period: {TEST_START_LOCAL} → {last_date}")
    print("Weather: ERA5 pre-2024 (training) / lead-2 forecast 2024+ (test)")

    runs: list[tuple[str, object, pd.DataFrame]] = [
        ("seasonal_naive", None, x_no_tso),   # naive doesn't fit
        ("ridge",          RidgeForecaster,   x_no_tso),
        ("ridge_tso",      RidgeTSO,          x_with_tso),
        ("lgbm_tso",       LGBMQuantileTSO,   x_with_tso),
    ]

    results: list[BacktestResult] = []
    for name, factory, x in runs:
        print(f"  Backtesting {name} ...")
        if factory is None:
            from src.models.baselines import SeasonalNaive
            factory = SeasonalNaive
        r = walk_forward_backtest(factory, x, y, test_start_utc)
        # rename in case factory returns a different .name
        results.append(BacktestResult(model_name=name, predictions=r.predictions))

    # Add TSO external benchmark
    if tso is not None:
        test_hours = results[0].predictions.index
        tso_series = tso.reindex(test_hours)
        results.append(BacktestResult(
            model_name="tso_forecast",
            predictions=pd.DataFrame(
                {"p10": float("nan"), "p50": tso_series, "p90": float("nan")},
                index=test_hours,
            ),
        ))

    table_2yr = summarize(results, y)
    print("\n2-year results:")
    print(table_2yr.round(2).to_string())

    # ------------------------------------------------------------------
    # Task 2: rolling-vs-expanding ablation (ridge and ridge_tso)
    # ------------------------------------------------------------------
    print("\n=== Rolling vs expanding window ablation ===")
    ablation_runs: list[tuple[str, object, pd.DataFrame, int]] = [
        ("ridge_rolling365",      RidgeForecaster, x_no_tso,   365),
        ("ridge_expanding",       RidgeForecaster, x_no_tso,   99999),
        ("ridge_tso_rolling365",  RidgeTSO,        x_with_tso, 365),
        ("ridge_tso_expanding",   RidgeTSO,        x_with_tso, 99999),
    ]

    ablation_results: list[BacktestResult] = []
    for name, factory, x, window in ablation_runs:
        print(f"  Ablation {name} (window={window}) ...")
        r = walk_forward_backtest(factory, x, y, test_start_utc,
                                  train_window_days=window)
        ablation_results.append(BacktestResult(model_name=name, predictions=r.predictions))

    # Add naive baseline for skill computation
    naive_res = [r for r in results if r.model_name == "seasonal_naive"]
    table_ablation = summarize(ablation_results + naive_res, y)
    print("\nAblation results:")
    print(table_ablation.round(2).to_string())

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    out_dir = Path("reports/backtests")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = str(pd.Timestamp.now(tz).date())

    # Predictions
    preds_dir = cfg.paths["data_processed"] / "backtest_preds_2yr"
    preds_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        r.predictions.to_parquet(preds_dir / f"{r.model_name}.parquet")

    # Summary CSVs
    table_2yr.to_csv(out_dir / f"{stamp}_2yr_summary.csv")
    table_ablation.to_csv(out_dir / f"{stamp}_2yr_ablation.csv")

    # Markdown reports
    _write_md(
        out_dir / f"{stamp}_2yr_summary.md",
        f"2-year backtest — {stamp}",
        f"Test period: {TEST_START_LOCAL} → {last_date}. "
        "Weather: ERA5 pre-2024 (training), lead-2 forecast 2024+ (honest test). "
        "TSO feature from PSE canonical (2023-01-01+).",
        table_2yr,
    )
    _write_md(
        out_dir / f"{stamp}_2yr_ablation.md",
        f"Rolling-vs-expanding ablation — {stamp}",
        f"Test period: {TEST_START_LOCAL} → {last_date}. "
        "Rolling: 365-day window. Expanding: all history from 2023-01-01.",
        table_ablation,
    )

    print(f"\nWritten to {out_dir}/")
    return 0


def _write_md(path: Path, title: str, subtitle: str, table: pd.DataFrame) -> None:
    lines = [f"# {title}", "", subtitle, "", table.round(2).to_markdown(), ""]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
