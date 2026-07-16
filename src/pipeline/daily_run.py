"""The daily dry run. One command: `make dry-run`.

Steps:
1. Fetch recent actual load, TSO forecast, weather.
2. Score yesterday: our naive vs TSO, against actuals.
3. Forecast tomorrow (P10/P50/P90), save CSV.
4. Write a short markdown report.

Run: python -m src.pipeline.daily_run [--date YYYY-MM-DD]
--date is "today" in local time. Default: now. Use it to re-run a past day.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from dotenv import load_dotenv

from src.clients.entsoe_client import fetch_load, fetch_tso_forecast
from src.clients.openmeteo_client import fetch_weather_forecast
from src.config import Config, load_config
from src.evaluation.metrics import mape
from src.models.naive import seasonal_naive_forecast
from src.pipeline.report import write_report


def shift_local_day(day: pd.Timestamp, n_days: int, tz: str) -> pd.Timestamp:
    """Local midnight n calendar days away. Calendar shift, not 24h — DST-safe."""
    return pd.Timestamp(day.date() + pd.Timedelta(days=n_days), tz=tz)


def local_day_hours_utc(day: pd.Timestamp, tz: str) -> pd.DatetimeIndex:
    """All hours of one local calendar day, as UTC timestamps.

    DST days have 23 or 25 hours. Adding Timedelta(days=1) to a tz-aware
    timestamp adds absolute 24h and lands on the wrong midnight — so the
    day boundary comes from calendar-date arithmetic instead.
    """
    start = pd.Timestamp(day.date(), tz=tz)
    end = shift_local_day(day, 1, tz)
    return pd.date_range(start, end, freq="1h", inclusive="left").tz_convert("UTC")


def run(cfg: Config, today_local: pd.Timestamp) -> str:
    tz = cfg.timezone_local
    yesterday = shift_local_day(today_local, -1, tz)
    tomorrow = shift_local_day(today_local, 1, tz)

    # 1. Fetch. History window ends now; naive only looks backwards.
    hist_start = today_local - pd.Timedelta(days=cfg.history_days)
    if cfg.data_source == "pse":
        from src.clients.pse_client import fetch_kse_load

        kse = fetch_kse_load(str(hist_start.date()), str(tomorrow.date()))
        load = kse["load_mw"].dropna()
        tso = kse["tso_forecast_mw"]
    else:
        load = fetch_load(
            cfg.zone, start=hist_start, end=today_local + pd.Timedelta(days=1)
        )
        tso = fetch_tso_forecast(
            cfg.zone, start=yesterday, end=tomorrow + pd.Timedelta(days=2)
        )
    city = cfg.cities[0]
    weather = fetch_weather_forecast(
        city.lat, city.lon, cfg.weather_vars, forecast_days=3, past_days=2
    )

    # Persist raw pulls for later inspection.
    cfg.paths["data_raw"].mkdir(parents=True, exist_ok=True)
    load.to_frame().to_parquet(cfg.paths["data_raw"] / "load_latest.parquet")
    tso.to_frame().to_parquet(cfg.paths["data_raw"] / "tso_forecast_latest.parquet")
    weather.to_parquet(cfg.paths["data_raw"] / f"weather_{city.name}_latest.parquet")

    # 2. Score yesterday.
    yhours = local_day_hours_utc(yesterday, tz)
    actual_y = load.reindex(yhours)
    naive_y = seasonal_naive_forecast(
        load[load.index < yhours[0]], yhours, cfg.naive_season_days, cfg.naive_n_seasons
    )
    scores = {
        "naive_mape": mape(actual_y, naive_y["p50"]),
        "tso_mape": mape(actual_y, tso.reindex(yhours)),
    }

    # 3. Forecast tomorrow.
    thours = local_day_hours_utc(tomorrow, tz)
    fc = seasonal_naive_forecast(load, thours, cfg.naive_season_days, cfg.naive_n_seasons)
    cfg.paths["forecasts"].mkdir(parents=True, exist_ok=True)
    fc_path = cfg.paths["forecasts"] / f"{tomorrow.date()}.csv"
    fc.rename_axis("time_utc").to_csv(fc_path, float_format="%.1f")

    # 3b. Shadow challenger (UAT): scored daily, promoted only by a human.
    challenger_note = None
    try:
        from src.ingestion.backfill import backfill_pse
        from src.pipeline.challenger import challenger_forecast

        backfill_pse(cfg)  # incremental: keeps the processed store current
        ch = challenger_forecast(cfg, today_local)
        ch_path = cfg.paths["forecasts"] / f"{tomorrow.date()}_challenger.csv"
        ch.rename_axis("time_utc").to_csv(ch_path, float_format="%.1f")
        # score yesterday's shadow forecast if it exists
        ch_y_path = cfg.paths["forecasts"] / f"{yesterday.date()}_challenger.csv"
        if ch_y_path.exists():
            ch_y = pd.read_csv(ch_y_path, index_col="time_utc", parse_dates=True)
            scores["challenger_mape"] = mape(actual_y, ch_y["p50"])
        else:
            challenger_note = "Challenger shadow started today; first score tomorrow."
    except Exception as exc:  # noqa: BLE001 — shadow must never kill the daily run
        challenger_note = f"Challenger failed: {exc}"

    # 4. Report, with the fan chart embedded.
    from src.viz.plots import plot_forecast_band

    fig_dir = cfg.paths["reports_daily"].parent / "figures" / "daily"
    plot_forecast_band(fc, str(tomorrow.date()), fig_dir / f"{tomorrow.date()}.png")

    # Living figure: yesterday's chart gets the realized load drawn on it
    # now that yesterday is fully observed. Old reports update in place.
    fc_y_path = cfg.paths["forecasts"] / f"{yesterday.date()}.csv"
    if fc_y_path.exists():
        fc_y = pd.read_csv(fc_y_path, index_col="time_utc", parse_dates=True)
        plot_forecast_band(
            fc_y, str(yesterday.date()), fig_dir / f"{yesterday.date()}.png",
            actual=actual_y,
        )

    oddities: list[str] = []
    n_missing = int(actual_y.isna().sum())
    if n_missing:
        oddities.append(f"{n_missing} of {len(yhours)} hours of yesterday's load are missing.")
    if fc["p50"].isna().any():
        oddities.append(f"{int(fc['p50'].isna().sum())} forecast hours have no history (NaN).")
    if challenger_note:
        oddities.append(challenger_note)

    # 3c. Price (Phase 2, shadow): scored next day against the realized
    # price. Isolated like the challenger — must never kill this report.
    price_lines: list[str] = []
    try:
        from src.pipeline.price_daily import price_daily_step

        price_scores, price_lines, price_odd = price_daily_step(cfg, today_local)
        scores.update(price_scores)
        oddities.extend(price_odd)
    except Exception as exc:  # noqa: BLE001 — price step must never kill the daily run
        oddities.append(f"Price step failed: {exc}")

    report_path = write_report(
        cfg=cfg,
        today_local=today_local,
        scores=scores,
        forecast=fc,
        weather=weather,
        oddities=oddities,
        extra_sections=price_lines,
    )
    return str(report_path)


def main() -> int:
    load_dotenv()
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="Local 'today' as YYYY-MM-DD.")
    args = parser.parse_args()
    today_local = (
        pd.Timestamp(args.date, tz=cfg.timezone_local)
        if args.date
        else pd.Timestamp.now(tz=cfg.timezone_local).normalize()
    )
    path = run(cfg, today_local)
    print(f"Report written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
