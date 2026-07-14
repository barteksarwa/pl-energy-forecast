"""Backfill history: weather (Open-Meteo archive) and load/TSO forecast (ENTSO-E).

Idempotent: existing parquet files are extended from their last timestamp,
not re-downloaded. Gaps are logged to data/processed/gap_log.csv.

Run: make backfill            # everything possible
     python -m src.ingestion.backfill --only weather
     python -m src.ingestion.backfill --only entsoe
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import Config, load_config
from src.ingestion.gaps import log_gaps

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEOUT_S = 60


def _resume_start(path: Path, default_start: pd.Timestamp) -> pd.Timestamp:
    """Existing file → continue after its last row. Otherwise full history."""
    if not path.exists():
        return default_start
    last = pd.read_parquet(path).index.max()
    return last + pd.Timedelta(hours=1)


def _merge_save(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        old = pd.read_parquet(path)
        combined = pd.concat([old, new])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new.sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path)
    return combined


def fetch_weather_archive(
    lat: float, lon: float, hourly_vars: list[str], start: str, end: str
) -> pd.DataFrame:
    params: dict[str, str | float] = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly_vars),
        "timezone": "UTC",
        "start_date": start,
        "end_date": end,
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=TIMEOUT_S)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    index = pd.DatetimeIndex(pd.to_datetime(hourly["time"], utc=True), name="time")
    return pd.DataFrame({v: hourly[v] for v in hourly_vars}, index=index)


def backfill_weather(cfg: Config) -> None:
    gap_log = cfg.paths["data_processed"] / "gap_log.csv"
    end_date = (
        pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=cfg.archive_lag_days)
    ).date()
    for city in cfg.cities:
        path = cfg.paths["data_raw"] / "weather" / f"{city.name}.parquet"
        start_ts = _resume_start(path, pd.Timestamp(cfg.backfill_start, tz="UTC"))
        if start_ts.date() > end_date:
            print(f"weather {city.name}: up to date")
            continue
        df = fetch_weather_archive(
            city.lat, city.lon, cfg.weather_vars, str(start_ts.date()), str(end_date)
        )
        combined = _merge_save(path, df)
        gaps = log_gaps(combined.iloc[:, 0], f"weather_{city.name}", gap_log)
        print(
            f"weather {city.name}: {len(df)} new rows, total {len(combined)}, "
            f"{len(gaps)} gap(s)"
        )
        time.sleep(cfg.request_sleep_s)


def backfill_entsoe(cfg: Config) -> None:
    if not os.environ.get("ENTSOE_API_TOKEN"):
        print("entsoe: skipped — ENTSOE_API_TOKEN not set in .env")
        return
    from src.clients.entsoe_client import fetch_load, fetch_tso_forecast

    gap_log = cfg.paths["data_processed"] / "gap_log.csv"
    now = pd.Timestamp.now(tz="UTC").floor("1h")
    targets = {
        "load": (fetch_load, cfg.paths["data_processed"] / "load.parquet"),
        "tso_forecast": (
            fetch_tso_forecast,
            cfg.paths["data_processed"] / "tso_forecast.parquet",
        ),
    }
    for name, (fetch, path) in targets.items():
        start = _resume_start(path, pd.Timestamp(cfg.backfill_start, tz="UTC"))
        chunk = pd.Timedelta(days=cfg.entsoe_chunk_days)
        while start < now:
            end = min(start + chunk, now)
            series = fetch(cfg.zone, start=start, end=end)
            combined = _merge_save(path, series.to_frame())
            print(f"{name}: {start.date()} → {end.date()}, total {len(combined)}")
            start = end
            time.sleep(cfg.request_sleep_s)
        if path.exists():
            gaps = log_gaps(pd.read_parquet(path).iloc[:, 0], name, gap_log)
            print(f"{name}: {len(gaps)} new gap(s) logged")


def main() -> int:
    load_dotenv()
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["weather", "entsoe"], default=None)
    args = parser.parse_args()
    if args.only in (None, "weather"):
        backfill_weather(cfg)
    if args.only in (None, "entsoe"):
        backfill_entsoe(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
