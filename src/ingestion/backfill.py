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
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
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
    # Multi-year archive pulls time out on cold CI runners — retry with backoff.
    resp = None
    for attempt in range(3):
        try:
            resp = requests.get(ARCHIVE_URL, params=params, timeout=TIMEOUT_S)
            resp.raise_for_status()
            break
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 2:
                raise
            time.sleep(10 * (attempt + 1))
    assert resp is not None
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
        # Yearly chunks: one 3.5-year request is what timed out on CI.
        combined = None
        chunk_start = start_ts.date()
        while chunk_start <= end_date:
            chunk_end = min(chunk_start + pd.Timedelta(days=365), end_date)
            df = fetch_weather_archive(
                city.lat, city.lon, cfg.weather_vars,
                str(chunk_start), str(chunk_end),
            )
            combined = _merge_save(path, df)
            chunk_start = chunk_end + pd.Timedelta(days=1)
            time.sleep(cfg.request_sleep_s)
        if combined is not None:
            gaps = log_gaps(combined.iloc[:, 0], f"weather_{city.name}", gap_log)
            print(f"weather {city.name}: total {len(combined)}, {len(gaps)} gap(s)")


def fetch_weather_forecast_archive(
    lat: float, lon: float, hourly_vars: list[str], leads: list[int], start: str, end: str
) -> pd.DataFrame:
    """Archived forecasts at fixed lead times (Previous Runs API).

    Column naming: temperature_2m_lead2d = value predicted 48h before valid time.
    """
    api_vars = [f"{v}_previous_day{d}" for v in hourly_vars for d in leads]
    params: dict[str, str | float] = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(api_vars),
        "timezone": "UTC",
        "start_date": start,
        "end_date": end,
    }
    resp = None
    for attempt in range(3):
        try:
            resp = requests.get(PREVIOUS_RUNS_URL, params=params, timeout=120)
            resp.raise_for_status()
            break
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    assert resp is not None
    hourly = resp.json()["hourly"]
    index = pd.DatetimeIndex(pd.to_datetime(hourly["time"], utc=True), name="time")
    frame = pd.DataFrame({v: hourly[v] for v in api_vars}, index=index)
    return frame.rename(
        columns={
            f"{v}_previous_day{d}": f"{v}_lead{d}d" for v in hourly_vars for d in leads
        }
    )


def backfill_weather_forecasts(cfg: Config) -> None:
    """Archived weather forecasts per city, yearly chunks, resumable."""
    gap_log = cfg.paths["data_processed"] / "gap_log.csv"
    end_date = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)).date()
    for city in cfg.cities:
        path = cfg.paths["data_raw"] / "weather_forecast" / f"{city.name}.parquet"
        start_ts = _resume_start(path, pd.Timestamp(cfg.forecast_start, tz="UTC"))
        if start_ts.date() > end_date:
            print(f"weather_forecast {city.name}: up to date")
            continue
        combined = None
        chunk_start = start_ts.date()
        while chunk_start <= end_date:
            chunk_end = min(chunk_start + pd.Timedelta(days=120), end_date)
            df = fetch_weather_forecast_archive(
                city.lat, city.lon, cfg.weather_vars, cfg.forecast_leads,
                str(chunk_start), str(chunk_end),
            )
            combined = _merge_save(path, df)
            chunk_start = chunk_end + pd.Timedelta(days=1)
            time.sleep(cfg.request_sleep_s)
        if combined is not None:
            gaps = log_gaps(combined.iloc[:, 0], f"weather_forecast_{city.name}", gap_log)
            print(
                f"weather_forecast {city.name}: total {len(combined)}, {len(gaps)} gap(s)"
            )


def backfill_pse(cfg: Config) -> None:
    """Load + TSO forecast from PSE API v2. Keyless. Data from 2024-06-14."""
    from src.clients.pse_client import fetch_kse_load

    gap_log = cfg.paths["data_processed"] / "gap_log.csv"
    load_path = cfg.paths["data_processed"] / "load.parquet"
    tso_path = cfg.paths["data_processed"] / "tso_forecast.parquet"
    pse_start = max(pd.Timestamp(cfg.backfill_start, tz="UTC"),
                    pd.Timestamp("2024-06-14", tz="UTC"))
    start_ts = _resume_start(load_path, pse_start)
    # TSO forecast exists through tomorrow; fetch that far.
    end_date = (pd.Timestamp.now(tz="Europe/Warsaw") + pd.Timedelta(days=1)).date()
    if start_ts.date() > end_date:
        print("pse: up to date")
        return
    chunk_start = start_ts.date()
    while chunk_start <= end_date:
        chunk_end = min(chunk_start + pd.Timedelta(days=60), end_date)
        df = fetch_kse_load(str(chunk_start), str(chunk_end))
        if not df.empty:
            _merge_save(load_path, df[["load_mw"]].dropna())
            _merge_save(tso_path, df[["tso_forecast_mw"]].dropna())
            print(f"pse: {chunk_start} → {chunk_end}, {len(df)} hours")
        else:
            print(f"pse: {chunk_start} → {chunk_end}, empty")
        chunk_start = chunk_end + pd.Timedelta(days=1)
        time.sleep(cfg.request_sleep_s)
    for name, path in [("load", load_path), ("tso_forecast", tso_path)]:
        if path.exists():
            gaps = log_gaps(pd.read_parquet(path).iloc[:, 0], f"pse_{name}", gap_log)
            print(f"pse {name}: {len(gaps)} new gap(s) logged")


PSE_PRICE_ENTITIES = {
    # entity: (api_col -> out_col, output file stem)
    "csdac-pln": ({"csdac_pln": "price_da_pln"}, "price_da"),
    "crb-rozl": ({"cen_cost": "price_bal_cen_pln"}, "price_balancing"),
    "his-wlk-cal": ({"pv": "gen_pv_mw", "wi": "gen_wind_mw", "jg": "gen_thermal_mw"},
                    "generation_mix"),
}


def backfill_pse_prices(cfg: Config) -> None:
    """Phase 2 data: day-ahead price, balancing price, generation mix. Keyless."""
    from src.clients.pse_client import fetch_entity_hourly

    gap_log = cfg.paths["data_processed"] / "gap_log.csv"
    end_date = (pd.Timestamp.now(tz="Europe/Warsaw") + pd.Timedelta(days=1)).date()
    for entity, (cols, stem) in PSE_PRICE_ENTITIES.items():
        path = cfg.paths["data_processed"] / f"{stem}.parquet"
        start_ts = _resume_start(path, pd.Timestamp("2024-06-14", tz="UTC"))
        if start_ts.date() > end_date:
            print(f"{stem}: up to date")
            continue
        chunk_start = start_ts.date()
        while chunk_start <= end_date:
            chunk_end = min(chunk_start + pd.Timedelta(days=60), end_date)
            df = fetch_entity_hourly(entity, cols, str(chunk_start), str(chunk_end))
            if not df.empty:
                _merge_save(path, df.dropna(how="all"))
            chunk_start = chunk_end + pd.Timedelta(days=1)
            time.sleep(cfg.request_sleep_s)
        if path.exists():
            combined = pd.read_parquet(path)
            gaps = log_gaps(combined.iloc[:, 0], stem, gap_log)
            print(f"{stem}: total {len(combined)}, {len(gaps)} new gap(s)")


def backfill_entsoe(cfg: Config) -> None:
    """ENTSO-E into its OWN store (data/processed/entsoe/).

    Role: deep history (2023+, PSE v2 only starts 2024-06-14) and an
    independent cross-check of the same series. PSE stays canonical for
    the overlap (DECISIONS 2026-07-14); merge happens in crosscheck.py.
    """
    if not os.environ.get("ENTSOE_API_TOKEN"):
        print("entsoe: skipped — ENTSOE_API_TOKEN not set in .env")
        return
    from src.clients.entsoe_client import fetch_load, fetch_tso_forecast

    gap_log = cfg.paths["data_processed"] / "gap_log.csv"
    now = pd.Timestamp.now(tz="UTC").floor("1h")
    entsoe_dir = cfg.paths["data_processed"] / "entsoe"
    targets = {
        "entsoe_load": (fetch_load, entsoe_dir / "load.parquet"),
        "entsoe_tso_forecast": (fetch_tso_forecast, entsoe_dir / "tso_forecast.parquet"),
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


def backfill_entsoe_prices(cfg: Config) -> None:
    """Day-ahead prices (EUR/MWh) from ENTSO-E, full config history.

    Stored in data/processed/price_da_eur.parquet. EUR is the raw unit of
    the ENTSO-E feed; PLN conversion happens at display time only.
    PSE (price_da.parquet, PLN) starts 2024-06-14; this series extends
    history back to cfg.backfill_start.

    No try/except around the fetch: a failed chunk must abort the run so
    the next resume (_resume_start) re-fetches it. Swallowing the error
    would advance past the hole and make the gap permanent.
    """
    if not os.environ.get("ENTSOE_API_TOKEN"):
        print("entsoe_prices: skipped — ENTSOE_API_TOKEN not set in .env")
        return
    from entsoe.exceptions import NoMatchingDataError

    from src.clients.entsoe_client import fetch_day_ahead_price

    gap_log = cfg.paths["data_processed"] / "gap_log.csv"
    # Day-ahead prices exist through the END OF TOMORROW once the D-1
    # auction clears (~13:00 CET) — not merely up to "now". Capping at
    # now starved the daily price forecast of yesterday's auction results.
    horizon = pd.Timestamp.now(tz="Europe/Warsaw").normalize() + pd.Timedelta(days=2)
    horizon = horizon.tz_convert("UTC")
    path = cfg.paths["data_processed"] / "price_da_eur.parquet"
    start = _resume_start(path, pd.Timestamp(cfg.backfill_start, tz="UTC"))
    chunk = pd.Timedelta(days=cfg.entsoe_chunk_days)
    while start < horizon:
        end = min(start + chunk, horizon)
        try:
            series = fetch_day_ahead_price(cfg.zone, start=start, end=end)
        except NoMatchingDataError:
            # nothing published for this window yet (e.g. tomorrow before
            # the auction). Store nothing; resume retries the same window.
            print(f"entsoe_prices: {start.date()} → {end.date()}, not published yet")
            break
        combined = _merge_save(path, series.to_frame())
        print(f"entsoe_prices: {start.date()} → {end.date()}, total {len(combined)}")
        start = end
        time.sleep(cfg.request_sleep_s)
    if path.exists():
        gaps = log_gaps(pd.read_parquet(path).iloc[:, 0], "price_da_eur", gap_log)
        print(f"entsoe_prices: {len(gaps)} new gap(s) logged")


def backfill_entsoe_res(cfg: Config) -> None:
    """TSO day-ahead wind + solar forecast (MW). Price driver #1.

    Stored in data/processed/res_forecast.parquet, full config history.
    Same no-try/except rule as prices: a failed chunk aborts so resume
    re-fetches it.
    """
    if not os.environ.get("ENTSOE_API_TOKEN"):
        print("entsoe_res: skipped — ENTSOE_API_TOKEN not set in .env")
        return
    from entsoe.exceptions import NoMatchingDataError

    from src.clients.entsoe_client import fetch_res_forecast

    gap_log = cfg.paths["data_processed"] / "gap_log.csv"
    # The TSO publishes tomorrow's wind/solar forecast on D-1 (~18:00);
    # fetch through end of tomorrow, same reasoning as entsoe_prices.
    horizon = pd.Timestamp.now(tz="Europe/Warsaw").normalize() + pd.Timedelta(days=2)
    horizon = horizon.tz_convert("UTC")
    path = cfg.paths["data_processed"] / "res_forecast.parquet"
    start = _resume_start(path, pd.Timestamp(cfg.backfill_start, tz="UTC"))
    chunk = pd.Timedelta(days=cfg.entsoe_chunk_days)
    while start < horizon:
        end = min(start + chunk, horizon)
        try:
            df = fetch_res_forecast(cfg.zone, start=start, end=end)
        except NoMatchingDataError:
            print(f"entsoe_res: {start.date()} → {end.date()}, not published yet")
            break
        combined = _merge_save(path, df)
        print(f"entsoe_res: {start.date()} → {end.date()}, total {len(combined)}")
        start = end
        time.sleep(cfg.request_sleep_s)
    if path.exists():
        gaps = log_gaps(pd.read_parquet(path).iloc[:, 0], "res_forecast", gap_log)
        print(f"entsoe_res: {len(gaps)} new gap(s) logged")


def backfill_entsoe_outages(cfg: Config) -> None:
    """Generation-unit outage messages, event-level store.

    data/processed/outages.parquet, deduped on (mrid, revision).
    Query windows cover the OUTAGE PERIOD, so incremental runs re-query
    a trailing window (revisions arrive late) and the dedupe absorbs
    the overlap. Publication time (created_doc_time) is kept — it is
    the leakage boundary for any feature built on this store.
    """
    if not os.environ.get("ENTSOE_API_TOKEN"):
        print("entsoe_outages: skipped — ENTSOE_API_TOKEN not set in .env")
        return
    from entsoe.exceptions import NoMatchingDataError

    from src.clients.entsoe_client import fetch_outages

    path = cfg.paths["data_processed"] / "outages.parquet"
    horizon = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=60)
    if path.exists():
        # incremental: revisions and new messages for ongoing/future periods
        start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=60)
        old = pd.read_parquet(path)
    else:
        start = pd.Timestamp(cfg.backfill_start, tz="UTC")
        old = None
    chunk = pd.Timedelta(days=30)
    frames = [] if old is None else [old]
    while start < horizon:
        end = min(start + chunk, horizon)
        try:
            df = fetch_outages(cfg.zone, start=start, end=end)
            frames.append(df)
            print(f"entsoe_outages: {start.date()} → {end.date()}, {len(df)} events")
        except NoMatchingDataError:
            print(f"entsoe_outages: {start.date()} → {end.date()}, none")
        start = end
        time.sleep(cfg.request_sleep_s)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["mrid", "revision"], keep="last")
    combined.to_parquet(path)
    print(f"entsoe_outages: total {len(combined)} events")


def backfill_fuel(cfg: Config) -> None:
    """TTF gas + EUA-proxy daily closes (yfinance). Full refetch — one
    request, seconds; idempotence via overwrite."""
    from src.clients.fuel_client import fetch_fuel_history

    path = cfg.paths["data_processed"] / "fuel_daily.parquet"
    df = fetch_fuel_history(start=cfg.backfill_start)
    df.to_parquet(path)
    print(f"fuel: {len(df)} trading days, {df.index.min().date()} → {df.index.max().date()}")


def main() -> int:
    load_dotenv()
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=[
            "weather", "weather_forecast", "pse", "pse_prices",
            "entsoe", "entsoe_prices", "entsoe_res", "entsoe_outages", "fuel",
        ],
        default=None,
    )
    args = parser.parse_args()
    if args.only in (None, "weather"):
        backfill_weather(cfg)
    if args.only in (None, "weather_forecast"):
        backfill_weather_forecasts(cfg)
    if args.only in (None, "pse"):
        backfill_pse(cfg)
    if args.only in (None, "pse_prices"):
        backfill_pse_prices(cfg)
    if args.only in (None, "entsoe"):
        backfill_entsoe(cfg)
    if args.only in (None, "entsoe_prices"):
        backfill_entsoe_prices(cfg)
    if args.only in (None, "entsoe_res"):
        backfill_entsoe_res(cfg)
    if args.only == "entsoe_outages":
        # opt-in ONLY: heavy, throttle-prone endpoint (503s on CI), and the
        # outage feature is research-tier (backtest verdict: flat) — the
        # daily ops loop does not need it.
        backfill_entsoe_outages(cfg)
    if args.only in (None, "fuel"):
        backfill_fuel(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
