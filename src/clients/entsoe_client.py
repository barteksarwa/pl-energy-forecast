"""ENTSO-E Transparency Platform client. Load, TSO load forecast, day-ahead prices.

All returns: hourly pandas Series, UTC tz-aware index.
Load/forecast unit: MW. Price unit: EUR/MWh.
ENTSO-E may deliver 15-min resolution; we resample to hourly mean.
"""

from __future__ import annotations

import os

import pandas as pd
from entsoe import EntsoePandasClient


def _client() -> EntsoePandasClient:
    token = os.environ.get("ENTSOE_API_TOKEN")
    if not token:
        raise RuntimeError(
            "ENTSOE_API_TOKEN not set. Copy .env.example to .env and add your token."
        )
    return EntsoePandasClient(api_key=token)


def _to_hourly_utc(obj: pd.Series | pd.DataFrame, name: str) -> pd.Series:
    series = obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj
    series = series.tz_convert("UTC").resample("1h").mean()
    series.name = name
    return series


def fetch_load(zone: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Actual total load for a bidding zone."""
    raw = _client().query_load(zone, start=start, end=end)
    return _to_hourly_utc(raw, "load_mw")


def fetch_tso_forecast(zone: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """TSO day-ahead load forecast. Our external benchmark."""
    raw = _client().query_load_forecast(zone, start=start, end=end)
    return _to_hourly_utc(raw, "tso_forecast_mw")


def fetch_day_ahead_price(
    zone: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    """Day-ahead market clearing price. Returns EUR/MWh, hourly UTC.

    ENTSO-E endpoint: Day Ahead Prices (12.1.D).
    Some periods return 15-min data; resampled to hourly mean.
    Store as EUR/MWh. Convert to PLN only at display time.
    """
    raw = _client().query_day_ahead_prices(zone, start=start, end=end)
    return _to_hourly_utc(raw, "price_da_eur")
