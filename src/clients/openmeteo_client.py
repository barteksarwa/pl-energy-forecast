"""Open-Meteo client. Free, no key.

Returns: hourly DataFrame, UTC tz-aware index, one column per weather variable.
"""

from __future__ import annotations

import pandas as pd
import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_S = 30


def fetch_weather_forecast(
    lat: float,
    lon: float,
    hourly_vars: list[str],
    forecast_days: int = 3,
    past_days: int = 2,
) -> pd.DataFrame:
    """Weather forecast (plus recent past) for one location."""
    params: dict[str, str | float | int] = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly_vars),
        "timezone": "UTC",
        "forecast_days": forecast_days,
        "past_days": past_days,
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=TIMEOUT_S)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    index = pd.DatetimeIndex(pd.to_datetime(hourly["time"], utc=True), name="time")
    return pd.DataFrame({v: hourly[v] for v in hourly_vars}, index=index)
