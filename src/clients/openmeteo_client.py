"""Open-Meteo client. Free, no key.

Returns: hourly DataFrame, UTC tz-aware index, one column per weather variable.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_S = 30
# The challenger fetches every config city in sequence; without retries a
# single transient timeout killed the whole shadow run for days
# (2026-07-17 → 07-27 gate stall).
RETRIES = 3
BACKOFF_S = 2.0


def fetch_weather_forecast(
    lat: float,
    lon: float,
    hourly_vars: list[str],
    forecast_days: int = 3,
    past_days: int = 2,
) -> pd.DataFrame:
    """Weather forecast (plus recent past) for one location.

    Retries transient network errors (timeouts, 5xx) with backoff;
    raises the last error if all attempts fail.
    """
    params: dict[str, str | float | int] = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly_vars),
        "timezone": "UTC",
        "forecast_days": forecast_days,
        "past_days": past_days,
    }
    last_error: Exception | None = None
    for attempt in range(RETRIES):
        try:
            resp = requests.get(FORECAST_URL, params=params, timeout=TIMEOUT_S)
            resp.raise_for_status()
            break
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt < RETRIES - 1:
                time.sleep(BACKOFF_S * 2**attempt)
    else:
        raise last_error  # type: ignore[misc]  # loop ran, so it is set
    hourly = resp.json()["hourly"]
    index = pd.DatetimeIndex(pd.to_datetime(hourly["time"], utc=True), name="time")
    return pd.DataFrame({v: hourly[v] for v in hourly_vars}, index=index)
