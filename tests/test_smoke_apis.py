"""One integration smoke test per API client. Marked `smoke` — not run in unit CI.

Run: pytest -m smoke
"""

import os

import pandas as pd
import pytest
from dotenv import load_dotenv

from src.clients.openmeteo_client import fetch_weather_forecast

load_dotenv()

HAS_TOKEN = bool(os.environ.get("ENTSOE_API_TOKEN"))


@pytest.mark.smoke
def test_openmeteo_returns_hourly_utc_frame() -> None:
    df = fetch_weather_forecast(52.23, 21.01, ["temperature_2m"], forecast_days=2, past_days=1)
    assert str(df.index.tz) == "UTC"
    assert "temperature_2m" in df.columns
    assert len(df) == 3 * 24


@pytest.mark.smoke
@pytest.mark.skipif(not HAS_TOKEN, reason="ENTSOE_API_TOKEN not set")
def test_entsoe_load_returns_hourly_utc_series() -> None:
    from src.clients.entsoe_client import fetch_load

    end = pd.Timestamp.now(tz="Europe/Warsaw").normalize()
    start = end - pd.Timedelta(days=3)
    load = fetch_load("PL", start=start, end=end)
    assert str(load.index.tz) == "UTC"
    assert load.notna().sum() > 24  # at least a day of real data
    assert 10_000 < load.dropna().median() < 35_000  # sane MW range for Poland
