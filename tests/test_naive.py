"""Unit tests for the seasonal naive model and day-hour helpers."""

import numpy as np
import pandas as pd
import pytest

from src.models.naive import seasonal_naive_forecast
from src.pipeline.daily_run import local_day_hours_utc

TZ = "Europe/Warsaw"


def _hourly_series(start: str, days: int) -> pd.Series:
    idx = pd.date_range(start, periods=days * 24, freq="1h", tz="UTC")
    return pd.Series(np.arange(len(idx), dtype=float), index=idx)


def test_p50_is_same_hour_last_week() -> None:
    load = _hourly_series("2026-05-01", 35)
    target = pd.DatetimeIndex([load.index[-1] + pd.Timedelta(hours=1)])
    fc = seasonal_naive_forecast(load, target, season_days=7, n_seasons=4)
    expected = load.loc[target[0] - pd.Timedelta(days=7)]
    assert fc.loc[target[0], "p50"] == expected


def test_quantiles_ordered() -> None:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-05-01", periods=35 * 24, freq="1h", tz="UTC")
    load = pd.Series(rng.normal(20000, 1000, len(idx)), index=idx)
    target = pd.DatetimeIndex([idx[-1] + pd.Timedelta(hours=1)])
    fc = seasonal_naive_forecast(load, target)
    assert fc.loc[target[0], "p10"] <= fc.loc[target[0], "p50"] <= fc.loc[target[0], "p90"]


def test_no_history_gives_nan() -> None:
    load = _hourly_series("2026-05-01", 3)  # less than one season
    target = pd.DatetimeIndex([load.index[-1] + pd.Timedelta(hours=1)])
    fc = seasonal_naive_forecast(load, target)
    assert fc.isna().all().all()


def test_naive_rejects_naive_timestamps() -> None:
    idx = pd.date_range("2026-05-01", periods=24, freq="1h")  # no tz
    load = pd.Series(1.0, index=idx)
    with pytest.raises(ValueError):
        seasonal_naive_forecast(load, idx)


def test_dst_spring_day_has_23_hours() -> None:
    # 2026-03-29: clocks jump forward in Europe/Warsaw.
    hours = local_day_hours_utc(pd.Timestamp("2026-03-29", tz=TZ), TZ)
    assert len(hours) == 23


def test_dst_autumn_day_has_25_hours() -> None:
    # 2026-10-25: clocks fall back in Europe/Warsaw.
    hours = local_day_hours_utc(pd.Timestamp("2026-10-25", tz=TZ), TZ)
    assert len(hours) == 25


def test_normal_day_has_24_hours() -> None:
    hours = local_day_hours_utc(pd.Timestamp("2026-07-15", tz=TZ), TZ)
    assert len(hours) == 24
    assert str(hours.tz) == "UTC"
