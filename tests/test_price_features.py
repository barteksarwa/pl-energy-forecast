"""Leakage and DST tests for the price feature pipeline.

The price cutoff is the first delivery hour of day D: all of D-1's prices
are known at bid time, nothing of day D is. These tests enforce that on
normal days AND on the 23/25-hour DST days, where a fixed minus-24h shift
would silently reach into day D.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.price_lags import lagged_price_features
from src.features.price_matrix import build_price_features
from src.pipeline.daily_run import local_day_hours_utc

TZ = "Europe/Warsaw"


def _price_history() -> pd.Series:
    idx = pd.date_range("2023-09-01", periods=90 * 24, freq="1h", tz="UTC")
    return pd.Series(np.arange(len(idx), dtype=float), index=idx)


def _day(day: str) -> tuple[pd.DatetimeIndex, pd.Timestamp]:
    hours = local_day_hours_utc(pd.Timestamp(day, tz=TZ), TZ)
    return hours, hours[0]


def test_price_lags_never_touch_target_day() -> None:
    price = _price_history()
    hours, cutoff = _day("2023-10-10")
    feats = lagged_price_features(price, hours, cutoff)
    assert feats[["price_lag_1d", "price_lag_2d", "price_lag_3d", "price_lag_7d"]].notna().all().all()


def test_post_cutoff_price_values_do_not_change_features() -> None:
    """Corrupt every day-D value; features must be identical."""
    price = _price_history()
    hours, cutoff = _day("2023-10-10")
    clean = lagged_price_features(price, hours, cutoff)
    corrupted = price.copy()
    corrupted[corrupted.index >= cutoff] = 1e9
    dirty = lagged_price_features(corrupted, hours, cutoff)
    pd.testing.assert_frame_equal(clean, dirty)


def test_autumn_dst_25h_day_no_leakage() -> None:
    """The bug this file exists for: on the 25-hour day, minus-24h from the
    last delivery hour lands inside day D. Local-day shifts must not."""
    price = _price_history()
    hours, cutoff = _day("2023-10-29")
    assert len(hours) == 25
    feats = lagged_price_features(price, hours, cutoff)
    # every non-NaN lag-1d source is strictly before the first delivery hour
    lag1 = feats["price_lag_1d"].dropna()
    source_values_from_day_d = price[price.index >= cutoff]
    assert not lag1.isin(source_values_from_day_d).any()


def test_spring_dst_23h_day_builds() -> None:
    price = _price_history()
    hours, cutoff = _day("2024-03-31")
    assert len(hours) == 23
    feats = lagged_price_features(price, hours, cutoff)
    assert len(feats) == 23


def test_day_after_autumn_switch_ambiguous_hour_is_nan() -> None:
    """Target on Oct 30 shifted -1 local day hits the ambiguous 02:00 of
    Oct 29 — that source must be NaN (honest), never a silent guess."""
    price = _price_history()
    hours, cutoff = _day("2023-10-30")
    feats = lagged_price_features(price, hours, cutoff)
    # exactly one target hour resolves to the ambiguous local 02:00
    assert feats["price_lag_1d"].isna().sum() == 1


def test_naked_timestamps_rejected() -> None:
    price = _price_history().tz_localize(None)
    hours, cutoff = _day("2023-10-10")
    with pytest.raises(ValueError, match="tz-aware"):
        lagged_price_features(price, hours, cutoff)


def test_build_price_features_shape_and_columns() -> None:
    price = _price_history()
    load = pd.Series(
        20000.0,
        index=pd.date_range("2023-09-01", periods=90 * 24, freq="1h", tz="UTC"),
    )
    hours, price_cutoff = _day("2023-10-10")
    load_cutoff = pd.Timestamp("2023-10-09 09:00", tz=TZ)
    tso = pd.Series(21000.0, index=hours)
    x = build_price_features(hours, price, load, price_cutoff, load_cutoff, tso=tso)
    assert len(x) == 24
    assert x.index.equals(hours)
    for col in ("price_lag_1d", "load_lag_168h", "is_holiday", "tso_forecast_mw"):
        assert col in x.columns
