"""Tests for calendar, weather, and lag features. Leakage tests are the point."""

import numpy as np
import pandas as pd
import pytest

from src.features.calendar import bridge_days, calendar_features
from src.features.lags import lagged_load_features
from src.features.matrix import build_features
from src.features.weather import add_degree_signals, population_weighted
from src.pipeline.daily_run import local_day_hours_utc

TZ = "Europe/Warsaw"


# --- calendar ---------------------------------------------------------------


def test_epiphany_is_holiday() -> None:
    hours = local_day_hours_utc(pd.Timestamp("2026-01-06", tz=TZ), TZ)
    cal = calendar_features(hours)
    assert cal["is_holiday"].all()


def test_bridge_friday_after_corpus_christi() -> None:
    # Corpus Christi 2026-06-04 is a Thursday → Friday 06-05 is a bridge day.
    assert pd.Timestamp("2026-06-05").date() in bridge_days([2026])


def test_bridge_day_flag_in_features() -> None:
    hours = local_day_hours_utc(pd.Timestamp("2026-06-05", tz=TZ), TZ)
    cal = calendar_features(hours)
    assert cal["is_bridge_day"].all()
    assert not cal["is_holiday"].any()


def test_hour_local_matches_warsaw_not_utc() -> None:
    # 2026-07-15 00:00 Warsaw = 2026-07-14 22:00 UTC (CEST = UTC+2).
    hours = local_day_hours_utc(pd.Timestamp("2026-07-15", tz=TZ), TZ)
    cal = calendar_features(hours)
    assert cal["hour_local"].iloc[0] == 0
    assert hours[0].hour == 22


def test_cyclic_encoding_wraps() -> None:
    hours = local_day_hours_utc(pd.Timestamp("2026-07-15", tz=TZ), TZ)
    cal = calendar_features(hours)
    # hour 23 and hour 0 must be close in (sin, cos) space.
    d = np.hypot(
        cal["hour_sin"].iloc[-1] - cal["hour_sin"].iloc[0],
        cal["hour_cos"].iloc[-1] - cal["hour_cos"].iloc[0],
    )
    assert d < 0.3


# --- weather ----------------------------------------------------------------


def _two_city_frames() -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2026-01-01", periods=24, freq="1h", tz="UTC")
    return {
        "A": pd.DataFrame({"temperature_2m": 0.0}, index=idx),
        "B": pd.DataFrame({"temperature_2m": 10.0}, index=idx),
    }


def test_population_weighting_math() -> None:
    out = population_weighted(_two_city_frames(), {"A": 3.0, "B": 1.0})
    assert out["temperature_2m"].iloc[0] == pytest.approx(2.5)  # (3*0 + 1*10) / 4


def test_weight_key_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        population_weighted(_two_city_frames(), {"A": 1.0})


def test_degree_signals_v_shape() -> None:
    idx = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    wx = pd.DataFrame({"temperature_2m": [-5.0, 18.0, 30.0]}, index=idx)
    out = add_degree_signals(wx)
    assert out["heating_degrees"].tolist() == [20.0, 0.0, 0.0]
    assert out["cooling_degrees"].tolist() == [0.0, 0.0, 8.0]


# --- lags: the leakage tests ------------------------------------------------


def _load_history() -> pd.Series:
    idx = pd.date_range("2026-05-01", periods=45 * 24, freq="1h", tz="UTC")
    return pd.Series(np.arange(len(idx), dtype=float), index=idx)


def test_lag_features_never_touch_post_cutoff_data() -> None:
    load = _load_history()
    target = local_day_hours_utc(pd.Timestamp("2026-06-10", tz=TZ), TZ)
    cutoff = pd.Timestamp("2026-06-09 09:00", tz=TZ)
    feats = lagged_load_features(load, target, cutoff)
    # Reconstruct source times and assert all strictly before cutoff.
    for lag in (48, 72, 168, 336):
        assert ((target - pd.Timedelta(hours=lag)) < cutoff).all()
    assert feats.notna().all().all()


def test_lag_24_rejected_as_leakage() -> None:
    load = _load_history()
    target = local_day_hours_utc(pd.Timestamp("2026-06-10", tz=TZ), TZ)
    cutoff = pd.Timestamp("2026-06-09 09:00", tz=TZ)
    with pytest.raises(ValueError, match="leakage"):
        lagged_load_features(load, target, cutoff, lags_h=(24,))


def test_post_cutoff_load_values_do_not_change_features() -> None:
    """Corrupt every post-cutoff value; features must be identical."""
    load = _load_history()
    target = local_day_hours_utc(pd.Timestamp("2026-06-10", tz=TZ), TZ)
    cutoff = pd.Timestamp("2026-06-09 09:00", tz=TZ)
    clean = lagged_load_features(load, target, cutoff)
    corrupted = load.copy()
    corrupted[corrupted.index >= cutoff] = 1e9
    dirty = lagged_load_features(corrupted, target, cutoff)
    pd.testing.assert_frame_equal(clean, dirty)


def test_build_features_shape_and_index() -> None:
    load = _load_history()
    target = local_day_hours_utc(pd.Timestamp("2026-06-10", tz=TZ), TZ)
    cutoff = pd.Timestamp("2026-06-09 09:00", tz=TZ)
    wx = pd.DataFrame(
        {"temperature_2m": 20.0}, index=pd.date_range(target[0], periods=24, freq="1h")
    )
    x = build_features(target, load, wx, cutoff)
    assert len(x) == 24
    assert x.index.equals(target)
    assert "load_lag_168h" in x.columns and "is_holiday" in x.columns
