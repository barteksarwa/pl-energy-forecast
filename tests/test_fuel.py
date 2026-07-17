"""Leakage tests for the fuel/carbon settlement features."""

import pandas as pd
import pytest

from src.features.fuel import fuel_features

TZ = "Europe/Warsaw"
HOURS = pd.date_range("2026-01-15 00:00", periods=24, freq="1h", tz="UTC")
CUTOFF = pd.Timestamp("2026-01-14 09:00", tz=TZ)  # D-1 09:00 for D=Jan 15


def _daily(rows: dict) -> pd.DataFrame:
    idx = pd.DatetimeIndex(sorted(rows))
    return pd.DataFrame({"ttf_eur_mwh": [rows[d] for d in sorted(rows)]}, index=idx)


def test_uses_close_of_d_minus_2() -> None:
    fuel = _daily({
        pd.Timestamp("2026-01-12"): 30.0,
        pd.Timestamp("2026-01-13"): 31.0,  # D-2: newest legal close
        pd.Timestamp("2026-01-14"): 99.0,  # D-1 close: NOT known at 09:00
    })
    out = fuel_features(fuel, HOURS, CUTOFF)
    assert (out["ttf_eur_mwh"] == 31.0).all()


def test_weekend_forward_fill() -> None:
    fuel = _daily({pd.Timestamp("2026-01-09"): 28.0})  # Friday only
    out = fuel_features(fuel, HOURS, CUTOFF)
    assert (out["ttf_eur_mwh"] == 28.0).all()


def test_future_corruption_invisible() -> None:
    clean = _daily({pd.Timestamp("2026-01-13"): 31.0})
    dirty = _daily({pd.Timestamp("2026-01-13"): 31.0,
                    pd.Timestamp("2026-01-20"): 1e9})
    a = fuel_features(clean, HOURS, CUTOFF)
    b = fuel_features(dirty, HOURS, CUTOFF)
    pd.testing.assert_frame_equal(a, b)


def test_naked_timestamps_rejected() -> None:
    fuel = _daily({pd.Timestamp("2026-01-13"): 31.0})
    with pytest.raises(ValueError, match="tz-aware"):
        fuel_features(fuel, HOURS.tz_localize(None), CUTOFF.tz_localize(None))
