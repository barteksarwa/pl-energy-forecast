"""Calendar features. Computed in local time, indexed in UTC.

Includes Polish holidays and bridge days. A bridge day is a workday squeezed
between a holiday and the weekend; many people take it off, load drops.
"""

from __future__ import annotations

import datetime as dt

import holidays as holidays_pkg
import numpy as np
import pandas as pd

LOCAL_TZ = "Europe/Warsaw"


def _country_holidays(years: list[int], country: str = "PL") -> set[dt.date]:
    return set(holidays_pkg.country_holidays(country, years=years).keys())


def bridge_days(years: list[int], country: str = "PL") -> set[dt.date]:
    """Friday after a Thursday holiday, Monday before a Tuesday holiday."""
    hols = _country_holidays(years, country)
    bridges: set[dt.date] = set()
    for h in hols:
        if h.weekday() == 3:  # Thursday
            bridges.add(h + dt.timedelta(days=1))
        if h.weekday() == 1:  # Tuesday
            bridges.add(h - dt.timedelta(days=1))
    return bridges - hols


def calendar_features(
    hours_utc: pd.DatetimeIndex, tz: str = LOCAL_TZ, country: str = "PL"
) -> pd.DataFrame:
    """One row per UTC hour. Categorical flags plus smooth cyclic encodings.

    Sin/cos pairs let linear models see that hour 23 neighbours hour 0.
    `country` is parametrized so Phase 2 can add neighbor calendars
    (DE/CZ/SK...) as extra feature columns without touching this code.
    """
    if hours_utc.tz is None:
        raise ValueError("hours_utc must be tz-aware")
    local = hours_utc.tz_convert(tz)
    years = sorted({t.year for t in local})
    hols = _country_holidays(years, country)
    bridges = bridge_days(years, country)

    dates = np.array([t.date() for t in local])
    hour = local.hour.to_numpy()
    dow = local.dayofweek.to_numpy()
    doy = local.dayofyear.to_numpy()

    return pd.DataFrame(
        {
            "hour_local": hour,
            "day_of_week": dow,
            "month": local.month.to_numpy(),
            "is_weekend": (dow >= 5).astype(int),
            "is_holiday": np.array([d in hols for d in dates], dtype=int),
            "is_bridge_day": np.array([d in bridges for d in dates], dtype=int),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=hours_utc,
    )
