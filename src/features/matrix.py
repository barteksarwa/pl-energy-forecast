"""Assemble the feature matrix for a target day. One place, one contract.

X = calendar + weather (forecast for the target day) + cutoff-safe load lags.
All indexes are UTC hours of the target day.
"""

from __future__ import annotations

import pandas as pd

from src.features.calendar import calendar_features
from src.features.lags import lagged_load_features


def build_features(
    target_hours: pd.DatetimeIndex,
    load: pd.Series,
    weather: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Feature matrix for the target hours, using only pre-cutoff load data.

    `weather` must be the forecast known at the cutoff (or archive data when
    training on actuals — see docs/DATA_CATALOG.md, weather leakage trap).
    """
    cal = calendar_features(target_hours)
    lags = lagged_load_features(load, target_hours, cutoff)
    wx = weather.reindex(target_hours)
    x = pd.concat([cal, lags, wx], axis=1)
    x.index.name = "time_utc"
    return x
