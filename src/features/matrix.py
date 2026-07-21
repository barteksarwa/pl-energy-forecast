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
    tso: pd.Series | None = None,
) -> pd.DataFrame:
    """Feature matrix for the target hours, using only pre-cutoff load data.

    `weather` must be the forecast known at the cutoff (or archive data when
    training on actuals — see docs/DATA_CATALOG.md, weather leakage trap).
    `tso`: the TSO day-ahead forecast. PSE publishes day D's forecast at
    ~09:00 on D-1 — at our cutoff — so it is a legal known-future covariate
    (DECISIONS.md 2026-07-15). Using it makes the model a forecast combiner.
    """
    cal = calendar_features(target_hours)
    lags = lagged_load_features(load, target_hours, cutoff)
    parts = [cal, lags, weather.reindex(target_hours)]
    if tso is not None:
        parts.append(tso.reindex(target_hours).rename("tso_forecast_mw"))
    x = pd.concat(parts, axis=1)
    x.index.name = "time_utc"
    return x
