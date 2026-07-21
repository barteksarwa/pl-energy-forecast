"""Fuel/carbon features from daily settlement closes, leakage-safe.

Timeline: the close of trading day T is published T evening. A forecast
for day D decided at 09:00 on D-1 can know closes dated D-2 or earlier
— never D-1's own close (it happens ~8 hours after the decision).

The feature is the last known close, forward-filled across weekends and
holidays. Gas and carbon move slowly day-to-day; the 1-2 day staleness
is the honest price of the timeline.
"""

from __future__ import annotations

import pandas as pd


def fuel_features(
    fuel_daily: pd.DataFrame,
    target_hours: pd.DatetimeIndex,
    cutoff: pd.Timestamp,
    tz: str = "Europe/Warsaw",
) -> pd.DataFrame:
    """Last close dated strictly before the cutoff's local DATE, per hour.

    cutoff 09:00 D-1 → the newest usable close is dated D-2 (its number
    existed the evening of D-2). Rows carry the same value for all 24
    target hours — level features, not shapes.
    """
    if target_hours.tz is None or cutoff.tz is None:
        raise ValueError("target_hours and cutoff must be tz-aware")

    cutoff_date = pd.Timestamp(cutoff.tz_convert(tz).date())
    known = fuel_daily[fuel_daily.index < cutoff_date]
    out = pd.DataFrame(index=target_hours)
    for col in fuel_daily.columns:
        out[col] = float(known[col].ffill().iloc[-1]) if len(known) else float("nan")
    return out
