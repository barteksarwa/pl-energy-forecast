"""Lagged load features that respect the forecast cutoff.

The forecast for day D is made at 09:00 local on D-1. At that moment,
yesterday-relative-to-D (lag 24h) is mostly unobserved — using it would be
leakage. Allowed lags for a day-ahead forecast start at 48h.

Every feature is asserted to lie strictly before the cutoff. This is a
hard rule of the project; the assert is the enforcement, tests prove it.
"""

from __future__ import annotations

import pandas as pd

# Same hour 2, 3, 7, 14, 21, 28 days back. All fully observed by 09:00 on D-1.
# The four weekly lags (168/336/504/672) double as the seasonal-naive ensemble.
DEFAULT_LAGS_H = (48, 72, 168, 336, 504, 672)


def lagged_load_features(
    load: pd.Series,
    target_hours: pd.DatetimeIndex,
    cutoff: pd.Timestamp,
    lags_h: tuple[int, ...] = DEFAULT_LAGS_H,
) -> pd.DataFrame:
    """Lag features for each target hour, using only data before `cutoff`.

    Raises if any requested lag would peek past the cutoff.
    Missing history stays NaN — models must handle it or drop the row.
    """
    if load.index.tz is None or target_hours.tz is None or cutoff.tz is None:
        raise ValueError("load, target_hours and cutoff must all be tz-aware")

    visible = load[load.index < cutoff]
    out = pd.DataFrame(index=target_hours)
    for lag in sorted(lags_h):
        source_times = target_hours - pd.Timedelta(hours=lag)
        if (source_times >= cutoff).any():
            raise ValueError(
                f"lag {lag}h reaches past the cutoff {cutoff} — leakage. "
                f"Earliest offending target: {target_hours[source_times >= cutoff][0]}"
            )
        out[f"load_lag_{lag}h"] = visible.reindex(source_times).to_numpy()

    # Mean of the last 7 fully observed days before the cutoff.
    week_before = visible[visible.index >= cutoff - pd.Timedelta(days=7)]
    out["load_mean_7d"] = week_before.mean()
    return out
