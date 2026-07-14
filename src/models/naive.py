"""Seasonal naive baseline with empirical quantiles.

P50: load at the same hour one season (default: 7 days) earlier.
P10/P90: empirical quantiles of the same hour over the last n seasons.
Missing history gives NaN. We never invent values.
"""

from __future__ import annotations

import pandas as pd

QUANTILE_COLS = ["p10", "p50", "p90"]


def seasonal_naive_forecast(
    load: pd.Series,
    target_hours: pd.DatetimeIndex,
    season_days: int = 7,
    n_seasons: int = 4,
) -> pd.DataFrame:
    """Forecast quantiles for each target hour from past same-hour values.

    load: hourly series, UTC tz-aware index.
    target_hours: UTC tz-aware hours to forecast.
    """
    if load.index.tz is None or target_hours.tz is None:
        raise ValueError("Both load index and target_hours must be tz-aware.")

    season = pd.Timedelta(days=season_days)
    rows = []
    for ts in target_hours:
        past = load.reindex([ts - k * season for k in range(1, n_seasons + 1)]).dropna()
        if past.empty:
            rows.append({"p10": float("nan"), "p50": float("nan"), "p90": float("nan")})
            continue
        p50 = past.iloc[0] if (ts - season) in past.index else past.median()
        rows.append(
            {"p10": past.quantile(0.10), "p50": float(p50), "p90": past.quantile(0.90)}
        )
    return pd.DataFrame(rows, index=target_hours)[QUANTILE_COLS]
