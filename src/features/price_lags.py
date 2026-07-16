"""Lagged day-ahead price features that respect the auction timeline.

The price timeline differs from the load timeline. The day-ahead auction
for delivery day D closes at 12:00 CET on D-1 (SDAC gate closure). At that
moment, prices for ALL hours of D-1 are already known — they were fixed at
the auction on D-2. So the youngest legal price lag for a day-D target
is one day (same hour yesterday). Load lags need 48h; price lags need 1 day.

Lags are LOCAL-CALENDAR day shifts, not fixed 24h offsets. On the 25-hour
autumn DST day, "minus 24 hours" from the last delivery hour lands inside
day D itself — that is leakage (day-D prices are what we are forecasting).
Shifting by local days keeps every source on day <= D-1 on every day of
the year. The DST-ambiguous hour (autumn, repeated 02:00) and the
nonexistent hour (spring, skipped 02:00) become NaN and the row drops —
about two hours per year per lag, logged honestly rather than fudged.

The cutoff passed here is the first delivery hour of day D (UTC). Every
source timestamp must lie strictly before it; the guard raise stays as
defense in depth.
"""

from __future__ import annotations

import pandas as pd

LOCAL_TZ = "Europe/Warsaw"

# Same local hour 1, 2, 3, 7 days back — the canonical LEAR lag set
# (Ziel & Weron 2018; Lago et al. 2021).
DEFAULT_PRICE_LAG_DAYS = (1, 2, 3, 7)


def _shift_local_days(
    target_hours: pd.DatetimeIndex, days: int, tz: str
) -> pd.DatetimeIndex:
    """Same local clock time `days` local days earlier, as UTC timestamps.

    DST-ambiguous or nonexistent local times map to NaT.
    """
    local_naive = target_hours.tz_convert(tz).tz_localize(None)
    shifted = local_naive - pd.Timedelta(days=days)
    return shifted.tz_localize(
        tz, ambiguous="NaT", nonexistent="NaT"
    ).tz_convert("UTC")


def lagged_price_features(
    price: pd.Series,
    target_hours: pd.DatetimeIndex,
    cutoff: pd.Timestamp,
    lag_days: tuple[int, ...] = DEFAULT_PRICE_LAG_DAYS,
    tz: str = LOCAL_TZ,
) -> pd.DataFrame:
    """Lag features for each target hour, using only prices before `cutoff`.

    `cutoff` must be the first delivery hour of the target day (UTC).
    Raises if any resolved source timestamp reaches past the cutoff.
    Missing history and DST edge hours stay NaN — models must handle it
    or drop the row.
    """
    if price.index.tz is None or target_hours.tz is None or cutoff.tz is None:
        raise ValueError("price, target_hours and cutoff must all be tz-aware")

    visible = price[price.index < cutoff]
    out = pd.DataFrame(index=target_hours)
    for days in sorted(lag_days):
        source_times = _shift_local_days(target_hours, days, tz)
        valid = source_times.notna()
        if (source_times[valid] >= cutoff).any():
            offending = target_hours[valid][source_times[valid] >= cutoff][0]
            raise ValueError(
                f"lag {days}d reaches past the cutoff {cutoff} — leakage. "
                f"Earliest offending target: {offending}"
            )
        out[f"price_lag_{days}d"] = visible.reindex(source_times).to_numpy()

    # Mean of the last 7 fully known days before the cutoff.
    week_before = visible[visible.index >= cutoff - pd.Timedelta(days=7)]
    out["price_mean_7d"] = week_before.mean()
    return out


def daily_price_vector(
    price: pd.Series,
    target_hours: pd.DatetimeIndex,
    cutoff: pd.Timestamp,
    tz: str = LOCAL_TZ,
) -> pd.DataFrame:
    """All 24 hourly prices of D-1 as one row-constant feature block.

    This is the core LEAR input (Lago et al. 2021): predicting hour h of
    day D uses the WHOLE shape of yesterday, not just hour h. Columns
    price_d1_h00 ... price_d1_h23, identical on every row of day D.

    DST handling on the source day D-1: the repeated autumn hour is
    averaged; the missing spring hour is linearly interpolated from its
    neighbours. Both are deterministic and stated here rather than hidden.
    """
    if price.index.tz is None or target_hours.tz is None or cutoff.tz is None:
        raise ValueError("price, target_hours and cutoff must all be tz-aware")

    visible = price[price.index < cutoff]
    d_minus_1 = (target_hours[0].tz_convert(tz) - pd.Timedelta(days=1)).date()
    local = visible.index.tz_convert(tz)
    day_mask = pd.Index(local.date) == d_minus_1
    yesterday = visible[day_mask]
    by_hour = yesterday.groupby(yesterday.index.tz_convert(tz).hour).mean()
    by_hour = by_hour.reindex(range(24)).interpolate(limit_direction="both")

    out = pd.DataFrame(
        {f"price_d1_h{h:02d}": by_hour.loc[h] for h in range(24)},
        index=target_hours,
    )
    return out
