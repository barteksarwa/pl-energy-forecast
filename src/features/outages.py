"""Unavailable-capacity feature from generation outage messages.

The scarcity signal our spike post-mortems said was missing: when a big
unit is out, the merit order shortens and the evening ramp gets
expensive. Desks read these messages (UMMs) all day.

Leakage rule: an outage message exists for the forecaster only from its
publication time (created_doc_time). For a day-D forecast, filter
messages published before the bid-time cutoff; later revisions of the
same message (mrid) supersede earlier ones — but only revisions
published before the cutoff count.
"""

from __future__ import annotations

import pandas as pd


def unavailable_capacity(
    events: pd.DataFrame,
    target_hours: pd.DatetimeIndex,
    cutoff: pd.Timestamp,
) -> pd.Series:
    """Sum of (nominal - available) MW over outages active at each hour.

    `events`: the outages.parquet store (one row per message revision).
    Only messages with created_doc_time < cutoff participate; per mrid,
    the latest such revision wins. Withdrawn messages (docstatus A13)
    are removed entirely.
    """
    if target_hours.tz is None or cutoff.tz is None:
        raise ValueError("target_hours and cutoff must be tz-aware")

    known = events[events["created_doc_time"] < cutoff]
    if len(known) == 0:
        return pd.Series(0.0, index=target_hours, name="outage_mw")
    latest = (
        known.sort_values("revision")
        .groupby("mrid", as_index=False)
        .last()
    )
    latest = latest[latest["docstatus"] != "A13"]

    reduction = (
        latest["nominal_power"].fillna(0.0) - latest["avail_qty"].fillna(0.0)
    ).clip(lower=0.0)
    start = latest["start"]
    end = latest["end"]

    out = pd.Series(0.0, index=target_hours, name="outage_mw")
    for hour in target_hours:
        active = (start <= hour) & (end > hour)
        out.loc[hour] = float(reduction[active].sum())
    return out
