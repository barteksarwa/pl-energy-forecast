"""Gap detection and logging. We record gaps, we never silently fill them."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

GAP_LOG_COLUMNS = ["series", "gap_start_utc", "gap_end_utc", "n_hours", "detected_at_utc"]


def find_gaps(series: pd.Series, freq: str = "1h") -> pd.DataFrame:
    """Find missing intervals in a tz-aware hourly series.

    A gap is one or more consecutive missing/NaN steps between the first and
    last observation. Returns one row per gap: start, end (inclusive), n_hours.
    """
    if series.index.tz is None:
        raise ValueError("series index must be tz-aware")
    if series.dropna().empty:
        return pd.DataFrame(columns=["gap_start_utc", "gap_end_utc", "n_hours"])

    observed = series.dropna()
    full = pd.date_range(observed.index[0], observed.index[-1], freq=freq)
    missing = full.difference(observed.index)
    if missing.empty:
        return pd.DataFrame(columns=["gap_start_utc", "gap_end_utc", "n_hours"])

    group = (missing.to_series().diff() != pd.Timedelta(freq)).cumsum()
    rows = []
    for _, chunk in missing.to_series().groupby(group):
        rows.append(
            {
                "gap_start_utc": chunk.iloc[0],
                "gap_end_utc": chunk.iloc[-1],
                "n_hours": len(chunk),
            }
        )
    return pd.DataFrame(rows)


def log_gaps(series: pd.Series, name: str, log_path: Path, freq: str = "1h") -> pd.DataFrame:
    """Append newly found gaps for `name` to the gap log CSV. Deduplicates."""
    gaps = find_gaps(series, freq)
    if gaps.empty:
        return gaps
    gaps = gaps.assign(series=name, detected_at_utc=pd.Timestamp.now(tz="UTC"))
    gaps = gaps[GAP_LOG_COLUMNS]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        old = pd.read_csv(log_path, parse_dates=["gap_start_utc", "gap_end_utc"])
        merged = pd.concat([old, gaps], ignore_index=True)
        merged = merged.drop_duplicates(subset=["series", "gap_start_utc", "gap_end_utc"])
    else:
        merged = gaps
    merged.to_csv(log_path, index=False)
    return gaps
