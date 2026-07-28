"""Zero-shot wrapper alignment: forecast hours must land on the right
timestamps even when the stored context does not end at midnight.

Regression for the 2026-07-27 incident: the 09:00 training cutoff left
the Chronos context ending 08:00 D-1; the wrapper stamped the next 24
forecast hours onto the target day — a 15-hour shift (MAE 21.9 → 55.9).
"""

import pandas as pd

from src.models.fm_common import HistoryContext, forecast_span

TZ = "UTC"


def _day(day: str) -> pd.DatetimeIndex:
    return pd.date_range(f"{day} 00:00", f"{day} 23:00", freq="1h", tz=TZ)


def test_span_aligned_context_ends_at_midnight() -> None:
    idx = _day("2026-01-10")
    context_end = pd.Timestamp("2026-01-09 23:00", tz=TZ)
    n, fc_idx = forecast_span(context_end, idx)
    assert n == 24
    assert fc_idx[0] == idx[0] and fc_idx[-1] == idx[-1]


def test_span_context_cut_at_0800_d_minus_1() -> None:
    """Context ends 08:00 D-1 → model must forecast 39 hours; the target
    day occupies slots 15..38."""
    idx = _day("2026-01-10")
    context_end = pd.Timestamp("2026-01-09 08:00", tz=TZ)
    n, fc_idx = forecast_span(context_end, idx)
    assert n == 39
    assert fc_idx[15] == idx[0]
    assert fc_idx[-1] == idx[-1]
    # reindex-based alignment picks exactly the target-day rows
    fc = pd.DataFrame({"p50": range(n)}, index=fc_idx)
    aligned = fc.reindex(idx)
    assert aligned["p50"].tolist() == list(range(15, 39))


def test_history_context_returns_tail_before_first_target_hour() -> None:
    hours = pd.date_range("2026-01-01", "2026-01-09 08:00", freq="1h", tz=TZ)
    y = pd.Series(range(len(hours)), index=hours, dtype=float)
    ctx = HistoryContext(context_hours=100)
    ctx.fit(y)
    out = ctx.context_before(pd.Timestamp("2026-01-10 00:00", tz=TZ))
    assert len(out) == 100
    assert out.index[-1] == hours[-1]
