"""Shared plumbing for zero-shot foundation-model wrappers."""

from __future__ import annotations

import pandas as pd


class HistoryContext:
    """Trailing target history for models that only need past values.

    `fit` stores the tail of y; `context_before` returns the context
    window ending just before the first forecast hour.
    """

    def __init__(self, context_hours: int):
        self.context_hours = context_hours
        self._history: pd.Series | None = None

    def fit(self, y: pd.Series) -> None:
        self._history = y.dropna().sort_index().tail(self.context_hours * 2)

    def context_before(self, first_ts: pd.Timestamp) -> pd.Series:
        assert self._history is not None, "fit first"
        return self._history[self._history.index < first_ts].tail(
            self.context_hours)


def forecast_span(
    context_end: pd.Timestamp, idx: pd.DatetimeIndex
) -> tuple[int, pd.DatetimeIndex]:
    """Hourly stretch a zero-shot model must forecast to cover `idx`.

    Foundation models forecast the hours immediately AFTER their context.
    If the context ends before midnight (e.g. the stored history was cut
    at a training cutoff), naively taking the first len(idx) forecast
    hours mis-stamps them onto the target day. Return the horizon length
    and the timestamps it covers, so callers can align by timestamp.
    """
    step = pd.Timedelta(hours=1)
    n_ahead = int((idx[-1] - context_end) / step)
    fc_idx = pd.date_range(context_end + step, periods=n_ahead, freq="1h")
    return n_ahead, fc_idx
