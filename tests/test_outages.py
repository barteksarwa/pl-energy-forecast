"""Leakage and supersede tests for the outage feature."""

import pandas as pd
import pytest

from src.features.outages import unavailable_capacity

TZ = "UTC"


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=TZ)


def _events(rows: list[dict]) -> pd.DataFrame:
    base = {
        "created_doc_time": _ts("2026-01-01 00:00"),
        "start": _ts("2026-01-10 00:00"),
        "end": _ts("2026-01-20 00:00"),
        "nominal_power": 500.0,
        "avail_qty": 0.0,
        "docstatus": None,
        "mrid": "m1",
        "revision": 1,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


HOURS = pd.date_range("2026-01-15 00:00", periods=24, freq="1h", tz=TZ)
CUTOFF = _ts("2026-01-14 00:00")


def test_active_outage_counts() -> None:
    ev = _events([{}])
    out = unavailable_capacity(ev, HOURS, CUTOFF)
    assert (out == 500.0).all()


def test_message_published_after_cutoff_is_invisible() -> None:
    ev = _events([{"created_doc_time": _ts("2026-01-14 12:00")}])
    out = unavailable_capacity(ev, HOURS, CUTOFF)
    assert (out == 0.0).all()


def test_later_revision_supersedes() -> None:
    ev = _events([
        {"revision": 1, "avail_qty": 0.0},
        {"revision": 2, "avail_qty": 400.0,  # unit mostly back
         "created_doc_time": _ts("2026-01-13 00:00")},
    ])
    out = unavailable_capacity(ev, HOURS, CUTOFF)
    assert (out == 100.0).all()


def test_post_cutoff_revision_does_not_supersede() -> None:
    ev = _events([
        {"revision": 1, "avail_qty": 0.0},
        {"revision": 2, "avail_qty": 500.0,
         "created_doc_time": _ts("2026-01-14 12:00")},  # after cutoff
    ])
    out = unavailable_capacity(ev, HOURS, CUTOFF)
    assert (out == 500.0).all()  # still sees revision 1


def test_withdrawn_message_removed() -> None:
    ev = _events([{"docstatus": "A13"}])
    out = unavailable_capacity(ev, HOURS, CUTOFF)
    assert (out == 0.0).all()


def test_outage_outside_window_is_zero() -> None:
    ev = _events([{"start": _ts("2026-02-01"), "end": _ts("2026-02-10")}])
    out = unavailable_capacity(ev, HOURS, CUTOFF)
    assert (out == 0.0).all()


def test_naked_timestamps_rejected() -> None:
    ev = _events([{}])
    with pytest.raises(ValueError, match="tz-aware"):
        unavailable_capacity(ev, HOURS.tz_localize(None), CUTOFF.tz_localize(None))
