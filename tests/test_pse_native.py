"""PSE ingest keeps the native 15-min resolution (MTU) next to hourly.

PL settles imbalance on 15-min periods and SDAC is moving to 15-min
MTU; resampling to hourly at ingest would throw that information away
(review finding, 2026-07-27).
"""

import pandas as pd

import src.clients.pse_client as pse


def _rows() -> list[dict]:
    # two hours = eight 15-min periods; dtime_utc marks the period END
    start = pd.Timestamp("2026-07-27 10:15")
    return [
        {
            "dtime_utc": str(start + pd.Timedelta(minutes=15 * i)),
            "load_actual": 20000 + 100 * i,
            "load_fcst": 19900 + 100 * i,
        }
        for i in range(8)
    ]


def test_native_keeps_all_quarters(monkeypatch) -> None:
    monkeypatch.setattr(pse, "_fetch_entity", lambda entity, flt: _rows())
    out = pse.fetch_kse_load_native("2026-07-27", "2026-07-27")
    assert len(out) == 8
    assert (out.index[1] - out.index[0]) == pd.Timedelta(minutes=15)
    # period-beginning labels: first END 10:15 → first START 10:00
    assert out.index[0] == pd.Timestamp("2026-07-27 10:00", tz="UTC")


def test_hourly_is_mean_of_four_quarters(monkeypatch) -> None:
    monkeypatch.setattr(pse, "_fetch_entity", lambda entity, flt: _rows())
    hourly = pse.fetch_kse_load("2026-07-27", "2026-07-27")
    assert len(hourly) == 2
    # first hour: quarters 0..3 → mean of 20000,20100,20200,20300
    assert hourly["load_mw"].iloc[0] == 20150.0


def test_empty_result_has_datetime_index(monkeypatch) -> None:
    monkeypatch.setattr(pse, "_fetch_entity", lambda entity, flt: [])
    native = pse.fetch_kse_load_native("2026-07-27", "2026-07-27")
    assert native.empty
    assert isinstance(native.index, pd.DatetimeIndex)
    # and the hourly path must not blow up on the empty frame
    assert pse.fetch_kse_load("2026-07-27", "2026-07-27").empty
