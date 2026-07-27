"""Open-Meteo client: transient failures retry instead of killing the run.

Regression for the 07-17 → 07-27 challenger stall: one timeout in one
city's fetch stopped the shadow gate from ever closing.
"""

import pandas as pd
import pytest
import requests

import src.clients.openmeteo_client as om


class _Resp:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        times = pd.date_range("2026-07-28", periods=3, freq="1h")
        return {
            "hourly": {
                "time": [t.isoformat() for t in times],
                "temperature_2m": [20.0, 21.0, 22.0],
            }
        }


def test_retries_then_succeeds(monkeypatch) -> None:
    calls = {"n": 0}

    def flaky_get(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.exceptions.ConnectTimeout("boom")
        return _Resp()

    monkeypatch.setattr(om.requests, "get", flaky_get)
    monkeypatch.setattr(om.time, "sleep", lambda s: None)
    out = om.fetch_weather_forecast(52.2, 21.0, ["temperature_2m"])
    assert calls["n"] == 3
    assert list(out.columns) == ["temperature_2m"]
    assert len(out) == 3


def test_raises_after_all_retries(monkeypatch) -> None:
    def always_fail(*args, **kwargs):
        raise requests.exceptions.ConnectTimeout("boom")

    monkeypatch.setattr(om.requests, "get", always_fail)
    monkeypatch.setattr(om.time, "sleep", lambda s: None)
    with pytest.raises(requests.exceptions.ConnectTimeout):
        om.fetch_weather_forecast(52.2, 21.0, ["temperature_2m"])
