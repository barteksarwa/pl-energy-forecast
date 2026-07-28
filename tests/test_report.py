"""Report writer safety: redact() keeps secrets out of committed reports.

A regex regression here would commit a live API token to a public repo,
so this is tested even though it is three lines of code.
"""

import src.pipeline.report as report


def test_redact_strips_security_token_query_param() -> None:
    msg = ("HTTP 401 for https://web-api.tp.entsoe.eu/api?"
           "securityToken=abc-123-SECRET&documentType=A65 — check token")
    out = report.redact(msg)
    assert "abc-123-SECRET" not in out
    assert "securityToken=REDACTED" in out
    assert "documentType=A65" in out  # only the token goes


def test_redact_strips_env_token_wherever_it_appears(monkeypatch) -> None:
    monkeypatch.setenv("ENTSOE_API_TOKEN", "tok-XYZ-987")
    out = report.redact("error body echoed the key: tok-XYZ-987 (raw)")
    assert "tok-XYZ-987" not in out
    assert "REDACTED" in out


def test_redact_no_token_env_is_a_noop(monkeypatch) -> None:
    monkeypatch.delenv("ENTSOE_API_TOKEN", raising=False)
    msg = "plain oddity, nothing secret"
    assert report.redact(msg) == msg


def _write(tmp_path, challenger_drivers):
    import numpy as np
    import pandas as pd

    from src.config import load_config
    from src.pipeline.report import write_report

    cfg = load_config()
    cfg.paths["reports_daily"] = tmp_path

    today = pd.Timestamp("2026-07-27", tz=cfg.timezone_local)
    hours = pd.date_range("2026-07-28", periods=24, freq="1h",
                          tz=cfg.timezone_local).tz_convert("UTC")
    fc = pd.DataFrame(
        {"p10": 18000.0, "p50": 20000.0, "p90": 22000.0}, index=hours
    )
    weather = pd.DataFrame(
        {"temperature_2m": np.linspace(12, 24, 24)}, index=hours
    )
    return write_report(
        cfg=cfg,
        today_local=today,
        scores={"naive_mape": 5.1, "tso_mape": 2.2},
        forecast=fc,
        weather=weather,
        oddities=[],
        challenger_drivers=challenger_drivers,
    )


def test_report_carries_measured_challenger_drivers(tmp_path) -> None:
    path = _write(tmp_path, ["the TSO's own day-ahead forecast",
                             "load last week at this hour",
                             "temperature"])
    text = path.read_text()
    assert "1. the TSO's own day-ahead forecast" in text
    assert "measured drivers" in text
    # the old hardcoded pseudo-driver must be gone
    assert "not yet used by the model" not in text


def test_report_without_challenger_still_honest(tmp_path) -> None:
    text = _write(tmp_path, []).read_text()
    assert "seasonal naive" in text
    assert "copies the same hour" in text
    assert "not yet used by the model" not in text
