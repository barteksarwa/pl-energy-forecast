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
