"""Daily report writer. Markdown, readable by a manager in 60 seconds."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import pandas as pd

from src.config import Config


def redact(text: str) -> str:
    """Strip secrets from text that will be committed.

    Exception messages can embed full request URLs — an ENTSO-E error
    once carried `securityToken=<our token>` straight into a report.
    Every oddity/error string must pass through here before writing.
    """
    text = re.sub(r"securityToken=[^&\s]+", "securityToken=REDACTED", text)
    token = os.environ.get("ENTSOE_API_TOKEN")
    if token:
        text = text.replace(token, "REDACTED")
    return text


def _fmt(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.2f}%"


def write_report(
    cfg: Config,
    today_local: pd.Timestamp,
    scores: dict[str, float],
    forecast: pd.DataFrame,
    weather: pd.DataFrame,
    oddities: list[str],
    extra_sections: list[str] | None = None,
    challenger_drivers: list[str] | None = None,
) -> Path:
    tz = cfg.timezone_local
    tomorrow = today_local.date() + pd.Timedelta(days=1)
    yesterday = today_local.date() - pd.Timedelta(days=1)

    local = forecast.tz_convert(tz)
    peak_hour = local["p50"].idxmax()
    temp_tomorrow = weather["temperature_2m"].tz_convert(tz).loc[str(tomorrow)]

    lines = [
        f"# Daily forecast report — {today_local.date()}",
        "",
        f"Model: seasonal naive — **BASELINE**, not the final model. "
        f"P50 copies the same hour {cfg.naive_season_days} days ago; the band is the "
        f"spread of the last {cfg.naive_n_seasons} weeks. Serves until a trained "
        f"model earns promotion (see docs/PLAN.md M4, UAT rules M9).",
        "",
        f"## Yesterday ({yesterday}) — how did we do?",
        "",
        "| Forecast | MAPE |",
        "|---|---|",
        f"| Ours (naive, incumbent) | {_fmt(scores['naive_mape'])} |",
        f"| Challenger (ridge+TSO, shadow) | {_fmt(scores.get('challenger_mape', float('nan')))} |",
        f"| TSO day-ahead | {_fmt(scores['tso_mape'])} |",
        "",
        f"![Yesterday: forecast vs realized](../figures/daily/{yesterday}.png)",
        "",
        f"## Tomorrow ({tomorrow}) — the forecast",
        "",
        f"- Expected peak: **{local['p50'].max():,.0f} MW** around "
        f"{peak_hour.strftime('%H:%M')} local time.",
        f"- Daily range (P50): {local['p50'].min():,.0f} – {local['p50'].max():,.0f} MW.",
        f"- Uncertainty band at peak: {local.loc[peak_hour, 'p10']:,.0f} – "
        f"{local.loc[peak_hour, 'p90']:,.0f} MW (P10–P90).",
        "",
        "### Top drivers (plain words)",
        "",
        "Published forecast (seasonal naive): it copies the same hour "
        "last week, so its only drivers are last week's load and the "
        f"day of week (tomorrow is a {pd.Timestamp(tomorrow).day_name()}).",
        "",
    ]
    if challenger_drivers:
        lines += [
            "Shadow challenger (ridge + TSO) — measured drivers of "
            "tomorrow's forecast (|coef x standardized feature|, exact "
            "for a linear model):",
            "",
        ]
        lines += [f"{i}. {d}" for i, d in enumerate(challenger_drivers, 1)]
        lines += [""]
    lines += [
        f"Context: Warsaw temperature tomorrow {temp_tomorrow.min():.0f} "
        f"to {temp_tomorrow.max():.0f} °C.",
        "",
        "### Oddities",
        "",
    ]
    lines += [f"- {redact(o)}" for o in oddities] if oddities else ["- None."]
    lines += [
        "",
        f"![Day-ahead forecast fan chart](../figures/daily/{tomorrow}.png)",
        "",
    ]
    if extra_sections:
        lines += extra_sections + [""]
    lines += [
        "_Full hourly quantiles: see `data/forecasts/`._",
        "",
    ]

    cfg.paths["reports_daily"].mkdir(parents=True, exist_ok=True)
    path = cfg.paths["reports_daily"] / f"{today_local.date()}.md"
    path.write_text("\n".join(lines))
    return path
