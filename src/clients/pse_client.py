"""PSE API v2 client (api.raporty.pse.pl). Free, no key.

Entity kse-load: 15-min actual load + the TSO's own demand forecast.
Data exists from 2024-06-14 (v2 launch). The forecast for day D publishes
around 09:00 local on D-1 — the same moment as our forecast cutoff.

Returns hourly, UTC tz-aware, MW, hour-beginning labels (ENTSO-E convention).
"""

from __future__ import annotations

import time

import pandas as pd
import requests

BASE_URL = "https://api.raporty.pse.pl/api"
TIMEOUT_S = 60
PAGE_SIZE = 20000


def _fetch_entity(entity: str, flt: str) -> list[dict]:
    url = f"{BASE_URL}/{entity}?$filter={flt}&$first={PAGE_SIZE}"
    rows: list[dict] = []
    while url:
        resp = None
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=TIMEOUT_S)
                resp.raise_for_status()
                break
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 2:
                    raise
                time.sleep(3 * (attempt + 1))
        assert resp is not None
        payload = resp.json()
        rows.extend(payload["value"])
        url = payload.get("nextLink")
    return rows


def fetch_entity_hourly(
    entity: str, value_cols: dict[str, str], start_date: str, end_date: str
) -> pd.DataFrame:
    """Generic 15-min entity → hourly UTC frame. value_cols: api_name → out_name."""
    flt = f"business_date ge '{start_date}' and business_date le '{end_date}'"
    rows = _fetch_entity(entity, flt)
    if not rows:
        return pd.DataFrame(columns=list(value_cols.values()))
    df = pd.DataFrame(rows)
    ts = pd.to_datetime(df["dtime_utc"], utc=True) - pd.Timedelta(minutes=15)
    out = pd.DataFrame(
        {out: pd.to_numeric(df[api], errors="coerce").to_numpy()
         for api, out in value_cols.items()},
        index=pd.DatetimeIndex(ts, name="time"),
    ).sort_index()
    return out.resample("1h").mean()


def fetch_kse_load(start_date: str, end_date: str) -> pd.DataFrame:
    """Hourly load_mw + tso_forecast_mw for [start_date, end_date] local days.

    dtime_utc marks the END of each 15-min period; we shift to period start
    before resampling so hourly rows are hour-beginning.
    """
    flt = f"business_date ge '{start_date}' and business_date le '{end_date}'"
    rows = _fetch_entity("kse-load", flt)
    if not rows:
        return pd.DataFrame(columns=["load_mw", "tso_forecast_mw"])
    df = pd.DataFrame(rows)
    ts = pd.to_datetime(df["dtime_utc"], utc=True) - pd.Timedelta(minutes=15)
    out = pd.DataFrame(
        {
            "load_mw": pd.to_numeric(df["load_actual"], errors="coerce").to_numpy(),
            "tso_forecast_mw": pd.to_numeric(df["load_fcst"], errors="coerce").to_numpy(),
        },
        index=pd.DatetimeIndex(ts, name="time"),
    ).sort_index()
    return out.resample("1h").mean()
