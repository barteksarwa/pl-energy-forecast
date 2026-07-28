"""PSE API v2 client (api.raporty.pse.pl). Free, no key.

Entity kse-load: 15-min actual load + the TSO's own demand forecast.
Data exists from 2024-06-14 (v2 launch). The forecast for day D publishes
around 09:00 local on D-1 — the same moment as our forecast cutoff.

Two views of every entity:
- `*_native`: the 15-min series as PSE publishes it. PL settles
  imbalance on 15-min periods and SDAC is moving to 15-min MTU —
  throwing the native resolution away at ingest would close that door.
- hourly (`fetch_entity_hourly`, `fetch_kse_load`): mean-aggregated,
  what the current hourly models consume.

All UTC tz-aware, MW, period-beginning labels (ENTSO-E convention).
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


def fetch_entity_native(
    entity: str, value_cols: dict[str, str], start_date: str, end_date: str
) -> pd.DataFrame:
    """Generic entity at native 15-min resolution. value_cols: api → out name.

    dtime_utc marks the END of each 15-min period; shifted to period
    start so rows are period-beginning.
    """
    flt = f"business_date ge '{start_date}' and business_date le '{end_date}'"
    rows = _fetch_entity(entity, flt)
    if not rows:
        return pd.DataFrame(
            columns=list(value_cols.values()),
            index=pd.DatetimeIndex([], tz="UTC", name="time"),
        )
    df = pd.DataFrame(rows)
    ts = pd.to_datetime(df["dtime_utc"], utc=True) - pd.Timedelta(minutes=15)
    return pd.DataFrame(
        {out: pd.to_numeric(df[api], errors="coerce").to_numpy()
         for api, out in value_cols.items()},
        index=pd.DatetimeIndex(ts, name="time"),
    ).sort_index()


def fetch_entity_hourly(
    entity: str, value_cols: dict[str, str], start_date: str, end_date: str
) -> pd.DataFrame:
    """Generic entity → hourly UTC frame (mean over the four 15-min periods)."""
    return fetch_entity_native(entity, value_cols, start_date, end_date).resample(
        "1h").mean()


KSE_LOAD_COLS = {"load_actual": "load_mw", "load_fcst": "tso_forecast_mw"}


def fetch_kse_load_native(start_date: str, end_date: str) -> pd.DataFrame:
    """15-min load_mw + tso_forecast_mw for [start_date, end_date] local days."""
    return fetch_entity_native("kse-load", KSE_LOAD_COLS, start_date, end_date)


def fetch_kse_load(start_date: str, end_date: str) -> pd.DataFrame:
    """Hourly load_mw + tso_forecast_mw for [start_date, end_date] local days."""
    return fetch_kse_load_native(start_date, end_date).resample("1h").mean()
