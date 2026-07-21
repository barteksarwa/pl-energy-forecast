"""ENTSO-E Transparency Platform client. Load, TSO load forecast, day-ahead prices.

All returns: hourly pandas Series, UTC tz-aware index.
Load/forecast unit: MW. Price unit: EUR/MWh.
ENTSO-E may deliver 15-min resolution; we resample to hourly mean.
"""

from __future__ import annotations

import os

import pandas as pd
from entsoe import EntsoePandasClient


def _client() -> EntsoePandasClient:
    token = os.environ.get("ENTSOE_API_TOKEN")
    if not token:
        raise RuntimeError(
            "ENTSOE_API_TOKEN not set. Copy .env.example to .env and add your token."
        )
    return EntsoePandasClient(api_key=token)


def _to_hourly_utc(obj: pd.Series | pd.DataFrame, name: str) -> pd.Series:
    series = obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj
    series = series.tz_convert("UTC").resample("1h").mean()
    series.name = name
    return series


def fetch_load(zone: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Actual total load for a bidding zone."""
    raw = _client().query_load(zone, start=start, end=end)
    return _to_hourly_utc(raw, "load_mw")


def fetch_tso_forecast(zone: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """TSO day-ahead load forecast. Our external benchmark."""
    raw = _client().query_load_forecast(zone, start=start, end=end)
    return _to_hourly_utc(raw, "tso_forecast_mw")


def fetch_day_ahead_price(
    zone: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    """Day-ahead market clearing price. Returns EUR/MWh, hourly UTC.

    ENTSO-E endpoint: Day Ahead Prices (12.1.D).
    Some periods return 15-min data; resampled to hourly mean.
    Store as EUR/MWh. Convert to PLN only at display time.
    """
    raw = _client().query_day_ahead_prices(zone, start=start, end=end)
    return _to_hourly_utc(raw, "price_da_eur")


RES_COLS = {
    "Solar": "solar_fcst_mw",
    "Wind Onshore": "wind_on_fcst_mw",
    "Wind Offshore": "wind_off_fcst_mw",
}


def fetch_res_forecast(
    zone: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """TSO day-ahead wind + solar generation forecast (14.1.D). MW, hourly UTC.

    Published by the TSO for delivery day D on D-1. Price driver #1:
    renewables displace the marginal plant (merit order).
    15-min data resampled to hourly mean. Offshore wind absent in early
    history — missing columns are filled with 0.0 (no capacity = 0 MW).
    """
    raw = _client().query_wind_and_solar_forecast(zone, start=start, end=end)
    raw = raw.tz_convert("UTC").resample("1h").mean()
    out = pd.DataFrame(index=raw.index)
    for src, dst in RES_COLS.items():
        out[dst] = raw[src] if src in raw.columns else 0.0
    # Offshore reports NaN in periods before the column existed: that is
    # absent capacity, not missing data. 0 MW is the true value.
    out["wind_off_fcst_mw"] = out["wind_off_fcst_mw"].fillna(0.0)
    out.index.name = "time"
    return out


OUTAGE_COLS = [
    "created_doc_time", "start", "end", "nominal_power", "avail_qty",
    "production_resource_name", "plant_type", "businesstype", "docstatus",
    "mrid", "revision",
]


def fetch_outages(zone: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Generation-unit unavailability messages (UMM-style, 15.1.A/B).

    Event-level rows, NOT a time series. created_doc_time is the
    publication timestamp — the leakage boundary: a forecast may only
    use messages published before its cutoff. Deduplication on
    (mrid, revision) happens in the store layer.
    """
    raw = _client().query_unavailability_of_generation_units(
        zone, start=start, end=end
    )
    if raw.empty:
        return pd.DataFrame(columns=OUTAGE_COLS)
    raw = raw.reset_index()
    raw["avail_qty"] = pd.to_numeric(raw["avail_qty"], errors="coerce")
    return raw[OUTAGE_COLS]
