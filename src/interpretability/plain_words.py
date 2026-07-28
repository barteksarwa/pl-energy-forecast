"""Feature → plain words, shared by every driver explanation.

Import-light on purpose (numpy only): the daily report writer uses it,
and pulling shap/matplotlib into the report path would be waste.
"""

from __future__ import annotations

import numpy as np

PLAIN_WORDS = {
    "load_lag_48h": "load two days ago at this hour",
    "load_lag_72h": "load three days ago at this hour",
    "load_lag_168h": "load last week at this hour",
    "load_lag_336h": "load two weeks ago at this hour",
    "load_lag_504h": "load three weeks ago at this hour",
    "load_lag_672h": "load four weeks ago at this hour",
    "load_mean_7d": "the average load level of the past week",
    "hour_local": "the hour of the day",
    "hour_sin": "the hour of the day",
    "hour_cos": "the hour of the day",
    "day_of_week": "the day of the week",
    "is_weekend": "weekend vs workday",
    "is_holiday": "a public holiday",
    "is_bridge_day": "a bridge day (workday squeezed next to a holiday)",
    "month": "the season",
    "doy_sin": "the time of year",
    "doy_cos": "the time of year",
    "temperature_2m": "temperature",
    "wind_speed_10m": "wind",
    "cloud_cover": "cloud cover",
    "shortwave_radiation": "sunshine",
    "relative_humidity_2m": "humidity",
    "heating_degrees": "heating demand (cold below 15°C)",
    "cooling_degrees": "cooling demand (heat above 22°C)",
    "tso_forecast_mw": "the TSO's own day-ahead forecast",
    # price features
    "price_lag_2d": "the price two days ago at this hour",
    "price_lag_3d": "the price three days ago at this hour",
    "price_lag_7d": "the price last week at this hour",
    "price_mean_7d": "the average price level of the past week",
    "solar_fcst_mw": "tomorrow's solar generation forecast",
    "wind_on_fcst_mw": "tomorrow's onshore wind forecast",
    "wind_off_fcst_mw": "tomorrow's offshore wind forecast",
    "ttf_eur_mwh": "the TTF gas price",
    "eua_proxy_eur": "the CO2 price (EUA proxy)",
}

# Column families that share one phrase; matched by prefix.
PREFIX_WORDS = {
    "price_d1_h": "yesterday's hourly price curve",
    "load_lag_": "recent load at this hour",
}


def phrase_for(column: str) -> str:
    """Plain-words phrase for a feature column, prefix families included."""
    if column in PLAIN_WORDS:
        return PLAIN_WORDS[column]
    for prefix, phrase in PREFIX_WORDS.items():
        if column.startswith(prefix):
            return phrase
    return column


def top_phrases(importance: np.ndarray, columns: list[str], n: int = 3) -> list[str]:
    """Top-n features by importance, deduplicated by plain-words phrase.

    Prefix families (e.g. the 24 price_d1_h* columns) pool their
    importance first, so one strong family beats scattered singles."""
    pooled: dict[str, float] = {}
    for imp, col in zip(importance, columns):
        phrase = phrase_for(col)
        pooled[phrase] = pooled.get(phrase, 0.0) + float(imp)
    ranked = sorted(pooled, key=pooled.get, reverse=True)
    return ranked[:n]
