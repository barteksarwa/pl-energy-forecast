"""Fuel and carbon price proxies via Yahoo Finance daily settlements.

Free proxies for the two marginal-cost drivers the merit order runs on:
- TTF=F  — Dutch TTF natural gas front-month future, EUR/MWh.
- CO2.MI — WisdomTree Carbon ETC (Milan), tracks EUA carbon futures, EUR.
           A tracking proxy, not the EUA settlement itself; documented
           in DATA_CATALOG terms as "free proxy, desk reality is a paid
           feed".

Daily closes only. The close of trading day T becomes known the evening
of T — the leakage rule lives in src/features/fuel.py, not here.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

TICKERS = {"TTF=F": "ttf_eur_mwh", "CO2.MI": "eua_proxy_eur"}


def fetch_fuel_history(start: str, end: str | None = None) -> pd.DataFrame:
    """Daily closes, indexed by trading date (naive dates, no tz)."""
    frames = {}
    for ticker, col in TICKERS.items():
        h = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
        if h.empty:
            raise RuntimeError(f"yfinance returned nothing for {ticker}")
        s = h["Close"]
        s.index = pd.DatetimeIndex(s.index.date)  # drop exchange tz, keep the date
        frames[col] = s
    out = pd.DataFrame(frames).sort_index()
    out.index.name = "date"
    return out
