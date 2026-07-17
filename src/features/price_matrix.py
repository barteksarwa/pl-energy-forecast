"""Assemble the price feature matrix for a target day. One place, one contract.

X = calendar + cutoff-safe price lags + cutoff-safe load lags
    + the TSO load forecast for the target day (known before the auction).

Two different cutoffs apply, because the information sets differ:
- price_cutoff: first delivery hour of day D. Prices for all of D-1 are
  known at bid time (fixed at the D-2 auction), so lag 24h is legal.
- load_cutoff: 09:00 local on D-1, same as the load-forecast pipeline.
  Load actuals arrive continuously; only pre-cutoff hours are usable.

All indexes are UTC hours of the target day.
"""

from __future__ import annotations

import pandas as pd

from src.features.calendar import calendar_features
from src.features.lags import lagged_load_features
from src.features.fuel import fuel_features
from src.features.outages import unavailable_capacity
from src.features.price_lags import daily_price_vector, lagged_price_features


def build_price_features(
    target_hours: pd.DatetimeIndex,
    price: pd.Series,
    load: pd.Series,
    price_cutoff: pd.Timestamp,
    load_cutoff: pd.Timestamp,
    tso: pd.Series | None = None,
    res: pd.DataFrame | None = None,
    outages: pd.DataFrame | None = None,
    fuel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Feature matrix for a target day's hours, leakage-safe on both series.

    `tso`: PSE publishes day D's load forecast ~09:00 on D-1, hours before
    the 12:00 gate closure — a legal known-future covariate. Load forecasts
    are a standard LEAR input (demand drives the merit-order position).

    `res`: the TSO day-ahead wind+solar forecast for day D. ENTSO-E
    publishes it ~18:00 on D-1 — a few hours AFTER gate closure. Using it
    is the standard proxy in the EPF literature (Lago et al. 2021 use this
    exact series): bidders have their own RES forecasts at bid time, and
    the TSO series stands in for that information set. Documented in
    DECISIONS; the model card repeats the caveat.
    """
    cal = calendar_features(target_hours)
    price_lags = lagged_price_features(price, target_hours, price_cutoff)
    day_vec = daily_price_vector(price, target_hours, price_cutoff)
    load_lags = lagged_load_features(load, target_hours, load_cutoff)
    parts = [cal, price_lags, day_vec, load_lags]
    if tso is not None:
        parts.append(tso.reindex(target_hours).rename("tso_forecast_mw"))
    if res is not None:
        parts.append(res.reindex(target_hours))
    if fuel is not None:
        parts.append(fuel_features(fuel, target_hours, load_cutoff))
    if outages is not None:
        # outage messages published before the LOAD cutoff (09:00 D-1,
        # conservative vs the 12:00 gate) — see src/features/outages.py
        parts.append(unavailable_capacity(outages, target_hours, load_cutoff))
    x = pd.concat(parts, axis=1)
    x.index.name = "time_utc"
    return x
