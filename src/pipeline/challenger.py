"""Shadow challenger for the daily loop: ridge + TSO forecast combiner.

UAT discipline (PLAN M9): the challenger forecasts every day alongside the
incumbent naive; both get scored; nothing is promoted by code. Promotion is
a human decision recorded in DECISIONS.md after enough shadow days.

Trains fresh each morning on the trailing year from the processed store
(kept current by the incremental PSE backfill).
"""

from __future__ import annotations

import pandas as pd

import src.models.baselines  # noqa: F401
from src.clients.openmeteo_client import fetch_weather_forecast
from src.config import Config
from src.evaluation.run_backtest import assemble_features
from src.features.matrix import build_features
from src.features.weather import (
    add_degree_signals,
    load_weather_forecast_history,
    population_weighted,
)
from src.models.base import REGISTRY
from src.pipeline.daily_run import local_day_hours_utc, shift_local_day

CHALLENGER = "ridge"


def live_weighted_weather(cfg: Config) -> pd.DataFrame:
    """Population-weighted live forecast over all config cities."""
    frames = {
        c.name: fetch_weather_forecast(c.lat, c.lon, cfg.weather_vars,
                                       forecast_days=3, past_days=1)
        for c in cfg.cities
    }
    weights = {c.name: c.weight for c in cfg.cities}
    return add_degree_signals(population_weighted(frames, weights))


def challenger_forecast(cfg: Config, today_local: pd.Timestamp) -> pd.DataFrame:
    """Train on trailing year, forecast tomorrow. Returns p10/p50/p90 frame."""
    tz = cfg.timezone_local
    tomorrow = shift_local_day(today_local, 1, tz)

    load = pd.read_parquet(cfg.paths["data_processed"] / "load.parquet").iloc[:, 0]
    tso = pd.read_parquet(cfg.paths["data_processed"] / "tso_forecast.parquet").iloc[:, 0]
    hist_weather = load_weather_forecast_history(cfg)

    train_end = shift_local_day(today_local, -1, tz)
    train_start = shift_local_day(train_end, -365, tz)
    x_train = assemble_features(load, hist_weather, tz, train_start, train_end, tso=tso)
    y_train = load.reindex(x_train.index)
    keep = x_train.dropna().index.intersection(y_train.dropna().index)
    x_train, y_train = x_train.loc[keep], y_train.loc[keep]

    model = REGISTRY[CHALLENGER]()
    model.fit(x_train, y_train)

    hours = local_day_hours_utc(tomorrow, tz)
    cutoff = pd.Timestamp(today_local.date(), tz=tz) + pd.Timedelta(hours=9)
    # TSO for tomorrow may not be published yet (PSE publishes ~09:00 local
    # D-1). Persist the same clock hour from the day before — shape-preserving,
    # unlike a flat ffill of the last value (which also fails when the whole
    # target day lies beyond the series end).
    from src.pipeline.price_daily import persist_24h

    tso_for_pred = persist_24h(tso, hours)
    x_tomorrow = build_features(hours, load, live_weighted_weather(cfg), cutoff,
                                tso=tso_for_pred)
    x_tomorrow = x_tomorrow[x_train.columns]
    return model.predict(x_tomorrow.ffill())
