"""Weather features: population-weighted aggregation over config cities.

Adds heating/cooling degree signals. Load responds to temperature in a V shape:
heating below ~15 °C, cooling above ~22 °C. Splitting the two lets linear
models fit each side separately.
"""

from __future__ import annotations

import pandas as pd

from src.config import Config

HEATING_BASE_C = 15.0
COOLING_BASE_C = 22.0


def population_weighted(
    city_frames: dict[str, pd.DataFrame], weights: dict[str, float]
) -> pd.DataFrame:
    """Weighted average of per-city hourly frames. Index: intersection of hours."""
    if set(city_frames) != set(weights):
        raise ValueError("city_frames and weights must have identical keys")
    total = sum(weights.values())
    parts = [df * (weights[name] / total) for name, df in city_frames.items()]
    out = parts[0]
    for p in parts[1:]:
        out = out.add(p, fill_value=None)  # NaN if any city missing → gap stays visible
    return out


def add_degree_signals(weather: pd.DataFrame) -> pd.DataFrame:
    """Append heating_degrees / cooling_degrees from temperature_2m."""
    out = weather.copy()
    temp = out["temperature_2m"]
    out["heating_degrees"] = (HEATING_BASE_C - temp).clip(lower=0.0)
    out["cooling_degrees"] = (temp - COOLING_BASE_C).clip(lower=0.0)
    return out


def _weighted_from_dir(cfg: Config, subdir: str) -> pd.DataFrame:
    frames: dict[str, pd.DataFrame] = {}
    for city in cfg.cities:
        path = cfg.paths["data_raw"] / subdir / f"{city.name}.parquet"
        if path.exists():
            frames[city.name] = pd.read_parquet(path)
    if not frames:
        raise FileNotFoundError(f"No data in data/raw/{subdir}. Run: make backfill")
    weights = {name: c.weight for c in cfg.cities for name in [c.name] if name in frames}
    return population_weighted(frames, weights)


def load_weather_history(cfg: Config) -> pd.DataFrame:
    """Backfilled weather actuals (ERA5), weighted + degree features."""
    return add_degree_signals(_weighted_from_dir(cfg, "weather"))


def load_weather_forecast_history(cfg: Config, lead_days: int = 2) -> pd.DataFrame:
    """Backfilled weather *forecasts* at fixed lead, renamed to base var names.

    lead 2 = known at the 09:00 D-1 cutoff (see DATA_CATALOG). Renaming to the
    base names keeps the feature matrix identical between training on actuals
    and evaluating on forecasts.
    """
    wide = _weighted_from_dir(cfg, "weather_forecast")
    suffix = f"_lead{lead_days}d"
    cols = {c: c.removesuffix(suffix) for c in wide.columns if c.endswith(suffix)}
    if not cols:
        raise KeyError(f"No columns with suffix {suffix} in weather_forecast data")
    return add_degree_signals(wide[list(cols)].rename(columns=cols))
