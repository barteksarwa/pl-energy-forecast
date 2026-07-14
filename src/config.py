"""Load and expose config/config.yaml. The only place that reads it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class City:
    name: str
    lat: float
    lon: float
    weight: float


@dataclass(frozen=True)
class Config:
    zone: str
    timezone_local: str
    data_source: str
    cities: list[City]
    weather_vars: list[str]
    paths: dict[str, Path]
    naive_season_days: int
    naive_n_seasons: int
    history_days: int
    backfill_start: str
    entsoe_chunk_days: int
    archive_lag_days: int
    request_sleep_s: float
    forecast_start: str
    forecast_leads: list[int]


def load_config(path: Path = CONFIG_PATH) -> Config:
    raw = yaml.safe_load(path.read_text())
    return Config(
        zone=raw["zone"],
        timezone_local=raw["timezone_local"],
        data_source=raw.get("data_source", "pse"),
        cities=[City(**c) for c in raw["cities"]],
        weather_vars=list(raw["weather_vars"]),
        paths={k: REPO_ROOT / v for k, v in raw["paths"].items()},
        naive_season_days=int(raw["naive"]["season_days"]),
        naive_n_seasons=int(raw["naive"]["n_seasons"]),
        history_days=int(raw["daily_run"]["history_days"]),
        backfill_start=str(raw["backfill"]["start_date"]),
        entsoe_chunk_days=int(raw["backfill"]["entsoe_chunk_days"]),
        archive_lag_days=int(raw["backfill"]["archive_lag_days"]),
        request_sleep_s=float(raw["backfill"]["request_sleep_s"]),
        forecast_start=str(raw["backfill"]["forecast_start_date"]),
        forecast_leads=[int(d) for d in raw["backfill"]["forecast_leads_days"]],
    )
