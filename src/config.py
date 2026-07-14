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
    cities: list[City]
    weather_vars: list[str]
    paths: dict[str, Path]
    naive_season_days: int
    naive_n_seasons: int
    history_days: int


def load_config(path: Path = CONFIG_PATH) -> Config:
    raw = yaml.safe_load(path.read_text())
    return Config(
        zone=raw["zone"],
        timezone_local=raw["timezone_local"],
        cities=[City(**c) for c in raw["cities"]],
        weather_vars=list(raw["weather_vars"]),
        paths={k: REPO_ROOT / v for k, v in raw["paths"].items()},
        naive_season_days=int(raw["naive"]["season_days"]),
        naive_n_seasons=int(raw["naive"]["n_seasons"]),
        history_days=int(raw["daily_run"]["history_days"]),
    )
