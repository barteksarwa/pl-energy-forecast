"""The model contract. Every forecaster implements it; the backtest engine,
daily run and reports consume it. New model = new file + one REGISTRY line.
"""

from __future__ import annotations

from typing import Callable, Protocol

import pandas as pd

QUANTILES = (0.1, 0.5, 0.9)
QUANTILE_COLS = ("p10", "p50", "p90")


class QuantileForecaster(Protocol):
    """Contract: fit on a feature matrix + target, predict three quantiles."""

    name: str

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None: ...

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with columns p10/p50/p90, index = x.index."""
        ...


# name -> zero-arg factory. Populated by src/models/baselines.py (and later
# gbm.py, lstm.py, ...) at import time.
REGISTRY: dict[str, Callable[[], "QuantileForecaster"]] = {}


def register(name: str) -> Callable:
    def wrap(factory: Callable[[], "QuantileForecaster"]) -> Callable:
        REGISTRY[name] = factory
        return factory

    return wrap
