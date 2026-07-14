"""Point and quantile metrics. Pure functions, no I/O."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _aligned(actual: pd.Series, forecast: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    df = pd.concat([actual.rename("a"), forecast.rename("f")], axis=1).dropna()
    return df["a"].to_numpy(), df["f"].to_numpy()


def mae(actual: pd.Series, forecast: pd.Series) -> float:
    a, f = _aligned(actual, forecast)
    return float(np.mean(np.abs(a - f))) if a.size else float("nan")


def rmse(actual: pd.Series, forecast: pd.Series) -> float:
    a, f = _aligned(actual, forecast)
    return float(np.sqrt(np.mean((a - f) ** 2))) if a.size else float("nan")


def mape(actual: pd.Series, forecast: pd.Series) -> float:
    """Mean absolute percentage error, in percent. Zero actuals are dropped."""
    a, f = _aligned(actual, forecast)
    mask = a != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((a[mask] - f[mask]) / a[mask]))) * 100.0


def pinball_loss(actual: pd.Series, forecast: pd.Series, quantile: float) -> float:
    """Pinball (quantile) loss. Lower is better.

    Penalizes under-prediction by `quantile`, over-prediction by `1 - quantile`.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    a, f = _aligned(actual, forecast)
    if not a.size:
        return float("nan")
    diff = a - f
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))
