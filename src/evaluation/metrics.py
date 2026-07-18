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


def winkler_score(
    actual: pd.Series,
    preds: pd.DataFrame,
    coverage: float = 0.8,
) -> float:
    """Winkler score for interval forecasts. Lower is better.

    Combines band sharpness with a penalty for missed coverage.
    preds must have 'p10' and 'p90' columns; alpha = 1 - coverage.

    W = (p90 - p10) + (2/alpha) * max(p10 - y, 0) + (2/alpha) * max(y - p90, 0)

    Mean over all hours. When y falls inside [p10, p90], only the width term
    contributes. When y falls outside, the penalty term dominates.
    """
    alpha = 1.0 - coverage
    y = actual.reindex(preds.index).dropna()
    p = preds.reindex(y.index)
    width = p["p90"] - p["p10"]
    lo_pen = np.maximum(p["p10"] - y, 0.0)
    hi_pen = np.maximum(y - p["p90"], 0.0)
    scores = width + (2.0 / alpha) * (lo_pen + hi_pen)
    return float(scores.mean())
