"""LightGBM quantile model tests on synthetic nonlinear data."""

import numpy as np
import pandas as pd

from src.models.gbm import LightGBMQuantile


def _nonlinear_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load with a V-shaped temperature response — linear models cannot fit it."""
    idx = pd.date_range("2025-01-01", periods=120 * 24, freq="1h", tz="UTC")
    rng = np.random.default_rng(3)
    hours = idx.hour.to_numpy()
    temp = rng.uniform(-15, 30, len(idx))
    y = (
        18000
        + 2000 * np.sin((hours - 6) / 24 * 2 * np.pi)
        + 120 * np.abs(temp - 17)  # the V shape
        + rng.normal(0, 200, len(idx))
    )
    x = pd.DataFrame(
        {"hour_local": hours, "temperature_2m": temp, "load_lag_168h": 18000.0},
        index=idx,
    )
    return x, pd.Series(y, index=idx)


def test_quantiles_ordered_and_calibrated() -> None:
    x, y = _nonlinear_data()
    split = len(x) * 3 // 4
    model = LightGBMQuantile()
    model.fit(x.iloc[:split], y.iloc[:split])
    pred = model.predict(x.iloc[split:])
    actual = y.iloc[split:]

    assert (pred["p10"] <= pred["p50"]).all()
    assert (pred["p50"] <= pred["p90"]).all()
    coverage = ((actual >= pred["p10"]) & (actual <= pred["p90"])).mean()
    assert 0.6 < coverage < 0.95  # nominal 0.8; loose bounds for small sample


def test_learns_nonlinear_temperature_response() -> None:
    x, y = _nonlinear_data()
    split = len(x) * 3 // 4
    model = LightGBMQuantile()
    model.fit(x.iloc[:split], y.iloc[:split])
    pred = model.predict(x.iloc[split:])
    mae = (y.iloc[split:] - pred["p50"]).abs().mean()
    # Noise floor is ~160 (mean |N(0,200)|); linear-in-temp fit would sit >600.
    assert mae < 400
