"""Unit tests for the price model classes (naives + LEAR).

The LEAR tests use a tiny synthetic matrix — enough hours per delivery
hour for LassoCV, small enough to stay fast. The z-clip regression test
guards the documented 38,000 EUR/MWh extrapolation blowup (May 2025).
"""

import numpy as np
import pandas as pd
import pytest

from src.models.price import (
    PRICE_LAG_COLS,
    PriceLEAR,
    PriceNaiveYesterday,
    PriceNaiveWeek,
)


def _lag_frame(n: int = 48) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {c: rng.uniform(50, 150, n) for c in PRICE_LAG_COLS}, index=idx
    )


def test_naive_yesterday_p50_is_lag1d_and_band_valid():
    x = _lag_frame()
    out = PriceNaiveYesterday().predict(x)
    assert (out["p50"] == x["price_lag_1d"]).all()
    assert (out["p10"] <= out["p50"]).all()
    assert (out["p50"] <= out["p90"]).all()


def test_naive_week_p50_is_lag7d_and_band_valid():
    x = _lag_frame()
    out = PriceNaiveWeek().predict(x)
    assert (out["p50"] == x["price_lag_7d"]).all()
    assert (out["p10"] <= out["p50"]).all()
    assert (out["p50"] <= out["p90"]).all()


def _lear_matrix(days: int = 25, seed: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic hourly matrix with the columns LEAR touches:
    hour_local, a few price_* features, one non-price feature."""
    idx = pd.date_range("2025-01-01", periods=days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    hour = pd.Series(idx.tz_convert("Europe/Warsaw").hour, index=idx)
    base = 100 + 30 * np.sin(2 * np.pi * hour.to_numpy() / 24)
    noise = rng.normal(0, 5, len(idx))
    y = pd.Series(base + noise, index=idx)
    x = pd.DataFrame(
        {
            "hour_local": hour,
            "price_lag_1d": np.roll(y.to_numpy(), 24),
            "price_lag_7d": np.roll(y.to_numpy(), 24 * 7),
            "price_d1_h12": np.repeat(
                y.groupby(pd.Index(idx.date)).mean().to_numpy(), 24
            )[: len(idx)],
            "tso_load_fcst": rng.uniform(15000, 25000, len(idx)),
        },
        index=idx,
    )
    return x, y


@pytest.fixture(scope="module")
def fitted_lear() -> tuple[PriceLEAR, pd.DataFrame, pd.Series]:
    x, y = _lear_matrix()
    model = PriceLEAR()
    model.fit(x, y)
    return model, x, y


def test_lear_transform_roundtrip(fitted_lear):
    model, _, y = fitted_lear
    p = y.to_numpy()
    assert np.allclose(model._from_z(model._to_z(p)), p)


def test_lear_band_valid_and_finite(fitted_lear):
    model, x, _ = fitted_lear
    out = model.predict(x.iloc[-48:])
    assert out.notna().all().all()
    assert (out["p10"] <= out["p50"]).all()
    assert (out["p50"] <= out["p90"]).all()


def test_lear_zclip_stops_extrapolation_blowup(fitted_lear):
    """Regression: features far outside the training range must not
    produce four-digit prices through sinh-back (the 38k EUR bug)."""
    model, x, y = fitted_lear
    crazy = x.iloc[-24:].copy()
    for c in crazy.columns:
        if c.startswith("price_"):
            crazy[c] = 1e5  # absurd feature values
    out = model.predict(crazy)
    # hard ceiling implied by the clip: z_max + margin, mapped back
    ceiling = model._from_z(np.array([model._z_clip[1]]))[0]
    assert out["p50"].max() <= ceiling + 1e-6
    assert out["p50"].max() < 10 * y.max()  # nowhere near 38k


def test_lear_unseen_hour_falls_back_to_nearest(fitted_lear):
    model, x, _ = fitted_lear
    # steal a fitted model away to simulate a DST-only hour
    stolen = model._models.pop(2)
    try:
        rows = x[x["hour_local"] == 2].iloc[-2:]
        out = model.predict(rows)
        assert out.notna().all().all()
    finally:
        model._models[2] = stolen


def test_lear_rejects_changed_feature_columns(fitted_lear):
    model, x, _ = fitted_lear
    wrong = x.rename(columns={"tso_load_fcst": "renamed"})
    with pytest.raises(AssertionError, match="feature columns"):
        model.predict(wrong)
