"""Backtest engine + baseline model tests on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import summarize, walk_forward_backtest
from src.models.base import REGISTRY
from src.models.baselines import Climatology, LassoAR, RidgeForecaster, SeasonalNaive

TZ = "Europe/Warsaw"


def _synthetic() -> tuple[pd.DataFrame, pd.Series]:
    """~8 months of load with daily+weekly pattern, plus matching features."""
    idx = pd.date_range("2025-11-01", "2026-07-01", freq="1h", tz="UTC")
    hours = idx.tz_convert(TZ).hour.to_numpy()
    dow = idx.tz_convert(TZ).dayofweek.to_numpy()
    rng = np.random.default_rng(7)
    y = (
        20000
        + 2500 * np.sin((hours - 6) / 24 * 2 * np.pi)
        - 1500 * (dow >= 5)
        + rng.normal(0, 300, len(idx))
    )
    y = pd.Series(y, index=idx, name="load_mw")

    x = pd.DataFrame(index=idx)
    x["hour_local"] = hours
    x["is_weekend"] = (dow >= 5).astype(int)
    x["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    x["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    for lag in (48, 72, 168, 336, 504, 672):
        x[f"load_lag_{lag}h"] = y.shift(lag)
    x["temperature_2m"] = rng.normal(10, 8, len(idx))
    return x, y


X, Y = _synthetic()
TEST_START = pd.Timestamp("2026-05-01", tz="UTC")


def test_registry_contains_all_baselines() -> None:
    assert {"seasonal_naive", "climatology", "ridge", "lasso_ar"} <= set(REGISTRY)


def test_walk_forward_produces_ordered_quantiles() -> None:
    res = walk_forward_backtest(SeasonalNaive, X, Y, TEST_START)
    p = res.predictions.dropna()
    assert (p["p10"] <= p["p50"]).all() and (p["p50"] <= p["p90"]).all()
    assert len(p) > 24 * 50


def test_ridge_beats_naive_on_learnable_pattern() -> None:
    naive = walk_forward_backtest(SeasonalNaive, X, Y, TEST_START)
    ridge = walk_forward_backtest(RidgeForecaster, X, Y, TEST_START, refit_every_days=30)
    table = summarize([naive, ridge], Y)
    # Pattern is linear in the features; ridge must clearly win.
    assert table.loc["ridge", "mae"] < table.loc["seasonal_naive", "mae"]
    assert table.loc["ridge", "skill_vs_naive"] > 0.1


def test_backtest_ignores_future_target_values() -> None:
    """Corrupt y after each prediction day; predictions must not change."""
    res_clean = walk_forward_backtest(Climatology, X, Y, TEST_START, refit_every_days=999)
    y_dirty = Y.copy()
    y_dirty[y_dirty.index >= TEST_START] = 1e9
    # Features stay clean (they are cutoff-safe by construction); only the
    # future *target* is corrupted. Training must never see it.
    res_dirty = walk_forward_backtest(Climatology, X, y_dirty, TEST_START, refit_every_days=999)
    pd.testing.assert_frame_equal(res_clean.predictions, res_dirty.predictions)


def test_summarize_has_skill_column_and_naive_zero() -> None:
    naive = walk_forward_backtest(SeasonalNaive, X, Y, TEST_START)
    table = summarize([naive], Y)
    assert table.loc["seasonal_naive", "skill_vs_naive"] == pytest.approx(0.0)


def test_lasso_ar_runs_and_orders_quantiles() -> None:
    res = walk_forward_backtest(LassoAR, X, Y, TEST_START, refit_every_days=60)
    p = res.predictions.dropna()
    assert (p["p10"] <= p["p90"]).all()
