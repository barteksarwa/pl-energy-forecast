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


def test_training_stops_at_0900_on_d_minus_1() -> None:
    """Regression (validation E2): the forecast for day D is decided at
    09:00 on D-1 — training data must not contain D-1 hours from 09:00
    local onward."""
    import numpy as np

    idx = pd.date_range("2024-01-01", periods=120 * 24, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"f": rng.normal(size=len(idx))}, index=idx)
    y = pd.Series(rng.normal(size=len(idx)), index=idx)

    seen: list[pd.Timestamp] = []

    class Spy:
        name = "spy"

        def fit(self, x_tr, y_tr):
            seen.append(x_tr.index.max())

        def predict(self, x_day):
            return pd.DataFrame(
                {"p10": 0.0, "p50": 0.0, "p90": 0.0}, index=x_day.index
            )

    test_start = pd.Timestamp("2024-04-01", tz="Europe/Warsaw")
    walk_forward_backtest(Spy, x, y, test_start.tz_convert("UTC"),
                          train_window_days=60, refit_every_days=1)
    assert seen, "no refits happened"
    tz = "Europe/Warsaw"
    for last in seen:
        local = last.tz_convert(tz)
        assert local.hour < 9 or local.time() < pd.Timestamp("09:00").time(), (
            f"training saw {local}, at/after the 09:00 D-1 decision moment"
        )


def test_published_target_mode_uses_full_d_minus_1_but_never_day_d() -> None:
    """Day-ahead prices for D-1 are public at decision time (fixed at
    the D-2 auction). In target_published mode, training may include
    all of D-1 — and still nothing from day D."""
    import numpy as np

    idx = pd.date_range("2024-01-01", periods=120 * 24, freq="1h", tz="UTC")
    rng = np.random.default_rng(1)
    x = pd.DataFrame({"f": rng.normal(size=len(idx))}, index=idx)
    y = pd.Series(rng.normal(size=len(idx)), index=idx)

    seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    class Spy:
        name = "spy"

        def fit(self, x_tr, y_tr):
            seen.append(x_tr.index.max())

        def predict(self, x_day):
            return pd.DataFrame(
                {"p10": 0.0, "p50": 0.0, "p90": 0.0}, index=x_day.index
            )

    test_start = pd.Timestamp("2024-04-01", tz="Europe/Warsaw")
    result = walk_forward_backtest(
        Spy, x, y, test_start.tz_convert("UTC"),
        train_window_days=60, refit_every_days=1,
        train_cutoff="target_published")
    tz = "Europe/Warsaw"
    # every refit sees D-1 evening hours (not cut at 09:00) ...
    assert all(last.tz_convert(tz).hour >= 21 for last in seen)
    # ... and even the final refit trains strictly before its target day
    pred_days = pd.Index(result.predictions.index.tz_convert(tz).date)
    assert max(seen).tz_convert(tz).date() < max(pred_days)
