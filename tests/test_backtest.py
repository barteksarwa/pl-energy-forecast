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
    """Corrupt y from a mid-test day onward; every DAILY refit before that
    day must be unaffected, so predictions up to it must not change.

    refit_every_days=1 on purpose: with one big fit before the test
    period (refit 999) this test would pass even if every later refit
    leaked — it must exercise refits that run right up against the
    corruption boundary."""
    cut = pd.Timestamp("2026-06-01", tz="UTC")
    res_clean = walk_forward_backtest(Climatology, X, Y, TEST_START, refit_every_days=1)
    y_dirty = Y.copy()
    y_dirty[y_dirty.index >= cut] = 1e9
    # Features stay clean (they are cutoff-safe by construction); only the
    # future *target* is corrupted. Training before `cut` must never see it.
    res_dirty = walk_forward_backtest(Climatology, X, y_dirty, TEST_START, refit_every_days=1)
    before = res_clean.predictions.index < cut
    pd.testing.assert_frame_equal(
        res_clean.predictions[before], res_dirty.predictions[before]
    )
    # sanity: the corruption is real — after the cut they must differ
    assert not res_clean.predictions[~before].equals(res_dirty.predictions[~before])


def test_summarize_has_skill_column_and_naive_zero() -> None:
    naive = walk_forward_backtest(SeasonalNaive, X, Y, TEST_START)
    table = summarize([naive], Y)
    assert table.loc["seasonal_naive", "skill_vs_naive"] == pytest.approx(0.0)


def test_lasso_ar_runs_and_orders_quantiles() -> None:
    res = walk_forward_backtest(LassoAR, X, Y, TEST_START, refit_every_days=60)
    p = res.predictions.dropna()
    assert (p["p10"] <= p["p90"]).all()


def _spy_backtest(**kwargs) -> list[pd.Timestamp]:
    """Run a spy model over noise; return the max training timestamp per fit."""
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
                          train_window_days=60, refit_every_days=1, **kwargs)
    assert seen, "no refits happened"
    return seen


def test_realtime_training_stops_at_0900_on_d_minus_1() -> None:
    """Regression (validation E2): a live-observed target decided at 09:00
    on D-1 — training data must not contain D-1 hours from 09:00 local
    onward."""
    for last in _spy_backtest():
        local = last.tz_convert("Europe/Warsaw")
        assert local.hour < 9 or local.time() < pd.Timestamp("09:00").time(), (
            f"training saw {local}, at/after the 09:00 D-1 decision moment"
        )


def test_day_ahead_training_sees_full_d_minus_1() -> None:
    """A day-ahead-published target (DA price): the whole D-1 curve is
    known at decision time. Training must reach 23:00 local D-1 — and
    never touch the target day."""
    for last in _spy_backtest(target_availability="day_ahead"):
        local = last.tz_convert("Europe/Warsaw")
        assert local.hour == 23, (
            f"daily refit should train through 23:00 D-1, saw {local}"
        )


def test_unknown_target_availability_rejected() -> None:
    with pytest.raises(ValueError, match="target_availability"):
        _spy_backtest(target_availability="typo")


def test_day_ahead_mode_never_sees_day_d() -> None:
    """Ported from the parallel session's `train_cutoff` fix: in
    day_ahead mode training may reach through D-1 but must never touch
    day D. Corrupt from a mid-test day on; daily-refit predictions
    before it must be identical."""
    cut = pd.Timestamp("2026-06-01", tz="UTC")
    clean = walk_forward_backtest(Climatology, X, Y, TEST_START,
                                  refit_every_days=1,
                                  target_availability="day_ahead")
    y_dirty = Y.copy()
    y_dirty[y_dirty.index >= cut] = 1e9
    dirty = walk_forward_backtest(Climatology, X, y_dirty, TEST_START,
                                  refit_every_days=1,
                                  target_availability="day_ahead")
    before = clean.predictions.index < cut
    pd.testing.assert_frame_equal(
        clean.predictions[before], dirty.predictions[before]
    )
