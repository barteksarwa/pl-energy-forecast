"""Unit tests for metrics. Hand-computed expectations."""

import math

import pandas as pd
import pytest

from src.evaluation.metrics import mae, mape, pinball_loss, rmse

IDX = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
ACTUAL = pd.Series([100.0, 200.0, 300.0, 400.0], index=IDX)
FORECAST = pd.Series([110.0, 190.0, 330.0, 360.0], index=IDX)


def test_mae() -> None:
    assert mae(ACTUAL, FORECAST) == pytest.approx((10 + 10 + 30 + 40) / 4)


def test_rmse() -> None:
    assert rmse(ACTUAL, FORECAST) == pytest.approx(math.sqrt((100 + 100 + 900 + 1600) / 4))


def test_mape_percent() -> None:
    expected = (10 / 100 + 10 / 200 + 30 / 300 + 40 / 400) / 4 * 100
    assert mape(ACTUAL, FORECAST) == pytest.approx(expected)


def test_mape_drops_zero_actuals() -> None:
    actual = pd.Series([0.0, 100.0], index=IDX[:2])
    forecast = pd.Series([50.0, 110.0], index=IDX[:2])
    assert mape(actual, forecast) == pytest.approx(10.0)


def test_pinball_penalizes_underprediction_more_at_high_quantile() -> None:
    actual = pd.Series([100.0], index=IDX[:1])
    under = pd.Series([90.0], index=IDX[:1])
    over = pd.Series([110.0], index=IDX[:1])
    assert pinball_loss(actual, under, 0.9) > pinball_loss(actual, over, 0.9)


def test_pinball_symmetric_at_median() -> None:
    actual = pd.Series([100.0], index=IDX[:1])
    under = pd.Series([90.0], index=IDX[:1])
    over = pd.Series([110.0], index=IDX[:1])
    assert pinball_loss(actual, under, 0.5) == pytest.approx(pinball_loss(actual, over, 0.5))


def test_misaligned_indexes_use_intersection() -> None:
    shifted = FORECAST.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=1)
    # Only 3 hours overlap after shift.
    assert not math.isnan(mae(ACTUAL, shifted))
