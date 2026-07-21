"""Unit tests for metrics. Hand-computed expectations."""

import math

import pandas as pd
import pytest

from src.evaluation.metrics import mae, mape, pinball_loss, rmse, winkler_score

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


def test_winkler_inside_band_is_width_only() -> None:
    """When y falls inside [p10, p90] the score equals band width exactly."""
    idx = pd.date_range("2026-01-01", periods=1, freq="1h", tz="UTC")
    y = pd.Series([100.0], index=idx)
    preds = pd.DataFrame({"p10": [90.0], "p50": [100.0], "p90": [110.0]}, index=idx)
    # y=100 inside [90,110] → Winkler = width = 20
    assert winkler_score(y, preds) == pytest.approx(20.0)


def test_winkler_outside_band_adds_penalty() -> None:
    """y above p90 adds (2/alpha) × overshoot."""
    idx = pd.date_range("2026-01-01", periods=1, freq="1h", tz="UTC")
    y = pd.Series([125.0], index=idx)  # 15 EUR above p90=110
    preds = pd.DataFrame({"p10": [90.0], "p50": [100.0], "p90": [110.0]}, index=idx)
    # alpha=0.2 → 2/alpha=10; penalty = 10*15=150; total = 20 + 150 = 170
    assert winkler_score(y, preds) == pytest.approx(170.0)


def test_winkler_wide_band_beats_narrow_when_coverage_equal() -> None:
    """Wide band scores worse than narrow when both cover the actual."""
    idx = pd.date_range("2026-01-01", periods=1, freq="1h", tz="UTC")
    y = pd.Series([100.0], index=idx)
    narrow = pd.DataFrame({"p10": [90.0], "p50": [100.0], "p90": [110.0]}, index=idx)
    wide = pd.DataFrame({"p10": [70.0], "p50": [100.0], "p90": [130.0]}, index=idx)
    assert winkler_score(y, narrow) < winkler_score(y, wide)
