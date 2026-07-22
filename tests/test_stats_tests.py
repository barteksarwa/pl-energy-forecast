"""DM + coverage tests: sanity on synthetic data with known answers."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.stats_tests import (
    band_violations,
    christoffersen_test,
    daily_losses,
    dm_test,
    kupiec_test,
)


def _daily(vals: np.ndarray) -> pd.Series:
    idx = pd.date_range("2025-01-01", periods=len(vals), freq="D").date
    return pd.Series(vals, index=pd.Index(idx))


def test_dm_equal_models_not_significant() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(100, 10, 400)
    a = _daily(base + rng.normal(0, 1, 400))
    b = _daily(base + rng.normal(0, 1, 400))
    _, p = dm_test(a, b)
    assert p > 0.05


def test_dm_better_model_detected() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(100, 10, 400)
    a = _daily(base)          # systematically lower loss
    b = _daily(base + 5.0)
    stat, p = dm_test(a, b)
    assert stat < 0 and p < 0.01


def test_dm_needs_enough_days() -> None:
    with pytest.raises(ValueError):
        dm_test(_daily(np.ones(10)), _daily(np.ones(10)))


def test_daily_losses_sums_24h() -> None:
    idx = pd.date_range("2025-06-01", periods=48, freq="1h", tz="UTC")
    y = pd.Series(1.0, index=idx)
    p50 = pd.Series(0.0, index=idx)
    dl = daily_losses(p50, y)
    # 48 UTC hours = 2 full local days in summer (UTC+2 shifts the edges)
    assert dl.sum() == 48.0


def test_kupiec_correct_rate_passes() -> None:
    rng = np.random.default_rng(2)
    v = pd.Series((rng.random(5000) < 0.2).astype(int))
    _, p = kupiec_test(v, coverage=0.8)
    assert p > 0.05


def test_kupiec_wrong_rate_fails() -> None:
    rng = np.random.default_rng(3)
    v = pd.Series((rng.random(5000) < 0.45).astype(int))
    _, p = kupiec_test(v, coverage=0.8)
    assert p < 0.001


def test_christoffersen_iid_passes_clustered_fails() -> None:
    rng = np.random.default_rng(4)
    iid = pd.Series((rng.random(5000) < 0.2).astype(int))
    _, p_iid = christoffersen_test(iid)
    assert p_iid > 0.05
    # clustered: violations come in runs
    runs = np.zeros(5000, dtype=int)
    i = 0
    while i < 5000:
        if rng.random() < 0.05:
            runs[i:i + 8] = 1
            i += 8
        else:
            i += 1
    _, p_runs = christoffersen_test(pd.Series(runs))
    assert p_runs < 0.001


def test_band_violations_alignment() -> None:
    idx = pd.date_range("2025-01-01", periods=4, freq="1h", tz="UTC")
    preds = pd.DataFrame(
        {"p10": [0.0] * 4, "p50": [5.0] * 4, "p90": [10.0] * 4}, index=idx)
    y = pd.Series([5.0, 20.0, -3.0, np.nan], index=idx)
    v = band_violations(preds, y)
    assert list(v) == [0, 1, 1] and len(v) == 3
