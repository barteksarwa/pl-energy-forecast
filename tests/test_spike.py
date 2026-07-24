"""Unit tests for the spike classifier (daily-report component).

The leakage property is the point: the top-5% threshold must come from
the TRAINING window only, never from test-period prices.
"""

import numpy as np
import pandas as pd

from src.models.spike import SpikeClassifier


def _data(n: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Separable toy set: high `driver` hours spike."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    driver = rng.uniform(0, 1, n)
    price = 80 + 40 * driver + rng.normal(0, 5, n)
    price[driver > 0.95] += 300  # spikes ride the driver
    x = pd.DataFrame(
        {"driver": driver, "noise": rng.normal(size=n)}, index=idx
    )
    return x, pd.Series(price, index=idx, name="price")


def test_threshold_comes_from_training_window_only():
    x, y = _data()
    clf = SpikeClassifier()
    clf.fit(x, y)
    assert clf.threshold_ == float(y.quantile(0.95))
    # test-period prices 10x higher must not move the stored threshold
    clf.predict_proba(x * 10)
    assert clf.threshold_ == float(y.quantile(0.95))


def test_deterministic_given_seed():
    x, y = _data()
    a = SpikeClassifier(seed=42)
    b = SpikeClassifier(seed=42)
    a.fit(x, y)
    b.fit(x, y)
    pd.testing.assert_series_equal(a.predict_proba(x), b.predict_proba(x))


def test_probabilities_valid_and_separate_spikes():
    x, y = _data()
    clf = SpikeClassifier()
    clf.fit(x, y)
    p = clf.predict_proba(x)
    assert ((p >= 0) & (p <= 1)).all()
    spike_hours = y >= clf.threshold_
    # a separable toy problem must rank spike hours far above calm ones
    assert p[spike_hours].mean() > p[~spike_hours].mean() + 0.5
