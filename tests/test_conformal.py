"""Conformal calibration: coverage repair + the no-future-leakage proof."""

import numpy as np
import pandas as pd

from src.evaluation.conformal import (
    conformity_scores,
    latest_offset,
    rolling_conformal,
)


def _make(n_days: int = 200, band: float = 5.0, noise: float = 10.0):
    """Truth = sine + noise; band deliberately too narrow for the noise."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=n_days * 24, freq="1h", tz="UTC")
    p50 = pd.Series(100 + 10 * np.sin(np.arange(len(idx)) / 24), index=idx)
    y = p50 + rng.normal(0, noise, len(idx))
    preds = pd.DataFrame({"p10": p50 - band, "p50": p50, "p90": p50 + band})
    return preds, y


def test_scores_sign_convention() -> None:
    idx = pd.date_range("2025-01-01", periods=3, freq="1h", tz="UTC")
    preds = pd.DataFrame({"p10": [90.0] * 3, "p50": [100.0] * 3, "p90": [110.0] * 3},
                         index=idx)
    y = pd.Series([100.0, 120.0, 80.0], index=idx)  # inside, above, below
    s = conformity_scores(preds, y)
    assert s.iloc[0] < 0 and s.iloc[1] == 10.0 and s.iloc[2] == 10.0


def test_coverage_repaired() -> None:
    preds, y = _make()
    raw_cov = ((y >= preds["p10"]) & (y <= preds["p90"])).mean()
    adj = rolling_conformal(preds, y)
    # score only the days that actually received a correction
    changed = adj["p90"] != preds["p90"]
    cov = ((y >= adj["p10"]) & (y <= adj["p90"]))[changed].mean()
    assert raw_cov < 0.5  # the setup really was broken
    assert 0.74 <= cov <= 0.86  # near nominal 80%


def test_overcovering_band_gets_tightened() -> None:
    preds, y = _make(band=50.0, noise=5.0)  # band far too wide
    adj = rolling_conformal(preds, y)
    changed = adj["p90"] != preds["p90"]
    assert changed.any()
    assert (adj.loc[changed, "p90"] < preds.loc[changed, "p90"]).all()


def test_no_future_leakage() -> None:
    """Corrupt the future: corrections before the corruption must not move."""
    preds, y = _make()
    adj_clean = rolling_conformal(preds, y)
    cut = y.index[len(y) // 2]
    y_dirty = y.copy()
    y_dirty[y_dirty.index >= cut] += 1000.0
    adj_dirty = rolling_conformal(preds, y_dirty)
    before = adj_clean.index < cut
    pd.testing.assert_frame_equal(adj_clean[before], adj_dirty[before])


def test_band_still_contains_p50() -> None:
    preds, y = _make()
    adj = rolling_conformal(preds, y)
    assert (adj["p10"] <= adj["p50"]).all()
    assert (adj["p90"] >= adj["p50"]).all()


def test_latest_offset_positive_for_narrow_band() -> None:
    preds, y = _make()
    assert latest_offset(preds, y) > 0
