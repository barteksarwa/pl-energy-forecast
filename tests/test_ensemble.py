import numpy as np
import pandas as pd

from src.evaluation.ensemble import blend, crps3, rolling_weights


def _preds(bias: float, n: int = 24 * 200) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    y = pd.Series(100.0 + np.sin(np.arange(n) / 24) * 10, index=idx)
    p50 = y + bias
    return pd.DataFrame({"p10": p50 - 10, "p50": p50, "p90": p50 + 10}), y


def test_crps3_lower_for_better_model() -> None:
    good, y = _preds(0.0)
    bad, _ = _preds(8.0)
    assert crps3(good, y).mean() < crps3(bad, y).mean()


def test_weights_favor_better_member_after_warmup() -> None:
    good, y = _preds(0.0)
    bad, _ = _preds(8.0)
    scores = {"good": crps3(good, y), "bad": crps3(bad, y)}
    days = sorted(set(good.index.tz_convert("Europe/Warsaw").date))
    w = rolling_weights(scores, days)
    assert np.isclose(w.iloc[0].sum(), 1.0)
    assert w["good"].iloc[-1] > w["bad"].iloc[-1]
    # warm-up rows are equal weights
    assert np.isclose(w["good"].iloc[0], 0.5)


def test_weights_are_past_only() -> None:
    good, y = _preds(0.0)
    bad, y2 = _preds(8.0)
    scores = {"g": crps3(good, y), "b": crps3(bad, y)}
    days = sorted(set(good.index.tz_convert("Europe/Warsaw").date))
    w_clean = rolling_weights(scores, days)
    # corrupt the future half of the bad member's scores
    s2 = dict(scores)
    cut = scores["b"].index[len(scores["b"]) // 2]
    dirty = scores["b"].copy()
    dirty[dirty.index >= cut] *= 100
    s2["b"] = dirty
    w_dirty = rolling_weights(s2, days)
    before = w_clean.index < cut.tz_convert("Europe/Warsaw").date()
    pd.testing.assert_frame_equal(w_clean[before], w_dirty[before])


def test_blend_ordering_and_range() -> None:
    a, y = _preds(0.0)
    b, _ = _preds(4.0)
    scores = {"a": crps3(a, y), "b": crps3(b, y)}
    days = sorted(set(a.index.tz_convert("Europe/Warsaw").date))
    w = rolling_weights(scores, days)
    ens = blend({"a": a, "b": b}, w)
    assert (ens["p10"] <= ens["p50"]).all()
    assert (ens["p50"] <= ens["p90"]).all()
    # blended p50 sits between the members'
    assert (ens["p50"] >= a["p50"].reindex(ens.index) - 1e-9).all()
    assert (ens["p50"] <= b["p50"].reindex(ens.index) + 1e-9).all()
