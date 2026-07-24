"""Unit tests for the canonical-series merge (PSE + ENTSO-E).

The precedence rule is load-bearing: a silent flip would overwrite the
canonical PSE hours with ENTSO-E values everywhere they overlap.
"""

import pandas as pd

from src.ingestion.crosscheck import compare, merge_canonical


def _series(start: str, hours: int, value: float) -> pd.Series:
    idx = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    return pd.Series(value, index=idx)


def test_pse_wins_on_overlap():
    pse = _series("2024-06-14", 48, 100.0)
    ent = _series("2024-06-14", 48, 999.0)
    merged = merge_canonical(pse, ent)
    assert (merged == 100.0).all()


def test_entsoe_fills_history_before_pse():
    ent = _series("2024-06-12", 96, 999.0)  # two days earlier
    pse = _series("2024-06-14", 48, 100.0)
    merged = merge_canonical(pse, ent)
    assert len(merged) == 96
    assert merged.index.is_monotonic_increasing
    assert (merged.loc[: "2024-06-13 23:00+00:00"] == 999.0).all()
    assert (merged.loc["2024-06-14":] == 100.0).all()


def test_merge_has_no_duplicate_hours():
    pse = _series("2024-06-14", 48, 100.0)
    ent = _series("2024-06-13", 72, 999.0)
    merged = merge_canonical(pse, ent)
    assert not merged.index.duplicated().any()


def test_compare_counts_overlap_and_differences():
    pse = _series("2024-06-14", 24, 100.0)
    ent = pse.copy()
    ent.iloc[0] = 110.0  # 10% off on one hour
    out = compare(pse, ent, "load")
    assert out["overlap_hours"] == 24
    assert out["hours_over_1pct"] == 1
    assert out["p99_abs_diff_mw"] > 0
