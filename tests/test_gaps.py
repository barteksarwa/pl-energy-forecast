"""Unit tests for gap detection."""

import pandas as pd
import pytest

from src.ingestion.gaps import find_gaps, log_gaps


def _series_with_holes() -> pd.Series:
    idx = pd.date_range("2026-01-01", periods=48, freq="1h", tz="UTC")
    s = pd.Series(1.0, index=idx)
    # one 3-hour hole and one single missing hour
    return s.drop(idx[10:13]).drop(idx[[30]])


def test_finds_two_gaps_with_correct_sizes() -> None:
    gaps = find_gaps(_series_with_holes())
    assert len(gaps) == 2
    assert gaps["n_hours"].tolist() == [3, 1]


def test_gap_bounds() -> None:
    gaps = find_gaps(_series_with_holes())
    idx = pd.date_range("2026-01-01", periods=48, freq="1h", tz="UTC")
    assert gaps.loc[0, "gap_start_utc"] == idx[10]
    assert gaps.loc[0, "gap_end_utc"] == idx[12]


def test_no_gaps_in_complete_series() -> None:
    idx = pd.date_range("2026-01-01", periods=24, freq="1h", tz="UTC")
    assert find_gaps(pd.Series(1.0, index=idx)).empty


def test_nan_counts_as_gap() -> None:
    idx = pd.date_range("2026-01-01", periods=24, freq="1h", tz="UTC")
    s = pd.Series(1.0, index=idx)
    s.iloc[5] = float("nan")
    gaps = find_gaps(s)
    assert len(gaps) == 1
    assert gaps.loc[0, "n_hours"] == 1


def test_rejects_naive_index() -> None:
    idx = pd.date_range("2026-01-01", periods=24, freq="1h")
    with pytest.raises(ValueError):
        find_gaps(pd.Series(1.0, index=idx))


def test_log_gaps_dedupes(tmp_path) -> None:
    log = tmp_path / "gap_log.csv"
    s = _series_with_holes()
    log_gaps(s, "load", log)
    log_gaps(s, "load", log)  # same gaps again
    stored = pd.read_csv(log)
    assert len(stored) == 2  # not 4


def test_log_gaps_separates_series_names(tmp_path) -> None:
    log = tmp_path / "gap_log.csv"
    s = _series_with_holes()
    log_gaps(s, "load", log)
    log_gaps(s, "weather_Warszawa", log)
    stored = pd.read_csv(log)
    assert len(stored) == 4
    assert set(stored["series"]) == {"load", "weather_Warszawa"}
