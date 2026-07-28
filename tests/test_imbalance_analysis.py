"""Unit tests for the imbalance-market analysis. Synthetic frames only.

No network, no data/ files. Only the pure logic: spread, sign convention,
implied-rate guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.run_imbalance_analysis import (
    compute_spread,
    implied_fx_rate,
    miss_cost,
    spread_by_group,
    spread_summary,
    summarize_miss_cost,
)


def _idx(n: int, start: str = "2025-01-01 00:00") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="h", tz="UTC")


# --------------------------------------------------------------------------
# spread
# --------------------------------------------------------------------------
def test_spread_is_balancing_minus_day_ahead():
    idx = _idx(3)
    da = pd.Series([100.0, 200.0, 300.0], index=idx)
    bal = pd.Series([150.0, 180.0, 300.0], index=idx)
    out = compute_spread(da, bal)
    assert out["spread_pln"].tolist() == [50.0, -20.0, 0.0]


def test_spread_drops_unpaired_and_nan_hours():
    idx = _idx(4)
    da = pd.Series([100.0, 200.0, np.nan, 400.0], index=idx)
    bal = pd.Series([110.0, 220.0, 330.0], index=idx[:3])
    out = compute_spread(da, bal)
    # hour 2 has no DA, hour 3 has no balancing
    assert out.index.tolist() == list(idx[:2])


def test_spread_output_is_sorted():
    idx = _idx(3)
    da = pd.Series([100.0, 200.0, 300.0], index=idx)[::-1]
    bal = pd.Series([110.0, 210.0, 310.0], index=idx)
    out = compute_spread(da, bal)
    assert out.index.is_monotonic_increasing


def test_spread_summary_numbers():
    s = pd.Series([-10.0, 0.0, 10.0, 20.0])
    head = spread_summary(s)
    assert head["n_hours"] == 4
    assert head["mean_pln"] == pytest.approx(5.0)
    assert head["median_pln"] == pytest.approx(5.0)
    assert head["mean_abs_pln"] == pytest.approx(10.0)
    # strictly above zero: 10 and 20 only
    assert head["share_bal_above_da"] == pytest.approx(0.5)


def test_spread_summary_empty_is_empty_dict():
    assert spread_summary(pd.Series(dtype=float)) == {}


def test_spread_by_group_splits_correctly():
    s = pd.Series([10.0, -30.0, 10.0, -30.0])
    groups = np.array([0, 1, 0, 1])
    out = spread_by_group(s, groups)
    assert out.loc[0, "mean_pln"] == pytest.approx(10.0)
    assert out.loc[1, "mean_pln"] == pytest.approx(-30.0)
    assert out.loc[1, "mean_abs_pln"] == pytest.approx(30.0)
    assert out.loc[0, "share_bal_above_da"] == pytest.approx(1.0)
    assert out.loc[1, "share_bal_above_da"] == pytest.approx(0.0)
    assert out["n_hours"].tolist() == [2, 2]


# --------------------------------------------------------------------------
# sign convention
# --------------------------------------------------------------------------
def test_under_forecast_pays_the_spread():
    """Forecast too low -> bought too little -> buys the rest at balancing."""
    idx = _idx(1)
    out = miss_cost(
        pd.Series([90.0], index=idx),   # p50
        pd.Series([100.0], index=idx),  # actual
        pd.Series([40.0], index=idx),   # spread bal - DA
    )
    assert out["direction"].iloc[0] == "under"
    assert out["cost_pln_per_mwh"].iloc[0] == pytest.approx(40.0)


def test_over_forecast_pays_the_negative_spread():
    """Forecast too high -> bought too much -> sells surplus at balancing."""
    idx = _idx(1)
    out = miss_cost(
        pd.Series([110.0], index=idx),
        pd.Series([100.0], index=idx),
        pd.Series([40.0], index=idx),
    )
    assert out["direction"].iloc[0] == "over"
    assert out["cost_pln_per_mwh"].iloc[0] == pytest.approx(-40.0)


def test_exact_hit_costs_nothing():
    idx = _idx(1)
    out = miss_cost(
        pd.Series([100.0], index=idx),
        pd.Series([100.0], index=idx),
        pd.Series([999.0], index=idx),
    )
    assert out["direction"].iloc[0] == "exact"
    assert out["cost_pln_per_mwh"].iloc[0] == pytest.approx(0.0)


def test_lucky_miss_has_negative_cost():
    """Under-forecast into a negative spread: balancing was cheaper."""
    idx = _idx(1)
    out = miss_cost(
        pd.Series([90.0], index=idx),
        pd.Series([100.0], index=idx),
        pd.Series([-25.0], index=idx),
    )
    assert out["cost_pln_per_mwh"].iloc[0] == pytest.approx(-25.0)


def test_cost_magnitude_is_the_spread_not_the_error():
    """A 1 MWh position: cost depends on the spread, not on error size."""
    idx = _idx(2)
    out = miss_cost(
        pd.Series([10.0, 99.0], index=idx),
        pd.Series([100.0, 100.0], index=idx),
        pd.Series([30.0, 30.0], index=idx),
    )
    assert out["cost_pln_per_mwh"].tolist() == pytest.approx([30.0, 30.0])


def test_miss_cost_aligns_and_drops_missing():
    idx = _idx(3)
    out = miss_cost(
        pd.Series([90.0, 90.0, 90.0], index=idx),
        pd.Series([100.0, np.nan, 100.0], index=idx),
        pd.Series([10.0, 10.0], index=idx[:2]),
    )
    assert out.index.tolist() == [idx[0]]


def test_summarize_miss_cost_decomposition():
    idx = _idx(4)
    out = miss_cost(
        pd.Series([90.0, 90.0, 110.0, 110.0], index=idx),
        pd.Series([100.0] * 4, index=idx),
        pd.Series([40.0, -20.0, 40.0, -20.0], index=idx),
    )
    s = summarize_miss_cost(out)
    # costs: +40, -20, -40, +20  -> mean 0
    assert s["mean_cost_pln"] == pytest.approx(0.0)
    assert s["mean_cost_under_pln"] == pytest.approx(10.0)
    assert s["mean_cost_over_pln"] == pytest.approx(-10.0)
    assert s["share_under_forecast"] == pytest.approx(0.5)
    assert s["share_costly_hours"] == pytest.approx(0.5)
    assert s["naive_cost_bound_pln"] == pytest.approx(30.0)
    assert s["mae_pln"] == pytest.approx(10.0)


def test_summarize_miss_cost_empty():
    assert summarize_miss_cost(pd.DataFrame()) == {}


# --------------------------------------------------------------------------
# implied FX rate guard
# --------------------------------------------------------------------------
def test_implied_rate_is_the_plain_ratio_when_prices_are_healthy():
    idx = _idx(2)
    rate = implied_fx_rate(
        pd.Series([425.0, 850.0], index=idx),
        pd.Series([100.0, 200.0], index=idx),
    )
    assert rate.tolist() == pytest.approx([4.25, 4.25])


def test_zero_eur_price_does_not_produce_inf():
    idx = _idx(3)
    rate = implied_fx_rate(
        pd.Series([425.0, 0.0, 425.0], index=idx),
        pd.Series([100.0, 0.0, 100.0], index=idx),
    )
    assert np.isfinite(rate).all()
    assert rate.iloc[1] == pytest.approx(4.25)  # local-day median fills it


def test_negative_eur_price_is_not_trusted():
    idx = _idx(3)
    rate = implied_fx_rate(
        pd.Series([425.0, -40.0, 425.0], index=idx),
        pd.Series([100.0, -10.0, 100.0], index=idx),
    )
    assert rate.tolist() == pytest.approx([4.25, 4.25, 4.25])


def test_tiny_eur_price_below_floor_is_replaced():
    idx = _idx(3)
    # hour 1: 4.0 EUR is below the 5.0 floor and the ratio is nonsense
    rate = implied_fx_rate(
        pd.Series([425.0, 900.0, 425.0], index=idx),
        pd.Series([100.0, 4.0, 100.0], index=idx),
        min_eur=5.0,
    )
    assert rate.iloc[1] == pytest.approx(4.25)


def test_rate_carries_across_days_when_a_day_has_no_trusted_hour():
    idx = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
    da = pd.Series(425.0, index=idx)
    eur = pd.Series(100.0, index=idx)
    eur.iloc[24:] = 0.0  # whole second day untrusted
    da.iloc[24:] = 0.0
    rate = implied_fx_rate(da, eur)
    assert rate.iloc[-1] == pytest.approx(4.25)
    assert np.isfinite(rate).all()


def test_implied_rate_raises_when_nothing_is_trustworthy():
    idx = _idx(2)
    with pytest.raises(ValueError):
        implied_fx_rate(
            pd.Series([0.0, 0.0], index=idx),
            pd.Series([0.0, 0.0], index=idx),
        )
