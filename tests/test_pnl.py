"""Accounting tests for the battery-arbitrage P&L engine."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.pnl import (
    Battery,
    daily_pnl,
    optimal_schedule,
    settle,
    summarize_pnl,
)

LOSSLESS = Battery(power_mw=1.0, capacity_mwh=1.0,
                   round_trip_eff=1.0, cycles_per_day=1.0)


def test_buy_low_sell_high_lossless():
    prices = np.array([10.0, 100.0])
    c, d = optimal_schedule(prices, LOSSLESS)
    assert c == pytest.approx([1.0, 0.0])
    assert d == pytest.approx([0.0, 1.0])
    assert settle(c, d, prices, LOSSLESS) == pytest.approx(90.0)


def test_efficiency_applied_on_discharge():
    bat = Battery(power_mw=1.0, capacity_mwh=1.0,
                  round_trip_eff=0.85, cycles_per_day=1.0)
    prices = np.array([10.0, 100.0])
    c, d = optimal_schedule(prices, bat)
    # buy 1 MWh at 10, deliver 0.85 MWh at 100
    assert settle(c, d, prices, bat) == pytest.approx(0.85 * 100 - 10)


def test_no_trade_when_spread_cannot_pay_losses():
    bat = Battery(round_trip_eff=0.85)
    prices = np.array([100.0, 110.0])  # 0.85*110 = 93.5 < 100
    c, d = optimal_schedule(prices, bat)
    assert settle(c, d, prices, bat) == pytest.approx(0.0)


def test_discharge_cannot_precede_charge():
    prices = np.array([100.0, 10.0])  # high first: nothing to sell yet
    c, d = optimal_schedule(prices, LOSSLESS)
    assert c.sum() == pytest.approx(0.0)
    assert d.sum() == pytest.approx(0.0)


def test_cycle_cap_limits_throughput():
    bat = Battery(power_mw=1.0, capacity_mwh=2.0,
                  round_trip_eff=1.0, cycles_per_day=1.0)
    # two profitable round trips available, cap allows 2 MWh charge total
    prices = np.array([10.0, 10.0, 100.0, 100.0, 10.0, 10.0, 100.0, 100.0])
    c, d = optimal_schedule(prices, bat)
    assert c.sum() == pytest.approx(2.0)
    assert d.sum() == pytest.approx(2.0)


def test_soc_never_negative_or_above_capacity():
    rng = np.random.default_rng(7)
    prices = rng.uniform(-50, 300, size=24)
    bat = Battery()
    c, d = optimal_schedule(prices, bat)
    soc = np.cumsum(c - d)
    assert soc.min() >= -1e-6
    assert soc.max() <= bat.capacity_mwh + 1e-6


def test_hourly_power_limit():
    prices = np.array([10.0, 100.0])
    bat = Battery(power_mw=1.0, capacity_mwh=5.0,
                  round_trip_eff=1.0, cycles_per_day=5.0)
    c, d = optimal_schedule(prices, bat)
    assert c.max() <= 1.0 + 1e-9
    assert d.max() <= 1.0 + 1e-9


def test_perfect_foresight_is_upper_bound():
    rng = np.random.default_rng(42)
    actual = rng.uniform(0, 200, size=24)
    forecast = actual + rng.normal(0, 40, size=24)
    bat = Battery()
    c_f, d_f = optimal_schedule(forecast, bat)
    c_p, d_p = optimal_schedule(actual, bat)
    assert settle(c_f, d_f, actual, bat) <= settle(c_p, d_p, actual, bat) + 1e-9


def test_daily_pnl_skips_incomplete_days_and_handles_dst():
    # 2025-03-30 is the 23-hour spring-forward day in Europe/Warsaw
    idx = pd.date_range("2025-03-28 23:00", "2025-03-31 21:00",
                        freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    actual = pd.Series(rng.uniform(20, 150, len(idx)), index=idx)
    fcst = actual.copy()
    fcst.iloc[5] = np.nan  # poke a hole in day one
    out = daily_pnl(fcst, actual)
    days = set(out.index)
    assert pd.Timestamp("2025-03-29").date() not in days  # gap -> skipped
    assert pd.Timestamp("2025-03-30").date() in days  # 23h day handled
    # forecast == actual, so capture must be exact
    assert summarize_pnl(out)["capture_rate"] == pytest.approx(1.0)
