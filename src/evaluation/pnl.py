"""Battery-arbitrage P&L: what a forecast is worth in EUR.

Scope: day-ahead market only. The battery bids a full day's schedule
at D-1 based on the P50 price forecast, then settles every hour at
the actual DA price. No intraday re-trading, no balancing exposure.
That understates real value but keeps the accounting honest.

Battery (fixed, from PLAN Phase 7): 1 MW power, 2 MWh capacity,
0.85 round-trip efficiency, at most 1 full cycle per day.
Efficiency convention: charging stores grid energy 1:1; discharging
delivers 0.85 MWh to the grid per MWh drawn from the battery. The
1 MW power limit applies to battery-side flow.

Per-day schedule: a small linear program (scipy HiGHS).
    max  sum_t price_t * (rte * d_t - c_t)
    s.t. 0 <= SoC_t = cumsum(c - d) <= capacity
         0 <= c_t, d_t <= power (hourly energy, MWh)
         sum_t c_t <= cycles_per_day * capacity
Days have 23/24/25 hours across DST switches; T is whatever the
local day contains.

Metrics:
- EUR/day: mean settled revenue.
- Capture rate: total revenue / perfect-foresight revenue (same LP
  run on actual prices). 1.0 = the forecast lost nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog

LOCAL_TZ = "Europe/Warsaw"


@dataclass(frozen=True)
class Battery:
    power_mw: float = 1.0
    capacity_mwh: float = 2.0
    round_trip_eff: float = 0.85
    cycles_per_day: float = 1.0


def optimal_schedule(
    prices: np.ndarray, battery: Battery = Battery()
) -> tuple[np.ndarray, np.ndarray]:
    """Charge/discharge energy (MWh) per hour maximizing revenue
    at the given prices. Returns (charge, discharge)."""
    t = len(prices)
    eff = battery.round_trip_eff
    # variables: [c_0..c_{T-1}, d_0..d_{T-1}]; linprog minimizes.
    # The 1e-6 activity penalty breaks degenerate ties (e.g. lossless
    # simultaneous charge+discharge) toward doing nothing.
    cost = np.concatenate([prices, -eff * prices]) + 1e-6

    lower_tri = np.tril(np.ones((t, t)))
    # SoC_t <= capacity  ->  cumsum(c) - cumsum(d) <= cap
    a_soc_hi = np.hstack([lower_tri, -lower_tri])
    # SoC_t >= 0         ->  cumsum(d) - cumsum(c) <= 0
    a_soc_lo = -a_soc_hi
    a_cycle = np.concatenate([np.ones(t), np.zeros(t)])[None, :]
    a_ub = np.vstack([a_soc_hi, a_soc_lo, a_cycle])
    b_ub = np.concatenate([
        np.full(t, battery.capacity_mwh),
        np.zeros(t),
        [battery.cycles_per_day * battery.capacity_mwh],
    ])

    res = linprog(cost, A_ub=a_ub, b_ub=b_ub,
                  bounds=(0.0, battery.power_mw), method="highs")
    if not res.success:  # pragma: no cover - HiGHS on a tiny feasible LP
        raise RuntimeError(f"battery LP failed: {res.message}")
    charge, discharge = res.x[:t], res.x[t:]
    # clip solver dust so accounting sums are exact
    charge[charge < 1e-9] = 0.0
    discharge[discharge < 1e-9] = 0.0
    return charge, discharge


def settle(
    charge: np.ndarray,
    discharge: np.ndarray,
    actual_prices: np.ndarray,
    battery: Battery = Battery(),
) -> float:
    """Revenue (EUR) of a schedule at the realized prices."""
    eff = battery.round_trip_eff
    return float(np.sum(actual_prices * (eff * discharge - charge)))


def daily_pnl(
    forecast_p50: pd.Series,
    actual: pd.Series,
    battery: Battery = Battery(),
    tz: str = LOCAL_TZ,
) -> pd.DataFrame:
    """Per-local-day P&L of trading the forecast, plus the
    perfect-foresight bound. Days with missing hours are skipped."""
    df = pd.DataFrame({"fcst": forecast_p50, "actual": actual}).dropna()
    days = pd.Index(df.index.tz_convert(tz).date)

    rows = []
    for day in sorted(set(days)):
        sub = df[days == day]
        # expected hour count for this local day (23/24/25 across DST)
        mid0 = pd.Timestamp(day).tz_localize(tz)
        mid1 = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(tz)
        expected = int((mid1 - mid0) / pd.Timedelta(hours=1))
        if len(sub) < expected:  # data gap or partial edge day
            continue
        c, d = optimal_schedule(sub["fcst"].to_numpy(), battery)
        pnl = settle(c, d, sub["actual"].to_numpy(), battery)
        c_pf, d_pf = optimal_schedule(sub["actual"].to_numpy(), battery)
        pnl_pf = settle(c_pf, d_pf, sub["actual"].to_numpy(), battery)
        rows.append({"day": day, "pnl_eur": pnl, "perfect_eur": pnl_pf,
                     "mwh_cycled": float(c.sum())})
    return pd.DataFrame(rows).set_index("day")


def summarize_pnl(daily: pd.DataFrame) -> dict[str, float]:
    """Headline numbers for one model's P&L series."""
    total = float(daily["pnl_eur"].sum())
    perfect = float(daily["perfect_eur"].sum())
    return {
        "eur_per_day": float(daily["pnl_eur"].mean()),
        "total_eur": total,
        "capture_rate": total / perfect if perfect > 0 else np.nan,
        "loss_days_pct": float((daily["pnl_eur"] < 0).mean() * 100),
        "n_days": int(len(daily)),
    }
