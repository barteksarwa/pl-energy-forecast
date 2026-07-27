"""Walk-forward backtest. The only honest way to score a forecaster.

Day by day: train on a trailing window that ends before the target day,
predict the target day, move on. Refit every few days to keep runtime sane —
between refits the last fitted model keeps predicting (like a real desk).

The feature matrix x must already be cutoff-safe (built via
src/features/matrix.py). This engine additionally guarantees the training
target never includes the target day or anything after it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Callable

import pandas as pd

from src.evaluation.metrics import mae, mape, pinball_loss, rmse

LOCAL_TZ = "Europe/Warsaw"


@dataclass(frozen=True)
class BacktestResult:
    model_name: str
    predictions: pd.DataFrame  # index: UTC hours, columns p10/p50/p90


def _local_dates(index: pd.DatetimeIndex, tz: str) -> pd.Index:
    return pd.Index(index.tz_convert(tz).date, name="local_date")


def walk_forward_backtest(
    model_factory: Callable[[], object],
    x: pd.DataFrame,
    y: pd.Series,
    test_start: pd.Timestamp,
    train_window_days: int = 365,
    refit_every_days: int = 7,
    tz: str = LOCAL_TZ,
    target_availability: str = "realtime",
) -> BacktestResult:
    """Predict every local day of `x` from `test_start` onward, walking forward.

    target_availability sets how much target history training may use at
    the 09:00 D-1 decision moment:

    - "realtime": the target is observed live (load actuals). Hours from
      09:00 D-1 onward do not exist yet, so training stops strictly
      before that moment (validation E2).
    - "day_ahead": the target is published one day ahead (DA auction
      prices clear on D-2 for delivery day D-1). The full D-1 curve is
      already public at 09:00 D-1, so training uses every day before the
      target day.
    """
    if target_availability not in ("realtime", "day_ahead"):
        raise ValueError(f"unknown target_availability: {target_availability!r}")
    if x.index.tz is None:
        raise ValueError("x must have a tz-aware index")
    dates = _local_dates(x.index, tz)
    test_days = sorted(set(dates[x.index >= test_start]))

    model = None
    last_fit_day: pd.Timestamp | None = None
    preds: list[pd.DataFrame] = []
    # Coverage accounting: .dropna() can silently delete whole stretches
    # of training data when one optional feature has ragged history, and
    # a skipped day is invisible in the output. Both get reported.
    skipped_days: list = []
    max_train_rows_dropped = 0

    for day in test_days:
        window_mask = dates >= day - pd.Timedelta(days=train_window_days)
        if target_availability == "day_ahead":
            train_mask = (dates < day) & window_mask
        else:
            # The decision moment is 09:00 local on D-1. `dates < day` alone
            # would let the training target include D-1 hours 09:00-23:00,
            # which do not exist yet for a live-observed target (validation
            # finding E2, 2026-07-27). 09:00 wall clock is safe to localize:
            # DST switches at 02:00-03:00.
            decision = pd.Timestamp(
                f"{day - timedelta(days=1)} 09:00").tz_localize(tz)
            train_mask = (x.index < decision.tz_convert("UTC")) & window_mask
        needs_refit = (
            model is None
            or (pd.Timestamp(day) - pd.Timestamp(last_fit_day)).days >= refit_every_days
        )
        if needs_refit:
            x_tr = x[train_mask].dropna()
            y_tr = y.reindex(x_tr.index).dropna()
            x_tr = x_tr.reindex(y_tr.index)
            max_train_rows_dropped = max(
                max_train_rows_dropped, int(train_mask.sum()) - len(x_tr)
            )
            if len(x_tr) < 24 * 30:
                skipped_days.append(day)
                continue  # not enough history yet; skip day, keep walking
            model = model_factory()
            model.fit(x_tr, y_tr)
            last_fit_day = day

        day_mask = dates == day
        x_day = x[day_mask].dropna()
        if x_day.empty:
            skipped_days.append(day)
            continue
        preds.append(model.predict(x_day))

    if not preds:
        raise ValueError("Backtest produced no predictions — not enough data?")
    if skipped_days:
        print(f"backtest: {len(skipped_days)} of {len(test_days)} test days "
              f"skipped (first {skipped_days[0]}, last {skipped_days[-1]}) — "
              "thin history or all-NaN features")
    if max_train_rows_dropped:
        print(f"backtest: up to {max_train_rows_dropped} training rows per "
              "refit dropped by NaN filtering — check optional-feature "
              "history if this is large")
    out = pd.concat(preds).sort_index()
    return BacktestResult(model_name=model_factory().name, predictions=out)


def summarize(results: list[BacktestResult], y: pd.Series) -> pd.DataFrame:
    """One row per model: point metrics on P50, pinball on each quantile."""
    rows = []
    for r in results:
        p = r.predictions
        rows.append(
            {
                "model": r.model_name,
                "mae": mae(y, p["p50"]),
                "rmse": rmse(y, p["p50"]),
                "mape_pct": mape(y, p["p50"]),
                "pinball_p10": pinball_loss(y, p["p10"], 0.1),
                "pinball_p50": pinball_loss(y, p["p50"], 0.5),
                "pinball_p90": pinball_loss(y, p["p90"], 0.9),
                "n_hours": int(p["p50"].notna().sum()),
            }
        )
    table = pd.DataFrame(rows).set_index("model")
    naive_mae = table.loc["seasonal_naive", "mae"] if "seasonal_naive" in table.index else None
    if naive_mae:
        table["skill_vs_naive"] = 1.0 - table["mae"] / naive_mae
    return table.sort_values("mae")
