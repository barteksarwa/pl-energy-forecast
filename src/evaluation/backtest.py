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
    train_cutoff: str = "decision_0900",
) -> BacktestResult:
    """Predict every local day of `x` from `test_start` onward, walking forward.

    `train_cutoff` encodes WHEN the training target becomes observable:
    - "decision_0900" (default, physical actuals like LOAD): the target
      for D-1 hours 09:00-23:00 does not exist at the 09:00 D-1 decision
      moment, so training stops at 09:00 D-1 (validation finding E2).
    - "target_published" (day-ahead PRICES): D-1's full 24h price vector
      was fixed at the D-2 auction (~13:00) and is public long before
      09:00 D-1 — training may use everything through the end of D-1.
      Cutting it at 09:00 costs real skill AND breaks context-based
      zero-shot models (validation follow-up, 2026-07-27).
    """
    if x.index.tz is None:
        raise ValueError("x must have a tz-aware index")
    if train_cutoff not in ("decision_0900", "target_published"):
        raise ValueError(f"unknown train_cutoff: {train_cutoff}")
    dates = _local_dates(x.index, tz)
    test_days = sorted(set(dates[x.index >= test_start]))

    model = None
    last_fit_day: pd.Timestamp | None = None
    preds: list[pd.DataFrame] = []

    for day in test_days:
        if train_cutoff == "decision_0900":
            # 09:00 wall clock is safe to localize: DST switches at 02-03.
            decision = pd.Timestamp(
                f"{day - timedelta(days=1)} 09:00").tz_localize(tz)
            upper = x.index < decision.tz_convert("UTC")
        else:  # target_published: everything through the end of D-1
            upper = dates < day
        train_mask = (
            upper & (dates >= day - pd.Timedelta(days=train_window_days))
        )
        needs_refit = (
            model is None
            or (pd.Timestamp(day) - pd.Timestamp(last_fit_day)).days >= refit_every_days
        )
        if needs_refit:
            x_tr = x[train_mask].dropna()
            y_tr = y.reindex(x_tr.index).dropna()
            x_tr = x_tr.reindex(y_tr.index)
            if len(x_tr) < 24 * 30:
                continue  # not enough history yet; skip day, keep walking
            model = model_factory()
            model.fit(x_tr, y_tr)
            last_fit_day = day

        day_mask = dates == day
        x_day = x[day_mask].dropna()
        if x_day.empty:
            continue
        preds.append(model.predict(x_day))

    if not preds:
        raise ValueError("Backtest produced no predictions — not enough data?")
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
