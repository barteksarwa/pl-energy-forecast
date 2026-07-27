"""Faithful-LEAR variant: day-vectors for D-2/3/7, one-hot calendar, AIC.

Each test pins one closed deviation from Ziel & Weron 2018 (LEAR model
card, deviations section).
"""

import numpy as np
import pandas as pd

from src.features.price_lags import daily_price_vector
from src.models.price import PriceLEARFull

TZ = "Europe/Warsaw"


def _price_series() -> pd.Series:
    idx = pd.date_range("2026-01-01", "2026-03-01", freq="1h", tz="UTC")
    hours = idx.tz_convert(TZ).hour.to_numpy()
    dow = idx.tz_convert(TZ).dayofweek.to_numpy()
    rng = np.random.default_rng(3)
    vals = 90 + 25 * np.sin((hours - 8) / 24 * 2 * np.pi) - 10 * (dow >= 5)
    return pd.Series(vals + rng.normal(0, 4, len(idx)), index=idx)


def test_day_vector_days_back_reads_the_right_day() -> None:
    price = _price_series()
    tz_day = pd.Timestamp("2026-02-10", tz=TZ)
    hours = pd.date_range(tz_day, periods=24, freq="1h").tz_convert("UTC")
    out = daily_price_vector(price, hours, hours[0], days_back=2)
    assert list(out.columns) == [f"price_d2_h{h:02d}" for h in range(24)]
    # row-constant, and equal to the actual price of D-2 at that hour
    src = pd.Timestamp("2026-02-08 13:00", tz=TZ).tz_convert("UTC")
    assert out["price_d2_h13"].nunique() == 1
    assert out["price_d2_h13"].iloc[0] == price.loc[src]


def test_lear_full_one_hot_replaces_ordinal_calendar() -> None:
    price = _price_series()
    idx = price.index[: 40 * 24]
    x = pd.DataFrame(
        {
            "hour_local": idx.tz_convert(TZ).hour,
            "day_of_week": idx.tz_convert(TZ).dayofweek,
            "month": idx.tz_convert(TZ).month,
            "price_lag_1d": price.shift(24).reindex(idx),
        },
        index=idx,
    )
    model = PriceLEARFull()
    model._med, model._mad = 90.0, 20.0
    xt = model._transform_x(x)
    assert "day_of_week" not in xt.columns and "month" not in xt.columns
    assert {f"dow_{d}" for d in range(7)} <= set(xt.columns)
    assert {f"month_{m}" for m in range(1, 13)} <= set(xt.columns)
    # dummies are exclusive: exactly one dow active per row
    assert (xt[[f"dow_{d}" for d in range(7)]].sum(axis=1) == 1).all()


def test_lear_full_fits_and_orders_quantiles() -> None:
    price = _price_series()
    idx = price.index
    x = pd.DataFrame(
        {
            "hour_local": idx.tz_convert(TZ).hour,
            "day_of_week": idx.tz_convert(TZ).dayofweek,
            "month": idx.tz_convert(TZ).month,
            "price_lag_1d": price.shift(24),
            "price_lag_7d": price.shift(24 * 7),
        },
        index=idx,
    ).dropna()
    y = price.reindex(x.index)
    model = PriceLEARFull()
    model.fit(x.iloc[:-24], y.iloc[:-24])
    pred = model.predict(x.iloc[-24:])
    assert (pred["p10"] <= pred["p50"]).all()
    assert (pred["p50"] <= pred["p90"]).all()
    # sane magnitude: within the observed price range plus margin
    assert pred["p50"].between(price.min() - 50, price.max() + 50).all()
