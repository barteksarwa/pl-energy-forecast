"""Driver attribution: plain-words mapping and exact linear contributions."""

import numpy as np
import pandas as pd

from src.interpretability.linear_drivers import linear_drivers
from src.interpretability.plain_words import phrase_for, top_phrases
from src.models.baselines import RidgeForecaster


def test_phrase_for_prefix_family() -> None:
    assert phrase_for("price_d1_h07") == "yesterday's hourly price curve"
    assert phrase_for("tso_forecast_mw") == "the TSO's own day-ahead forecast"
    assert phrase_for("unknown_feature") == "unknown_feature"


def test_top_phrases_pools_prefix_family() -> None:
    cols = ["price_d1_h00", "price_d1_h01", "solar_fcst_mw"]
    # two family members at 0.4 each must outrank one single at 0.6
    out = top_phrases(np.array([0.4, 0.4, 0.6]), cols, n=2)
    assert out[0] == "yesterday's hourly price curve"
    assert out[1] == "tomorrow's solar generation forecast"


def test_linear_drivers_finds_the_dominant_feature() -> None:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=500, freq="1h", tz="UTC")
    x = pd.DataFrame(
        {
            "tso_forecast_mw": rng.normal(20000, 2000, len(idx)),
            "temperature_2m": rng.normal(5, 8, len(idx)),
            "cloud_cover": rng.uniform(0, 100, len(idx)),
        },
        index=idx,
    )
    y = pd.Series(
        x["tso_forecast_mw"] * 1.0 + rng.normal(0, 50, len(idx)), index=idx
    )
    model = RidgeForecaster()
    model.fit(x, y)
    drivers = linear_drivers(model, x.tail(24))
    assert drivers[0] == "the TSO's own day-ahead forecast"
