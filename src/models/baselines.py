"""Baseline models. Every new model must beat these first."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base import register

WEEKLY_LAG_COLS = ["load_lag_168h", "load_lag_336h", "load_lag_504h", "load_lag_672h"]


class SeasonalNaive:
    """P50 = same hour last week. Band = spread of the last 4 weeks."""

    name = "seasonal_naive"

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        pass  # nothing to learn

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        weekly = x[WEEKLY_LAG_COLS]
        p50 = x["load_lag_168h"]
        # keep the band around p50
        return pd.DataFrame(
            {
                "p10": weekly.quantile(0.1, axis=1).clip(upper=p50),
                "p50": p50,
                "p90": weekly.quantile(0.9, axis=1).clip(lower=p50),
            },
            index=x.index,
        )


class Climatology:
    """Long-run average by hour and weekday/weekend."""

    name = "climatology"

    def __init__(self) -> None:
        self._table: pd.DataFrame | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        df = pd.DataFrame(
            {"y": y, "hour": x["hour_local"], "weekend": x["is_weekend"]}
        )
        self._table = df.groupby(["hour", "weekend"])["y"].quantile([0.1, 0.5, 0.9]).unstack()

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        assert self._table is not None, "fit first"
        keys = pd.MultiIndex.from_arrays([x["hour_local"], x["is_weekend"]])
        rows = self._table.reindex(keys)
        rows.index = x.index
        return rows.rename(columns={0.1: "p10", 0.5: "p50", 0.9: "p90"})


class _ResidualBandModel:
    """Point model + residual quantile band."""

    name = "residual_band_base"

    def __init__(self, estimator) -> None:
        self._pipe = Pipeline([("scale", StandardScaler()), ("est", estimator)])
        self._resid_q: tuple[float, float] | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self._pipe.fit(x.to_numpy(), y.to_numpy())
        resid = y.to_numpy() - self._pipe.predict(x.to_numpy())
        self._resid_q = (float(np.quantile(resid, 0.1)), float(np.quantile(resid, 0.9)))

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        assert self._resid_q is not None, "fit first"
        point = self._pipe.predict(x.to_numpy())
        lo, hi = self._resid_q
        return pd.DataFrame(
            {"p10": point + lo, "p50": point, "p90": point + hi}, index=x.index
        )


class RidgeForecaster(_ResidualBandModel):
    """Linear baseline."""

    name = "ridge"

    def __init__(self) -> None:
        super().__init__(Ridge(alpha=1.0))


class LassoAR(_ResidualBandModel):
    """LEAR-style LASSO. Picks its own features. Alpha from 5-fold CV."""

    name = "lasso_ar"

    def __init__(self) -> None:
        super().__init__(LassoCV(cv=5, alphas=50, max_iter=5000, random_state=0))


register("seasonal_naive")(SeasonalNaive)
register("climatology")(Climatology)
register("ridge")(RidgeForecaster)
register("lasso_ar")(LassoAR)
