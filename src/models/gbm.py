"""LightGBM quantile model. One booster per quantile, pinball loss."""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from src.models.base import QUANTILES, register

PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 40,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "random_state": 0,
    "verbosity": -1,
}


class LightGBMQuantile:
    name = "lgbm_quantile"

    def __init__(self) -> None:
        self._models: dict[float, lgb.LGBMRegressor] = {}
        self._columns: list[str] | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self._columns = list(x.columns)
        for q in QUANTILES:
            model = lgb.LGBMRegressor(objective="quantile", alpha=q, **PARAMS)
            model.fit(x, y)
            self._models[q] = model

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        assert self._models, "fit first"
        x = x[self._columns]
        out = pd.DataFrame(
            {
                "p10": self._models[0.1].predict(x),
                "p50": self._models[0.5].predict(x),
                "p90": self._models[0.9].predict(x),
            },
            index=x.index,
        )
        # quantiles can cross, force p10 <= p50 <= p90
        out["p10"] = out[["p10", "p50"]].min(axis=1)
        out["p90"] = out[["p90", "p50"]].max(axis=1)
        return out


register("lgbm_quantile")(LightGBMQuantile)
