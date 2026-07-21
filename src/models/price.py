"""Price models: naive baselines and LEAR (LASSO, Ziel & Weron 2018).
Target is asinh-transformed because prices spike and go negative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base import register

PRICE_LAG_COLS = ["price_lag_1d", "price_lag_2d", "price_lag_3d", "price_lag_7d"]


class PriceNaiveYesterday:
    """P50 = same hour yesterday. Band = spread of the price lags."""

    name = "price_naive_yesterday"

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        pass  # nothing to learn

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        lags = x[PRICE_LAG_COLS]
        p50 = x["price_lag_1d"]
        return pd.DataFrame(
            {
                "p10": lags.quantile(0.1, axis=1).clip(upper=p50),
                "p50": p50,
                "p90": lags.quantile(0.9, axis=1).clip(lower=p50),
            },
            index=x.index,
        )


class PriceNaiveWeek:
    """P50 = same hour last week."""

    name = "price_naive_week"

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        pass

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        lags = x[PRICE_LAG_COLS]
        p50 = x["price_lag_7d"]
        return pd.DataFrame(
            {
                "p10": lags.quantile(0.1, axis=1).clip(upper=p50),
                "p50": p50,
                "p90": lags.quantile(0.9, axis=1).clip(lower=p50),
            },
            index=x.index,
        )


class PriceLEAR:
    """LEAR: one LASSO per delivery hour, 24 models in total.
    Each sees the full D-1 price vector. Target: z = asinh((p - med) / MAD).
    Band from per-hour residuals, mapped back with sinh.
    """

    name = "lear"

    # z-headroom above the training range before clipping
    Z_CLIP_MARGIN = 0.5

    def __init__(self) -> None:
        self._models: dict[int, Pipeline] = {}
        self._resid_q: dict[int, tuple[float, float]] = {}
        self._feature_cols: list[str] | None = None
        self._med: float = 0.0
        self._mad: float = 1.0
        self._z_clip: tuple[float, float] = (-float("inf"), float("inf"))

    @staticmethod
    def _make_pipe() -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("est", LassoCV(cv=5, alphas=50, max_iter=20000, random_state=0)),
            ]
        )

    def _to_z(self, p: np.ndarray) -> np.ndarray:
        return np.arcsinh((p - self._med) / self._mad)

    def _from_z(self, z: np.ndarray) -> np.ndarray:
        return self._med + self._mad * np.sinh(z)

    def _transform_x(self, x: pd.DataFrame) -> pd.DataFrame:
        out = x.copy()
        price_cols = [c for c in x.columns if c.startswith("price_")]
        out[price_cols] = self._to_z(out[price_cols].to_numpy())
        return out

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self._feature_cols = list(x.columns)
        self._med = float(np.median(y))
        # 1.4826 * MAD ~ sigma, guard against 0
        self._mad = max(1.4826 * float(np.median(np.abs(y - self._med))), 1e-6)
        xt = self._transform_x(x)
        z = pd.Series(self._to_z(y.to_numpy()), index=y.index)
        # extrapolation guard: clip predictions to training z-range + margin
        self._z_clip = (
            float(z.min()) - self.Z_CLIP_MARGIN,
            float(z.max()) + self.Z_CLIP_MARGIN,
        )
        self._models, self._resid_q = {}, {}
        for hour, x_h in xt.groupby(xt["hour_local"].astype(int)):
            z_h = z.reindex(x_h.index)
            pipe = self._make_pipe()
            pipe.fit(x_h.to_numpy(), z_h.to_numpy())
            resid = z_h.to_numpy() - pipe.predict(x_h.to_numpy())
            self._models[hour] = pipe
            self._resid_q[hour] = (
                float(np.quantile(resid, 0.1)),
                float(np.quantile(resid, 0.9)),
            )

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        assert self._models, "fit first"
        assert list(x.columns) == self._feature_cols, "feature columns changed"
        xt = self._transform_x(x)
        out = pd.DataFrame(index=x.index, columns=["p10", "p50", "p90"], dtype=float)
        for hour, x_h in xt.groupby(xt["hour_local"].astype(int)):
            pipe = self._models.get(hour)
            if pipe is None:  # unseen hour (DST edge): use nearest
                pipe = self._models[min(self._models, key=lambda k: abs(k - hour))]
                lo, hi = self._resid_q[min(self._resid_q, key=lambda k: abs(k - hour))]
            else:
                lo, hi = self._resid_q[hour]
            z = np.clip(pipe.predict(x_h.to_numpy()), *self._z_clip)
            out.loc[x_h.index, "p10"] = self._from_z(z + lo)
            out.loc[x_h.index, "p50"] = self._from_z(z)
            out.loc[x_h.index, "p90"] = self._from_z(z + hi)
        return out


register("price_naive_yesterday")(PriceNaiveYesterday)
register("price_naive_week")(PriceNaiveWeek)
register("lear")(PriceLEAR)
