"""TimesFM 2.5 zero-shot wrapper (Google, 200M, torch).

Univariate like Chronos — sees only past prices. Same fairness note
applies: the champion also sees RES/TSO/calendar. Quantile head gives
deciles; we take q10/q50/q90 directly.

Install: uv sync --extra fm. Run with --refit-days 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import register
from src.models.fm_common import HistoryContext

CONTEXT_HOURS = 2048
MODEL_ID = "google/timesfm-2.5-200m-pytorch"


class TimesFMZS:
    name = "timesfm_zs"

    def __init__(self) -> None:
        self._model = None
        self._ctx = HistoryContext(CONTEXT_HOURS)

    def _load(self):
        if self._model is None:
            import timesfm

            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID)
            model.compile(timesfm.ForecastConfig(
                max_context=CONTEXT_HOURS, max_horizon=24,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                fix_quantile_crossing=True,
            ))
            self._model = model
        return self._model

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self._ctx.fit(y)

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        idx = x.index.sort_values()
        out = pd.DataFrame(index=idx, columns=["p10", "p50", "p90"],
                           dtype=float)
        context = self._ctx.context_before(idx[0])
        if len(context) < 168:
            return out
        model = self._load()
        _, quantiles = model.forecast(
            horizon=len(idx), inputs=[context.to_numpy(dtype=np.float32)])
        # quantiles: (1, horizon, 10) = mean + deciles q10..q90
        q = quantiles[0]
        out["p10"], out["p50"], out["p90"] = q[:, 1], q[:, 5], q[:, 9]
        arr = np.sort(out[["p10", "p50", "p90"]].to_numpy(), axis=1)
        out[["p10", "p50", "p90"]] = arr
        return out


register("timesfm_zs")(TimesFMZS)
