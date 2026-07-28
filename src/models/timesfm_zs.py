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
from src.models.fm_common import HistoryContext, forecast_span

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
                # 64 not 24: the forecast may start before midnight when
                # the stored context ends early (see forecast_span), and
                # DST gives 25-hour local days.
                max_context=CONTEXT_HOURS, max_horizon=64,
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
        n_ahead, fc_idx = forecast_span(context.index[-1], idx)
        _, quantiles = model.forecast(
            horizon=n_ahead, inputs=[context.to_numpy(dtype=np.float32)])
        # quantiles: (1, horizon, 10) = mean + deciles q10..q90
        q = quantiles[0][:n_ahead]
        fc = pd.DataFrame(
            q[:, [1, 5, 9]], index=fc_idx, columns=["p10", "p50", "p90"])
        out[["p10", "p50", "p90"]] = fc.reindex(idx).to_numpy()
        arr = np.sort(out[["p10", "p50", "p90"]].to_numpy(), axis=1)
        out[["p10", "p50", "p90"]] = arr
        return out


register("timesfm_zs")(TimesFMZS)
