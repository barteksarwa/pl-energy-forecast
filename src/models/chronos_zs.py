"""Chronos-Bolt zero-shot wrapper — a foundation model in the standard
backtest.

Chronos-Bolt (Amazon) is a time-series foundation model: pretrained on
billions of series, applied here with NO training on our data. `fit`
only stores price history; `predict` runs one forward pass per day.

Fairness note, stated wherever this model is quoted: Chronos is
univariate. It sees only past prices. The champion also sees RES
forecasts, the TSO load forecast, and calendar features. Chronos is
expected to lose — the measured gap is the deliverable: what does a
covariate-blind foundation model buy on PL prices?

Install: uv sync --extra fm. Registered as "chronos_bolt_zs"; run with
--refit-days 1 so the stored context is always fresh (refits cost
nothing here).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import register

CONTEXT_HOURS = 2048  # Bolt's maximum context
MODEL_ID = "amazon/chronos-bolt-base"


class ChronosBoltZS:
    name = "chronos_bolt_zs"

    def __init__(self) -> None:
        self._pipeline = None
        self._history: pd.Series | None = None

    def _load(self):
        if self._pipeline is None:
            from chronos import BaseChronosPipeline

            from src.models.deep.train import device
            dev = str(device())
            self._pipeline = BaseChronosPipeline.from_pretrained(
                MODEL_ID, device_map=dev)
        return self._pipeline

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        """Ignores X entirely (univariate). Stores the trailing history."""
        self._history = y.dropna().sort_index().tail(CONTEXT_HOURS * 2)

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        import torch

        assert self._history is not None, "fit first"
        pipe = self._load()
        idx = x.index.sort_values()
        out = pd.DataFrame(index=idx, columns=["p10", "p50", "p90"],
                           dtype=float)
        # one call per contiguous local day (the engine passes one day)
        first = idx[0]
        context = self._history[self._history.index < first].tail(CONTEXT_HOURS)
        if len(context) < 168:
            return out  # not enough history — NaN row, engine skips
        quantiles, _ = pipe.predict_quantiles(
            torch.tensor(context.to_numpy(), dtype=torch.float32),
            prediction_length=len(idx),
            quantile_levels=[0.1, 0.5, 0.9],
        )
        q = quantiles[0].cpu().numpy()  # (len(idx), 3)
        out["p10"], out["p50"], out["p90"] = q[:, 0], q[:, 1], q[:, 2]
        # enforce quantile ordering, same trick as gbm.py
        arr = np.sort(out[["p10", "p50", "p90"]].to_numpy(), axis=1)
        out[["p10", "p50", "p90"]] = arr
        return out


register("chronos_bolt_zs")(ChronosBoltZS)
