"""Spike classifier: probability that an hour lands in the price top 5%.

Why a classifier: band calibration cannot fix spike misses (asymmetric
CQR and the GPD tail both failed the spike-coverage gate — DECISIONS
2026-07-17 and 2026-07-22). Spikes are conditional events; the honest
route is a dedicated probability, shown in the daily report next to
the band.

Leakage rule: the "top 5%" threshold is computed from the TRAINING
window at each refit — never from the test period. An evaluation-side
pooled threshold would look better and be a lie.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd

SPIKE_QUANTILE = 0.95

PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 40,
    "verbosity": -1,
}


class SpikeClassifier:
    """LightGBM binary classifier over the price feature matrix."""

    name = "spike_classifier"

    def __init__(self, quantile: float = SPIKE_QUANTILE, seed: int = 42):
        self.quantile = quantile
        self.seed = seed
        self._model: lgb.LGBMClassifier | None = None
        self.threshold_: float | None = None

    def fit(self, x: pd.DataFrame, y_price: pd.Series) -> None:
        """`y_price` is the raw price; the label is derived here so the
        threshold can only see the training window."""
        self.threshold_ = float(y_price.quantile(self.quantile))
        label = (y_price >= self.threshold_).astype(int)
        self._model = lgb.LGBMClassifier(random_state=self.seed, **PARAMS)
        self._model.fit(x, label)

    def predict_proba(self, x: pd.DataFrame) -> pd.Series:
        assert self._model is not None, "fit first"
        return pd.Series(
            self._model.predict_proba(x)[:, 1], index=x.index, name="p_spike"
        )
