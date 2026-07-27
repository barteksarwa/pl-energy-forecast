"""Exact driver attribution for the linear challenger.

A _ResidualBandModel prediction is intercept + sum(coef_i * z_i) on
standardized features, so |coef_i * z_i| IS feature i's contribution —
no approximation needed. Ranked over the forecast hours and translated
to plain words for the daily report (hard rule 3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.interpretability.plain_words import top_phrases


def linear_drivers(model: object, x: pd.DataFrame, n: int = 3) -> list[str]:
    """Top-n plain-word drivers of `model`'s forecast over the rows of x.

    `model` is a fitted _ResidualBandModel (scale + linear estimator
    pipeline). Raises AttributeError for anything else — callers treat
    that as "no drivers available", never as a report failure.
    """
    pipe = model._pipe
    z = pipe.named_steps["scale"].transform(x.to_numpy())
    coef = np.asarray(pipe.named_steps["est"].coef_).ravel()
    contribution = np.abs(z * coef).mean(axis=0)
    return top_phrases(contribution, list(x.columns), n)


def lear_drivers(model: object, x: pd.DataFrame, n: int = 3) -> list[str]:
    """Top-n plain-word drivers of a PriceLEAR forecast over the rows of x.

    LEAR is 24 per-hour LASSO pipelines on asinh-standardized features;
    each hour's contribution is |coef * scaled feature| for that hour's
    model, averaged across the forecast hours. Exact per hour model.
    """
    xt = model._transform_x(x[model._feature_cols])
    total = np.zeros(len(model._feature_cols))
    rows = 0
    for hour, x_h in xt.groupby(xt["hour_local"].astype(int)):
        pipe = model._models.get(hour)
        if pipe is None:  # DST edge — same fallback as predict()
            pipe = model._models[min(model._models,
                                     key=lambda k: abs(k - hour))]
        z = pipe.named_steps["scale"].transform(x_h.to_numpy())
        coef = np.asarray(pipe.named_steps["est"].coef_).ravel()
        total += np.abs(z * coef).sum(axis=0)
        rows += len(x_h)
    return top_phrases(total / max(rows, 1), model._feature_cols, n)
