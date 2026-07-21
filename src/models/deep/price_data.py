"""Sample builder for the price target.
Cutoff = first delivery hour of day D (all D-1 prices known at bid time).
Encoder length is a parameter. Covariates: calendar + RES + TSO + anchor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.features.calendar import calendar_features
from src.models.deep.data import FUTURE_CAL_COLS, DaySamples
from src.pipeline.daily_run import local_day_hours_utc

TARGET_HOURS = 24


def build_price_samples(
    price: pd.Series,
    res: pd.DataFrame,
    tso: pd.Series,
    days: list,
    encoder_hours: int,
    tz: str = "Europe/Warsaw",
) -> DaySamples:
    enc_l, fut_l, y_l, anchor_l, mean_l, std_l, kept = [], [], [], [], [], [], []

    for day in days:
        day_ts = pd.Timestamp(day, tz=tz)
        hours = local_day_hours_utc(day_ts, tz)
        if len(hours) != TARGET_HOURS:
            continue  # skip DST days in training, same policy as load
        cutoff_utc = hours[0]

        enc_idx = pd.date_range(
            end=cutoff_utc - pd.Timedelta(hours=1), periods=encoder_hours, freq="1h"
        )
        enc_price = price.reindex(enc_idx)
        target = price.reindex(hours)
        anchor = price.reindex(hours - pd.Timedelta(hours=168))
        fut_res = res.reindex(hours)
        fut_tso = tso.reindex(hours)
        if (
            enc_price.isna().any() or target.isna().any() or anchor.isna().any()
            or fut_res.isna().any().any() or fut_tso.isna().any()
        ):
            continue

        mu, sd = float(enc_price.mean()), float(enc_price.std()) or 1.0
        cal = calendar_features(hours)[FUTURE_CAL_COLS]

        enc = ((enc_price.to_numpy() - mu) / sd)[:, None]
        fut = np.column_stack([
            cal.to_numpy(dtype=float),
            fut_res.to_numpy(),
            fut_tso.to_numpy()[:, None],
            ((anchor.to_numpy() - mu) / sd)[:, None],
        ])
        enc_l.append(enc)
        fut_l.append(fut)
        y_l.append((target.to_numpy() - mu) / sd)
        anchor_l.append((anchor.to_numpy() - mu) / sd)
        mean_l.append(mu)
        std_l.append(sd)
        kept.append(day)

    def t(x, dtype=torch.float32):
        return torch.tensor(np.array(x), dtype=dtype)

    return DaySamples(
        enc=t(enc_l), fut=t(fut_l), y=t(y_l), anchor=t(anchor_l),
        mean=t(mean_l), std=t(std_l), days=kept,
    )
