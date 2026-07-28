"""Sample builder for sequence models on the PRICE target.

Differences from the load builder (data.py):
- Cutoff: the first delivery hour of day D. All of D-1's prices are
  known at bid time (fixed at the D-2 auction), so the encoder window
  ends at D-1 23:00 — one day closer than the load cutoff allows.
- Encoder length is a PARAMETER, not a constant. The whole point of the
  attention model is that it can look far further back than the 14 days
  the tabular lags encode. We sweep it.
- Known-future covariates: calendar + wind/solar forecast + TSO load
  forecast + the lag-168 price anchor.

Instance normalization: price channels scaled per sample by the encoder
window's mean/std, same trick as load. RES and TSO are z-scored with
TRAIN statistics via standardize_covariates (n_tail=1: only the anchor
is instance-normalized).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.features.calendar import calendar_features
from src.models.deep.data import FUTURE_CAL_COLS, DaySamples
from src.pipeline.daily_run import local_day_hours_utc

# Canonical fut-tensor column order, matching the np.column_stack below.
# Importance/VSN scripts must label by THIS list — two of them once kept
# their own diverging copies (review finding, 2026-07-27).
PRICE_FUT_COLS = FUTURE_CAL_COLS + [
    "solar_fcst_mw", "wind_on_fcst_mw", "wind_off_fcst_mw",
    "tso_forecast_mw", "price_anchor_lag168",
]

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
