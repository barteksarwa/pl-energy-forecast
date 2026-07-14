"""Sample builder for sequence models. One sample = one forecast day.

Encoder sees the past (load + weather) up to the 09:00 D-1 cutoff.
Decoder sees known-future covariates (calendar + weather forecast) for the
24 target hours of day D. Target = the 24 loads of day D.

Instance normalization: load channels are scaled per sample by the encoder
window's mean/std. Window-normalized nets lose the absolute level; the
seasonal-naive anchor (lag-168 load per target hour) restores it as a
decoder covariate — the "raw recency anchor" trick.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.features.calendar import calendar_features
from src.pipeline.daily_run import local_day_hours_utc, shift_local_day

ENCODER_HOURS = 336  # 14 days
TARGET_HOURS = 24    # normal day; DST days are skipped in training samples

FUTURE_CAL_COLS = [
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "is_weekend", "is_holiday", "is_bridge_day",
]


@dataclass
class DaySamples:
    """Tensors for a set of forecast days."""

    enc: torch.Tensor      # (n, ENCODER_HOURS, enc_features)
    fut: torch.Tensor      # (n, 24, fut_features)
    y: torch.Tensor        # (n, 24) normalized load
    anchor: torch.Tensor   # (n, 24) lag-168 load, normalized
    mean: torch.Tensor     # (n,) per-sample denorm
    std: torch.Tensor      # (n,)
    days: list             # local dates, for traceability


def build_samples(
    load: pd.Series,
    weather: pd.DataFrame,
    days: list,
    tz: str = "Europe/Warsaw",
) -> DaySamples:
    enc_l, fut_l, y_l, anchor_l, mean_l, std_l, kept = [], [], [], [], [], [], []
    wx_cols = list(weather.columns)

    for day in days:
        day_ts = pd.Timestamp(day, tz=tz)
        hours = local_day_hours_utc(day_ts, tz)
        if len(hours) != TARGET_HOURS:
            continue  # skip DST days in training; production handles them via fallback
        cutoff = shift_local_day(day_ts, -1, tz) + pd.Timedelta(hours=9)
        cutoff_utc = cutoff.tz_convert("UTC").floor("1h")

        enc_idx = pd.date_range(
            end=cutoff_utc - pd.Timedelta(hours=1), periods=ENCODER_HOURS, freq="1h"
        )
        enc_load = load.reindex(enc_idx)
        enc_wx = weather.reindex(enc_idx)
        target = load.reindex(hours)
        anchor = load.reindex(hours - pd.Timedelta(hours=168))
        fut_wx = weather.reindex(hours)
        if (
            enc_load.isna().any() or target.isna().any()
            or anchor.isna().any() or enc_wx.isna().any().any()
            or fut_wx.isna().any().any()
        ):
            continue

        mu, sd = float(enc_load.mean()), float(enc_load.std()) or 1.0
        cal = calendar_features(hours)[FUTURE_CAL_COLS]

        enc = np.column_stack([
            (enc_load.to_numpy() - mu) / sd,
            enc_wx.to_numpy(),
        ])
        fut = np.column_stack([
            cal.to_numpy(dtype=float),
            fut_wx.to_numpy(),
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


def standardize_covariates(train: DaySamples, *others: DaySamples) -> None:
    """Z-score non-load covariates using TRAIN statistics only (no leakage).

    Column 0 of enc is the instance-normalized load — left untouched.
    The last fut column is the normalized anchor — left untouched.
    """
    e_mu = train.enc[:, :, 1:].mean(dim=(0, 1), keepdim=True)
    e_sd = train.enc[:, :, 1:].std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    f_mu = train.fut[:, :, :-1].mean(dim=(0, 1), keepdim=True)
    f_sd = train.fut[:, :, :-1].std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    for s in (train, *others):
        s.enc[:, :, 1:] = (s.enc[:, :, 1:] - e_mu) / e_sd
        s.fut[:, :, :-1] = (s.fut[:, :, :-1] - f_mu) / f_sd
