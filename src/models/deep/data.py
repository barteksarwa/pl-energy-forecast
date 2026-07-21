"""Sample builder for sequence models. One sample = one forecast day.
Encoder: the past up to the 09:00 D-1 cutoff. Decoder: known-future covariates.
Load is instance-normalized, the lag-168 anchor restores the level.
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
    tso: pd.Series | None = None,
    origin_offsets_h: tuple[int, ...] = (0,),
) -> DaySamples:
    """origin_offsets_h: extra training origins per day. Evaluation uses (0,)."""
    enc_l, fut_l, y_l, anchor_l, mean_l, std_l, kept = [], [], [], [], [], [], []

    for day in days:
      for off in origin_offsets_h:
        day_ts = pd.Timestamp(day, tz=tz)
        hours = local_day_hours_utc(day_ts, tz)
        if len(hours) != TARGET_HOURS:
            continue  # skip DST days in training; production handles them via fallback
        cutoff = shift_local_day(day_ts, -1, tz) + pd.Timedelta(hours=9 + off)
        cutoff_utc = cutoff.tz_convert("UTC").floor("1h")

        enc_idx = pd.date_range(
            end=cutoff_utc - pd.Timedelta(hours=1), periods=ENCODER_HOURS, freq="1h"
        )
        enc_load = load.reindex(enc_idx)
        enc_wx = weather.reindex(enc_idx)
        target = load.reindex(hours)
        anchor = load.reindex(hours - pd.Timedelta(hours=168))
        fut_wx = weather.reindex(hours)
        tso_fut = tso.reindex(hours) if tso is not None else None
        if (
            enc_load.isna().any() or target.isna().any()
            or anchor.isna().any() or enc_wx.isna().any().any()
            or fut_wx.isna().any().any()
            or (tso_fut is not None and tso_fut.isna().any())
        ):
            continue

        mu, sd = float(enc_load.mean()), float(enc_load.std()) or 1.0
        cal = calendar_features(hours)[FUTURE_CAL_COLS]

        enc = np.column_stack([
            (enc_load.to_numpy() - mu) / sd,
            enc_wx.to_numpy(),
        ])
        fut_cols = [
            cal.to_numpy(dtype=float),
            fut_wx.to_numpy(),
            ((anchor.to_numpy() - mu) / sd)[:, None],
        ]
        if tso_fut is not None:
            fut_cols.append(((tso_fut.to_numpy() - mu) / sd)[:, None])
        fut = np.column_stack(fut_cols)
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


def standardize_covariates(
    train: DaySamples, *others: DaySamples, n_tail: int = 1
) -> dict:
    """Z-score covariates with TRAIN stats only (no leakage).
    Instance-normalized columns are left untouched.
    Near-constant training columns are zeroed everywhere (zero-variance guard).
    Prediction-time samples must go through apply_covariate_stats too.
    """
    e_mu = train.enc[:, :, 1:].mean(dim=(0, 1), keepdim=True)
    e_sd_raw = train.enc[:, :, 1:].std(dim=(0, 1), keepdim=True)
    e_sd = e_sd_raw.clamp_min(1e-6)
    f_mu = train.fut[:, :, :-n_tail].mean(dim=(0, 1), keepdim=True)
    f_sd_raw = train.fut[:, :, :-n_tail].std(dim=(0, 1), keepdim=True)
    f_sd = f_sd_raw.clamp_min(1e-6)
    f_zero_mask = f_sd_raw < 1e-4   # constant in training → zero out everywhere
    stats = {"e_mu": e_mu, "e_sd": e_sd, "f_mu": f_mu, "f_sd": f_sd,
             "f_zero_mask": f_zero_mask, "n_tail": n_tail}
    for s in (train, *others):
        apply_covariate_stats(s, stats)
    return stats


def apply_covariate_stats(s: DaySamples, stats: dict) -> None:
    """Apply stored train-window covariate stats to a sample (in place)."""
    n_tail = stats["n_tail"]
    if s.enc.shape[-1] > 1:
        s.enc[:, :, 1:] = (s.enc[:, :, 1:] - stats["e_mu"]) / stats["e_sd"]
    s.fut[:, :, :-n_tail] = (s.fut[:, :, :-n_tail] - stats["f_mu"]) / stats["f_sd"]
    if stats.get("f_zero_mask") is not None and stats["f_zero_mask"].any():
        s.fut[:, :, :-n_tail] = s.fut[:, :, :-n_tail].masked_fill(
            stats["f_zero_mask"], 0.0
        )
