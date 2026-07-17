"""Rolling conformal calibration for quantile bands.

The problem it fixes: both price models produce good P50s but bands that
cover 51% (LightGBM) / 72% (LEAR) instead of the nominal 80%. A band
nobody can trust is a band nobody can size a position on.

Method: split-conformal on quantile regression (CQR, Romano, Patterson
& Candès 2019), applied on a rolling window so it stays walk-forward
honest. For each target day D:

1. Take the trailing `window_days` of PAST out-of-sample forecasts and
   their realized values (errors the desk had already seen by D-1).
2. Conformity score per past hour: E = max(p10 - y, y - p90).
   Positive = the actual fell outside the band by that much.
3. Q = the `coverage` empirical quantile of those scores (with the
   standard finite-sample +1 correction).
4. Widen day D's band: p10 - Q, p90 + Q. (Q can be negative — a band
   that over-covers gets tightened.)

No future information: day D's correction uses only errors observable
before D. The first `min_days` of the series carry no correction (not
enough history) and are flagged by the caller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOCAL_TZ = "Europe/Warsaw"


def conformity_scores(preds: pd.DataFrame, y: pd.Series) -> pd.Series:
    """CQR score per hour: how far outside [p10, p90] the actual fell.

    Negative = inside the band (distance to the nearest edge).
    """
    y = y.reindex(preds.index)
    return pd.concat([preds["p10"] - y, y - preds["p90"]], axis=1).max(axis=1)


def rolling_conformal(
    preds: pd.DataFrame,
    y: pd.Series,
    window_days: int = 90,
    min_days: int = 30,
    coverage: float = 0.8,
    tz: str = LOCAL_TZ,
) -> pd.DataFrame:
    """Return a copy of `preds` with the band conformally adjusted.

    Day-by-day: each local day's correction comes from the trailing
    `window_days` of past scores only. Days with fewer than `min_days`
    of history keep the raw band.
    """
    scores = conformity_scores(preds, y).dropna()
    out = preds.copy()
    days = pd.Index(preds.index.tz_convert(tz).date)
    score_days = pd.Index(scores.index.tz_convert(tz).date)

    for day in sorted(set(days)):
        past = scores[
            (score_days < day)
            & (score_days >= day - pd.Timedelta(days=window_days))
        ]
        if len(past) < min_days * 24:
            continue
        # finite-sample corrected quantile level, capped at 1
        level = min(coverage * (len(past) + 1) / len(past), 1.0)
        q = float(np.quantile(past.to_numpy(), level))
        mask = days == day
        out.loc[mask, "p10"] = preds.loc[mask, "p10"] - q
        out.loc[mask, "p90"] = preds.loc[mask, "p90"] + q
    # a widened band must still contain the P50
    out["p10"] = out[["p10", "p50"]].min(axis=1)
    out["p90"] = out[["p90", "p50"]].max(axis=1)
    return out


def latest_offset(
    preds: pd.DataFrame,
    y: pd.Series,
    window_days: int = 90,
    coverage: float = 0.8,
) -> float:
    """The correction the NEXT day would receive. For the daily loop:
    computed from the trailing window of stored out-of-sample errors,
    applied to tomorrow's fresh band."""
    scores = conformity_scores(preds, y).dropna()
    cutoff = scores.index.max() - pd.Timedelta(days=window_days)
    recent = scores[scores.index >= cutoff]
    level = min(coverage * (len(recent) + 1) / len(recent), 1.0)
    return float(np.quantile(recent.to_numpy(), level))


def rolling_conformal_asymmetric(
    preds: pd.DataFrame,
    y: pd.Series,
    window_days: int = 90,
    min_days: int = 30,
    coverage: float = 0.8,
    tz: str = LOCAL_TZ,
) -> pd.DataFrame:
    """Asymmetric CQR: separate corrections for lower and upper tails.

    The existing `rolling_conformal` adds the same offset Q to both tails.
    This function computes two independent offsets:
      Q_lo = quantile of (p10 - y)  — how much P10 overshoots below
      Q_hi = quantile of (y - p90)  — how much P90 undershoots above

    Use case: negative-price hours cause lower-tail miscalibration that
    symmetric CQR cannot fix without also over-widening the upper tail.

    Coverage guarantee: individual tail coverage >= alpha/2 by construction.
    Total coverage >= 1 - alpha = 0.8 (union bound is tighter than alpha).
    """
    y = y.reindex(preds.index)
    lo_scores = (preds["p10"] - y).dropna()  # positive = p10 too high
    hi_scores = (y - preds["p90"]).dropna()  # positive = p90 too low

    out = preds.copy()
    days = pd.Index(preds.index.tz_convert(tz).date)
    lo_days = pd.Index(lo_scores.index.tz_convert(tz).date)
    hi_days = pd.Index(hi_scores.index.tz_convert(tz).date)

    alpha_half = (1.0 - coverage) / 2.0

    for day in sorted(set(days)):
        past_lo = lo_scores[
            (lo_days < day)
            & (lo_days >= day - pd.Timedelta(days=window_days))
        ]
        past_hi = hi_scores[
            (hi_days < day)
            & (hi_days >= day - pd.Timedelta(days=window_days))
        ]
        if len(past_lo) < min_days * 24:
            continue
        n = len(past_lo)
        # finite-sample corrected level per tail
        level_lo = min((1.0 - alpha_half) * (n + 1) / n, 1.0)
        level_hi = min((1.0 - alpha_half) * (n + 1) / n, 1.0)
        q_lo = float(np.quantile(past_lo.to_numpy(), level_lo))
        q_hi = float(np.quantile(past_hi.to_numpy(), level_hi))

        mask = days == day
        out.loc[mask, "p10"] = preds.loc[mask, "p10"] - q_lo
        out.loc[mask, "p90"] = preds.loc[mask, "p90"] + q_hi

    out["p10"] = out[["p10", "p50"]].min(axis=1)
    out["p90"] = out[["p90", "p50"]].max(axis=1)
    return out


def latest_offset_asymmetric(
    preds: pd.DataFrame,
    y: pd.Series,
    window_days: int = 90,
    coverage: float = 0.8,
) -> tuple[float, float]:
    """Asymmetric offsets for the NEXT day. Returns (q_lo, q_hi).

    For the daily loop: apply as p10_new = p10 - q_lo, p90_new = p90 + q_hi.
    """
    y = y.reindex(preds.index)
    lo_scores = (preds["p10"] - y).dropna()
    hi_scores = (y - preds["p90"]).dropna()

    cutoff = preds.index.max() - pd.Timedelta(days=window_days)
    recent_lo = lo_scores[lo_scores.index >= cutoff]
    recent_hi = hi_scores[hi_scores.index >= cutoff]

    alpha_half = (1.0 - coverage) / 2.0
    n = len(recent_lo)
    level = min((1.0 - alpha_half) * (n + 1) / n, 1.0)

    q_lo = float(np.quantile(recent_lo.to_numpy(), level))
    q_hi = float(np.quantile(recent_hi.to_numpy(), level))
    return q_lo, q_hi
