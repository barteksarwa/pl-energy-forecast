"""Cross-model quantile ensemble with rolling CRPS weights.

Score: `crps3` — mean pinball loss over the three stored quantiles
(0.1, 0.5, 0.9). With only three quantiles this is the honest CRPS
proxy; the name says so.

Weights for day D come from each member's mean crps3 over the trailing
`window_days` before D (inverse-score, normalized). The first
`warmup_days` use equal weights. No future information anywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOCAL_TZ = "Europe/Warsaw"
QS = (0.1, 0.5, 0.9)
COLS = ("p10", "p50", "p90")


def crps3(preds: pd.DataFrame, y: pd.Series) -> pd.Series:
    y = y.reindex(preds.index)
    losses = []
    for q, col in zip(QS, COLS):
        diff = y - preds[col]
        losses.append(np.maximum(q * diff, (q - 1) * diff))
    return pd.concat(losses, axis=1).mean(axis=1)


def rolling_weights(
    scores: dict[str, pd.Series],
    days: list,
    window_days: int = 60,
) -> pd.DataFrame:
    """Per-day member weights from trailing mean crps3. Equal weights
    until every member has `window_days` of history."""
    daily = pd.DataFrame({
        name: s.groupby(pd.Index(s.index.tz_convert(LOCAL_TZ).date)).mean()
        for name, s in scores.items()
    })
    weights = {}
    for day in days:
        past = daily[(daily.index < day)
                     & (daily.index >= day - pd.Timedelta(days=window_days))]
        past = past.dropna()
        if len(past) < window_days * 0.8:
            w = pd.Series(1.0 / len(scores), index=daily.columns)
        else:
            inv = 1.0 / past.mean()
            w = inv / inv.sum()
        weights[day] = w
    return pd.DataFrame(weights).T


def blend(
    members: dict[str, pd.DataFrame],
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Weighted quantile-wise average, ordering re-enforced."""
    any_pred = next(iter(members.values()))
    idx = any_pred.index
    for p in members.values():
        idx = idx.intersection(p.index)
    idx = idx.sort_values()
    days = pd.Index(idx.tz_convert(LOCAL_TZ).date)

    out = pd.DataFrame(0.0, index=idx, columns=list(COLS))
    wsum = pd.Series(0.0, index=idx)
    for name, p in members.items():
        w = weights[name].reindex(days).to_numpy()
        for col in COLS:
            out[col] += p[col].reindex(idx).to_numpy() * w
        wsum += w
    for col in COLS:
        out[col] /= wsum
    arr = np.sort(out.to_numpy(), axis=1)
    return pd.DataFrame(arr, index=idx, columns=list(COLS))
