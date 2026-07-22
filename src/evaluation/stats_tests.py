"""Statistical tests for forecast comparison and band validation.

Why: point metrics without significance tests are anecdotes. The EPF
protocol paper (Lago, Marcjasz, De Schutter & Weron 2021, Applied
Energy 293) prescribes the Diebold-Mariano test on DAILY loss
differentials — the multivariate version, because hourly errors within
a day are correlated and would inflate significance.

Band validation: Kupiec (1995) unconditional coverage and
Christoffersen (1998) independence/conditional coverage likelihood-
ratio tests. A band can hit 80% on average while its violations
cluster on exactly the days that hurt — Christoffersen catches that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

LOCAL_TZ = "Europe/Warsaw"


def daily_losses(
    pred_p50: pd.Series, y: pd.Series, tz: str = LOCAL_TZ
) -> pd.Series:
    """L1 norm of the 24-hour error vector per local day (Lago eq. DM)."""
    err = (y.reindex(pred_p50.index) - pred_p50).abs()
    err = err.dropna()
    days = pd.Index(err.index.tz_convert(tz).date)
    return err.groupby(days).sum()


def dm_test(
    loss_a: pd.Series, loss_b: pd.Series
) -> tuple[float, float]:
    """Multivariate Diebold-Mariano on daily loss series.

    H0: equal accuracy. Returns (statistic, one_sided_p) where the
    one-sided alternative is "A is MORE accurate than B" (negative
    statistic favors A). Asymptotically standard normal.
    """
    common = loss_a.index.intersection(loss_b.index)
    d = (loss_a.reindex(common) - loss_b.reindex(common)).dropna()
    n = len(d)
    if n < 30:
        raise ValueError(f"only {n} common days — DM needs more")
    stat = float(d.mean() / (d.std(ddof=1) / np.sqrt(n)))
    p_one_sided = float(norm.cdf(stat))  # P(A better) small stat -> small p
    return stat, p_one_sided


def band_violations(preds: pd.DataFrame, y: pd.Series) -> pd.Series:
    """1 where the actual fell outside [p10, p90], aligned + NaN-free."""
    y = y.reindex(preds.index)
    ok = y.notna() & preds["p10"].notna()
    out = ~((y >= preds["p10"]) & (y <= preds["p90"]))
    return out[ok].astype(int)


def kupiec_test(violations: pd.Series, coverage: float = 0.8) -> tuple[float, float]:
    """Unconditional coverage LR test. H0: violation rate == 1-coverage.

    Returns (LR statistic, p). LR ~ chi2(1) under H0.
    """
    v = violations.to_numpy()
    n, x = len(v), int(v.sum())
    pi_hat = x / n
    pi0 = 1.0 - coverage
    if pi_hat in (0.0, 1.0):
        return float("inf"), 0.0
    ll0 = x * np.log(pi0) + (n - x) * np.log(1 - pi0)
    ll1 = x * np.log(pi_hat) + (n - x) * np.log(1 - pi_hat)
    lr = float(-2.0 * (ll0 - ll1))
    return lr, float(chi2.sf(lr, df=1))


def christoffersen_test(violations: pd.Series) -> tuple[float, float]:
    """Independence LR test. H0: violations do not cluster.

    First-order Markov alternative: P(violation | violation yesterday)
    != P(violation | calm yesterday). LR ~ chi2(1) under H0.
    """
    v = violations.to_numpy()
    pairs = np.stack([v[:-1], v[1:]], axis=1)
    n00 = int(((pairs[:, 0] == 0) & (pairs[:, 1] == 0)).sum())
    n01 = int(((pairs[:, 0] == 0) & (pairs[:, 1] == 1)).sum())
    n10 = int(((pairs[:, 0] == 1) & (pairs[:, 1] == 0)).sum())
    n11 = int(((pairs[:, 0] == 1) & (pairs[:, 1] == 1)).sum())
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    if 0.0 in (pi01, pi11, pi) or 1.0 in (pi01, pi11, pi):
        # degenerate table: fall back to "cannot reject" only when there
        # is genuinely no clustering signal (no 1->1 transitions)
        return 0.0, 1.0
    ll0 = (n01 + n11) * np.log(pi) + (n00 + n10) * np.log(1 - pi)
    ll1 = (n01 * np.log(pi01) + n00 * np.log(1 - pi01)
           + n11 * np.log(pi11) + n10 * np.log(1 - pi11))
    lr = float(-2.0 * (ll0 - ll1))
    return lr, float(chi2.sf(lr, df=1))
