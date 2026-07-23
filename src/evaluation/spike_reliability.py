"""Reliability analysis of the spike classifier probabilities.

Run: python -m src.evaluation.spike_reliability
"""

from __future__ import annotations

import sys

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import load_config
from src.models.spike import SPIKE_QUANTILE


def main() -> int:
    cfg = load_config()
    proc = cfg.paths["data_processed"]
    tz = cfg.timezone_local

    proba = pd.read_parquet(
        proc / "backtest_preds_spike/spike_proba_s42.parquet").iloc[:, 0]
    price = pd.read_parquet(proc / "price_da_eur.parquet").iloc[:, 0]
    y = price.reindex(proba.index).dropna()
    proba = proba.reindex(y.index)
    label = (y >= y.quantile(SPIKE_QUANTILE)).astype(int)

    print(f"{len(proba)} hours, spike rate {label.mean():.3f}")
    print(f"ROC-AUC {roc_auc_score(label, proba):.3f}  "
          f"PR-AUC {average_precision_score(label, proba):.3f} "
          f"(base rate = {label.mean():.3f})")

    years = proba.index.tz_convert(tz).year
    for yr in sorted(set(years)):
        m = years == yr
        if label[m].sum() < 10:
            continue
        print(f"  {yr}: ROC-AUC {roc_auc_score(label[m], proba[m]):.3f}  "
              f"PR-AUC {average_precision_score(label[m], proba[m]):.3f}  "
              f"spikes {label[m].sum()}")

    print("\nreliability (predicted bin -> observed spike rate):")
    bins = pd.cut(proba, [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    tbl = label.groupby(bins, observed=True).agg(["mean", "count"])
    print(tbl.rename(columns={"mean": "observed_rate"}).to_string())

    print("\nthreshold table:")
    for t in (0.3, 0.5, 0.7):
        flag = proba >= t
        if not flag.any():
            continue
        prec = label[flag].mean()
        rec = label[flag].sum() / label.sum()
        print(f"  p>={t}: precision {prec:.2f}  recall {rec:.2f}  "
              f"flagged {flag.sum()} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
