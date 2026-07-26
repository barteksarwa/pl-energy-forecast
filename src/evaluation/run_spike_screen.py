"""Walk-forward screen of the spike classifier.

Weekly refits, trailing 365-day window, same feature matrix as the
price models. The label threshold comes from each training window
(see src/models/spike.py — the leakage rule).

Metrics:
- AUC: can the model rank spike hours above normal hours at all?
- Brier score: quality of the probability itself.
- precision@2/day: of the 2 highest-probability hours each day, how
  many were real spikes (only days that had a spike count).

Gate (PLAN Phase 5 S3): AUC >= 0.80 -> p(spike) becomes a daily-report
column (after a 3-seed confirm). Below: honest negative, stop.

Run: python -m src.evaluation.run_spike_screen [--test-start 2024-07-16]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.config import load_config
from src.evaluation.run_price_backtest import assemble_price_features
from src.models.spike import SPIKE_QUANTILE, SpikeClassifier


def walk_forward_proba(
    x: pd.DataFrame, y: pd.Series, test_start: pd.Timestamp, tz: str,
    seed: int, train_days: int = 365, refit_days: int = 7,
) -> pd.Series:
    dates = pd.Index(x.index.tz_convert(tz).date)
    test_days = sorted(set(dates[x.index >= test_start]))
    model, last_fit, out = None, None, []
    for day in test_days:
        if model is None or (pd.Timestamp(day) - pd.Timestamp(last_fit)).days >= refit_days:
            tr = (dates < day) & (dates >= day - pd.Timedelta(days=train_days))
            x_tr = x[tr].dropna()
            model = SpikeClassifier(seed=seed)
            model.fit(x_tr, y.reindex(x_tr.index))
            last_fit = day
        x_day = x[dates == day].dropna()
        if not x_day.empty:
            out.append(model.predict_proba(x_day))
    return pd.concat(out)


def score(proba: pd.Series, y: pd.Series, tz: str) -> dict:
    y = y.reindex(proba.index)
    ok = y.notna()
    proba, y = proba[ok], y[ok]
    # Evaluation-side label: pooled test quantile is fine HERE (it only
    # defines what we score against, not what the model saw).
    label = (y >= y.quantile(SPIKE_QUANTILE)).astype(int)
    days = pd.Index(proba.index.tz_convert(tz).date)
    hits, spike_days = 0, 0
    for day in sorted(set(days)):
        m = days == day
        if label[m].sum() == 0:
            continue
        spike_days += 1
        top2 = proba[m].nlargest(2).index
        hits += int(label.loc[top2].sum())
    return {
        "auc": roc_auc_score(label, proba),
        "brier": brier_score_loss(label, proba),
        "precision_at2": hits / (2 * spike_days) if spike_days else float("nan"),
        "spike_days": spike_days,
        "n_hours": int(len(proba)),
        "spike_rate": float(label.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-start", default="2024-07-16")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--train-days", type=int, default=365)
    args = parser.parse_args()

    cfg = load_config()
    proc = cfg.paths["data_processed"]
    tz = cfg.timezone_local
    price = pd.read_parquet(proc / "price_da_eur.parquet").iloc[:, 0]
    load = pd.read_parquet(proc / "load.parquet").iloc[:, 0]
    tso = pd.read_parquet(proc / "tso_forecast.parquet").iloc[:, 0]
    res = pd.read_parquet(proc / "res_forecast.parquet")

    test_start = pd.Timestamp(args.test_start, tz=tz)
    first = test_start - pd.Timedelta(days=args.train_days + 40)
    last = min(price.index[-1], load.index[-1]).tz_convert(tz) - pd.Timedelta(days=1)
    print(f"Assembling features {first.date()} → {last.date()} ...")
    x = assemble_price_features(
        price, load, tso, tz,
        pd.Timestamp(first.date(), tz=tz), pd.Timestamp(last.date(), tz=tz),
        res=res)
    y = price.reindex(x.index)

    rows = []
    (proc / "backtest_preds_spike").mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        print(f"walk-forward, seed {seed} ...")
        proba = walk_forward_proba(
            x, y, test_start.tz_convert("UTC"), tz, seed, args.train_days)
        proba.to_frame().to_parquet(
            proc / f"backtest_preds_spike/spike_proba_s{seed}.parquet")
        rows.append({"seed": seed, **score(proba, y, tz)})
        print(f"  AUC {rows[-1]['auc']:.3f}  Brier {rows[-1]['brier']:.4f}  "
              f"p@2 {rows[-1]['precision_at2']:.3f}")

    table = pd.DataFrame(rows).set_index("seed").round(4)
    out_dir = proc.parent.parent / "reports" / "backtests"
    stamp = f"{pd.Timestamp.now(tz).date()}_spike_screen"
    table.to_csv(out_dir / f"{stamp}.csv")
    md = [
        f"# Spike classifier screen — {stamp}", "",
        f"Top-{int((1 - SPIKE_QUANTILE) * 100)}% price hours. Walk-forward, "
        f"weekly refits, {args.train_days}d train window, "
        f"test from {args.test_start}.",
        "Label threshold from each training window (leakage-safe).",
        "Gate: AUC >= 0.80 for a daily-report column.", "",
        table.to_markdown(), "",
    ]
    (out_dir / f"{stamp}.md").write_text("\n".join(md))
    print(f"\nWritten {out_dir}/{stamp}.(csv|md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
