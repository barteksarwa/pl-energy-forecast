"""SHAP drivers for the LightGBM price model. Hard rule 3: every shipped
model explains itself.

Fits the P50 booster on the most recent 365 days (same window the daily
loop would use), computes SHAP on the last 90 days, writes:
- reports/sensitivity/shap_summary_price_lgbm.png
- reports/sensitivity/shap_importance_price.csv

Run: python -m src.evaluation.run_price_shap
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.config import load_config
from src.evaluation.run_price_backtest import assemble_price_features
from src.models.gbm import LightGBMQuantile


def main() -> int:
    cfg = load_config()
    proc = cfg.paths["data_processed"]
    price = pd.read_parquet(proc / "price_da_eur.parquet").iloc[:, 0]
    load = pd.read_parquet(proc / "load.parquet").iloc[:, 0]
    tso = pd.read_parquet(proc / "tso_forecast.parquet").iloc[:, 0]
    res = pd.read_parquet(proc / "res_forecast.parquet")

    tz = cfg.timezone_local
    last = min(price.index[-1], load.index[-1]).tz_convert(tz) - pd.Timedelta(days=1)
    start = last - pd.Timedelta(days=365)
    print(f"Features {start.date()} → {last.date()} ...")
    x = assemble_price_features(
        price, load, tso, tz,
        pd.Timestamp(start.date(), tz=tz), pd.Timestamp(last.date(), tz=tz),
        res=res,
    )
    x = x.dropna()
    y = price.reindex(x.index)

    model = LightGBMQuantile()
    model.fit(x, y)
    booster = model._models[0.5]

    x_recent = x[x.index >= x.index[-1] - pd.Timedelta(days=90)]
    explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(x_recent)

    out_dir = proc.parent.parent / "reports" / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    imp = (
        pd.DataFrame({"feature": x.columns, "mean_abs_shap_eur": abs(sv).mean(axis=0)})
        .sort_values("mean_abs_shap_eur", ascending=False)
        .reset_index(drop=True)
    )
    imp.to_csv(out_dir / "shap_importance_price.csv", index=False)

    plt.figure()
    shap.summary_plot(sv, x_recent, show=False, max_display=20)
    plt.title("SHAP — LightGBM price P50, last 90 days", loc="left")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_price_lgbm.png", dpi=150)
    plt.close()

    print(imp.head(10).to_string(index=False))
    print(f"\nWritten to {out_dir}/shap_*price*.{{png,csv}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
