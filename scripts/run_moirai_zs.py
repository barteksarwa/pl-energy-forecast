"""Moirai zero-shot walk-forward, run from the SCRATCH venv.

uni2ts pins numpy<1.27 and cannot live in the main environment
(demandlib needs numpy>=2). This script has no src.* imports; it reads
the data parquets directly and writes predictions parquet that the
main env scores like any other model.

Usage (from the repo root):
  <scratch>/moirai_env/bin/python scripts/run_moirai_zs.py [--covariates]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MODEL_ID = "Salesforce/moirai-1.1-R-base"
CONTEXT = 2048
TZ = "Europe/Warsaw"
TEST_START = "2024-07-16"
COV_COLS = ["solar_fcst_mw", "wind_on_fcst_mw", "tso_forecast_mw"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--covariates", action="store_true")
    parser.add_argument("--days", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    proc = Path("data/processed")
    price = pd.read_parquet(proc / "price_da_eur.parquet").iloc[:, 0]
    res = pd.read_parquet(proc / "res_forecast.parquet")
    tso = pd.read_parquet(proc / "tso_forecast.parquet").iloc[:, 0]
    cov = pd.concat([res[["solar_fcst_mw", "wind_on_fcst_mw"]],
                     tso.rename("tso_forecast_mw")], axis=1)[COV_COLS]

    n_cov = len(COV_COLS) if args.covariates else 0
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(MODEL_ID),
        prediction_length=24,
        context_length=CONTEXT,
        patch_size="auto",
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=n_cov,
        past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=1)

    days = sorted(set(price.tz_convert(TZ).index.date))
    test_days = [d for d in days if d >= pd.Timestamp(TEST_START).date()][:-1]
    if args.days:
        test_days = test_days[:args.days]

    name = "moirai_cov" if args.covariates else "moirai_zs"
    rows = []
    torch.manual_seed(42)  # sampling forecast: fix the sample draw
    for i, day in enumerate(test_days):
        start = pd.Timestamp(day, tz=TZ).tz_convert("UTC")
        hours = pd.date_range(start, periods=24, freq="1h")
        # local-day boundary: take exactly the hours of this local day
        hours = price.index[(price.index >= start)
                            & (price.index < start + pd.Timedelta(hours=25))]
        hours = hours[pd.Index(hours.tz_convert(TZ).date) == day]
        if len(hours) < 23:
            continue
        ctx = price[price.index < hours[0]].dropna().tail(CONTEXT)
        if len(ctx) < 168:
            continue
        entry = {"target": ctx.to_numpy(np.float32),
                 "start": pd.Period(ctx.index[0], freq="h")}
        if n_cov:
            span = cov.reindex(ctx.index.append(hours)).ffill().fillna(0.0)
            entry["feat_dynamic_real"] = span.to_numpy(np.float32).T
        fcst = next(iter(predictor.predict([entry])))
        samples = fcst.samples[:, :len(hours)]
        q = np.quantile(samples, [0.1, 0.5, 0.9], axis=0).T
        q.sort(axis=1)
        rows.append(pd.DataFrame(q, index=hours,
                                 columns=["p10", "p50", "p90"]))
        if i % 50 == 0:
            print(f"{i}/{len(test_days)} {day}", flush=True)

    preds = pd.concat(rows)
    out_dir = proc / "backtest_preds_price_moirai2yr"
    out_dir.mkdir(exist_ok=True)
    preds.to_parquet(out_dir / f"{name}.parquet")
    print(f"wrote {len(preds)} hours -> {out_dir}/{name}.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
