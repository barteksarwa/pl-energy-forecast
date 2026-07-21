"""Walk-forward evaluation for a deep model — the README-grade exam.

Monthly refits (nets are slower than LightGBM; cadence logged) over the same
12-month test period as the classic backtest. Predictions land in
data/processed/backtest_preds_fcst/ so summarize/diagnostics pick them up.

Run: python -m src.models.deep.run_walkforward --variant enc_dec --hidden 64
     [--with-tso]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.config import REPO_ROOT, load_config
from src.evaluation.metrics import mae, mape, pinball_loss
from src.features.weather import load_weather_forecast_history
from src.models.deep.data import build_samples, standardize_covariates
from src.models.deep.nets import LADDER, VARIANTS
from src.models.deep.run_campaign import flat_series
from src.models.deep.train import predict_mw, train_variant

CKPT_DIR = REPO_ROOT / "outputs" / "checkpoints"


def month_starts(first: str, last: str) -> list[str]:
    return [str(d.date()) for d in pd.date_range(first, last, freq="MS")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="enc_dec")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--with-tso", action="store_true")
    parser.add_argument("--train-days", type=int, default=365)
    args = parser.parse_args()

    cfg = load_config()
    load = pd.read_parquet(cfg.paths["data_processed"] / "load.parquet").iloc[:, 0]
    weather = load_weather_forecast_history(cfg)
    tso = (
        pd.read_parquet(cfg.paths["data_processed"] / "tso_forecast.parquet").iloc[:, 0]
        if args.with_tso else None
    )
    n_tail = 2 if args.with_tso else 1

    all_days = sorted(set(load.index.tz_convert(cfg.timezone_local).date))[15:-1]
    last_day = all_days[-1]
    test_first = str(pd.Timestamp(str(last_day)) - pd.Timedelta(days=365))[:10]
    refits = month_starts(test_first, str(last_day))
    name = f"lstm_{args.variant}_h{args.hidden}" + ("_tso" if args.with_tso else "")
    print(f"{name}: {len(refits)} monthly refits, test {test_first} → {last_day}",
          flush=True)

    cls = VARIANTS.get(args.variant) or LADDER[args.variant][0]
    preds_frames = []
    for i, refit in enumerate(refits):
        block_end = refits[i + 1] if i + 1 < len(refits) else str(
            pd.Timestamp(str(last_day)) + pd.Timedelta(days=1))[:10]
        tr_days = [d for d in all_days
                   if str(pd.Timestamp(refit) - pd.Timedelta(days=args.train_days))[:10]
                   <= str(d) < refit]
        # validation = last 30 train days (early stopping), never the test block
        va_days, tr_days = tr_days[-30:], tr_days[:-30]
        bl_days = [d for d in all_days if refit <= str(d) < block_end]
        if not bl_days or len(tr_days) < 120:
            continue
        tr = build_samples(load, weather, tr_days, tso=tso)
        va = build_samples(load, weather, va_days, tso=tso)
        bl = build_samples(load, weather, bl_days, tso=tso)
        if not bl.days:
            continue
        standardize_covariates(tr, va, bl, n_tail=n_tail)
        net = cls(tr.enc.shape[-1], tr.fut.shape[-1], hidden=args.hidden)
        train_variant(net, tr, va, str(CKPT_DIR / f"wf_{name}_{refit}.pt"),
                      seed=42, max_epochs=60, patience=8)
        pred = predict_mw(net, bl)
        preds_frames.append(pd.DataFrame({
            "p10": flat_series(pred, bl, 0),
            "p50": flat_series(pred, bl, 1),
            "p90": flat_series(pred, bl, 2),
        }))
        print(f"block {refit}: {len(bl.days)} days done", flush=True)

    preds = pd.concat(preds_frames).sort_index()
    out_dir = cfg.paths["data_processed"] / "backtest_preds_fcst"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out_dir / f"{name}.parquet")

    y = load.reindex(preds.index)
    print(f"\n{name} walk-forward: "
          f"MAPE {mape(y, preds['p50']):.2f}%  MAE {mae(y, preds['p50']):.0f} MW  "
          f"pinball p10/p50/p90 "
          f"{pinball_loss(y, preds['p10'], 0.1):.1f}/"
          f"{pinball_loss(y, preds['p50'], 0.5):.1f}/"
          f"{pinball_loss(y, preds['p90'], 0.9):.1f}  "
          f"n={int(preds['p50'].notna().sum())}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
