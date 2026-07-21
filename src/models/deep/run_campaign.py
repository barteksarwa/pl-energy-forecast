"""Overnight LSTM screening campaign. Sequential queue, one MPS job at a time.

4 variants x 3 seeds. Temporal split (screening only — the winner earns a
proper walk-forward backtest afterwards):
  train: start .. 2026-01-31
  val:   2026-02-01 .. 2026-04-15   (early stopping)
  test:  2026-04-16 .. last full day (never touched during training)

Results appended to outputs/deep_campaign_results.csv after every run —
a killed batch loses nothing. Checkpoints in outputs/checkpoints/.

Run: nohup caffeinate -i uv run python -u -m src.models.deep.run_campaign \
       > outputs/logs/campaign.log 2>&1 &
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.config import REPO_ROOT, load_config
from src.evaluation.metrics import mape, pinball_loss
from src.features.weather import load_weather_forecast_history
from src.models.deep.data import build_samples, standardize_covariates
from src.models.deep.nets import VARIANTS
from src.models.deep.train import predict_mw, train_variant

RESULTS = REPO_ROOT / "outputs" / "deep_campaign_results.csv"
CKPT_DIR = REPO_ROOT / "outputs" / "checkpoints"
SEEDS = [42, 7, 2026]
VAL_START, TEST_START = "2026-02-01", "2026-04-16"


def flat_series(arr: np.ndarray, samples, q_idx: int) -> pd.Series:
    from src.pipeline.daily_run import local_day_hours_utc

    parts = []
    for i, day in enumerate(samples.days):
        hours = local_day_hours_utc(pd.Timestamp(day, tz="Europe/Warsaw"), "Europe/Warsaw")
        parts.append(pd.Series(arr[i, :, q_idx], index=hours))
    return pd.concat(parts).sort_index()


def main() -> int:
    cfg = load_config()
    load = pd.read_parquet(cfg.paths["data_processed"] / "load.parquet").iloc[:, 0]
    weather = load_weather_forecast_history(cfg)

    all_days = sorted(set(load.index.tz_convert(cfg.timezone_local).date))[15:-1]
    train_days = [d for d in all_days if str(d) < VAL_START]
    val_days = [d for d in all_days if VAL_START <= str(d) < TEST_START]
    test_days = [d for d in all_days if str(d) >= TEST_START]
    print(f"days: train {len(train_days)}, val {len(val_days)}, test {len(test_days)}",
          flush=True)

    tr = build_samples(load, weather, train_days)
    va = build_samples(load, weather, val_days)
    te = build_samples(load, weather, test_days)
    standardize_covariates(tr, va, te)
    print(f"samples: train {len(tr.days)}, val {len(va.days)}, test {len(te.days)}",
          flush=True)

    y_test = pd.concat([
        pd.Series(te.y[i].numpy() * float(te.std[i]) + float(te.mean[i]))
        for i in range(len(te.days))
    ])

    enc_feat, fut_feat = tr.enc.shape[-1], tr.fut.shape[-1]
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    done = set()
    if RESULTS.exists():
        prev = pd.read_csv(RESULTS)
        done = set(zip(prev["variant"], prev["seed"]))
        print(f"resume: {len(done)} runs already done", flush=True)

    for vname, cls in VARIANTS.items():
        for seed in SEEDS:
            if (vname, seed) in done:
                continue
            print(f"\n=== {vname} seed {seed} ===", flush=True)
            net = cls(enc_feat, fut_feat)
            ckpt = CKPT_DIR / f"{vname}_seed{seed}.pt"
            info = train_variant(net, tr, va, str(ckpt), seed)

            pred = predict_mw(net, te)
            actual = flat_series(
                np.stack([te.y.numpy()] * 3, axis=-1)
                * te.std.numpy()[:, None, None] + te.mean.numpy()[:, None, None],
                te, 1,
            )
            p10 = flat_series(pred, te, 0)
            p50 = flat_series(pred, te, 1)
            p90 = flat_series(pred, te, 2)
            row = {
                "variant": vname, **info,
                "test_mape": round(mape(actual, p50), 3),
                "test_pinball_p10": round(pinball_loss(actual, p10, 0.1), 2),
                "test_pinball_p50": round(pinball_loss(actual, p50, 0.5), 2),
                "test_pinball_p90": round(pinball_loss(actual, p90, 0.9), 2),
                "checkpoint": str(ckpt.relative_to(REPO_ROOT)),
            }
            header = not RESULTS.exists()
            pd.DataFrame([row]).to_csv(RESULTS, mode="a", header=header, index=False)
            print(f"logged: {row}", flush=True)

    print("\nCampaign complete.", flush=True)
    print(pd.read_csv(RESULTS).groupby("variant")["test_mape"].agg(["mean", "std"])
          .round(3).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
