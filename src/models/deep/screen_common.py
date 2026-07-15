"""Shared plumbing for screening scripts: data, splits, train+eval+log."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import REPO_ROOT, load_config
from src.evaluation.metrics import mape, pinball_loss
from src.features.weather import load_weather_forecast_history
from src.models.deep.data import DaySamples
from src.models.deep.run_campaign import TEST_START, VAL_START, flat_series
from src.models.deep.train import predict_mw, train_variant

CKPT_DIR = REPO_ROOT / "outputs" / "checkpoints"


def load_data():
    cfg = load_config()
    load = pd.read_parquet(cfg.paths["data_processed"] / "load.parquet").iloc[:, 0]
    weather = load_weather_forecast_history(cfg)
    tso = pd.read_parquet(cfg.paths["data_processed"] / "tso_forecast.parquet").iloc[:, 0]
    days = sorted(set(load.index.tz_convert(cfg.timezone_local).date))[15:-1]
    split = {
        "train": [d for d in days if str(d) < VAL_START],
        "val": [d for d in days if VAL_START <= str(d) < TEST_START],
        "test": [d for d in days if str(d) >= TEST_START],
    }
    return load, weather, tso, split


def actual_series(te: DaySamples) -> pd.Series:
    return flat_series(
        np.stack([te.y.numpy()] * 3, axis=-1)
        * te.std.numpy()[:, None, None] + te.mean.numpy()[:, None, None],
        te, 1,
    )


def eval_and_log(net, tr, va, te, results_csv, row_base: dict, seed: int) -> dict:
    n_params = sum(p.numel() for p in net.parameters())
    ckpt = CKPT_DIR / f"{row_base['variant']}_{row_base.get('tag','')}_seed{seed}.pt"
    info = train_variant(net, tr, va, str(ckpt), seed)
    pred = predict_mw(net, te)
    p10, p50, p90 = (flat_series(pred, te, i) for i in range(3))
    actual = actual_series(te)
    row = {
        **row_base, "n_params": n_params, **info,
        "test_mape": round(mape(actual, p50), 3),
        "test_pinball_p10": round(pinball_loss(actual, p10, 0.1), 2),
        "test_pinball_p50": round(pinball_loss(actual, p50, 0.5), 2),
        "test_pinball_p90": round(pinball_loss(actual, p90, 0.9), 2),
    }
    pd.DataFrame([row]).to_csv(results_csv, mode="a",
                               header=not results_csv.exists(), index=False)
    print(f"logged: {row}", flush=True)
    return row
