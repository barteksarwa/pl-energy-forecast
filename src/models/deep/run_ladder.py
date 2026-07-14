"""LSTM complexity ladder (v3): vanilla -> BiLSTM -> seq2seq+attention.

Fills the rungs below/above the v2 shapes. Same split, seeds, metric, and
n_params logging as v2 — all three CSVs concatenate for one readout.

Run: nohup caffeinate -i uv run python -u -m src.models.deep.run_ladder \
       > outputs/logs/ladder_v3.log 2>&1 &
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.config import REPO_ROOT, load_config
from src.evaluation.metrics import mape, pinball_loss
from src.features.weather import load_weather_forecast_history
from src.models.deep.data import build_samples, standardize_covariates
from src.models.deep.nets import LADDER
from src.models.deep.run_campaign import TEST_START, VAL_START, flat_series
from src.models.deep.train import predict_mw, train_variant

RESULTS = REPO_ROOT / "outputs" / "deep_campaign_v3.csv"
CKPT_DIR = REPO_ROOT / "outputs" / "checkpoints"
SEEDS = [42, 7]


def main() -> int:
    cfg = load_config()
    load = pd.read_parquet(cfg.paths["data_processed"] / "load.parquet").iloc[:, 0]
    weather = load_weather_forecast_history(cfg)

    all_days = sorted(set(load.index.tz_convert(cfg.timezone_local).date))[15:-1]
    tr = build_samples(load, weather, [d for d in all_days if str(d) < VAL_START])
    va = build_samples(
        load, weather, [d for d in all_days if VAL_START <= str(d) < TEST_START]
    )
    te = build_samples(load, weather, [d for d in all_days if str(d) >= TEST_START])
    standardize_covariates(tr, va, te)
    print(f"samples: train {len(tr.days)}, val {len(va.days)}, test {len(te.days)}",
          flush=True)

    actual = flat_series(
        np.stack([te.y.numpy()] * 3, axis=-1)
        * te.std.numpy()[:, None, None] + te.mean.numpy()[:, None, None],
        te, 1,
    )
    enc_feat, fut_feat = tr.enc.shape[-1], tr.fut.shape[-1]

    done = set()
    if RESULTS.exists():
        prev = pd.read_csv(RESULTS)
        done = set(zip(prev["variant"], prev["hidden"], prev["seed"]))
        print(f"resume: {len(done)} runs done", flush=True)

    for vname, (cls, hiddens) in LADDER.items():
        for hidden in hiddens:
            for seed in SEEDS:
                if (vname, hidden, seed) in done:
                    continue
                print(f"\n=== {vname} h{hidden} seed {seed} ===", flush=True)
                net = cls(enc_feat, fut_feat, hidden=hidden)
                n_params = sum(p.numel() for p in net.parameters())
                ckpt = CKPT_DIR / f"v3_{vname}_h{hidden}_seed{seed}.pt"
                info = train_variant(net, tr, va, str(ckpt), seed)
                pred = predict_mw(net, te)
                p10, p50, p90 = (flat_series(pred, te, i) for i in range(3))
                row = {
                    "variant": vname, "hidden": hidden, "n_params": n_params, **info,
                    "test_mape": round(mape(actual, p50), 3),
                    "test_pinball_p10": round(pinball_loss(actual, p10, 0.1), 2),
                    "test_pinball_p50": round(pinball_loss(actual, p50, 0.5), 2),
                    "test_pinball_p90": round(pinball_loss(actual, p90, 0.9), 2),
                }
                pd.DataFrame([row]).to_csv(
                    RESULTS, mode="a", header=not RESULTS.exists(), index=False
                )
                print(f"logged: {vname} h{hidden} params={n_params} "
                      f"mape={row['test_mape']}", flush=True)

    print("\nLadder complete.", flush=True)
    df = pd.read_csv(RESULTS)
    print(df.groupby(["variant", "hidden", "n_params"])["test_mape"]
          .mean().round(3).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
