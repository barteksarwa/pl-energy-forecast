"""PatchTST capacity sweep at 730-day windows — last root-cause check.

The original loss diagnosis blamed "small capacity (197k params) +
short windows". The window half is proven (+1.2 EUR/MWh). This tests
the capacity half: d_model {64, 96, 128, 192} at 730d windows, same
1-year walk-forward as everything else, seed 42 screen -> best config
confirmed on 3 seeds -> median ens-3.

Run: uv run python -m src.models.deep.run_patchtst730_capacity [--smoke]
Expected: ~30 min on MPS (PatchTST refits are ~30s).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.deep.patchtst import PatchTST
from src.models.deep.patchtst_feature_analysis import (
    CTX,
    PATCH_LEN,
    STRIDE,
    load_inputs,
    score_preds,
    walk_forward_ablate,
)
from src.models.deep.price_data import build_price_samples
from src.models.deep.train import device

TZ = "Europe/Warsaw"
OUT = Path("reports/sensitivity/patchtst")
TEST_START = "2025-07-16"
TRAIN_DAYS = 730
D_MODELS = [64, 96, 128, 192]


def factory(d_model: int):
    def make() -> PatchTST:
        return PatchTST(enc_feat=1, fut_feat=12, d_model=d_model,
                        patch_len=PATCH_LEN, stride=STRIDE).to(device())
    return make


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    csv = OUT / "capacity730.csv"
    print(f"[{pd.Timestamp.now()}] PatchTST-730 capacity sweep | "
          f"device={device()}", flush=True)
    price, res, tso = load_inputs()
    all_dates = sorted(set(price.index.tz_convert(TZ).date))
    test_days = [d for d in all_dates
                 if d >= pd.Timestamp(TEST_START).date()]
    kw = {"max_epochs": 2, "patience": 2} if args.smoke else {}
    if args.smoke:
        test_days = test_days[:35]

    t0 = time.time()
    first_needed = test_days[0] - pd.Timedelta(days=TRAIN_DAYS + 1)
    master = build_price_samples(
        price, res, tso, [d for d in all_dates if d >= first_needed], CTX, TZ)
    print(f"master: {len(master.days)} days ({time.time() - t0:.0f}s)",
          flush=True)

    done = set()
    if csv.exists():
        done = {(int(r.d_model), int(r.seed))
                for r in pd.read_csv(csv).itertuples()}

    def run(d_model: int, seed: int) -> dict | None:
        pred = walk_forward_ablate(
            master, price, "full", seed, test_days, train_days=TRAIN_DAYS,
            net_factory=factory(d_model),
            name=f"cap730_d{d_model}_s{seed}", **kw)
        if pred is None:
            return None
        pred.to_parquet(OUT / f"preds_cap730_d{d_model}_s{seed}.parquet")
        row = {"d_model": d_model, "seed": seed, **score_preds(pred, price)}
        pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(),
                                   index=False)
        print(f"  -> d{d_model} s{seed} MAE {row['mae']:.2f} "
              f"rMAE {row['rmae']:.3f} cov {row['coverage_80_pct']:.1f}%",
              flush=True)
        return row

    for d_model in D_MODELS:
        if (d_model, 42) in done:
            continue
        print(f"\n=== capacity d_model={d_model} seed=42 ===", flush=True)
        run(d_model, 42)

    df = pd.read_csv(csv)
    best = int(df[df.seed == 42].sort_values("mae").iloc[0]["d_model"])
    print(f"\n=== confirm: best d_model = {best} ===", flush=True)
    for seed in (7, 2026):
        if (best, seed) in done:
            continue
        print(f"\n=== confirm d{best} seed={seed} ===", flush=True)
        run(best, seed)

    ps = [pd.read_parquet(OUT / f"preds_cap730_d{best}_s{s}.parquet")
          for s in (42, 7, 2026)]
    idx = ps[0].index
    ens = pd.DataFrame(
        {q: np.median([p[q].to_numpy() for p in ps], axis=0)
         for q in ("p10", "p50", "p90")}, index=idx)
    ens[:] = np.sort(ens.to_numpy(), axis=1)
    ens.to_parquet(OUT / f"preds_cap730_d{best}_ens3.parquet")
    row = {"d_model": best, "seed": -1, **score_preds(ens, price)}
    pd.DataFrame([row]).to_csv(csv, mode="a", header=False, index=False)
    print(f"\nens-3 d{best}: MAE {row['mae']:.2f} rMAE {row['rmae']:.3f} "
          f"cov {row['coverage_80_pct']:.1f}%", flush=True)
    print(f"[{pd.Timestamp.now()}] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
