"""TFT config sweep at 730-day training windows.

The HPO winner (d_model=128, lstm=2, dropout=0.183, lr=1.7e-3) was tuned
FOR 365-day windows. At 730d it reaches ens-3 MAE 18.31 vs champion
LGBM 17.66 on the same window. Question: does a config tuned for the
longer window close the remaining 0.65 EUR/MWh?

Protocol: 1-year walk-forward (test 2025-07-16 on, monthly refits,
train_days=730) — the same protocol the 18.31 came from, NOT a single
split (screening splits mis-ranked models twice in this project).

Stage sweep: 8 configs x seed 42. Stage confirm: best sweep config x
seeds {7, 2026}, then median ens-3, scored against the champion.
Incremental CSVs, idempotent. Preds saved per run for ensembling.

Run: uv run python -m src.models.deep.run_tft730_sweep [--stage all]
Expected: sweep ~2.5-3 h, confirm ~1 h on MPS.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.deep.patchtst_feature_analysis import (
    load_inputs,
    score_preds,
    walk_forward_ablate,
)
from src.models.deep.price_data import build_price_samples
from src.models.deep.tft import TFT
from src.models.deep.train import device

TZ = "Europe/Warsaw"
OUT = Path("reports/sensitivity/tft")
TEST_START = "2025-07-16"
TRAIN_DAYS = 730

# name -> (d_model, lstm_layers, dropout, lr). First entry = HPO winner
# (baseline, rerun so every row shares one code path).
CONFIGS: dict[str, tuple[int, int, float, float]] = {
    "hpo365_base": (128, 2, 0.183, 1.7e-3),
    "dr010":       (128, 2, 0.10, 1.7e-3),
    "dr030":       (128, 2, 0.30, 1.7e-3),
    "d192":        (192, 2, 0.183, 1.7e-3),
    "d64":         (64, 2, 0.183, 1.7e-3),
    "lstm1":       (128, 1, 0.183, 1.7e-3),
    "lr08":        (128, 2, 0.183, 8e-4),
    "lr30":        (128, 2, 0.183, 3e-3),
}
CTX = 1344
N_HEADS = 8
BATCH = 32


def factory(d_model: int, lstm_layers: int, dropout: float):
    def make() -> TFT:
        return TFT(enc_feat=1, fut_feat=12, d_model=d_model, n_heads=N_HEADS,
                   lstm_layers=lstm_layers, dropout=dropout).to(device())
    return make


def run_one(master, price, cfg_name: str, seed: int, test_days, smoke: bool,
            ) -> dict | None:
    d_model, lstm_layers, dropout, lr = CONFIGS[cfg_name]
    kw = {"max_epochs": 2, "patience": 2} if smoke else {}
    t0 = time.time()
    pred = walk_forward_ablate(
        master, price, "full", seed, test_days, train_days=TRAIN_DAYS,
        net_factory=factory(d_model, lstm_layers, dropout), lr=lr,
        batch=BATCH, name=f"tft730_{cfg_name}_s{seed}", **kw)
    if pred is None:
        return None
    pred.to_parquet(OUT / f"preds_sweep_{cfg_name}_s{seed}.parquet")
    return {"config": cfg_name, "seed": seed, "d_model": d_model,
            "lstm_layers": lstm_layers, "dropout": dropout, "lr": lr,
            **score_preds(pred, price),
            "wall_min": round((time.time() - t0) / 60, 1)}


def ensemble(cfg_name: str, seeds: list[int], price) -> dict:
    ps = [pd.read_parquet(OUT / f"preds_sweep_{cfg_name}_s{s}.parquet")
          for s in seeds]
    idx = ps[0].index
    ens = pd.DataFrame(
        {q: np.median([p[q].to_numpy() for p in ps], axis=0)
         for q in ("p10", "p50", "p90")}, index=idx)
    ens[:] = np.sort(ens.to_numpy(), axis=1)
    ens.to_parquet(OUT / f"preds_sweep_{cfg_name}_ens{len(seeds)}.parquet")
    d_model, lstm_layers, dropout, lr = CONFIGS[cfg_name]
    # same column set as run_one rows — appending with fewer keys shifts
    # the header-less CSV append and corrupts the row
    return {"config": f"{cfg_name}_ens{len(seeds)}", "seed": -1,
            "d_model": d_model, "lstm_layers": lstm_layers,
            "dropout": dropout, "lr": lr, **score_preds(ens, price),
            "wall_min": 0.0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all",
                        choices=["all", "sweep", "confirm"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    csv = OUT / "sweep730.csv"
    print(f"[{pd.Timestamp.now()}] TFT-730 sweep | device={device()} "
          f"stage={args.stage}", flush=True)

    price, res, tso = load_inputs()
    all_dates = sorted(set(price.index.tz_convert(TZ).date))
    test_days = [d for d in all_dates
                 if d >= pd.Timestamp(TEST_START).date()]
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
        done = {(r.config, int(r.seed))
                for r in pd.read_csv(csv).itertuples()}
        print(f"sweep: {len(done)} rows exist, skipping them")

    def record(row: dict) -> None:
        pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(),
                                   index=False)
        print(f"  -> {row['config']} s{row['seed']} MAE {row['mae']:.2f} "
              f"rMAE {row['rmae']:.3f} cov {row['coverage_80_pct']:.1f}%",
              flush=True)

    if args.stage in ("all", "sweep"):
        for cfg_name in CONFIGS:
            if (cfg_name, 42) in done:
                continue
            print(f"\n=== sweep {cfg_name} seed=42 ===", flush=True)
            row = run_one(master, price, cfg_name, 42, test_days, args.smoke)
            if row:
                record(row)
                done.add((cfg_name, 42))

    if args.stage in ("all", "confirm"):
        df = pd.read_csv(csv)
        single = df[(df.seed == 42) & (df.config.isin(CONFIGS))]
        best = single.sort_values("mae").iloc[0]["config"]
        print(f"\n=== confirm: best sweep config = {best} ===", flush=True)
        for seed in (7, 2026):
            if (best, seed) in done:
                continue
            print(f"\n=== confirm {best} seed={seed} ===", flush=True)
            row = run_one(master, price, best, seed, test_days, args.smoke)
            if row:
                record(row)
                done.add((best, seed))
        record(ensemble(best, [42, 7, 2026], price))
        print("\nChampion LGBM same window: MAE 17.66 "
              "(reports/backtests/2026-07-20_price_group_ablation.csv)")
    print(f"[{pd.Timestamp.now()}] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
