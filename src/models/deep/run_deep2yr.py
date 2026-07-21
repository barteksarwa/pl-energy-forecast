"""Deep re-benchmark on the FULL 2-year test at 730-day windows.

Unblocked by the 2015+ deep-history backfill: 730d training windows now
have data before 2024-07. Removes the "1-yr window only" asterisk from
the deep verdict (TFT-730 ens-3 18.31 vs champion 17.66 was 1-yr only).

One config per architecture (the proven best), 3 seeds, median ensemble,
monthly refits — the same protocol the 18.31 came from. Per-year rows
answer the regime question: do deep models close the gap anywhere?

Run: uv run python -m src.models.deep.run_deep2yr --model tft [--smoke]
     uv run python -m src.models.deep.run_deep2yr --model patchtst
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
from src.models.deep.tft import TFT
from src.models.deep.train import device

TZ = "Europe/Warsaw"
SEEDS = [42, 7, 2026]

# Proven-best configs from the 730d campaigns (sweep730 / capacity730).
TFT_CFG = {"d_model": 128, "n_heads": 8, "lstm_layers": 2,
           "dropout": 0.183, "lr": 1.7e-3, "batch": 32}
PATCHTST_D_MODEL = 128


def make_factory(model: str):
    if model == "tft":
        def make() -> TFT:
            return TFT(enc_feat=1, fut_feat=12, d_model=TFT_CFG["d_model"],
                       n_heads=TFT_CFG["n_heads"],
                       lstm_layers=TFT_CFG["lstm_layers"],
                       dropout=TFT_CFG["dropout"]).to(device())
        return make, {"lr": TFT_CFG["lr"], "batch": TFT_CFG["batch"]}
    def make() -> PatchTST:
        return PatchTST(enc_feat=1, fut_feat=12, d_model=PATCHTST_D_MODEL,
                        patch_len=PATCH_LEN, stride=STRIDE).to(device())
    return make, {}


def score_by_year(pred: pd.DataFrame, price: pd.Series) -> list[dict]:
    years = pd.Index(pred.index.tz_convert(TZ).year)
    rows = []
    for yr in sorted(set(years)):
        sub = pred[np.asarray(years == yr)]
        if sub["p50"].notna().sum() < 24 * 28:
            continue
        rows.append({"period": str(yr), **score_preds(sub, price)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["tft", "patchtst"])
    parser.add_argument("--test-start", default="2024-07-16")
    parser.add_argument("--train-days", type=int, default=730)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    out = Path("reports/sensitivity") / args.model
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    csv = out / f"deep2yr_{args.model}{suffix}.csv"
    net_factory, extra = make_factory(args.model)
    kw = dict(extra)
    if args.smoke:
        kw.update(max_epochs=2, patience=2)

    print(f"[{pd.Timestamp.now()}] deep2yr {args.model} | device={device()} "
          f"test_start={args.test_start} train_days={args.train_days}",
          flush=True)

    price, res, tso = load_inputs()
    all_dates = sorted(set(price.index.tz_convert(TZ).date))
    test_days = [d for d in all_dates
                 if d >= pd.Timestamp(args.test_start).date()]
    if args.smoke:
        test_days = test_days[:35]

    t0 = time.time()
    first_needed = test_days[0] - pd.Timedelta(days=args.train_days + 1)
    master = build_price_samples(
        price, res, tso, [d for d in all_dates if d >= first_needed], CTX, TZ)
    print(f"master: {len(master.days)} days ({time.time() - t0:.0f}s)",
          flush=True)

    done = set()
    if csv.exists():
        done = {(r.config, int(r.seed), str(r.period))
                for r in pd.read_csv(csv).itertuples()}

    def record(config: str, seed: int, pred: pd.DataFrame) -> None:
        rows = [{"period": "pooled", **score_preds(pred, price)}]
        rows += score_by_year(pred, price)
        for r in rows:
            pd.DataFrame([{"config": config, "seed": seed, **r}]).to_csv(
                csv, mode="a", header=not csv.exists(), index=False)
        pooled = rows[0]
        print(f"  -> {config} s{seed} MAE {pooled['mae']:.2f} "
              f"rMAE {pooled['rmae']:.3f} cov {pooled['coverage_80_pct']:.1f}%",
              flush=True)

    cfg_name = f"{args.model}730_2yr"
    for seed in SEEDS:
        if (cfg_name, seed, "pooled") in done:
            print(f"skip {cfg_name} s{seed} (done)", flush=True)
            continue
        print(f"\n=== {cfg_name} seed={seed} ===", flush=True)
        t0 = time.time()
        pred = walk_forward_ablate(
            master, price, "full", seed, test_days,
            train_days=args.train_days, net_factory=net_factory,
            name=f"{cfg_name}_s{seed}", **kw)
        if pred is None:
            print(f"  seed {seed} returned no predictions", flush=True)
            continue
        pred.to_parquet(out / f"preds_{cfg_name}_s{seed}.parquet")
        print(f"  wall {round((time.time() - t0) / 60, 1)} min", flush=True)
        record(cfg_name, seed, pred)

    paths = [out / f"preds_{cfg_name}_s{s}.parquet" for s in SEEDS]
    have = [p for p in paths if p.exists()]
    if len(have) == len(SEEDS) and (f"{cfg_name}_ens3", -1, "pooled") not in done:
        ps = [pd.read_parquet(p) for p in have]
        ens = pd.DataFrame(
            {q: np.median([p[q].to_numpy() for p in ps], axis=0)
             for q in ("p10", "p50", "p90")}, index=ps[0].index)
        ens[:] = np.sort(ens.to_numpy(), axis=1)
        ens.to_parquet(out / f"preds_{cfg_name}_ens3.parquet")
        record(f"{cfg_name}_ens3", -1, ens)

    print(f"[{pd.Timestamp.now()}] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
