"""TFT group-ablation walk-forward — does the best deep model use history?

The PatchTST feature analysis showed its 56-day price-history encoder is
fully redundant with the known-future covariates. Open question: does
TFT — the best deep model (walk-forward MAE 19.71 EUR/MWh) — actually
extract value from the encoder, or does its edge over PatchTST come from
capacity spent on the covariates?

Same protocol as the PatchTST ablation (zero one input group after
standardization, retrain, walk forward), same fut layout, same master-
sample slicing. Model: the TFT HPO winner (ctx=1344h, d_model=128,
2 LSTM layers, ~1.27M params) from data/processed/tft_hpo.db.

Cost control: TFT refits are ~10x PatchTST, so default is 1 seed and a
1-year test window (2025-07-16 on, ~13 monthly refits per run). Groups
run in information order (full, encoder, res_fcst, anchor168, tso_load,
calendar) so an interrupted queue still answers the main question.
The CSV is incremental and idempotent.

Run: uv run python -m src.models.deep.run_tft_ablation
     [--seeds 42] [--smoke]
Expected: ~1 h per (group, seed) on MPS; 6 groups x 1 seed ~ 6 h.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import pandas as pd

from src.models.deep.patchtst_feature_analysis import (
    GROUPS,
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
HPO_DB = "data/processed/tft_hpo.db"

# information order: an interrupted queue still answers the main question
GROUP_ORDER = ["full", "encoder", "res_fcst", "anchor168", "tso_load",
               "calendar"]


def hpo_params() -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name="tft_price_hpo",
                              storage=f"sqlite:///{HPO_DB}")
    return study.best_params


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--groups", nargs="+", default=GROUP_ORDER,
                        choices=GROUP_ORDER)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    p = hpo_params()
    ctx = p["encoder_hours"]
    print(f"[{pd.Timestamp.now()}] TFT ablation | HPO params {p} | "
          f"device={device()} seeds={args.seeds}", flush=True)

    def net_factory() -> TFT:
        return TFT(
            enc_feat=1, fut_feat=12, d_model=p["d_model"],
            n_heads=p["n_heads"], lstm_layers=p["lstm_layers"],
            dropout=p["dropout"],
        ).to(device())

    price, res, tso = load_inputs()
    all_dates = sorted(set(price.index.tz_convert(TZ).date))
    test_start = pd.Timestamp(TEST_START).date()
    test_days = [d for d in all_dates if d >= test_start]
    kw = {}
    if args.smoke:
        test_days = test_days[:35]
        kw = {"max_epochs": 2, "patience": 2}

    suffix = "" if args.train_days == 365 else f"_w{args.train_days}"
    csv = OUT / f"ablation_walkforward{suffix}.csv"
    done = set()
    if csv.exists():
        prev = pd.read_csv(csv)
        done = {(r.group, int(r.seed)) for r in prev.itertuples()}
        print(f"tft ablation: {len(done)} runs already done, skipping them")

    first_needed = test_days[0] - pd.Timedelta(days=args.train_days + 1)
    t0 = time.time()
    master = build_price_samples(
        price, res, tso, [d for d in all_dates if d >= first_needed], ctx, TZ)
    print(f"tft ablation: master samples {len(master.days)} days "
          f"({time.time() - t0:.0f}s)", flush=True)

    for seed in args.seeds:
        for group in args.groups:
            if (group, seed) in done:
                continue
            print(f"\n=== tft ablation {group} seed={seed} "
                  f"w{args.train_days} ===", flush=True)
            t0 = time.time()
            pred = walk_forward_ablate(
                master, price, group, seed, test_days,
                train_days=args.train_days,
                net_factory=net_factory, lr=p["lr"], batch=p["batch"],
                name=f"tft_abl{suffix}_{group}_s{seed}", **kw)
            if pred is None:
                print(f"  {group} s{seed}: no predictions, skipped", flush=True)
                continue
            if group == "full":   # keep hourly preds: ensembling needs them
                pred.to_parquet(
                    OUT / f"preds_full_w{args.train_days}_s{seed}.parquet")
            row = {"group": group, "seed": seed, **score_preds(pred, price),
                   "mean_val_pinball": pred.attrs["mean_val_pinball"],
                   "wall_min": round((time.time() - t0) / 60, 1)}
            pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(),
                                       index=False)
            print(f"  -> MAE {row['mae']:.2f} rMAE {row['rmae']:.3f} "
                  f"({row['wall_min']} min)", flush=True)

    # plot
    if csv.exists():
        df = pd.read_csv(csv)
        agg = df.groupby("group")["mae"].agg(["mean", "std"])
        if "full" in agg.index:
            base = agg.loc["full", "mean"]
            agg["delta_mae"] = agg["mean"] - base
            d = agg.drop(index="full").sort_values("delta_mae", ascending=False)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.barh(d.index, d["delta_mae"], xerr=d["std"].fillna(0),
                    color="#4878a8")
            ax.axvline(0, color="k", lw=0.8)
            ax.set_xlabel("ΔMAE vs full TFT (EUR/MWh) — higher = more important")
            ax.set_title(f"TFT group ablation, walk-forward {TEST_START}→ | "
                         f"train {args.train_days}d | "
                         f"full MAE {base:.2f} EUR/MWh")
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUT / f"ablation_delta_mae{suffix}.png", dpi=150)
            plt.close(fig)
    print(f"[{pd.Timestamp.now()}] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
