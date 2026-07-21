"""PatchTST screening sweep on the day-ahead price task.

Hypothesis: patching allows attention over months of price history
without quadratic attention cost. Test: does the patch mechanism improve
on the raw TFT result at similar or smaller parameter counts?

Sweep design (screening tier — 1 seed, single split):
  patch_len : {12, 24, 48}  — hours per patch (grain of attention)
  stride    : {6, 12, 24}   — hop size; smaller = more overlapping patches
  ctx       : {672, 1344, 2016} — total encoder hours (28d, 56d, 84d)

All combinations = 27 configs. Fast because PatchTST is ~46k params vs
TFT's ~725k. Filter: keep configs where n_patches ≤ 256 (attention cost).

After screening: take top-3 by val pinball, run walk-forward, write
comparison table vs TFT HPO winner and tabular models.

Run: uv run python -m src.models.deep.run_patchtst_sweep
     [--seed 42]
     [--d_model 64]   # fixed for this sweep; HPO if needed
Expected: ~2-4 h on MPS (27 configs × ~5 min).
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.models.deep.data import apply_covariate_stats, standardize_covariates
from src.models.deep.patchtst import PatchTST
from src.models.deep.price_data import build_price_samples
from src.models.deep.train import device, train_variant, predict_mw
from src.pipeline.daily_run import local_day_hours_utc

TZ = "Europe/Warsaw"
SEED = 42
TRAIN_END = "2026-01-01"
MAX_PATCHES = 256


def _samples(price, res, tso, encoder_hours):
    all_dates = sorted(set(price.index.tz_convert(TZ).date))
    split = pd.Timestamp(TRAIN_END).date()
    last = all_dates[-2]
    train_dates = [d for d in all_dates if d < split]
    val_dates = [d for d in all_dates if split <= d <= last]
    tr = build_price_samples(price, res, tso, train_dates, encoder_hours, TZ)
    va = build_price_samples(price, res, tso, val_dates, encoder_hours, TZ)
    stats = standardize_covariates(tr, va, n_tail=1)
    return tr, va, stats


def walk_forward_patchtst(
    price, res, tso, patch_len, stride, encoder_hours, d_model, seed,
    test_days, all_dates, train_days=365, refit_every=30,
):
    name = f"patch{patch_len}_s{stride}_ctx{encoder_hours}"
    ckpt_dir = Path("data/processed/patchtst_ckpts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    net = None
    last_refit = None
    stats = None
    preds = []

    for test_day in test_days:
        needs_refit = net is None or (test_day - last_refit).days >= refit_every
        if needs_refit:
            t_end = test_day - pd.Timedelta(days=1)
            t_start = test_day - pd.Timedelta(days=train_days)
            window = [d for d in all_dates if t_start <= d <= t_end]
            split = int(0.85 * len(window))
            tr = build_price_samples(price, res, tso, window[:split], encoder_hours, TZ)
            va = build_price_samples(price, res, tso, window[split:], encoder_hours, TZ)
            if len(tr.days) < 100 or len(va.days) < 10:
                continue
            stats = standardize_covariates(tr, va, n_tail=1)
            net = PatchTST(
                enc_feat=1,
                fut_feat=tr.fut.shape[-1],
                d_model=d_model,
                patch_len=patch_len,
                stride=stride,
            ).to(device())
            result = train_variant(
                net, tr, va,
                checkpoint=str(ckpt_dir / f"{name}_s{seed}.pt"),
                seed=seed, max_epochs=60, patience=8, lr=5e-4, batch=32,
            )
            print(
                f"  [{name}] @{test_day} val={result['best_val_pinball_norm']:.4f} "
                f"epoch={result['best_epoch']} {result['train_s']:.0f}s",
                flush=True,
            )
            last_refit = test_day

        if net is None or stats is None:
            continue
        sample = build_price_samples(price, res, tso, [test_day], encoder_hours, TZ)
        if len(sample.days) == 0:
            continue
        apply_covariate_stats(sample, stats)
        p = predict_mw(net, sample)[0]
        hours = local_day_hours_utc(pd.Timestamp(test_day, tz=TZ), TZ)
        if len(hours) == 24:
            preds.append(
                pd.DataFrame({"p10": p[:, 0], "p50": p[:, 1], "p90": p[:, 2]}, index=hours)
            )

    if not preds:
        return None
    return pd.concat(preds).sort_index()


def _screen_config(price, res, tso, patch_len, stride, encoder_hours, d_model, seed, cache):
    n_patches = (encoder_hours - patch_len) // stride + 1
    if n_patches > MAX_PATCHES:
        return None, f"n_patches={n_patches} > {MAX_PATCHES}, skip"

    if encoder_hours not in cache:
        cache[encoder_hours] = _samples(price, res, tso, encoder_hours)
    tr, va, _ = cache[encoder_hours]

    net = PatchTST(
        enc_feat=1, fut_feat=tr.fut.shape[-1], d_model=d_model,
        patch_len=patch_len, stride=stride,
    ).to(device())
    n_params = sum(p.numel() for p in net.parameters())
    result = train_variant(
        net, tr, va,
        checkpoint=f"data/processed/patchtst_ckpts/screen_p{patch_len}_s{stride}_ctx{encoder_hours}.pt",
        seed=seed, max_epochs=50, patience=6, lr=5e-4, batch=32,
    )
    return result["best_val_pinball_norm"], (
        f"patch={patch_len} stride={stride} ctx={encoder_hours} "
        f"n_patches={n_patches} params={n_params} "
        f"val={result['best_val_pinball_norm']:.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--walkforward", action="store_true",
                        help="After screening, run walk-forward on top-3 configs")
    args = parser.parse_args()

    Path("data/processed/patchtst_ckpts").mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    proc = cfg.paths["data_processed"]
    price = pd.read_parquet(proc / "price_da_eur.parquet").iloc[:, 0]
    res = pd.read_parquet(proc / "res_forecast.parquet")
    tso = pd.read_parquet(proc / "tso_forecast.parquet").iloc[:, 0]

    patch_lens = [12, 24, 48]
    strides = [6, 12, 24]
    contexts = [672, 1344, 2016]
    configs = list(itertools.product(patch_lens, strides, contexts))
    print(f"PatchTST sweep: {len(configs)} configs | d_model={args.d_model} seed={args.seed}")

    cache: dict = {}
    rows = []
    t0 = time.time()

    for patch_len, stride, ctx in configs:
        label = f"patch{patch_len}_s{stride}_ctx{ctx}"
        print(f"\n--- {label} ---", flush=True)
        val, msg = _screen_config(price, res, tso, patch_len, stride, ctx,
                                  args.d_model, args.seed, cache)
        print(msg)
        if val is not None:
            rows.append({
                "config": label, "patch_len": patch_len, "stride": stride,
                "ctx": ctx, "val_pinball": val,
            })

    screen_h = (time.time() - t0) / 3600
    df = pd.DataFrame(rows).sort_values("val_pinball")
    stamp = f"{pd.Timestamp.now(TZ).date()}_patchtst_sweep"
    out = Path("reports/backtests")
    df.to_csv(out / f"{stamp}.csv", index=False)

    print(f"\n=== SCREENING DONE ({screen_h:.1f} h) ===")
    print(df.to_string(index=False))
    print(f"\nTop-3:\n{df.head(3).to_string(index=False)}")

    if args.walkforward and len(df) > 0:
        all_dates = sorted(set(price.index.tz_convert(TZ).date))
        last = all_dates[-2]
        test_start = pd.Timestamp("2024-07-16").date()
        test_days = [d for d in all_dates if d >= test_start]

        wf_rows = []
        for _, row in df.head(3).iterrows():
            p, s, c = int(row["patch_len"]), int(row["stride"]), int(row["ctx"])
            print(f"\n=== Walk-forward: patch{p}_s{s}_ctx{c} ===", flush=True)
            pred = walk_forward_patchtst(
                price, res, tso, p, s, c, args.d_model, args.seed,
                test_days, all_dates,
            )
            if pred is None:
                continue
            y = price.reindex(pred.index)
            naive1d = price.reindex(pred.index - pd.Timedelta(hours=24))
            naive1d.index = pred.index
            mae = float((pred["p50"] - y).abs().mean())
            rmae = mae / float((naive1d - y).abs().mean())
            spike_cut = y.quantile(0.95)
            spike = y >= spike_cut
            cov = 100.0 * ((y >= pred["p10"]) & (y <= pred["p90"])).mean()
            wf_rows.append({
                "model": f"patchtst_p{p}_s{s}_ctx{c}",
                "mae": mae, "rmae": rmae,
                "coverage_80_pct": cov,
                "spike_mae": float((pred.loc[spike, "p50"] - y[spike]).abs().mean()),
            })

        if wf_rows:
            wf_df = pd.DataFrame(wf_rows).sort_values("mae")
            wf_df.to_csv(out / f"{stamp}_walkforward.csv", index=False)
            print(f"\n=== WALK-FORWARD ===\n{wf_df.to_string(index=False)}")

    md = [
        f"# PatchTST screening sweep — {stamp}",
        "",
        f"Seed={args.seed} d_model={args.d_model} | {len(configs)} configs | {screen_h:.1f} h",
        "",
        "## Top 10 (val pinball, lower = better)",
        "",
        df.head(10).round(4).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- Compare best PatchTST val vs best TFT HPO val (0.1184) to gauge",
        "  whether patching adds value beyond the TFT architecture.",
        "- Walk-forward the top-3 with --walkforward flag.",
        "- Quote ONLY walk-forward numbers in results tables.",
    ]
    (out / f"{stamp}.md").write_text("\n".join(md))
    print(f"\nSaved → {out}/{stamp}.(csv|md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
