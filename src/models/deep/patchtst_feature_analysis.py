"""PatchTST feature analysis — explain the negative result.

PatchTST lost the walk-forward (MAE 22.98 vs TFT 19.71, LGBM 17.8).
This script answers: WHERE does the signal come from, and what does the
model actually use? Four analyses on the best config (patch24_s24_ctx1344):

1. ablation  — group-ablation walk-forward. Zero one input group
   (calendar, RES forecast, TSO load, anchor, whole encoder) after
   standardization, retrain, measure MAE delta. 3 seeds.
2. perm      — permutation importance on the screening split
   (train < 2026-01-01, val 2026+) with the screening checkpoint.
3. pca       — PCA on raw 24h price patches and on the learned pooled
   representation. Shows what the encoder compresses.
4. attention — mean attention map over val days. With patch=stride=24h,
   one patch = one day: the map shows which past days the model reads.

Outputs → reports/sensitivity/patchtst/. Ablation CSV is written
incrementally; finished (group, seed) pairs are skipped on restart.

Run: uv run python -m src.models.deep.patchtst_feature_analysis
     [--stage all|ablation|perm|pca|attention|report]
     [--seeds 42 7 2026] [--smoke]
Expected: ablation ~6 h on MPS (18 walk-forwards), rest ~20 min.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.models.deep.data import (
    FUTURE_CAL_COLS,
    DaySamples,
    apply_covariate_stats,
    standardize_covariates,
)
from src.models.deep.patchtst import PatchTST
from src.models.deep.price_data import build_price_samples
from src.models.deep.train import QUANTILES, device, pinball, predict_mw, train_variant
from src.pipeline.daily_run import local_day_hours_utc

TZ = "Europe/Warsaw"
OUT = Path("reports/sensitivity/patchtst")
CKPT_DIR = Path("data/processed/patchtst_ckpts")

# Best sweep config (reports/backtests/2026-07-17_patchtst_sweep.csv, rank 1)
PATCH_LEN = 24
STRIDE = 24
CTX = 1344
D_MODEL = 64
TRAIN_END = "2026-01-01"          # screening split boundary
TEST_START = "2024-07-16"         # walk-forward test period start

# fut column layout, must match build_price_samples order
FUT_COLS = FUTURE_CAL_COLS + [
    "solar_fcst_mw", "wind_on_fcst_mw", "wind_off_fcst_mw",
    "tso_load_fcst", "price_anchor_lag168",
]

# name -> fut column indices; "encoder" zeroes the past-price channel instead
GROUPS: dict[str, list[int]] = {
    "calendar": list(range(0, 7)),
    "res_fcst": [7, 8, 9],
    "solar": [7],          # res_fcst split: which RES component carries it?
    "wind_on": [8],        # (wind_off is structurally zero pre-2026 — skipped)
    "tso_load": [10],
    "anchor168": [11],
    "encoder": [],
}


def slice_samples(master: DaySamples, idx: list[int]) -> DaySamples:
    """Clone a subset of days out of a master sample set. Clones tensors so
    per-refit standardization and ablation never touch the master copy."""
    ii = torch.tensor(idx, dtype=torch.long)
    return DaySamples(
        enc=master.enc[ii].clone(), fut=master.fut[ii].clone(),
        y=master.y[ii].clone(), anchor=master.anchor[ii].clone(),
        mean=master.mean[ii].clone(), std=master.std[ii].clone(),
        days=[master.days[i] for i in idx],
    )


def zero_group(s: DaySamples, group: str) -> None:
    """Ablate one input group in place. Call AFTER covariate standardization
    so zero = 'held at training mean' for z-scored columns."""
    if group == "full":
        return
    if group == "encoder":
        s.enc.zero_()
        return
    for c in GROUPS[group]:
        s.fut[:, :, c] = 0.0


def load_inputs():
    cfg = load_config()
    proc = cfg.paths["data_processed"]
    price = pd.read_parquet(proc / "price_da_eur.parquet").iloc[:, 0]
    res = pd.read_parquet(proc / "res_forecast.parquet")
    tso = pd.read_parquet(proc / "tso_forecast.parquet").iloc[:, 0]
    return price, res, tso


def make_net() -> PatchTST:
    return PatchTST(
        enc_feat=1, fut_feat=len(FUT_COLS), d_model=D_MODEL,
        patch_len=PATCH_LEN, stride=STRIDE,
    ).to(device())


def screening_split(price, res, tso):
    """Same split as run_patchtst_sweep._samples: train < 2026, val 2026+."""
    all_dates = sorted(set(price.index.tz_convert(TZ).date))
    split = pd.Timestamp(TRAIN_END).date()
    last = all_dates[-2]
    train_dates = [d for d in all_dates if d < split]
    val_dates = [d for d in all_dates if split <= d <= last]
    tr = build_price_samples(price, res, tso, train_dates, CTX, TZ)
    va = build_price_samples(price, res, tso, val_dates, CTX, TZ)
    stats = standardize_covariates(tr, va, n_tail=1)
    return tr, va, stats


# ---------------------------------------------------------------- ablation


def walk_forward_ablate(
    master: DaySamples, price, group: str, seed: int,
    test_days, train_days=365, refit_every=30,
    max_epochs=60, patience=8,
    net_factory=None, lr=5e-4, batch=32, name=None,
) -> pd.DataFrame | None:
    """Walk-forward identical to run_patchtst_sweep.walk_forward_patchtst,
    plus zero_group() on every sample set. Slices a prebuilt master sample
    set instead of rebuilding samples per refit (same tensors, ~10x faster).
    net_factory: () -> nn.Module; defaults to the PatchTST best config.
    Returns hourly p10/p50/p90."""
    if net_factory is None:
        net_factory = make_net
    name = name or f"abl_{group}_s{seed}"
    day_pos = {d: i for i, d in enumerate(master.days)}
    net, last_refit, stats, preds, val_scores = None, None, None, [], []

    for test_day in test_days:
        needs_refit = net is None or (test_day - last_refit).days >= refit_every
        if needs_refit:
            t_end = test_day - pd.Timedelta(days=1)
            t_start = test_day - pd.Timedelta(days=train_days)
            window = [i for d, i in day_pos.items() if t_start <= d <= t_end]
            split = int(0.85 * len(window))
            tr = slice_samples(master, window[:split])
            va = slice_samples(master, window[split:])
            if len(tr.days) < 100 or len(va.days) < 10:
                continue
            stats = standardize_covariates(tr, va, n_tail=1)
            zero_group(tr, group)
            zero_group(va, group)
            torch.manual_seed(seed)  # net INIT must be seeded too, not only training
            net = net_factory()
            result = train_variant(
                net, tr, va,
                checkpoint=str(CKPT_DIR / f"{name}.pt"),
                seed=seed, max_epochs=max_epochs, patience=patience,
                lr=lr, batch=batch,
            )
            val_scores.append(result["best_val_pinball_norm"])
            print(f"  [{name}] @{test_day} val={result['best_val_pinball_norm']:.4f} "
                  f"epoch={result['best_epoch']} {result['train_s']:.0f}s", flush=True)
            last_refit = test_day

        if net is None or stats is None or test_day not in day_pos:
            continue
        sample = slice_samples(master, [day_pos[test_day]])
        apply_covariate_stats(sample, stats)
        zero_group(sample, group)
        p = predict_mw(net, sample)[0]
        hours = local_day_hours_utc(pd.Timestamp(test_day, tz=TZ), TZ)
        if len(hours) == 24:
            preds.append(pd.DataFrame(
                {"p10": p[:, 0], "p50": p[:, 1], "p90": p[:, 2]}, index=hours))

    if not preds:
        return None
    df = pd.concat(preds).sort_index()
    df.attrs["mean_val_pinball"] = float(np.mean(val_scores)) if val_scores else np.nan
    return df


def score_preds(pred: pd.DataFrame, price: pd.Series) -> dict:
    y = price.reindex(pred.index)
    naive = price.reindex(pred.index - pd.Timedelta(hours=24))
    naive.index = pred.index
    mae = float((pred["p50"] - y).abs().mean())
    spike = y >= y.quantile(0.95)
    return {
        "mae": mae,
        "rmae": mae / float((naive - y).abs().mean()),
        "coverage_80_pct": 100.0 * float(((y >= pred["p10"]) & (y <= pred["p90"])).mean()),
        "spike_mae": float((pred.loc[spike, "p50"] - y[spike]).abs().mean()),
        "n_hours": int(len(y)),
    }


def stage_ablation(price, res, tso, seeds: list[int], smoke: bool,
                   train_days: int = 365) -> None:
    """train_days != 365 writes to a suffixed CSV (robustness check: is the
    group ranking stable when the training window doubles?). The test
    period then starts 2025-07-16 so the longer window has history."""
    suffix = "" if train_days == 365 else f"_w{train_days}"
    csv = OUT / f"ablation_walkforward{suffix}.csv"
    test_start_s = TEST_START if train_days == 365 else "2025-07-16"
    done = set()
    if csv.exists():
        prev = pd.read_csv(csv)
        done = {(r.group, int(r.seed)) for r in prev.itertuples()}
        print(f"ablation: {len(done)} runs already done, skipping them")

    all_dates = sorted(set(price.index.tz_convert(TZ).date))
    test_start = pd.Timestamp(test_start_s).date()
    test_days = [d for d in all_dates if d >= test_start]
    kw = {}
    if smoke:
        test_days = test_days[:35]
        kw = {"max_epochs": 2, "patience": 2}

    # one master build for every walk-forward run (raw covariates, no stats)
    first_needed = test_days[0] - pd.Timedelta(days=train_days + 1)
    t0 = time.time()
    master = build_price_samples(
        price, res, tso, [d for d in all_dates if d >= first_needed], CTX, TZ)
    print(f"ablation: master samples {len(master.days)} days "
          f"({time.time() - t0:.0f}s)", flush=True)

    groups = ["full"] + list(GROUPS.keys())
    for seed in seeds:                      # seed-major: full picture per seed
        for group in groups:
            if (group, seed) in done:
                continue
            print(f"\n=== ablation {group} seed={seed} w{train_days} ===",
                  flush=True)
            t0 = time.time()
            pred = walk_forward_ablate(master, price, group, seed,
                                       test_days, train_days=train_days,
                                       name=f"abl{suffix}_{group}_s{seed}",
                                       **kw)
            if pred is None:
                print(f"  {group} s{seed}: no predictions, skipped", flush=True)
                continue
            if group == "full":   # keep hourly preds: ensembling needs them
                pred.to_parquet(OUT / f"preds_full_w{train_days}_s{seed}.parquet")
            row = {"group": group, "seed": seed, **score_preds(pred, price),
                   "mean_val_pinball": pred.attrs["mean_val_pinball"],
                   "wall_min": round((time.time() - t0) / 60, 1)}
            pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(),
                                       index=False)
            print(f"  -> MAE {row['mae']:.2f} rMAE {row['rmae']:.3f} "
                  f"({row['wall_min']} min)", flush=True)

    _plot_ablation(csv, suffix, test_start_s)


def _plot_ablation(csv: Path, suffix: str = "", test_start: str = TEST_START,
                   ) -> None:
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    agg = df.groupby("group")["mae"].agg(["mean", "std", "count"])
    if "full" not in agg.index:
        return
    base = agg.loc["full", "mean"]
    agg["delta_mae"] = agg["mean"] - base
    agg = agg.drop(index="full").sort_values("delta_mae", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(agg.index, agg["delta_mae"], xerr=agg["std"], color="#4878a8")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("ΔMAE vs full model (EUR/MWh) — higher = more important")
    ax.set_title(f"PatchTST group ablation, walk-forward {test_start}→ | "
                 f"full MAE {base:.2f} EUR/MWh")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"ablation_delta_mae{suffix}.png", dpi=150)
    plt.close(fig)


def stage_window(price, res, tso, seeds: list[int], smoke: bool) -> None:
    """Test the stated root cause of the PatchTST loss: '365-day training
    windows overfit'. Same walk-forward, train_days in {365, 730}, full
    inputs, paired test days (2025-07-16 on, so 730d of history exists)."""
    csv = OUT / "window_walkforward.csv"
    done = set()
    if csv.exists():
        prev = pd.read_csv(csv)
        done = {(int(r.train_days), int(r.seed)) for r in prev.itertuples()}
        print(f"window: {len(done)} runs already done, skipping them")

    all_dates = sorted(set(price.index.tz_convert(TZ).date))
    test_start = pd.Timestamp("2025-07-16").date()
    test_days = [d for d in all_dates if d >= test_start]
    kw = {}
    if smoke:
        test_days = test_days[:35]
        kw = {"max_epochs": 2, "patience": 2}

    first_needed = test_days[0] - pd.Timedelta(days=731)
    t0 = time.time()
    master = build_price_samples(
        price, res, tso, [d for d in all_dates if d >= first_needed], CTX, TZ)
    print(f"window: master samples {len(master.days)} days "
          f"({time.time() - t0:.0f}s)", flush=True)

    for seed in seeds:
        for train_days in (365, 730):
            if (train_days, seed) in done:
                continue
            print(f"\n=== window train_days={train_days} seed={seed} ===",
                  flush=True)
            t0 = time.time()
            pred = walk_forward_ablate(
                master, price, "full", seed, test_days,
                train_days=train_days, name=f"win{train_days}_s{seed}", **kw)
            if pred is None:
                continue
            row = {"train_days": train_days, "seed": seed,
                   **score_preds(pred, price),
                   "mean_val_pinball": pred.attrs["mean_val_pinball"],
                   "wall_min": round((time.time() - t0) / 60, 1)}
            pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(),
                                       index=False)
            print(f"  -> MAE {row['mae']:.2f} rMAE {row['rmae']:.3f} "
                  f"({row['wall_min']} min)", flush=True)


# ---------------------------------------------------------------- perm


def _val_scores(net, va: DaySamples) -> tuple[float, float]:
    """(pinball_norm, mae_eur) on a sample set."""
    dev = device()
    net.eval()
    with torch.no_grad():
        out = net(va.enc.to(dev), va.fut.to(dev), va.anchor.to(dev)).cpu()
    pb = float(pinball(out, va.y, QUANTILES))
    mw = out * va.std.view(-1, 1, 1) + va.mean.view(-1, 1, 1)
    mw, _ = torch.sort(mw, dim=-1)
    y_mw = va.y * va.std.view(-1, 1) + va.mean.view(-1, 1)
    mae = float((mw[:, :, 1] - y_mw).abs().mean())
    return pb, mae


def stage_perm(price, res, tso, smoke: bool, n_rep: int = 10) -> None:
    tr, va, _ = screening_split(price, res, tso)
    ckpt = CKPT_DIR / f"screen_p{PATCH_LEN}_s{STRIDE}_ctx{CTX}.pt"
    net = make_net()
    if ckpt.exists():
        net.load_state_dict(torch.load(ckpt, map_location=device()))
        print(f"perm: loaded {ckpt}")
    else:
        print(f"perm: {ckpt} missing, retraining")
        train_variant(net, tr, va, checkpoint=str(ckpt), seed=42,
                      max_epochs=2 if smoke else 50, patience=6, lr=5e-4, batch=32)

    if smoke:
        n_rep = 2
    base_pb, base_mae = _val_scores(net, va)
    print(f"perm: baseline val pinball {base_pb:.4f}  MAE {base_mae:.2f} EUR/MWh "
          f"({len(va.days)} val days)")

    rng = np.random.default_rng(42)
    rows = []
    features = [("enc_price_history", "enc", None)] + [
        (FUT_COLS[c], "fut", c) for c in range(len(FUT_COLS))
    ]
    for fname, kind, col in features:
        d_pb, d_mae = [], []
        for _ in range(n_rep):
            vp = copy.deepcopy(va)
            perm = torch.from_numpy(rng.permutation(len(vp.days)))
            if kind == "enc":
                vp.enc[:] = vp.enc[perm]
            else:
                vp.fut[:, :, col] = vp.fut[perm][:, :, col]
            pb, mae = _val_scores(net, vp)
            d_pb.append(pb - base_pb)
            d_mae.append(mae - base_mae)
        rows.append({"feature": fname,
                     "delta_pinball": float(np.mean(d_pb)),
                     "delta_pinball_std": float(np.std(d_pb)),
                     "delta_mae_eur": float(np.mean(d_mae)),
                     "delta_mae_std": float(np.std(d_mae))})
        print(f"  {fname:24s} Δpinball {rows[-1]['delta_pinball']:+.4f} "
              f"ΔMAE {rows[-1]['delta_mae_eur']:+.2f}", flush=True)

    df = pd.DataFrame(rows).sort_values("delta_pinball", ascending=False)
    df.to_csv(OUT / "permutation_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    d = df.sort_values("delta_pinball")
    ax.barh(d["feature"], d["delta_pinball"], xerr=d["delta_pinball_std"],
            color="#4878a8")
    ax.set_xlabel("Δ val pinball when permuted (normalized units)")
    ax.set_title(f"PatchTST permutation importance | val 2026+, {n_rep} shuffles\n"
                 f"baseline pinball {base_pb:.4f}, MAE {base_mae:.1f} EUR/MWh")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "permutation_importance.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- pca


def stage_pca(price, res, tso, smoke: bool) -> None:
    from sklearn.decomposition import PCA

    tr, va, _ = screening_split(price, res, tso)
    ckpt = CKPT_DIR / f"screen_p{PATCH_LEN}_s{STRIDE}_ctx{CTX}.pt"
    net = make_net()
    if ckpt.exists():
        net.load_state_dict(torch.load(ckpt, map_location=device()))

    # 1. raw patches: every 24h standardized price patch in val encoders
    with torch.no_grad():
        patches = va.enc.permute(0, 2, 1).reshape(-1, CTX)
        patches = patches.unfold(1, PATCH_LEN, STRIDE).reshape(-1, PATCH_LEN).numpy()
    pca_in = PCA(n_components=10).fit(patches)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(np.cumsum(pca_in.explained_variance_ratio_) * 100, marker="o")
    axes[0].set_xlabel("component")
    axes[0].set_ylabel("cumulative explained variance (%)")
    axes[0].set_title("PCA of raw 24h price patches (val, standardized)")
    axes[0].grid(alpha=0.3)
    for i in range(4):
        axes[1].plot(pca_in.components_[i], label=f"PC{i+1} "
                     f"({pca_in.explained_variance_ratio_[i]*100:.0f}%)")
    axes[1].set_xlabel("hour of patch (patch = 1 local day)")
    axes[1].set_title("top-4 patch components = daily price shapes")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "pca_patches.png", dpi=150)
    plt.close(fig)

    pd.DataFrame(
        pca_in.components_[:4].T,
        columns=[f"PC{i+1}" for i in range(4)],
    ).to_csv(OUT / "pca_patch_components.csv", index_label="hour")

    # 2. learned pooled representation per val day
    dev = device()
    net.eval()
    with torch.no_grad():
        B, T, C = va.enc.shape
        chans = va.enc.permute(0, 2, 1).reshape(B * C, T).to(dev)
        p = chans.unfold(1, PATCH_LEN, STRIDE)
        tok = net.patch_embed(p) + net.pos[:, : p.shape[1]]
        z = net.encoder(tok)
        pooled = z.mean(dim=1).reshape(B, C, -1).mean(dim=1).cpu().numpy()

    pca_rep = PCA(n_components=min(10, pooled.shape[1])).fit(pooled)
    proj = pca_rep.transform(pooled)
    days = pd.to_datetime([str(d) for d in va.days])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sc = axes[0].scatter(proj[:, 0], proj[:, 1], c=days.month, cmap="viridis", s=18)
    plt.colorbar(sc, ax=axes[0], label="month")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("pooled encoder representation per val day")
    wd = days.weekday
    sc2 = axes[1].scatter(proj[:, 0], proj[:, 1], c=wd, cmap="coolwarm", s=18)
    plt.colorbar(sc2, ax=axes[1], label="weekday (0=Mon)")
    axes[1].set_xlabel("PC1")
    axes[1].set_title("same, colored by weekday")
    fig.suptitle(f"PatchTST learned representations | "
                 f"PC1+PC2 = {sum(pca_rep.explained_variance_ratio_[:2])*100:.0f}% var")
    fig.tight_layout()
    fig.savefig(OUT / "pca_representations.png", dpi=150)
    plt.close(fig)

    pd.DataFrame({
        "day": [str(d) for d in va.days],
        "pc1": proj[:, 0], "pc2": proj[:, 1],
    }).to_csv(OUT / "pca_representation_proj.csv", index=False)
    ev = pd.DataFrame({
        "component": range(1, len(pca_rep.explained_variance_ratio_) + 1),
        "explained_var_ratio_reps": pca_rep.explained_variance_ratio_,
    })
    ev["explained_var_ratio_patches"] = pca_in.explained_variance_ratio_[: len(ev)]
    ev.to_csv(OUT / "pca_explained_variance.csv", index=False)
    print(f"pca: patches PC1-4 explain "
          f"{sum(pca_in.explained_variance_ratio_[:4])*100:.0f}%, "
          f"reps PC1-2 explain "
          f"{sum(pca_rep.explained_variance_ratio_[:2])*100:.0f}%")


# ---------------------------------------------------------------- attention


def stage_attention(price, res, tso, smoke: bool) -> None:
    tr, va, _ = screening_split(price, res, tso)
    ckpt = CKPT_DIR / f"screen_p{PATCH_LEN}_s{STRIDE}_ctx{CTX}.pt"
    net = make_net()
    if ckpt.exists():
        net.load_state_dict(torch.load(ckpt, map_location=device()))
    dev = device()
    net.eval()

    B, T, C = va.enc.shape
    if smoke:
        B = min(B, 8)
    x = va.enc[:B].permute(0, 2, 1).reshape(B * C, T).to(dev)
    p = x.unfold(1, PATCH_LEN, STRIDE)
    n_patches = p.shape[1]

    maps = []
    with torch.no_grad():
        h = net.patch_embed(p) + net.pos[:, :n_patches]
        for layer in net.encoder.layers:
            a = layer.norm1(h)
            attn_out, attn_w = layer.self_attn(
                a, a, a, need_weights=True, average_attn_weights=True)
            maps.append(attn_w.mean(dim=0).cpu().numpy())   # (P, P)
            h = h + attn_out
            h = h + layer._ff_block(layer.norm2(h))

    # patch index -> age in days (patch 0 = oldest). age of patch i:
    ages = [(n_patches - 1 - i) for i in range(n_patches)]  # days before delivery-1

    fig, axes = plt.subplots(1, len(maps) + 1, figsize=(5 * (len(maps) + 1), 4.2))
    for li, m in enumerate(maps):
        im = axes[li].imshow(m, aspect="auto", cmap="viridis", origin="lower")
        axes[li].set_title(f"layer {li} mean attention")
        axes[li].set_xlabel("key patch (0 = oldest day)")
        axes[li].set_ylabel("query patch")
        plt.colorbar(im, ax=axes[li])
    recv = maps[-1].mean(axis=0)                            # attention per key patch
    axes[-1].bar(ages, recv, color="#4878a8")
    axes[-1].set_xlabel("day age (days before cutoff; 0 = yesterday)")
    axes[-1].set_ylabel("mean attention received (last layer)")
    axes[-1].set_title("which past days the model reads")
    axes[-1].invert_xaxis()
    axes[-1].grid(alpha=0.3)
    fig.suptitle(f"PatchTST attention | patch=stride=24h → 1 patch = 1 day | "
                 f"ctx {CTX}h = {n_patches} days | {B} val days avg")
    fig.tight_layout()
    fig.savefig(OUT / "attention_patterns.png", dpi=150)
    plt.close(fig)

    pd.DataFrame({"day_age": ages, "mean_attention_last_layer": recv}).to_csv(
        OUT / "attention_by_day_age.csv", index=False)
    top = np.argsort(recv)[::-1][:5]
    print("attention: top-5 most-attended day ages:",
          [ages[i] for i in top])


# ---------------------------------------------------------------- report


def stage_report() -> None:
    lines = [
        "# PatchTST feature analysis",
        "",
        f"Config: patch{PATCH_LEN}_s{STRIDE}_ctx{CTX}, d_model={D_MODEL}.",
        "Context: PatchTST lost the 2-year walk-forward "
        "(MAE 22.98 vs TFT 19.71 vs LGBM 17.8 EUR/MWh).",
        "This analysis shows where its signal comes from.",
        "",
    ]
    abl = OUT / "ablation_walkforward.csv"
    if abl.exists():
        df = pd.read_csv(abl)
        agg = df.groupby("group").agg(
            mae_mean=("mae", "mean"), mae_std=("mae", "std"),
            rmae_mean=("rmae", "mean"), cov_mean=("coverage_80_pct", "mean"),
            seeds=("seed", "count"),
        ).sort_values("mae_mean")
        if "full" in agg.index:
            agg["delta_mae"] = agg["mae_mean"] - agg.loc["full", "mae_mean"]
        lines += ["## Group ablation (walk-forward, 3 seeds)", "",
                  "Zero one input group, retrain, rerun 2-year walk-forward.",
                  "ΔMAE vs full = importance of that group.", "",
                  agg.round(3).to_markdown(), "",
                  "![ablation](ablation_delta_mae.png)", ""]
    perm = OUT / "permutation_importance.csv"
    if perm.exists():
        df = pd.read_csv(perm)
        lines += ["## Permutation importance (screening split, val 2026+)", "",
                  df.round(4).to_markdown(index=False), "",
                  "![perm](permutation_importance.png)", ""]
    att = OUT / "attention_by_day_age.csv"
    if att.exists():
        df = pd.read_csv(att).sort_values("mean_attention_last_layer",
                                          ascending=False)
        lines += ["## Attention", "",
                  "Top-5 most-attended past days (last layer):", "",
                  df.head(5).round(4).to_markdown(index=False), "",
                  "![attention](attention_patterns.png)", ""]
    if (OUT / "pca_explained_variance.csv").exists():
        lines += ["## PCA", "",
                  "![patches](pca_patches.png)", "",
                  "![reps](pca_representations.png)", ""]
    # auto_report.md, not README.md: README holds the hand-written analysis
    (OUT / "auto_report.md").write_text("\n".join(lines))
    print(f"report -> {OUT}/auto_report.md")


# ---------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all",
                        choices=["all", "ablation", "window", "perm", "pca",
                                 "attention", "report"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2026])
    parser.add_argument("--train-days", type=int, default=365,
                        help="ablation stage only: walk-forward train window")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run to verify code paths")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{pd.Timestamp.now()}] stage={args.stage} seeds={args.seeds} "
          f"smoke={args.smoke} device={device()}", flush=True)

    price, res, tso = load_inputs()
    n_fut_expected = len(FUT_COLS)
    probe = build_price_samples(
        price, res, tso,
        sorted(set(price.index.tz_convert(TZ).date))[400:401], CTX, TZ)
    assert probe.fut.shape[-1] == n_fut_expected, (
        f"fut layout changed: {probe.fut.shape[-1]} != {n_fut_expected}")

    # cheap stages first: guaranteed artifacts even if ablation dies
    if args.stage in ("all", "perm"):
        stage_perm(price, res, tso, args.smoke)
    if args.stage in ("all", "pca"):
        stage_pca(price, res, tso, args.smoke)
    if args.stage in ("all", "attention"):
        stage_attention(price, res, tso, args.smoke)
    if args.stage in ("all", "ablation"):
        stage_ablation(price, res, tso, args.seeds, args.smoke,
                       train_days=args.train_days)
    if args.stage == "window":       # not in "all": separate follow-up question
        stage_window(price, res, tso, args.seeds, args.smoke)
    stage_report()
    print(f"[{pd.Timestamp.now()}] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
