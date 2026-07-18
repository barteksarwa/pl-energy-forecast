"""PatchTST feature analysis — overnight run (~12h).

Sections (run in order, or select with --section):
  1. permutation  — permutation importance using last checkpoint (~30 min)
  2. attention    — attention weight visualization (~20 min)
  3. pca          — PCA on fut covariates + encoder representations (~10 min)
  4. ablation     — feature-group ablation walk-forward (~7.5 h, 5 runs × 90 min)

Usage:
  uv run python -m src.models.deep.patchtst_feature_analysis
  uv run python -m src.models.deep.patchtst_feature_analysis --section permutation
  uv run python -m src.models.deep.patchtst_feature_analysis --section ablation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import load_config
from src.models.deep.data import standardize_covariates, apply_covariate_stats
from src.models.deep.patchtst import PatchTST
from src.models.deep.price_data import build_price_samples, TARGET_HOURS
from src.models.deep.run_patchtst_sweep import walk_forward_patchtst
from src.pipeline.daily_run import local_day_hours_utc

OUT = Path("reports/figures/patchtst_features")
OUT.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("reports/backtests")

BEST_CONFIG = {"patch_len": 24, "stride": 24, "ctx": 1344}
CKPT_PATH = "data/processed/patchtst_ckpts/patch24_s24_ctx1344_s42.pt"
TZ = "Europe/Warsaw"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 42


# ─── Data loading ────────────────────────────────────────────────────────────

def load_price_data():
    cfg = load_config()
    price = pd.read_parquet("data/processed/price_da_eur.parquet")["price_da_eur"]
    price.index = pd.to_datetime(price.index, utc=True)
    res = pd.read_parquet("data/processed/res_forecast.parquet")
    res.index = pd.to_datetime(res.index, utc=True)
    tso = pd.read_parquet("data/processed/tso_forecast.parquet")["tso_forecast_mw"]
    tso.index = pd.to_datetime(tso.index, utc=True)
    return price, res, tso


def get_test_days(price: pd.Series, start: str = "2025-09-01") -> list:
    all_dates = sorted(set(price.index.tz_convert(TZ).date))
    t0 = pd.Timestamp(start).date()
    return [d for d in all_dates if d >= t0]


# ─── Feature column names ─────────────────────────────────────────────────────

def fut_col_names(sample: object) -> list[str]:
    from src.models.deep.data import FUTURE_CAL_COLS
    n_cal = len(FUTURE_CAL_COLS)
    n_fut = sample.fut.shape[-1]
    # fut layout: [cal_cols, res_cols, tso, anchor]
    res_cols = n_fut - n_cal - 2  # tso + anchor = 2
    names = list(FUTURE_CAL_COLS)
    names += [f"res_{i}" for i in range(res_cols)]
    names += ["tso_fcst", "price_lag168"]
    return names


# ─── Feature group definitions ───────────────────────────────────────────────

from src.models.deep.data import FUTURE_CAL_COLS

def get_feature_groups(n_fut: int) -> dict[str, list[int]]:
    n_cal = len(FUTURE_CAL_COLS)
    res_end = n_fut - 2
    return {
        "calendar": list(range(0, n_cal)),
        "res_forecast": list(range(n_cal, res_end)),
        "tso_forecast": [res_end],
        "price_lag168": [res_end + 1],
    }


# ─── Section 1: Permutation importance ───────────────────────────────────────

def run_permutation_importance(price, res, tso) -> None:
    print("\n=== Section 1: Permutation importance ===")
    ctx = BEST_CONFIG["ctx"]
    patch_len = BEST_CONFIG["patch_len"]
    stride = BEST_CONFIG["stride"]

    test_days = get_test_days(price, start="2025-10-01")
    print(f"Test window: {test_days[0]} → {test_days[-1]} ({len(test_days)} days)")

    # Build sample set for test period
    all_dates = sorted(set(price.index.tz_convert(TZ).date))
    t_end = test_days[0] - pd.Timedelta(days=1)
    t_start = t_end - pd.Timedelta(days=365)
    window = [d for d in all_dates if t_start <= d <= t_end]
    split = int(0.85 * len(window))
    tr = build_price_samples(price, res, tso, window[:split], ctx, TZ)
    va = build_price_samples(price, res, tso, window[split:], ctx, TZ)
    stats = standardize_covariates(tr, va, n_tail=1)

    te = build_price_samples(price, res, tso, test_days, ctx, TZ)
    apply_covariate_stats(te, stats)

    # Load checkpoint
    net = PatchTST(
        enc_feat=1,
        fut_feat=tr.fut.shape[-1],
        d_model=64,
        patch_len=patch_len,
        stride=stride,
    ).to(DEVICE)
    net.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    net.eval()

    from src.models.deep.train import predict_mw as _predict_mw_all

    def predict_te(te_samples) -> np.ndarray:
        import copy
        s = copy.copy(te_samples)
        # predict_mw returns (n, 24, 3) in EUR/MWh
        preds_all = _predict_mw_all(net, s)  # (n, 24, 3)
        return preds_all[:, :, 1]  # p50

    # Baseline MAE
    baseline_preds = predict_te(te)
    actuals = []
    for day in te.days:
        hours = local_day_hours_utc(pd.Timestamp(day, tz=TZ), TZ)
        if len(hours) == 24:
            actuals.append(price.reindex(hours).values)
    actuals = np.array(actuals)
    n = min(len(baseline_preds), len(actuals))
    baseline_mae = float(np.abs(baseline_preds[:n] - actuals[:n]).mean())
    print(f"Baseline MAE: {baseline_mae:.3f} EUR/MWh")

    # Permute each fut feature group
    import copy
    groups = get_feature_groups(tr.fut.shape[-1])
    col_names = fut_col_names(tr)
    results = [{"feature": "baseline", "delta_mae": 0.0, "mae": baseline_mae}]
    torch.manual_seed(42)

    def permute_cols(te_samples, cols):
        te_perm = copy.copy(te_samples)
        fut_clone = te_samples.fut.clone()
        perm_idx = torch.randperm(fut_clone.shape[0])
        fut_clone[:, :, cols] = fut_clone[perm_idx][:, :, cols]
        te_perm.fut = fut_clone
        return te_perm

    for group_name, cols in groups.items():
        te_perm = permute_cols(te, cols)
        preds_perm = predict_te(te_perm)
        mae_perm = float(np.abs(preds_perm[:n] - actuals[:n]).mean())
        delta = mae_perm - baseline_mae
        results.append({"feature": group_name, "delta_mae": delta, "mae": mae_perm})
        print(f"  {group_name}: MAE={mae_perm:.3f} (Δ={delta:+.3f})")

    # Also permute individual calendar columns
    for ci, cname in enumerate(col_names[:len(FUTURE_CAL_COLS)]):
        te_perm = permute_cols(te, [ci])
        preds_perm = predict_te(te_perm)
        mae_perm = float(np.abs(preds_perm[:n] - actuals[:n]).mean())
        delta = mae_perm - baseline_mae
        results.append({"feature": f"cal:{cname}", "delta_mae": delta, "mae": mae_perm})
        print(f"  cal:{cname}: MAE={mae_perm:.3f} (Δ={delta:+.3f})")

    df = pd.DataFrame(results).sort_values("delta_mae", ascending=False)
    df.to_csv(RESULTS_DIR / "patchtst_permutation_importance.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    non_base = df[df["feature"] != "baseline"]
    bars = ax.barh(non_base["feature"], non_base["delta_mae"],
                   color=["#e74c3c" if x > 0.5 else "#3498db" for x in non_base["delta_mae"]])
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("ΔMAE when feature permuted (EUR/MWh, positive = important)")
    ax.set_title(f"PatchTST permutation importance\nBaseline MAE={baseline_mae:.2f} EUR/MWh")
    fig.tight_layout()
    fig.savefig(OUT / "perm_importance.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved perm_importance.png")


# ─── Section 2: Attention weight analysis ────────────────────────────────────

def run_attention_analysis(price, res, tso) -> None:
    print("\n=== Section 2: Attention pattern analysis ===")
    ctx = BEST_CONFIG["ctx"]
    patch_len = BEST_CONFIG["patch_len"]
    stride = BEST_CONFIG["stride"]
    n_patches = (ctx - patch_len) // stride + 1

    test_days = get_test_days(price, start="2026-04-01")[:60]
    print(f"Test window: {test_days[0]} → {test_days[-1]} ({len(test_days)} days)")

    all_dates = sorted(set(price.index.tz_convert(TZ).date))
    t_end = test_days[0] - pd.Timedelta(days=1)
    t_start = t_end - pd.Timedelta(days=365)
    window = [d for d in all_dates if t_start <= d <= t_end]
    split = int(0.85 * len(window))
    tr = build_price_samples(price, res, tso, window[:split], ctx, TZ)
    va = build_price_samples(price, res, tso, window[split:], ctx, TZ)
    stats = standardize_covariates(tr, va, n_tail=1)
    te = build_price_samples(price, res, tso, test_days, ctx, TZ)
    apply_covariate_stats(te, stats)

    net = PatchTST(
        enc_feat=1,
        fut_feat=tr.fut.shape[-1],
        d_model=64,
        patch_len=patch_len,
        stride=stride,
    ).to(DEVICE)
    net.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    net.eval()

    attn_maps = []
    for i in range(len(te.days)):
        enc_i = te.enc[[i]].to(DEVICE)   # (1, T, C)
        with torch.no_grad():
            B, T, C = enc_i.shape
            chans = enc_i.permute(0, 2, 1).reshape(B * C, T)  # (B*C, T)
            patches = net._patch(chans)                        # (B*C, P, L)
            tokens = net.patch_embed(patches)                  # (B*C, P, d)
            tokens = tokens + net.pos[:, :tokens.shape[1]]
            attn_out = None
            for layer in net.encoder.layers:
                src = tokens
                _, attn_w = layer.self_attn(src, src, src, need_weights=True, average_attn_weights=False)
                attn_out = attn_w.cpu().numpy()[0]  # [n_heads, P, P]
                tokens = layer(tokens)
            if attn_out is not None:
                attn_maps.append(attn_out.mean(axis=0))  # avg over heads: [P, P]

    if not attn_maps:
        print("No attention weights collected")
        return

    avg_attn = np.mean(attn_maps, axis=0)  # [n_patches, n_patches]

    # Plot 1: Average attention heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im = axes[0].imshow(avg_attn, aspect="auto", cmap="hot_r", origin="upper")
    plt.colorbar(im, ax=axes[0])
    axes[0].set_xlabel("Key patch index (past → recent)")
    axes[0].set_ylabel("Query patch index")
    axes[0].set_title(f"Average attention weights\n(n_patches={n_patches}, patch={patch_len}h)")

    # Plot 2: Row-sum (how much each patch is attended to)
    attended_to = avg_attn.sum(axis=0)  # sum over queries for each key
    patch_ages_days = np.arange(n_patches) * stride / 24  # age of each patch in days (0=most recent)
    axes[1].bar(patch_ages_days[::-1], attended_to, width=stride / 24 * 0.8, color="#3498db", alpha=0.7)
    axes[1].set_xlabel("Patch age (days before forecast day)")
    axes[1].set_ylabel("Total attention received")
    axes[1].set_title("Which past periods get attended to most?")
    axes[1].invert_xaxis()

    fig.suptitle(f"PatchTST attention analysis — {len(attn_maps)} samples", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "attention_analysis.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved attention_analysis.png")

    # Save mean attention
    np.save(OUT / "avg_attention.npy", avg_attn)
    print("Saved avg_attention.npy")


# ─── Section 3: PCA on features ──────────────────────────────────────────────

def run_pca_analysis(price, res, tso) -> None:
    print("\n=== Section 3: PCA on input features ===")
    ctx = BEST_CONFIG["ctx"]
    patch_len = BEST_CONFIG["patch_len"]
    stride = BEST_CONFIG["stride"]

    # Use a representative training sample
    all_dates = sorted(set(price.index.tz_convert(TZ).date))
    t0 = pd.Timestamp("2024-01-01").date()
    t1 = pd.Timestamp("2025-12-31").date()
    days = [d for d in all_dates if t0 <= d <= t1]
    split = int(0.85 * len(days))
    tr = build_price_samples(price, res, tso, days[:split], ctx, TZ)
    va = build_price_samples(price, res, tso, days[split:], ctx, TZ)
    stats = standardize_covariates(tr, va, n_tail=1)

    col_names = fut_col_names(tr)
    n_samples = tr.fut.shape[0]

    # PCA on future covariates (shape: [n_samples * 24, n_fut_cols])
    fut_flat = tr.fut.reshape(-1, tr.fut.shape[-1])
    scaler = StandardScaler()
    fut_scaled = scaler.fit_transform(fut_flat)
    pca = PCA(n_components=min(10, fut_flat.shape[-1]))
    pca.fit(fut_scaled)

    print(f"Explained variance ratio (top-10): {pca.explained_variance_ratio_}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Explained variance
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    axes[0].bar(range(1, len(cumvar) + 1), pca.explained_variance_ratio_ * 100, color="#3498db", alpha=0.7)
    axes[0].plot(range(1, len(cumvar) + 1), cumvar * 100, "ro-", lw=1.5)
    axes[0].axhline(80, color="gray", ls="--", lw=0.8, label="80%")
    axes[0].set_xlabel("PCA component")
    axes[0].set_ylabel("Variance explained (%)")
    axes[0].set_title("PCA on future covariates\n(calendar + RES forecast + TSO + lag-168)")
    axes[0].legend()

    # Loading of PC1 on each feature
    pc1_load = np.abs(pca.components_[0])
    axes[1].barh(col_names, pc1_load, color="#e67e22", alpha=0.8)
    axes[1].set_xlabel("|Loading| on PC1")
    axes[1].set_title("PC1 loadings by feature")
    fig.tight_layout()
    fig.savefig(OUT / "pca_features.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved pca_features.png")

    # Save explained variance
    df_var = pd.DataFrame({
        "component": range(1, len(cumvar) + 1),
        "explained_var_pct": pca.explained_variance_ratio_ * 100,
        "cumvar_pct": cumvar * 100,
    })
    df_var.to_csv(RESULTS_DIR / "patchtst_pca_variance.csv", index=False)
    df_load = pd.DataFrame({"feature": col_names, "pc1_loading": pca.components_[0]})
    df_load.to_csv(RESULTS_DIR / "patchtst_pca_loadings.csv", index=False)
    print("PCA results saved")


# ─── Section 4: Feature-group ablation walk-forward ──────────────────────────

ABLATION_GROUPS = {
    "full": None,
    "drop_calendar": "calendar",
    "drop_res": "res_forecast",
    "drop_tso": "tso_forecast",
    "drop_lag168": "price_lag168",
}


def build_price_samples_ablated(price, res, tso, days, ctx, tz, drop_group):
    """Build price samples with one feature group zeroed out."""
    from src.models.deep.data import FUTURE_CAL_COLS
    samples = build_price_samples(price, res, tso, days, ctx, tz)
    if drop_group is None or len(samples.days) == 0:
        return samples

    n_fut = samples.fut.shape[-1]
    groups = get_feature_groups(n_fut)
    if drop_group not in groups:
        return samples

    import copy
    s = copy.copy(samples)
    s.fut = samples.fut.copy()
    for ci in groups[drop_group]:
        s.fut[:, :, ci] = 0.0
    return s


def run_ablation_walkforward(price, res, tso) -> None:
    print("\n=== Section 4: Feature-group ablation walk-forward ===")
    patch_len = BEST_CONFIG["patch_len"]
    stride = BEST_CONFIG["stride"]
    ctx = BEST_CONFIG["ctx"]

    all_dates = sorted(set(price.index.tz_convert(TZ).date))
    test_start = pd.Timestamp("2024-07-16").date()
    test_days = [d for d in all_dates if d >= test_start]
    print(f"Test period: {test_days[0]} → {test_days[-1]} ({len(test_days)} days)")
    print(f"Ablation configs: {list(ABLATION_GROUPS.keys())}")
    print("Each config ~90 min. Total ~7.5 h.")

    results = []
    for label, drop_group in ABLATION_GROUPS.items():
        print(f"\n--- Ablation: {label} (drop: {drop_group}) ---", flush=True)

        # Patch build_price_samples in the sweep module's own namespace
        import src.models.deep.run_patchtst_sweep as sweep_mod
        orig_fn = sweep_mod.build_price_samples

        if drop_group is not None:
            def _patched(p, r, t, days, enc_h, tz_):
                return build_price_samples_ablated(p, r, t, days, enc_h, tz_, drop_group)
            sweep_mod.build_price_samples = _patched

        try:
            pred = walk_forward_patchtst(
                price, res, tso,
                patch_len=patch_len, stride=stride,
                encoder_hours=ctx, d_model=64, seed=SEED,
                test_days=test_days, all_dates=all_dates,
            )
        finally:
            sweep_mod.build_price_samples = orig_fn

        if pred is None:
            print(f"  {label}: no predictions")
            continue

        y = price.reindex(pred.index)
        naive1d = price.reindex(pred.index - pd.Timedelta(hours=24))
        naive1d.index = pred.index
        mae = float((pred["p50"] - y).abs().mean())
        rmae = mae / float((naive1d - y).abs().mean())
        cov = 100.0 * ((y >= pred["p10"]) & (y <= pred["p90"])).mean()
        spike_cut = y.quantile(0.95)
        spike = y >= spike_cut
        spike_mae = float((pred.loc[spike, "p50"] - y[spike]).abs().mean())

        row = {
            "ablation": label,
            "drop_group": str(drop_group),
            "mae": mae, "rmae": rmae,
            "coverage_80_pct": float(cov),
            "spike_mae": spike_mae,
        }
        results.append(row)
        print(f"  MAE={mae:.3f} rMAE={rmae:.4f} cov={cov:.1f}%")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "patchtst_ablation_walkforward.csv", index=False)
    print(f"\nAblation results saved to {RESULTS_DIR}/patchtst_ablation_walkforward.csv")
    print(df.to_string(index=False))

    # Plot ablation impact
    if len(df) > 1:
        baseline_mae = df.loc[df["ablation"] == "full", "mae"].iloc[0]
        non_full = df[df["ablation"] != "full"].copy()
        non_full["delta_mae"] = non_full["mae"] - baseline_mae

        fig, ax = plt.subplots(figsize=(9, 4))
        colors = ["#e74c3c" if d > 0 else "#2ecc71" for d in non_full["delta_mae"]]
        bars = ax.barh(non_full["ablation"], non_full["delta_mae"], color=colors, alpha=0.8)
        ax.bar_label(bars, fmt="%+.2f", padding=3, fontsize=8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("ΔMAE vs full model (EUR/MWh)")
        ax.set_title(f"PatchTST feature ablation walk-forward\nFull MAE={baseline_mae:.2f} EUR/MWh")
        fig.tight_layout()
        fig.savefig(OUT / "ablation_walkforward.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        print("Saved ablation_walkforward.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

SECTIONS = {
    "permutation": run_permutation_importance,
    "attention": run_attention_analysis,
    "pca": run_pca_analysis,
    "ablation": run_ablation_walkforward,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section", choices=list(SECTIONS.keys()), default=None,
        help="Run only this section. Default: all sections in order.",
    )
    args = parser.parse_args()

    print("Loading price/RES/TSO data...")
    price, res, tso = load_price_data()

    if args.section:
        SECTIONS[args.section](price, res, tso)
    else:
        for name, fn in SECTIONS.items():
            fn(price, res, tso)

    print(f"\nAll results saved to {OUT}/ and {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
