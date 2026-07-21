"""Channel correlation + PCA — the PatchTST channel-independence check.

PatchTST treats each input channel as an independent univariate series
(its 'channel independence' design). That is a MODELING BET: it pays
when channels are weakly coupled, and throws away signal when they are
strongly coupled. Before building PatchTST for this market, measure the
coupling.

Outputs (reports/sensitivity/):
- channels_corr.png / .csv — hourly correlation matrix of the channels
- channels_pca.png / channels_pca.csv — PCA explained variance +
  loadings on standardized channels
- verdict lines printed and appended to the csv header comment

Run: uv run python -m src.evaluation.run_channel_analysis  (~1 min CPU)
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import load_config
from src.viz.style import apply_style


def main() -> int:
    cfg = load_config()
    proc = cfg.paths["data_processed"]
    channels = pd.DataFrame({
        "price": pd.read_parquet(proc / "price_da_eur.parquet").iloc[:, 0],
        "load": pd.read_parquet(proc / "load.parquet").iloc[:, 0],
        "tso_fcst": pd.read_parquet(proc / "tso_forecast.parquet").iloc[:, 0],
    })
    res = pd.read_parquet(proc / "res_forecast.parquet")
    channels["solar_fcst"] = res["solar_fcst_mw"]
    channels["wind_fcst"] = res["wind_on_fcst_mw"] + res["wind_off_fcst_mw"]
    fuel = pd.read_parquet(proc / "fuel_daily.parquet")
    fuel.index = pd.DatetimeIndex(fuel.index).tz_localize("UTC")
    channels["ttf_gas"] = fuel["ttf_eur_mwh"].reindex(channels.index, method="ffill")
    channels["eua_proxy"] = fuel["eua_proxy_eur"].reindex(channels.index, method="ffill")
    channels = channels.dropna()
    print(f"{len(channels)} common hours, {channels.shape[1]} channels")

    out = proc.parent.parent / "reports" / "sensitivity"
    apply_style()

    corr = channels.corr()
    corr.to_csv(out / "channels_corr.csv")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
    ax.set_title("Hourly channel correlation — the channel-independence bet",
                 loc="left")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "channels_corr.png")
    plt.close(fig)

    z = StandardScaler().fit_transform(channels.to_numpy())
    pca = PCA().fit(z)
    ev = pca.explained_variance_ratio_
    loadings = pd.DataFrame(
        pca.components_.T, index=channels.columns,
        columns=[f"PC{i+1}" for i in range(len(ev))],
    )
    loadings["explained_var_of_pc"] = 0.0
    pd.concat([
        loadings,
        pd.DataFrame([ev], columns=loadings.columns[:-1], index=["explained_var"]),
    ]).to_csv(out / "channels_pca.csv")

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.bar(range(1, len(ev) + 1), 100 * ev, color="#0072B2")
    ax.plot(range(1, len(ev) + 1), 100 * np.cumsum(ev), color="#E69F00",
            marker="o", label="cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("PCA on standardized channels", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "channels_pca.png")
    plt.close(fig)

    k80 = int(np.searchsorted(np.cumsum(ev), 0.80) + 1)
    strong = (corr.abs().where(~np.eye(len(corr), dtype=bool)) > 0.5).sum().sum() // 2
    n_pairs = len(corr) * (len(corr) - 1) // 2
    coupled = k80 <= len(ev) // 2 or strong > n_pairs // 3
    print(f"components for 80% variance: {k80} of {len(ev)}")
    print(f"channel pairs with |corr| > 0.5: {strong} of {n_pairs}")
    if coupled:
        verdict = ("COUPLED: strict channel independence discards signal — "
                   "PatchTST here needs cross-channel mixing or covariates.")
    else:
        verdict = ("WEAKLY COUPLED at hourly resolution: channel independence "
                   "is a defensible bet — cross-channel signal is thin outside "
                   "load-vs-TSO, so per-channel patching loses little.")
    print("Verdict:", verdict)
    (out / "channels_verdict.txt").write_text(
        f"k80={k80}/{len(ev)}, strong_pairs={strong}/{n_pairs}\n{verdict}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
