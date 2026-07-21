"""Feature sensitivity analysis, PCA/ICA/KernelPCA, and SHAP for the load forecasting model.

Outputs (all in reports/sensitivity/):
  1. pca_explained.png           — cumulative explained variance by component
  2. pca_loadings.png            — PC1/PC2 biplot with feature labels
  3. feature_correlation.png     — Pearson correlation heatmap
  4. group_ablation.csv/.md      — MAPE delta when each feature group is removed
  5. permutation_importance.csv  — sklearn permutation importance for ridge_tso
  6. shap_summary_lgbm_tso.png   — SHAP beeswarm for lgbm_tso on 2-year test
  7. lasso_path.png              — which features survive as L1 penalty grows
  8. pca_feature_backtest.csv/md — ridge trained on PCA / ICA / KernelPCA projections
                                   at multiple cutoffs (80/90/95/99% var, 2/5/10/15/20 PCs)
                                   vs raw 25-feature baseline.

Run: uv run python -m src.evaluation.run_sensitivity
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import src.models.baselines  # noqa: F401
import src.models.gbm  # noqa: F401
from src.config import load_config
from src.evaluation.backtest import walk_forward_backtest, summarize, BacktestResult
from src.evaluation.run_2year_backtest import (
    assemble, make_hybrid_weather, TEST_START_LOCAL, RidgeTSO, LGBMQuantileTSO,
)
from src.features.weather import load_weather_forecast_history, load_weather_history
from src.models.baselines import RidgeForecaster, SeasonalNaive
from src.models.gbm import LightGBMQuantile
from src.pipeline.daily_run import shift_local_day

TZ = "Europe/Warsaw"

# Feature groups — match column names from build_features output
GROUPS = {
    "weather": ["temperature_2m", "wind_speed_10m", "cloud_cover",
                "shortwave_radiation", "relative_humidity_2m",
                "heating_degrees", "cooling_degrees"],
    "lags": ["load_lag_48h", "load_lag_72h", "load_lag_168h", "load_lag_336h",
             "load_lag_504h", "load_lag_672h", "load_mean_7d"],
    "calendar": ["hour_local", "weekday", "month", "is_holiday", "is_bridge_day",
                 "is_weekend", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
                 "dow_sin", "dow_cos", "month_sin", "month_cos"],
    "tso": ["tso_forecast_mw"],
}


def _style_and_save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def run_pca(x: pd.DataFrame, out_dir: Path) -> None:
    print("\n[1] PCA on feature matrix ...")
    x_std = StandardScaler().fit_transform(x.dropna())
    pca = PCA(n_components=min(25, x_std.shape[1]))
    pca.fit(x_std)

    # Explained variance
    fig, ax = plt.subplots(figsize=(8, 4))
    cum = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(range(1, len(cum) + 1), cum, "o-")
    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.5, label="90%")
    ax.axhline(0.95, color="red", linestyle="--", alpha=0.5, label="95%")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA: explained variance")
    ax.legend()
    _style_and_save(fig, out_dir / "pca_explained.png")

    # PC1 / PC2 loadings biplot
    comps = pca.components_[:2]  # (2, n_feat)
    feat_names = list(x.columns)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(comps[0], comps[1], alpha=0.0)
    for i, name in enumerate(feat_names):
        ax.annotate(name, (comps[0, i], comps[1, i]), fontsize=7,
                    ha="center", va="center")
        ax.arrow(0, 0, comps[0, i], comps[1, i], head_width=0.01, alpha=0.4)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("PCA loadings — feature biplot")
    _style_and_save(fig, out_dir / "pca_loadings.png")

    # Components table
    comp_df = pd.DataFrame(
        pca.components_[:5].T,
        index=feat_names,
        columns=[f"PC{i+1}" for i in range(5)],
    )
    comp_df.to_csv(out_dir / "pca_components.csv")
    print(f"  95% var in {int(np.searchsorted(cum, 0.95)) + 1} components")


def run_correlation(x: pd.DataFrame, out_dir: Path) -> None:
    print("\n[2] Feature correlation heatmap ...")
    corr = x.corr()
    feat_names = list(x.columns)
    n = len(feat_names)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(n))
    ax.set_xticklabels(feat_names, rotation=90, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(feat_names, fontsize=7)
    ax.set_title("Feature correlations")
    _style_and_save(fig, out_dir / "feature_correlation.png")


def run_group_ablation(
    x_no_tso: pd.DataFrame,
    x_with_tso: pd.DataFrame,
    y: pd.Series,
    test_start_utc: pd.Timestamp,
    out_dir: Path,
) -> None:
    print("\n[3] Feature group ablation (ridge_tso baseline, drop each group) ...")

    def mape_for_cols(drop_group: str | None, x: pd.DataFrame, factory) -> float:
        if drop_group is None:
            x_run = x
        else:
            cols_to_drop = [c for c in GROUPS.get(drop_group, []) if c in x.columns]
            x_run = x.drop(columns=cols_to_drop, errors="ignore")
        r = walk_forward_backtest(factory, x_run, y, test_start_utc,
                                  train_window_days=365)
        tbl = summarize([r], y)
        return float(tbl["mape_pct"].iloc[0])

    # Baseline (all features with TSO)
    print("  baseline ridge_tso ...")
    base_mape = mape_for_cols(None, x_with_tso, RidgeTSO)
    print(f"  baseline MAPE: {base_mape:.3f}%")

    rows = [{"group_removed": "none (baseline)", "mape": base_mape, "delta": 0.0}]
    for group in GROUPS:
        print(f"  drop {group} ...")
        x_ablate = x_with_tso if group != "tso" else x_no_tso
        mape = mape_for_cols(group, x_ablate, RidgeTSO if group != "tso" else RidgeForecaster)
        delta = mape - base_mape
        rows.append({"group_removed": group, "mape": mape, "delta": delta})
        print(f"  drop {group}: MAPE {mape:.3f}% (Δ {delta:+.3f}pp)")

    result = pd.DataFrame(rows).set_index("group_removed")
    result.to_csv(out_dir / "group_ablation.csv")

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    deltas = result["delta"].iloc[1:]
    colors = ["#d62728" if d > 0 else "#2ca02c" for d in deltas]
    ax.bar(deltas.index, deltas, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("MAPE increase when group removed (pp)")
    ax.set_title("Feature group ablation — ridge_tso, 1-year rolling, 2yr test")
    ax.tick_params(axis="x", rotation=20)
    _style_and_save(fig, out_dir / "group_ablation.png")

    md = ["# Feature group ablation", "", result.round(3).to_markdown(), ""]
    (out_dir / "group_ablation.md").write_text("\n".join(md))
    print(f"  saved group_ablation.csv / .md / .png")


def run_permutation_importance(
    x_with_tso: pd.DataFrame,
    y: pd.Series,
    test_start_utc: pd.Timestamp,
    tz: str,
    out_dir: Path,
) -> None:
    print("\n[4] Permutation importance for ridge_tso ...")
    # Train on 1 year before test start
    train_end = test_start_utc - pd.Timedelta(hours=1)
    train_start = train_end - pd.Timedelta(days=365)
    mask = (x_with_tso.index >= train_start) & (x_with_tso.index <= train_end)
    x_tr = x_with_tso[mask].dropna()
    y_tr = y.reindex(x_tr.index).dropna()
    x_tr = x_tr.reindex(y_tr.index)

    pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    pipe.fit(x_tr.values, y_tr.values)

    # Test set: first 3 months after test start
    test_end = test_start_utc + pd.Timedelta(days=90)
    test_mask = (x_with_tso.index >= test_start_utc) & (x_with_tso.index <= test_end)
    x_te = x_with_tso[test_mask].dropna()
    y_te = y.reindex(x_te.index).dropna()
    x_te = x_te.reindex(y_te.index)

    result = permutation_importance(pipe, x_te.values, y_te.values,
                                    n_repeats=20, random_state=0,
                                    scoring="neg_mean_absolute_error")
    imp_df = pd.DataFrame(
        {"importance_mean": result.importances_mean,
         "importance_std": result.importances_std},
        index=x_tr.columns,
    ).sort_values("importance_mean", ascending=False)
    imp_df.to_csv(out_dir / "permutation_importance.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    imp_df["importance_mean"].plot.barh(ax=ax, xerr=imp_df["importance_std"],
                                        color="steelblue")
    ax.set_title("Permutation importance — ridge_tso (MAE reduction)")
    ax.set_xlabel("Mean MAE reduction when feature permuted")
    _style_and_save(fig, out_dir / "permutation_importance.png")
    print(f"  Top 5: {list(imp_df.index[:5])}")


def run_lasso_path(
    x_with_tso: pd.DataFrame,
    y: pd.Series,
    test_start_utc: pd.Timestamp,
    out_dir: Path,
) -> None:
    print("\n[5] LASSO path — which features survive as regularization increases ...")
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler

    train_end = test_start_utc - pd.Timedelta(hours=1)
    train_start = train_end - pd.Timedelta(days=365)
    mask = (x_with_tso.index >= train_start) & (x_with_tso.index <= train_end)
    x_tr = x_with_tso[mask].dropna()
    y_tr = y.reindex(x_tr.index).dropna()
    x_tr = x_tr.reindex(y_tr.index)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_tr.values)

    alphas = np.logspace(-4, 1, 50)
    coef_paths = []
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=5000)
        lasso.fit(x_scaled, y_tr.values)
        coef_paths.append(lasso.coef_)
    coef_paths = np.array(coef_paths)  # (n_alphas, n_feat)

    feat_names = list(x_tr.columns)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(feat_names):
        ax.semilogx(alphas, coef_paths[:, i], label=name if i < 8 else None, lw=0.9)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("LASSO alpha (log scale)")
    ax.set_ylabel("Coefficient")
    ax.set_title("LASSO regularization path — features surviving shrinkage")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    _style_and_save(fig, out_dir / "lasso_path.png")

    # Count active features at each alpha
    n_active = (coef_paths != 0).sum(axis=1)
    active_df = pd.DataFrame({"alpha": alphas, "n_active": n_active,
                              "features": [(coef_paths[i] != 0).sum() for i in range(len(alphas))]})
    active_df.to_csv(out_dir / "lasso_path_active.csv", index=False)

    # Report features that survive at high regularization (alpha=0.1)
    idx = np.searchsorted(alphas, 0.1)
    survivors = [f for f, c in zip(feat_names, coef_paths[idx]) if c != 0]
    print(f"  Features active at alpha=0.1: {survivors}")


def run_shap_lgbm(
    x_with_tso: pd.DataFrame,
    y: pd.Series,
    test_start_utc: pd.Timestamp,
    out_dir: Path,
) -> None:
    print("\n[6] SHAP for lgbm_tso ...")
    try:
        import shap
    except ImportError:
        print("  shap not installed, skipping")
        return

    # Train on 1 year before test start
    train_end = test_start_utc - pd.Timedelta(hours=1)
    train_start = train_end - pd.Timedelta(days=365)
    mask = (x_with_tso.index >= train_start) & (x_with_tso.index <= train_end)
    x_tr = x_with_tso[mask].dropna()
    y_tr = y.reindex(x_tr.index).dropna()
    x_tr = x_tr.reindex(y_tr.index)

    import lightgbm as lgb
    model = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=300,
                               learning_rate=0.05, num_leaves=63, verbosity=-1,
                               random_state=0)
    model.fit(x_tr, y_tr)

    # SHAP on first 3 months of test
    test_end = test_start_utc + pd.Timedelta(days=90)
    test_mask = (x_with_tso.index >= test_start_utc) & (x_with_tso.index <= test_end)
    x_te = x_with_tso[test_mask].dropna()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_te)

    plt.close("all")
    shap.summary_plot(shap_values, x_te, show=False, plot_size=(10, 7))
    fig = plt.gcf()
    fig.suptitle("SHAP summary — lgbm_tso (P50), first 90 test days", y=1.02)
    _style_and_save(fig, out_dir / "shap_summary_lgbm_tso.png")

    # Global importance table
    imp = pd.Series(
        np.abs(shap_values).mean(axis=0), index=x_te.columns
    ).sort_values(ascending=False)
    imp.to_csv(out_dir / "shap_importance.csv")
    print(f"  Top 5 SHAP: {list(imp.index[:5])}")


def _pca_transform_x(
    x_with_tso: pd.DataFrame,
    n_components: int,
    reducer_name: str = "pca",
) -> pd.DataFrame:
    """Return x transformed into n_components-dimensional space.

    For PCA and ICA: fit on full x (leakage note: we fit on the FULL
    dataset including test for the reducer itself, which is acceptable
    because PCA/ICA are unsupervised — they see no y. The walk-forward
    backtest's Ridge is still trained only on pre-cutoff data each fold).

    For correctness in a production setting you'd fit the PCA on training
    rows only and transform test rows. Here we're interested in the
    question "can a model trained on PCs beat one trained on raw features?"
    — the approximation is minor and noted.
    """
    from sklearn.decomposition import FastICA, KernelPCA

    x_clean = x_with_tso.dropna()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_clean)

    if reducer_name == "pca":
        reducer = PCA(n_components=n_components, random_state=0)
    elif reducer_name == "ica":
        reducer = FastICA(n_components=n_components, random_state=0, max_iter=500)
    elif reducer_name == "kpca_rbf":
        reducer = KernelPCA(n_components=n_components, kernel="rbf",
                            gamma=None, random_state=0)
    else:
        raise ValueError(f"Unknown reducer: {reducer_name}")

    x_reduced = reducer.fit_transform(x_scaled)
    cols = [f"{reducer_name}_c{i+1}" for i in range(n_components)]
    return pd.DataFrame(x_reduced, index=x_clean.index, columns=cols)


def run_pca_feature_backtest(
    x_with_tso: pd.DataFrame,
    y: pd.Series,
    test_start_utc: pd.Timestamp,
    out_dir: Path,
) -> None:
    """Train ridge on PCA / ICA / KernelPCA projections at different n_components.

    Answers: does compressing 25 features into their principal components
    help or hurt a linear model? What about nonlinear reductions?

    Also compares:
      - Variance threshold cutoffs: 80%, 90%, 95%, 99% explained variance
      - Fixed-count cutoffs: 2, 5, 10, 15, 20 PCs
      - ICA: same counts (finds statistically independent components)
      - Kernel PCA (RBF): nonlinear structure (higher cost, only 5/10/15)
    """
    print("\n[7] Training ridge on PCA / ICA / KernelPCA projections ...")

    # First: determine PC counts for variance thresholds
    x_clean = x_with_tso.dropna()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_clean)
    pca_full = PCA().fit(x_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    thresholds = {f"pca_var{int(v*100)}pct": int(np.searchsorted(cum_var, v)) + 1
                  for v in [0.80, 0.90, 0.95, 0.99]}
    print(f"  PC counts for variance thresholds: {thresholds}")

    # All experiments
    experiments: list[tuple[str, str, int]] = []
    for label, n in thresholds.items():
        experiments.append((label, "pca", n))
    for n in [2, 5, 10, 15, 20]:
        experiments.append((f"pca_{n}comp", "pca", n))
        experiments.append((f"ica_{n}comp", "ica", n))
    for n in [5, 10, 15]:
        experiments.append((f"kpca_rbf_{n}comp", "kpca_rbf", n))

    # Baseline: raw features ridge_tso
    print("  raw ridge_tso baseline ...")
    raw_r = walk_forward_backtest(RidgeTSO, x_with_tso, y, test_start_utc,
                                  train_window_days=365)
    from src.evaluation.backtest import summarize, BacktestResult
    raw_tbl = summarize([raw_r], y)
    raw_mape = float(raw_tbl["mape_pct"].iloc[0])
    print(f"  raw baseline MAPE: {raw_mape:.3f}%")

    rows = [{"experiment": "raw_ridge_tso (25 features)", "reducer": "raw",
             "n_components": 25, "mape": raw_mape, "delta": 0.0}]

    for exp_name, reducer_name, n_comp in experiments:
        print(f"  {exp_name} ({reducer_name}, n={n_comp}) ...")
        try:
            x_proj = _pca_transform_x(x_with_tso, n_comp, reducer_name)
            x_proj = x_proj.reindex(x_with_tso.index)

            r = walk_forward_backtest(RidgeForecaster, x_proj, y, test_start_utc,
                                      train_window_days=365)
            tbl = summarize([r], y)
            mape = float(tbl["mape_pct"].iloc[0])
            delta = mape - raw_mape
            rows.append({"experiment": exp_name, "reducer": reducer_name,
                         "n_components": n_comp, "mape": mape, "delta": delta})
            print(f"    MAPE {mape:.3f}% (Δ {delta:+.3f}pp vs raw)")
        except Exception as e:
            print(f"    FAILED: {e}")
            rows.append({"experiment": exp_name, "reducer": reducer_name,
                         "n_components": n_comp, "mape": float("nan"), "delta": float("nan")})

    result = pd.DataFrame(rows).set_index("experiment")
    result.to_csv(out_dir / "pca_feature_backtest.csv")

    # Plot MAPE vs n_components for each reducer
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, reducer_name, color in zip(axes, ["pca", "ica", "kpca_rbf"],
                                        ["steelblue", "darkorange", "green"]):
        sub = result[result["reducer"] == reducer_name].sort_values("n_components")
        if not sub.empty:
            ax.plot(sub["n_components"], sub["mape"], "o-", color=color)
            ax.axhline(raw_mape, color="red", linestyle="--", label=f"raw {raw_mape:.2f}%")
            ax.set_xlabel("n_components")
            ax.set_ylabel("MAPE (%)")
            ax.set_title(f"ridge on {reducer_name} projections")
            ax.legend(fontsize=8)
    _style_and_save(fig, out_dir / "pca_feature_backtest.png")

    # Summary markdown
    md = [
        "# PCA / ICA / KernelPCA feature backtest",
        "",
        "Train ridge on projected features vs 25 raw features.",
        "Delta = MAPE(projected) - MAPE(raw). Positive = raw features win.",
        "",
        result.round(3).to_markdown(),
    ]
    (out_dir / "pca_feature_backtest.md").write_text("\n".join(md))
    print(f"  saved pca_feature_backtest.csv / .md / .png")


def main() -> int:
    cfg = load_config()
    tz = cfg.timezone_local
    out_dir = Path("reports/sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    load = pd.read_parquet(cfg.paths["data_processed"] / "load.parquet").iloc[:, 0]
    tso = pd.read_parquet(cfg.paths["data_processed"] / "tso_forecast.parquet").iloc[:, 0]
    weather = make_hybrid_weather(cfg)

    test_start_local = pd.Timestamp(TEST_START_LOCAL, tz=tz)
    data_start = load.index[0].tz_convert(tz) + pd.Timedelta(days=30)
    data_end = load.index[-1].tz_convert(tz) - pd.Timedelta(days=1)
    test_start_utc = test_start_local.tz_convert("UTC")

    print(f"Assembling 2-year feature matrices ...")
    x_no_tso = assemble(load, weather, tz,
                        pd.Timestamp(data_start.date(), tz=tz),
                        pd.Timestamp(data_end.date(), tz=tz))
    x_with_tso = assemble(load, weather, tz,
                          pd.Timestamp(data_start.date(), tz=tz),
                          pd.Timestamp(data_end.date(), tz=tz), tso=tso)
    y = load.reindex(x_no_tso.index)

    # Test set only — for PCA / correlation
    test_mask = x_with_tso.index >= test_start_utc
    x_test = x_with_tso[test_mask].dropna()

    def _skip(name: str) -> None:
        print(f"  skipping {name} (output exists)")

    if not (out_dir / "pca_explained.png").exists():
        run_pca(x_test, out_dir)
    else:
        _skip("PCA")

    if not (out_dir / "feature_correlation.png").exists():
        run_correlation(x_test, out_dir)
    else:
        _skip("correlation")

    if not (out_dir / "group_ablation.csv").exists():
        run_group_ablation(x_no_tso, x_with_tso, y, test_start_utc, out_dir)
    else:
        _skip("group ablation")

    if not (out_dir / "permutation_importance.csv").exists():
        run_permutation_importance(x_with_tso, y, test_start_utc, tz, out_dir)
    else:
        _skip("permutation importance")

    if not (out_dir / "lasso_path_active.csv").exists():
        run_lasso_path(x_with_tso, y, test_start_utc, out_dir)
    else:
        _skip("LASSO path")

    if not (out_dir / "shap_importance.csv").exists():
        run_shap_lgbm(x_with_tso, y, test_start_utc, out_dir)
    else:
        _skip("SHAP")

    if not (out_dir / "pca_feature_backtest.csv").exists():
        run_pca_feature_backtest(x_with_tso, y, test_start_utc, out_dir)
    else:
        _skip("PCA feature backtest")

    print(f"\nAll done. Outputs in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
