"""TFT hyperparameter optimization on the price task (Optuna).

Searches jointly over architecture AND context length — the
hypothesis is that attention models have headroom here; this run
measures it instead of arguing about it.

Search space: encoder_hours {336..2016}, d_model, n_heads, lstm_layers,
dropout, lr, batch. Objective: validation pinball (normalized) on a
single temporal split — SCREENING tier. Repo lesson: single splits
flatter nets by 0.6-0.9pp, so the winner must be confirmed by
walk-forward before entering any table (next session).

After the study, the best config is retrained once and its Variable
Selection Network weights are exported — TFT's built-in feature
selection — to reports/sensitivity/tft_vsn_weights.csv.

Resume-safe: Optuna study lives in data/processed/tft_hpo.db.

Run: uv run python -m src.models.deep.run_tft_hpo [--trials 60]
Expected: ~4-8 h on MPS for 60 trials (long-context trials dominate).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna
import pandas as pd
import torch

from src.config import load_config
from src.models.deep.data import apply_covariate_stats, standardize_covariates
from src.models.deep.price_data import build_price_samples
from src.models.deep.tft import TFT
from src.models.deep.train import device, train_variant

TZ = "Europe/Warsaw"
SEED = 42
TRAIN_END = "2026-01-01"   # temporal split: train <, val >=
FUT_FEATURE_NAMES = [
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "is_weekend", "is_holiday", "is_bridge_day",
    "solar_fcst_mw", "wind_on_fcst_mw", "wind_off_fcst_mw",
    "tso_forecast_mw", "anchor_price_lag168",
]


def _load_data():
    cfg = load_config()
    proc = cfg.paths["data_processed"]
    price = pd.read_parquet(proc / "price_da_eur.parquet").iloc[:, 0]
    res = pd.read_parquet(proc / "res_forecast.parquet")
    tso = pd.read_parquet(proc / "tso_forecast.parquet").iloc[:, 0]
    return price, res, tso


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=60)
    args = parser.parse_args()

    price, res, tso = _load_data()
    ckpt_dir = Path("data/processed/tft_hpo_ckpts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[int, tuple] = {}  # encoder_hours -> (tr, va, stats)

    def objective(trial: optuna.Trial) -> float:
        enc_h = trial.suggest_categorical("encoder_hours", [336, 672, 1008, 1344, 2016])
        d_model = trial.suggest_categorical("d_model", [32, 48, 64, 96, 128])
        n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
        lstm_layers = trial.suggest_int("lstm_layers", 1, 2)
        dropout = trial.suggest_float("dropout", 0.05, 0.3)
        lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
        batch = trial.suggest_categorical("batch", [16, 32, 64])

        if enc_h not in cache:
            cache[enc_h] = _samples(price, res, tso, enc_h)
        tr, va, _ = cache[enc_h]

        net = TFT(enc_feat=1, fut_feat=tr.fut.shape[-1], d_model=d_model,
                  n_heads=n_heads, lstm_layers=lstm_layers, dropout=dropout
                  ).to(device())
        result = train_variant(
            net, tr, va, checkpoint=str(ckpt_dir / f"trial{trial.number}.pt"),
            seed=SEED, max_epochs=50, patience=6, lr=lr, batch=batch,
        )
        return result["best_val_pinball_norm"]

    study = optuna.create_study(
        study_name="tft_price_hpo",
        storage="sqlite:///data/processed/tft_hpo.db",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    df = study.trials_dataframe().sort_values("value")
    out_dir = Path("reports/backtests")
    stamp = f"{pd.Timestamp.now(TZ).date()}_tft_hpo"
    df.to_csv(out_dir / f"{stamp}_trials.csv", index=False)
    print(df.head(10).to_string())
    print(f"\nbest: {study.best_params} -> {study.best_value:.4f}")

    # --- VSN export for the best config (TFT-native feature selection) ---
    bp = study.best_params
    enc_h = bp["encoder_hours"]
    if enc_h not in cache:
        cache[enc_h] = _samples(price, res, tso, enc_h)
    tr, va, _ = cache[enc_h]
    net = TFT(enc_feat=1, fut_feat=tr.fut.shape[-1], d_model=bp["d_model"],
              n_heads=bp["n_heads"], lstm_layers=bp["lstm_layers"],
              dropout=bp["dropout"]).to(device())
    train_variant(net, tr, va, checkpoint=str(ckpt_dir / "best.pt"), seed=SEED,
                  max_epochs=50, patience=6, lr=bp["lr"], batch=bp["batch"])
    net.eval()
    with torch.no_grad():
        dev = device()
        net(va.enc.to(dev), va.fut.to(dev), va.anchor.to(dev))
        fut_w = net.fut_vsn.weights.mean(dim=0).cpu().numpy()
    vsn = pd.DataFrame({
        "feature": FUT_FEATURE_NAMES[: len(fut_w)],
        "vsn_weight": fut_w,
    }).sort_values("vsn_weight", ascending=False)
    vsn.to_csv("reports/sensitivity/tft_vsn_weights.csv", index=False)
    print("\nVSN feature selection (known-future covariates):")
    print(vsn.to_string(index=False))

    md = [
        f"# TFT HPO on price — {stamp}",
        "",
        f"{len(study.trials)} trials, screening split (train < {TRAIN_END} <= val).",
        "Winner must survive walk-forward before entering any results table",
        "(single splits flatter nets — measured 0.6-0.9pp on load).",
        "",
        f"Best: `{study.best_params}` -> val pinball {study.best_value:.4f}",
        "",
        "Top 10 trials: see the CSV. VSN weights:",
        "reports/sensitivity/tft_vsn_weights.csv",
        "",
    ]
    (out_dir / f"{stamp}.md").write_text("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
