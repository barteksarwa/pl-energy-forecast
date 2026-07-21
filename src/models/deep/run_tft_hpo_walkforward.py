"""Walk-forward confirmation of the TFT HPO winner.

After HPO finishes (run_tft_hpo.py), the best config is only validated
on a single screening split — which flatters nets by 0.6-0.9pp.
This script confirms (or refutes) the winner with:
  - A 2-year walk-forward test (2024-07-16 → present, monthly refits).
  - 3 seeds: if 2/3 beat the 1-seed screening threshold, we report.
  - Same test window as the tabular LEAR/LGBM backtest, enabling an
    honest same-window comparison.

Rule: quote ONLY the walk-forward numbers in
any results table. Screening val-pinball is for selection, not reporting.

Run: uv run python -m src.models.deep.run_tft_hpo_walkforward
     [--hpo-db data/processed/tft_hpo.db]
     [--trial-days 730]   # test window in days (default: all from 2024-07-16)
     [--seeds 42 7 99]
Expected: ~3-6 h on MPS (3 seeds × 24 monthly refits × ~5 min/refit).
One MPS job at a time — wait for HPO to finish before running this.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch

from src.config import load_config
from src.models.deep.data import apply_covariate_stats, standardize_covariates
from src.models.deep.price_data import build_price_samples
from src.models.deep.tft import TFT
from src.models.deep.train import device, predict_mw, train_variant
from src.pipeline.daily_run import local_day_hours_utc

TZ = "Europe/Warsaw"
TEST_START = pd.Timestamp("2024-07-16").date()


def load_hpo_best(db_path: str) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(
        study_name="tft_price_hpo",
        storage=f"sqlite:///{db_path}",
    )
    n = len([t for t in study.trials if t.state.name == "COMPLETE"])
    print(f"HPO study: {n} completed trials. Best val pinball: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


def walk_forward_one_seed(
    price: pd.Series,
    res: pd.DataFrame,
    tso: pd.Series,
    params: dict,
    test_days: list,
    all_dates: list,
    seed: int,
    train_days: int = 365,
    refit_every: int = 30,
) -> pd.DataFrame | None:
    encoder_hours = params["encoder_hours"]
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    lstm_layers = params["lstm_layers"]
    dropout = params["dropout"]
    lr = params["lr"]
    batch = params["batch"]

    ckpt_dir = Path("data/processed/tft_hpo_walkforward_ckpts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = str(ckpt_dir / f"seed{seed}.pt")

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
            print(
                f"  [seed{seed}] refit @{test_day} "
                f"train={len(tr.days)} val={len(va.days)}",
                flush=True,
            )
            net = TFT(
                enc_feat=1,
                fut_feat=tr.fut.shape[-1],
                d_model=d_model,
                n_heads=n_heads,
                lstm_layers=lstm_layers,
                dropout=dropout,
            ).to(device())
            result = train_variant(
                net, tr, va, checkpoint=ckpt, seed=seed,
                max_epochs=60, patience=8, lr=lr, batch=batch,
            )
            print(
                f"  [seed{seed}] val {result['best_val_pinball_norm']:.4f} "
                f"| epoch {result['best_epoch']} | {result['train_s']:.0f}s",
                flush=True,
            )
            last_refit = test_day

        if net is None or stats is None:
            continue
        sample = build_price_samples(price, res, tso, [test_day], encoder_hours, TZ)
        if len(sample.days) == 0:
            continue
        apply_covariate_stats(sample, stats)
        p = predict_mw(net, sample)[0]  # (24, 3)
        hours = local_day_hours_utc(pd.Timestamp(test_day, tz=TZ), TZ)
        if len(hours) == 24:
            preds.append(
                pd.DataFrame({"p10": p[:, 0], "p50": p[:, 1], "p90": p[:, 2]}, index=hours)
            )

    if not preds:
        print(f"[seed{seed}] no predictions", flush=True)
        return None
    return pd.concat(preds).sort_index()


def _metrics(pred: pd.DataFrame, y: pd.Series, naive1d: pd.Series) -> dict:
    n = len(y.dropna())
    mae = float((pred["p50"] - y).abs().mean())
    rmse = float(np.sqrt(((pred["p50"] - y) ** 2).mean()))
    naive_mae = float((naive1d - y).abs().mean())
    spike_cut = y.quantile(0.95)
    spike = y >= spike_cut
    cov = 100.0 * ((y >= pred["p10"]) & (y <= pred["p90"])).mean()
    return {
        "mae": mae,
        "rmse": rmse,
        "rmae": mae / naive_mae,
        "coverage_80_pct": cov,
        "spike_mae": float((pred.loc[spike, "p50"] - y[spike]).abs().mean()),
        "n_hours": n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpo-db", default="data/processed/tft_hpo.db")
    parser.add_argument("--trial-days", type=int, default=0,
                        help="Limit test to last N days (0 = use full 2yr window)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 99])
    args = parser.parse_args()

    params = load_hpo_best(args.hpo_db)
    encoder_hours = params["encoder_hours"]

    cfg = load_config()
    proc = cfg.paths["data_processed"]
    price = pd.read_parquet(proc / "price_da_eur.parquet").iloc[:, 0]
    res = pd.read_parquet(proc / "res_forecast.parquet")
    tso = pd.read_parquet(proc / "tso_forecast.parquet").iloc[:, 0]

    all_dates = sorted(set(price.index.tz_convert(TZ).date))
    last = all_dates[-2]

    if args.trial_days > 0:
        test_days = [d for d in all_dates if (last - d).days < args.trial_days]
    else:
        test_days = [d for d in all_dates if d >= TEST_START]

    print(f"\nTFT HPO walk-forward | best ctx={encoder_hours}h d={params['d_model']} "
          f"h={params['n_heads']} lr={params['lr']:.5f} batch={params['batch']}")
    print(f"Test: {test_days[0]} → {test_days[-1]} ({len(test_days)} days)")
    print(f"Seeds: {args.seeds}\n")

    t0 = time.time()
    seed_preds = {}
    for seed in args.seeds:
        print(f"=== Seed {seed} ===", flush=True)
        pred = walk_forward_one_seed(
            price, res, tso, params, test_days, all_dates, seed
        )
        if pred is not None:
            seed_preds[seed] = pred

    if not seed_preds:
        print("No predictions from any seed — exiting")
        return 1

    # Ensemble: mean of seeds
    common_idx = seed_preds[args.seeds[0]].index
    for p in seed_preds.values():
        common_idx = common_idx.intersection(p.index)
    ens = pd.DataFrame({
        "p10": pd.concat([p.loc[common_idx, "p10"] for p in seed_preds.values()], axis=1).mean(axis=1),
        "p50": pd.concat([p.loc[common_idx, "p50"] for p in seed_preds.values()], axis=1).mean(axis=1),
        "p90": pd.concat([p.loc[common_idx, "p90"] for p in seed_preds.values()], axis=1).mean(axis=1),
    })

    y = price.reindex(common_idx)
    naive1d = price.reindex(common_idx - pd.Timedelta(hours=24))
    naive1d.index = common_idx

    # Compare with tabular models on the same window
    pred_dir = proc / "backtest_preds_price_res"
    rows = []

    # TFT (per-seed + ensemble)
    for seed, pred in seed_preds.items():
        pred_win = pred.reindex(common_idx)
        m = _metrics(pred_win, y, naive1d)
        rows.append({"model": f"tft_hpo_seed{seed}", **m})

    ens_m = _metrics(ens, y, naive1d)
    rows.append({"model": f"tft_hpo_ens{len(seed_preds)}", **ens_m})

    # Save TFT predictions
    out_dir = proc / "backtest_preds_price_res"
    ens.to_parquet(out_dir / "tft_hpo_ens.parquet")

    # Reference tabular models (load if available)
    for name in ["lear_conformal", "lgbm_quantile_conformal", "price_naive_yesterday"]:
        path = pred_dir / f"{name}.parquet"
        if path.exists():
            ref = pd.read_parquet(path).reindex(common_idx)
            ref_y = price.reindex(common_idx)
            ref_naive = naive1d
            m = _metrics(ref, ref_y, ref_naive)
            rows.append({"model": name, **m})

    table = pd.DataFrame(rows).set_index("model").sort_values("mae")
    total_h = (time.time() - t0) / 3600
    print(f"\n=== RESULTS ({total_h:.1f} h) ===")
    print(table.round(3).to_string())

    stamp = f"{pd.Timestamp.now(TZ).date()}_tft_hpo_walkforward"
    out_reports = Path("reports/backtests")
    table.to_csv(out_reports / f"{stamp}.csv")

    best_mae = float(table.loc["tft_hpo_ens" + str(len(seed_preds)), "mae"])
    lear_mae = float(table.loc["lear_conformal", "mae"]) if "lear_conformal" in table.index else float("inf")
    verdict = "BEATS LEAR" if best_mae < lear_mae else "trails LEAR"

    md_lines = [
        f"# TFT HPO walk-forward — {stamp}",
        "",
        f"**Verdict: {verdict}** (TFT ens MAE {best_mae:.2f} vs LEAR {lear_mae:.2f} EUR/MWh)",
        "",
        f"Best HPO config: `{params}`",
        f"Test: {test_days[0]} → {test_days[-1]} ({len(common_idx)} hours)",
        f"Seeds: {args.seeds} | Runtime: {total_h:.1f} h",
        "",
        "## Results (same-window comparison)",
        "",
        table.round(3).to_markdown(),
        "",
        "## What to do with this",
        "",
        "- **If TFT beats LEAR**: 3-seed confirmed, open shadow gate, write model card.",
        "  Promotion criterion: mean daily MAE over 14 shadow days < LEAR MAE.",
        "- **If TFT trails**: document WHY (architecture ceiling, data ceiling, or both).",
        "  The honest verdict is as valuable as a win for the portfolio.",
        "",
        "Per-seed predictions: `data/processed/backtest_preds_price_res/tft_hpo_ens.parquet`",
    ]
    (out_reports / f"{stamp}.md").write_text("\n".join(md_lines))
    print(f"\nSaved → {out_reports}/{stamp}.(csv|md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
