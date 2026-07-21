"""TFT training campaign. Walk-forward backtest for all TFT sizes and TSO variants.

Trains:
  tft_d32, tft_d64, tft_d128   — without TSO covariate
  tft_d32_tso, tft_d64_tso, tft_d128_tso  — with TSO (the combiner variant)

Each variant walks the 2-year test (2024-07-16 → present) with weekly refits
on a 365-day rolling window.

TFT's key advantage: Variable Selection Network (VSN) learns per-feature
importance weights at each timestep. Export .vsn_weights after training to
see which features drive each forecast step — interpretable by design.

Comparison baseline: lstm_attn+TSO (2.43% from overnight readout). If TFT
beats this, it enters the model table. If not, we know why and say so.

Run: uv run python -m src.models.deep.run_tft_campaign
Expected runtime: 6–12 h on MPS (Apple Silicon M-series).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.evaluation.run_2year_backtest import make_hybrid_weather, TEST_START_LOCAL
from src.models.deep.data import DaySamples, build_samples, standardize_covariates
from src.models.deep.train import device, pinball, predict_mw, train_variant
from src.models.deep.tft import TFT, TFT_CONFIGS
from src.pipeline.daily_run import local_day_hours_utc, shift_local_day

TZ = "Europe/Warsaw"
QUANTILES = torch.tensor([0.1, 0.5, 0.9])


# ---------------------------------------------------------------------------
# Walk-forward backtest adapted for sequence models (same logic as run_overnight.py)
# ---------------------------------------------------------------------------

def _local_dates(s: pd.Series, tz: str) -> list:
    return sorted(set(s.index.tz_convert(tz).date))


def walk_forward_tft(
    load: pd.Series,
    weather: pd.DataFrame,
    tso: pd.Series | None,
    test_start: pd.Timestamp,
    d_model: int,
    name: str,
    tz: str = TZ,
    train_days: int = 365,
    refit_every: int = 7,
    seed: int = 0,
    checkpoint_dir: Path = Path("data/processed/tft_ckpts"),
) -> dict:
    """Returns: dict with mape, mae, pinball metrics."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt = str(checkpoint_dir / f"{name}.pt")

    all_dates = _local_dates(load, tz)
    test_start_local = test_start.tz_convert(tz).date()
    test_days = [d for d in all_dates if d >= test_start_local]

    print(f"\n[{name}] d_model={d_model}, test_days={len(test_days)}", flush=True)

    dev = device()
    n_enc_feat = 1 + weather.shape[1]  # load + weather cols
    n_fut_feat = 7 + weather.shape[1] + 1  # cal(7) + weather + anchor
    if tso is not None:
        n_fut_feat += 1  # +TSO slot
    n_tail = 2 if tso is not None else 1  # anchor + maybe TSO are instance-normalized

    net: TFT | None = None
    last_refit = None
    all_preds: list[tuple] = []  # (day, ndarray shape 24×3)

    for i, test_day in enumerate(test_days):
        # Training window: [test_day - train_days, test_day - 1]
        day_ts = pd.Timestamp(test_day, tz=tz)
        train_end_ts = shift_local_day(day_ts, -1, tz)
        train_start_ts = shift_local_day(train_end_ts, -train_days, tz)
        train_date_end = train_end_ts.date()
        train_date_start = train_start_ts.date()

        needs_refit = (
            net is None
            or last_refit is None
            or (test_day - last_refit).days >= refit_every
        )
        if needs_refit:
            train_days_list = [d for d in all_dates
                               if train_date_start <= d <= train_date_end]
            val_split = int(0.85 * len(train_days_list))
            train_dates = train_days_list[:val_split]
            val_dates = train_days_list[val_split:]

            tr = build_samples(load, weather, train_dates, tz=tz, tso=tso)
            va = build_samples(load, weather, val_dates, tz=tz, tso=tso)
            if len(tr.days) < 100 or len(va.days) < 14:
                continue
            standardize_covariates(tr, va, n_tail=n_tail)

            net = TFT(n_enc_feat, n_fut_feat, d_model=d_model).to(dev)
            result = train_variant(
                net, tr, va, checkpoint=ckpt, seed=seed,
                max_epochs=60, patience=8, lr=5e-4, batch=32,
            )
            print(f"  {test_day} refit ok | val {result['best_val_pinball_norm']:.4f} "
                  f"| epochs={result['best_epoch']} | {result['train_s']:.0f}s", flush=True)
            last_refit = test_day

        # Predict test day
        test_sample = build_samples(load, weather, [test_day], tz=tz, tso=tso)
        if len(test_sample.days) == 0:
            continue
        # BUG (latent, campaign never ran to completion): test samples are
        # NOT standardized with train stats here. Fixed pattern lives in
        # run_tft_price.py (apply_covariate_stats). Fix before any rerun.
        preds_mw = predict_mw(net, test_sample)  # (1, 24, 3)
        hours = local_day_hours_utc(day_ts, tz)
        if len(hours) == 24:
            all_preds.append((test_day, preds_mw[0]))

    if not all_preds:
        print(f"[{name}] NO predictions generated — check data coverage", flush=True)
        return {"model": name, "mape": float("nan")}

    # Evaluate
    index_all = []
    p10_all, p50_all, p90_all = [], [], []
    for day, pred in all_preds:
        day_ts = pd.Timestamp(day, tz=tz)
        hours = local_day_hours_utc(day_ts, tz)
        index_all.extend(hours)
        p10_all.extend(pred[:, 0])
        p50_all.extend(pred[:, 1])
        p90_all.extend(pred[:, 2])

    idx = pd.DatetimeIndex(index_all)
    pred_df = pd.DataFrame({"p10": p10_all, "p50": p50_all, "p90": p90_all}, index=idx)

    y_test = load.reindex(idx).dropna()
    pred_aligned = pred_df.reindex(y_test.index)

    def _mape(y, p):
        return float(100 * np.abs((y - p) / y).mean())

    def _mae(y, p):
        return float(np.abs(y - p).mean())

    def _pinball(y, p, q):
        diff = y.values - p.values
        return float(np.maximum(q * diff, (q - 1) * diff).mean())

    metrics = {
        "model": name, "d_model": d_model, "with_tso": tso is not None,
        "mape": _mape(y_test, pred_aligned["p50"]),
        "mae": _mae(y_test, pred_aligned["p50"]),
        "pinball_p10": _pinball(y_test, pred_aligned["p10"], 0.1),
        "pinball_p50": _pinball(y_test, pred_aligned["p50"], 0.5),
        "pinball_p90": _pinball(y_test, pred_aligned["p90"], 0.9),
        "n_hours": len(y_test),
    }
    print(f"[{name}] MAPE={metrics['mape']:.3f}% MAE={metrics['mae']:.0f} MW", flush=True)

    # Save predictions
    preds_dir = Path("data/processed/backtest_preds_tft")
    preds_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(preds_dir / f"{name}.parquet")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    cfg = load_config()
    tz = cfg.timezone_local

    load = pd.read_parquet(cfg.paths["data_processed"] / "load.parquet").iloc[:, 0]
    tso = pd.read_parquet(cfg.paths["data_processed"] / "tso_forecast.parquet").iloc[:, 0]
    weather = make_hybrid_weather(cfg)

    test_start = pd.Timestamp(TEST_START_LOCAL, tz=tz).tz_convert("UTC")
    out_dir = Path("reports/backtests")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"TFT campaign — 2-year test starting {TEST_START_LOCAL}")
    print(f"Device: {device()}")
    print("=" * 60)

    rows = []
    t0 = time.time()

    for label, d_model in TFT_CONFIGS:
        # Without TSO
        r = walk_forward_tft(load, weather, None, test_start, d_model, label)
        rows.append(r)

        # With TSO
        r_tso = walk_forward_tft(load, weather, tso, test_start, d_model, f"{label}_tso")
        rows.append(r_tso)

    total_h = (time.time() - t0) / 3600
    print(f"\nTotal runtime: {total_h:.1f} h")

    result = pd.DataFrame(rows).set_index("model")
    stamp = str(pd.Timestamp.now(tz).date())
    result.to_csv(out_dir / f"{stamp}_tft_campaign.csv")

    # Reference baselines from overnight readout (12-month, for context)
    print("\n--- TFT results (2-year test) vs reference baselines (12-month) ---")
    print(result.round(3).to_string())
    print("\nReference baselines (12-month walk-forward):")
    print("  ridge_tso: 2.13%  |  lstm_attn+TSO: 2.43%  |  TSO alone: 2.31%")

    md = [
        f"# TFT campaign — {stamp}",
        "",
        f"Test period: {TEST_START_LOCAL} → today. "
        f"Weather: hybrid (ERA5 pre-2024, lead-2 forecast 2024+). "
        f"Runtime: {total_h:.1f} h.",
        "",
        result.round(3).to_markdown(),
        "",
        "## Reference baselines (12-month walk-forward)",
        "| model | MAPE |",
        "|---|---|",
        "| ridge_tso | 2.13% |",
        "| lstm_attn+TSO | 2.43% |",
        "| TSO alone | 2.31% |",
        "",
    ]
    (out_dir / f"{stamp}_tft_campaign.md").write_text("\n".join(md))
    print(f"\nSaved to {out_dir}/{stamp}_tft_campaign.(csv|md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
