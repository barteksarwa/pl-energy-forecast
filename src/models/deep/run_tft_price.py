"""TFT on the day-ahead PRICE, with a long-context sweep.

The question this run answers: does attention over a MUCH longer past
(4 weeks, 12 weeks) beat the tabular models, whose lags reach back at
most 7 days? Tabular LEAR/LGBM cannot see last month's regime; a
sequence model can. If long context doesn't pay here, the "more context"
argument is dead for this market and we say so.

Configs (screening, 1 seed — repo rule: 1 seed to screen, 3 to confirm):
  tft_price_ctx168   — 1-week encoder (parity with tabular lag reach)
  tft_price_ctx672   — 4-week encoder
  tft_price_ctx2016  — 12-week encoder

Walk-forward: last 180 days, refit every 30 days, 365-day train window.
Benchmarks in the readout: naive-1d, LEAR (18.5 MAE), LGBM (17.8 MAE)
from reports/backtests/2026-07-16_price_res_summary.md (same test
period family, honest comparison caveat: different test span is noted).

Run: uv run python -m src.models.deep.run_tft_price [--days 180]
Expected: ~1-4 h on MPS. Checkpoints in data/processed/tft_price_ckpts/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.models.deep.price_data import build_price_samples
from src.models.deep.data import apply_covariate_stats, standardize_covariates
from src.models.deep.tft import TFT
from src.models.deep.train import device, predict_mw, train_variant
from src.pipeline.daily_run import local_day_hours_utc

TZ = "Europe/Warsaw"
CONTEXTS = [168, 672, 2016]
D_MODEL = 64
SEED = 42


def walk_forward_tft_price(
    price: pd.Series, res: pd.DataFrame, tso: pd.Series,
    encoder_hours: int, test_days: list, all_dates: list,
    train_days: int = 365, refit_every: int = 30, tz: str = TZ,
) -> pd.DataFrame | None:
    name = f"tft_price_ctx{encoder_hours}"
    ckpt_dir = Path("data/processed/tft_price_ckpts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    net = None
    last_refit = None
    stats = None
    preds = []
    for test_day in test_days:
        needs_refit = (
            net is None or (test_day - last_refit).days >= refit_every
        )
        if needs_refit:
            t_end = test_day - pd.Timedelta(days=1)
            t_start = test_day - pd.Timedelta(days=train_days)
            window = [d for d in all_dates if t_start <= d <= t_end]
            split = int(0.85 * len(window))
            tr = build_price_samples(price, res, tso, window[:split], encoder_hours, tz)
            va = build_price_samples(price, res, tso, window[split:], encoder_hours, tz)
            if len(tr.days) < 100 or len(va.days) < 10:
                continue
            stats = standardize_covariates(tr, va, n_tail=1)
            print(f"  {name} train samples: {len(tr.days)} / val: {len(va.days)}", flush=True)
            net = TFT(enc_feat=1, fut_feat=tr.fut.shape[-1], d_model=D_MODEL).to(device())
            result = train_variant(
                net, tr, va, checkpoint=str(ckpt_dir / f"{name}.pt"), seed=SEED,
                max_epochs=60, patience=8, lr=5e-4, batch=32,
            )
            print(f"  {name} refit @{test_day} | val {result['best_val_pinball_norm']:.4f}"
                  f" | epochs={result['best_epoch']} | {result['train_s']:.0f}s", flush=True)
            last_refit = test_day

        if net is None:
            continue
        sample = build_price_samples(price, res, tso, [test_day], encoder_hours, tz)
        if len(sample.days) == 0:
            continue
        apply_covariate_stats(sample, stats)
        p = predict_mw(net, sample)[0]  # (24, 3) EUR/MWh
        hours = local_day_hours_utc(pd.Timestamp(test_day, tz=tz), tz)
        if len(hours) == 24:
            preds.append(pd.DataFrame(
                {"p10": p[:, 0], "p50": p[:, 1], "p90": p[:, 2]}, index=hours
            ))

    if not preds:
        print(f"[{name}] no predictions", flush=True)
        return None
    out = pd.concat(preds).sort_index()
    out_dir = Path("data/processed/backtest_preds_price_res")
    out.to_parquet(out_dir / f"{name}.parquet")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=180)
    args = parser.parse_args()

    cfg = load_config()
    proc = cfg.paths["data_processed"]
    price = pd.read_parquet(proc / "price_da_eur.parquet").iloc[:, 0]
    res = pd.read_parquet(proc / "res_forecast.parquet")
    tso = pd.read_parquet(proc / "tso_forecast.parquet").iloc[:, 0]

    print("=" * 60)
    print(f"TFT price campaign | contexts={CONTEXTS} d_model={D_MODEL} "
          f"seed={SEED} test_days={args.days} | device={device()}")
    print("=" * 60, flush=True)

    all_dates = sorted(set(price.index.tz_convert(TZ).date))
    last = all_dates[-2]  # last fully priced day
    test_days = [d for d in all_dates if (last - d).days < args.days]

    rows, t0 = [], time.time()
    naive = price.reindex(price.index)  # for rMAE below
    for ctx in CONTEXTS:
        pred = walk_forward_tft_price(price, res, tso, ctx, test_days, all_dates)
        if pred is None:
            continue
        y = price.reindex(pred.index)
        mae = float((pred["p50"] - y).abs().mean())
        naive_p = price.reindex(pred.index - pd.Timedelta(hours=24)).to_numpy()
        naive_mae = float(pd.Series(naive_p, index=pred.index).sub(y).abs().mean())
        spike_cut = y.quantile(0.95)
        spike = y >= spike_cut
        rows.append({
            "model": f"tft_price_ctx{ctx}",
            "mae": mae,
            "rmae_vs_naive1d": mae / naive_mae,
            "coverage_80_pct": 100.0 * ((y >= pred["p10"]) & (y <= pred["p90"])).mean(),
            "spike_mae": float((pred.loc[spike, "p50"] - y[spike]).abs().mean()),
            "n_hours": len(y),
        })
        print(f"[ctx{ctx}] MAE {mae:.2f} | rMAE {mae/naive_mae:.3f}", flush=True)

    total_h = (time.time() - t0) / 3600
    table = pd.DataFrame(rows).set_index("model")
    out_dir = Path("reports/backtests")
    stamp = f"{pd.Timestamp.now(TZ).date()}_tft_price"
    table.to_csv(out_dir / f"{stamp}.csv")
    md = [
        f"# TFT price campaign (long-context screening) — {stamp}",
        "",
        f"Test: last {args.days} days, monthly refits, 1 seed (screening).",
        "Question: does attention over 4-12 weeks of past prices beat",
        "tabular models limited to 7-day lags?",
        f"Runtime: {total_h:.1f} h. Tabular references on the 2-yr test:",
        "LGBM MAE 17.8 (rMAE 0.638), LEAR 18.5 (0.660) — different test",
        "span, direction-of-effect comparison only.",
        "",
        table.round(3).to_markdown(),
        "",
    ]
    (out_dir / f"{stamp}.md").write_text("\n".join(md))
    print(table.round(3).to_string())
    print(f"Saved to {out_dir}/{stamp}.(csv|md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
