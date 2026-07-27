"""CRPS-weighted ensemble over stored predictions.

Members are conformally calibrated variants where they exist. TFT is
excluded from the 2-yr table (its stored preds cover other windows);
the two-window discipline from RESULTS.md applies.

Run: python -m src.evaluation.run_price_ensemble
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.evaluation.backtest import BacktestResult
from src.evaluation.conformal import rolling_conformal
from src.evaluation.ensemble import LOCAL_TZ, blend, crps3, rolling_weights
from src.evaluation.run_price_backtest import summarize_price

PROC = Path("data/processed")
MEMBERS = {
    "lgbm": PROC / "backtest_preds_price_res/lgbm_quantile_conformal.parquet",
    "lear": PROC / "backtest_preds_price_res/lear_conformal.parquet",
    "chronos": PROC / "backtest_preds_price_chronos2yr/chronos_bolt_zs.parquet",
    "timesfm": PROC / "backtest_preds_price_timesfm2yr/timesfm_zs.parquet",
    "moirai_cov": PROC / "backtest_preds_price_moirai2yr/moirai_cov.parquet",
    # 1095d-window variants (deep-history campaign follow-up)
    "lgbm_1095": PROC / "backtest_preds_price_res1095/lgbm_quantile.parquet",
    "lear_1095": PROC / "backtest_preds_price_res1095/lear.parquet",
    # best deep model, full-2yr preds (survive under reports/, not data/)
    "tft": Path("reports/sensitivity/tft/preds_tft730_2yr_ens3.parquet"),
}
# raw bands get CQR before blending (FMs ship raw; the 1095d runs are
# stored raw so the shared calibration script never sees their offsets)
CALIBRATE = ("chronos", "timesfm", "moirai_cov", "lgbm_1095", "lear_1095",
             "tft")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    # pre-declared default (PLAN Phase 6 S5): champion + LEAR + best FM
    parser.add_argument("--members", default="lgbm,lear,chronos")
    parser.add_argument("--suffix", default="",
                        help="suffix for output names, e.g. '1095' — keeps "
                             "variant runs from clobbering the shipped blend")
    args = parser.parse_args()
    sfx = f"_{args.suffix}" if args.suffix else ""

    cfg = load_config()
    y = pd.read_parquet(PROC / "price_da_eur.parquet").iloc[:, 0]

    members = {}
    for name in args.members.split(","):
        path = MEMBERS[name]
        if not path.exists():
            print(f"skip {name} (no preds)")
            continue
        p = pd.read_parquet(path)
        if name in CALIBRATE:
            p = rolling_conformal(p, y)
        members[name] = p

    scores = {n: crps3(p, y) for n, p in members.items()}
    idx = None
    for p in members.values():
        idx = p.index if idx is None else idx.intersection(p.index)
    days = sorted(set(idx.tz_convert(LOCAL_TZ).date))

    w = rolling_weights(scores, days)
    ens = blend(members, w)
    equal = blend(members, w * 0 + 1.0 / len(members))
    # the raw blend over-covers (84.2% vs nominal 80): a second CQR pass
    # on the blended band tightens it (Q goes negative when over-covered)
    ens_cqr = rolling_conformal(ens, y)

    results = [BacktestResult("ens_crps", ens),
               BacktestResult("ens_crps_cqr", ens_cqr),
               BacktestResult("ens_equal", equal)]
    for n, p in members.items():
        results.append(BacktestResult(n, p.reindex(ens.index)))
    naive = pd.read_parquet(
        PROC / "backtest_preds_price_res/price_naive_yesterday.parquet")
    results.append(BacktestResult("price_naive_yesterday",
                                  naive.reindex(ens.index)))

    table = summarize_price(results, y.reindex(ens.index))
    tz = cfg.timezone_local
    stamp = f"{pd.Timestamp.now(tz).date()}_price_ensemble{sfx}"
    out_dir = Path("reports/backtests")
    table.to_csv(out_dir / f"{stamp}_summary.csv")
    ens.to_parquet(PROC / f"backtest_preds_price_res/ens_crps{sfx}.parquet")
    ens_cqr.to_parquet(
        PROC / f"backtest_preds_price_res/ens_crps_cqr{sfx}.parquet")
    w.to_csv(out_dir / f"{stamp}_weights.csv")

    md = [
        f"# CRPS-weighted ensemble — {stamp}", "",
        f"Members: {', '.join(members)}. Weights: inverse trailing-60d",
        "crps3 (mean pinball over the 3 quantiles), equal during warm-up.",
        "FM members conformalized before blending.", "",
        table.round(3).to_markdown(), "",
    ]
    (out_dir / f"{stamp}_summary.md").write_text("\n".join(md))
    print(table.round(3).to_string())
    print(f"\nWritten {out_dir}/{stamp}_summary.(csv|md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
