"""Apply the EPF-standard statistical tests to stored predictions.

DM tests (daily multivariate, Lago et al. 2021 protocol) on the key
model claims, plus Kupiec/Christoffersen validation of the shipped
conformal bands. Reads parquet only; nothing is re-run.

Run: python -m src.evaluation.run_stats_tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.evaluation.stats_tests import (
    band_violations,
    christoffersen_test,
    daily_losses,
    dm_test,
    kupiec_test,
)

COMPARISONS = [
    # (label, path_a, path_b) — one-sided H1: A more accurate than B
    ("lgbm_win1095 vs lgbm_win365",
     "data/processed/backtest_preds_price_win1095/lgbm_quantile.parquet",
     "data/processed/backtest_preds_price_win365/lgbm_quantile.parquet"),
    ("lgbm vs lear (2-yr)",
     "data/processed/backtest_preds_price_res/lgbm_quantile.parquet",
     "data/processed/backtest_preds_price_res/lear.parquet"),
    ("lgbm vs tft730_ens3 (2-yr)",
     "data/processed/backtest_preds_price_res/lgbm_quantile.parquet",
     "reports/sensitivity/tft/preds_tft730_2yr_ens3.parquet"),
    ("lgbm vs naive (2-yr)",
     "data/processed/backtest_preds_price_res/lgbm_quantile.parquet",
     "data/processed/backtest_preds_price_res/price_naive_yesterday.parquet"),
]

BANDS = [
    ("lgbm_quantile_conformal",
     "data/processed/backtest_preds_price_res/lgbm_quantile_conformal.parquet"),
    ("lear_conformal",
     "data/processed/backtest_preds_price_res/lear_conformal.parquet"),
]


def main() -> int:
    cfg = load_config()
    tz = cfg.timezone_local
    y = pd.read_parquet(
        cfg.paths["data_processed"] / "price_da_eur.parquet").iloc[:, 0]

    dm_rows = []
    for label, pa, pb in COMPARISONS:
        if not (Path(pa).exists() and Path(pb).exists()):
            print(f"skip {label} (missing preds)")
            continue
        la = daily_losses(pd.read_parquet(pa)["p50"], y, tz)
        lb = daily_losses(pd.read_parquet(pb)["p50"], y, tz)
        stat, p = dm_test(la, lb)
        dm_rows.append({
            "comparison": label, "dm_stat": round(stat, 3),
            "p_one_sided": f"{p:.2e}",
            "n_days": len(la.index.intersection(lb.index)),
            "verdict": "A significantly better" if p < 0.05
            else "no significant difference",
        })

    cov_rows = []
    for label, path in BANDS:
        if not Path(path).exists():
            continue
        v = band_violations(pd.read_parquet(path), y)
        lr_uc, p_uc = kupiec_test(v, coverage=0.8)
        lr_ind, p_ind = christoffersen_test(v)
        cov_rows.append({
            "band": label, "violation_rate": round(float(v.mean()), 4),
            "kupiec_p": f"{p_uc:.3f}",
            "christoffersen_p": f"{p_ind:.2e}",
            "n_hours": len(v),
        })

    dm_tbl = pd.DataFrame(dm_rows).set_index("comparison")
    cov_tbl = pd.DataFrame(cov_rows).set_index("band")
    out_dir = Path("reports/backtests")
    stamp = f"{pd.Timestamp.now(tz).date()}_stats_tests"
    md = [
        f"# Statistical tests — {stamp}", "",
        "Diebold-Mariano on daily L1 loss differentials (multivariate",
        "version, Lago et al. 2021). One-sided H1: model A more accurate.", "",
        dm_tbl.to_markdown(), "",
        "Band validation, shipped conformal bands (nominal 20% violations):",
        "Kupiec = unconditional coverage; Christoffersen = violations",
        "independent (low p = violations CLUSTER).", "",
        cov_tbl.to_markdown(), "",
    ]
    (out_dir / f"{stamp}.md").write_text("\n".join(md))
    dm_tbl.to_csv(out_dir / f"{stamp}_dm.csv")
    cov_tbl.to_csv(out_dir / f"{stamp}_coverage.csv")
    print(dm_tbl.to_string())
    print()
    print(cov_tbl.to_string())
    print(f"\nWritten {out_dir}/{stamp}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
