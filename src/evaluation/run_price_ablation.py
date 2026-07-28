"""Feature-group ablation for the price model — the SHAP cross-check.

SHAP attributes credit within one fitted model; correlated features
split credit, so SHAP rank is NOT value-of-information. This script
answers the retrain question instead: drop a whole feature group,
walk the model forward again, measure the MAE it costs.

Both numbers belong together: SHAP explains the shipped model,
ablation prices each input group. When they disagree, redundancy is
usually the reason (see docs/notes/learning/13_shap_vs_ablation.tex).

Walk-forward on the last `--days` days (default 180), weekly refits,
LightGBM P50 booster only (MAE needs no band).

Run: python -m src.evaluation.run_price_ablation [--days 180]
"""

from __future__ import annotations

import argparse
import sys

import lightgbm as lgb
import pandas as pd

from src.config import load_config
from src.evaluation.backtest import walk_forward_backtest
from src.evaluation.run_price_backtest import assemble_price_features
from src.models.gbm import PARAMS

GROUPS = {
    "price_lags": lambda c: c.startswith(("price_lag_", "price_mean", "price_d")),
    "res_forecast": lambda c: c in ("solar_fcst_mw", "wind_on_fcst_mw", "wind_off_fcst_mw"),
    "tso_load_fcst": lambda c: c == "tso_forecast_mw",
    "load_lags": lambda c: c.startswith("load_"),
    "calendar": lambda c: c in (
        "hour_local", "day_of_week", "month", "is_weekend", "is_holiday",
        "is_bridge_day", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    ),
}


class _P50Booster:
    """Median-only LGBM in the engine's model contract.

    The ablation only needs P50 MAE; p10/p90 just repeat p50 so the
    shared walk-forward engine (one cutoff implementation, not two —
    validation lesson) can be reused as-is.
    """

    name = "ablation_p50"

    def __init__(self) -> None:
        self._m = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **PARAMS)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self._m.fit(x, y)

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        p = pd.Series(self._m.predict(x), index=x.index)
        return pd.DataFrame({"p10": p, "p50": p, "p90": p})


def walk_forward_mae(
    x: pd.DataFrame, y: pd.Series, test_start: pd.Timestamp, tz: str
) -> float:
    result = walk_forward_backtest(
        _P50Booster, x, y, test_start.tz_convert("UTC"),
        train_window_days=365, refit_every_days=7, tz=tz,
        target_availability="day_ahead",
    )
    pred = result.predictions["p50"]
    return float((pred - y.reindex(pred.index)).abs().mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=180)
    args = parser.parse_args()

    cfg = load_config()
    proc = cfg.paths["data_processed"]
    price = pd.read_parquet(proc / "price_da_eur.parquet").iloc[:, 0]
    load = pd.read_parquet(proc / "load.parquet").iloc[:, 0]
    tso = pd.read_parquet(proc / "tso_forecast.parquet").iloc[:, 0]
    res = pd.read_parquet(proc / "res_forecast.parquet")

    tz = cfg.timezone_local
    last = min(price.index[-1], load.index[-1]).tz_convert(tz) - pd.Timedelta(days=1)
    test_start = pd.Timestamp((last - pd.Timedelta(days=args.days)).date(), tz=tz)
    first = test_start - pd.Timedelta(days=380)

    print(f"Features {first.date()} → {last.date()}, test from {test_start.date()}")
    x = assemble_price_features(
        price, load, tso, tz,
        pd.Timestamp(first.date(), tz=tz), pd.Timestamp(last.date(), tz=tz), res=res,
    )
    y = price.reindex(x.index)

    rows = []
    base = walk_forward_mae(x, y, test_start, tz)
    rows.append({"config": "full", "mae": base, "delta_vs_full": 0.0})
    print(f"full: MAE {base:.2f}")
    for name, match in GROUPS.items():
        cols = [c for c in x.columns if not match(c)]
        dropped = len(x.columns) - len(cols)
        mae = walk_forward_mae(x[cols], y, test_start, tz)
        rows.append({"config": f"drop {name} ({dropped} cols)", "mae": mae,
                     "delta_vs_full": mae - base})
        print(f"drop {name}: MAE {mae:.2f} (Δ {mae - base:+.2f})")

    table = pd.DataFrame(rows).set_index("config").round(3)
    out_dir = proc.parent.parent / "reports" / "backtests"
    stamp = f"{pd.Timestamp.now(tz).date()}_price_group_ablation"
    table.to_csv(out_dir / f"{stamp}.csv")
    md = [
        f"# Price LGBM feature-group ablation — {stamp}",
        "",
        f"Walk-forward, weekly refits, last {args.days} days, P50 MAE (EUR/MWh).",
        "Retrain ablation = value of information. Compare with SHAP rank",
        "(reports/sensitivity/shap_importance_price.csv) — they answer",
        "different questions; the gap between them measures redundancy.",
        "",
        table.to_markdown(),
        "",
    ]
    (out_dir / f"{stamp}.md").write_text("\n".join(md))
    print(f"\nWritten to {out_dir}/{stamp}.(csv|md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
