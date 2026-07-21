"""Price backtest runner: LEAR and naives on PL day-ahead prices.

Same walk-forward engine as the load campaign. Differences:
- Target: price_da_eur.parquet (EUR/MWh, ENTSO-E).
- No MAPE. Prices cross zero; MAPE is meaningless. Headline metric is
  rMAE — MAE relative to naive-yesterday (Lago et al. 2021 convention).
- Band coverage reported: share of hours where the actual fell inside
  [p10, p90]. Nominal is 80%. Spikes live in the tails; coverage shows
  whether the band is honest there.

Run: python -m src.evaluation.run_price_backtest [--test-start 2024-07-16]
     [--models price_naive_yesterday,price_naive_week,lear]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

import src.models.gbm  # noqa: F401  (populates REGISTRY)
import src.models.price  # noqa: F401
from src.config import Config, load_config
from src.evaluation.backtest import BacktestResult, walk_forward_backtest
from src.evaluation.metrics import mae, pinball_loss, rmse, winkler_score
from src.features.price_matrix import build_price_features
from src.models.base import REGISTRY
from src.pipeline.daily_run import local_day_hours_utc, shift_local_day

PRICE_MODELS = ["price_naive_yesterday", "price_naive_week", "lear", "lgbm_quantile"]


def assemble_price_features(
    price: pd.Series, load: pd.Series, tso: pd.Series, tz: str,
    start: pd.Timestamp, end: pd.Timestamp, res: pd.DataFrame | None = None,
    outages: pd.DataFrame | None = None, fuel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Cutoff-safe X for every local day in [start, end]. One build per day."""
    frames = []
    day = start
    while day <= end:
        hours = local_day_hours_utc(day, tz)
        price_cutoff = hours[0]  # first delivery hour: all of D-1 is known
        load_cutoff = shift_local_day(day, -1, tz) + pd.Timedelta(hours=9)
        frames.append(
            build_price_features(
                hours, price, load, price_cutoff, load_cutoff, tso=tso, res=res,
                outages=outages, fuel=fuel,
            )
        )
        day = shift_local_day(day, 1, tz)
    return pd.concat(frames)


def summarize_price(results: list[BacktestResult], y: pd.Series) -> pd.DataFrame:
    """One row per model. rMAE vs naive-yesterday instead of MAPE.

    Spike columns evaluate the top 5% priciest hours (by actual price)
    within the evaluated period: a model can look fine on pooled MAE and
    still miss every spike.
    """
    y = y.reindex(results[0].predictions.index)
    spike_cut = y.quantile(0.95)
    rows = []
    for r in results:
        p = r.predictions
        y_r = y.reindex(p.index)
        inside = (y_r >= p["p10"]) & (y_r <= p["p90"])
        spike = y_r >= spike_cut
        rows.append(
            {
                "model": r.model_name,
                "mae": mae(y, p["p50"]),
                "rmse": rmse(y, p["p50"]),
                "pinball_p10": pinball_loss(y, p["p10"], 0.1),
                "pinball_p50": pinball_loss(y, p["p50"], 0.5),
                "pinball_p90": pinball_loss(y, p["p90"], 0.9),
                "coverage_80_pct": 100.0 * inside.mean(),
                "winkler": winkler_score(y_r, p),
                "spike_mae": mae(y_r[spike], p.loc[spike, "p50"]),
                "spike_cover_pct": 100.0 * (y_r[spike] <= p.loc[spike, "p90"]).mean(),
                "n_hours": int(p["p50"].notna().sum()),
            }
        )
    table = pd.DataFrame(rows).set_index("model")
    if "price_naive_yesterday" in table.index:
        table["rmae"] = table["mae"] / table.loc["price_naive_yesterday", "mae"]
    return table.sort_values("mae")


def main() -> int:
    cfg: Config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(PRICE_MODELS))
    parser.add_argument("--test-start", default=None, help="YYYY-MM-DD local")
    parser.add_argument("--tag", default=None, help="suffix for output names")
    parser.add_argument("--with-outages", action="store_true",
                        help="add the unavailable-capacity feature")
    parser.add_argument("--with-fuel", action="store_true",
                        help="add TTF gas + EUA-proxy settlement features")
    args = parser.parse_args()

    proc = cfg.paths["data_processed"]
    price_path = proc / "price_da_eur.parquet"
    if not price_path.exists():
        print("Missing price_da_eur.parquet — run "
              "`python -m src.ingestion.backfill --only entsoe_prices` first.")
        return 1
    price = pd.read_parquet(price_path).iloc[:, 0]
    load = pd.read_parquet(proc / "load.parquet").iloc[:, 0]
    tso = pd.read_parquet(proc / "tso_forecast.parquet").iloc[:, 0]
    res_path = proc / "res_forecast.parquet"
    res = pd.read_parquet(res_path) if res_path.exists() else None
    if res is None:
        print("res_forecast.parquet missing — running WITHOUT wind/solar features")
    outages = None
    if args.with_outages:
        outages = pd.read_parquet(proc / "outages.parquet")
    fuel = None
    if args.with_fuel:
        fuel = pd.read_parquet(proc / "fuel_daily.parquet")

    tz = cfg.timezone_local
    first = price.index[0].tz_convert(tz) + pd.Timedelta(days=30)
    last = min(price.index[-1], load.index[-1]).tz_convert(tz) - pd.Timedelta(days=1)
    test_start = (
        pd.Timestamp(args.test_start, tz=tz)
        if args.test_start
        else shift_local_day(last, -365, tz)
    )

    print(f"Assembling price features {first.date()} → {last.date()} ...")
    x = assemble_price_features(
        price, load, tso, tz,
        pd.Timestamp(first.date(), tz=tz), pd.Timestamp(last.date(), tz=tz),
        res=res, outages=outages, fuel=fuel,
    )
    y = price.reindex(x.index)

    results: list[BacktestResult] = []
    for name in args.models.split(","):
        print(f"Backtesting {name} ...")
        results.append(
            walk_forward_backtest(REGISTRY[name], x, y, test_start.tz_convert("UTC"))
        )

    tag = "price" + (f"_{args.tag}" if args.tag else "")
    preds_dir = proc / f"backtest_preds_{tag}"
    preds_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        r.predictions.to_parquet(preds_dir / f"{r.model_name}.parquet")

    table = summarize_price(results, y)
    out_dir = proc.parent.parent / "reports" / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{pd.Timestamp.now(tz).date()}_{tag}"
    table.to_csv(out_dir / f"{stamp}_summary.csv")
    md = [
        f"# Price backtest summary — {stamp}",
        "",
        f"Target: PL day-ahead price, EUR/MWh (ENTSO-E). "
        f"Test period: {test_start.date()} → {last.date()}.",
        "rMAE = MAE / MAE(naive-yesterday). No MAPE: prices cross zero.",
        "coverage_80_pct: share of actuals inside [P10, P90]; nominal 80.",
        "",
        table.round(3).to_markdown(),
        "",
    ]
    (out_dir / f"{stamp}_summary.md").write_text("\n".join(md))
    print(table.round(3).to_string())
    print(f"\nWritten to {out_dir}/{stamp}_summary.(csv|md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
