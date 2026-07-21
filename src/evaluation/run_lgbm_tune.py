"""Q3b: LightGBM mini-tuning — two configs beyond the default, walk-forward.

Not a grid search. Two informed bets:
- deeper: more leaves + more trees, lower lr (fits sharper interactions)
- regular: fewer leaves, stronger subsampling (fights the tail overfit)
Each runs the same honest walk-forward as the default did.
"""

from __future__ import annotations

import sys

import pandas as pd

from src.config import load_config
from src.evaluation.backtest import BacktestResult, summarize, walk_forward_backtest
from src.evaluation.run_backtest import assemble_features
from src.features.weather import load_weather_forecast_history
from src.models.gbm import LightGBMQuantile
from src.pipeline.daily_run import shift_local_day

CONFIGS = {
    "lgbm_deeper": {"n_estimators": 900, "learning_rate": 0.03, "num_leaves": 127,
                    "min_child_samples": 30, "subsample": 0.9, "subsample_freq": 1,
                    "colsample_bytree": 0.9, "random_state": 0, "verbosity": -1},
    "lgbm_regular": {"n_estimators": 600, "learning_rate": 0.04, "num_leaves": 31,
                     "min_child_samples": 60, "subsample": 0.7, "subsample_freq": 1,
                     "colsample_bytree": 0.7, "random_state": 0, "verbosity": -1},
}


def make_factory(name: str, params: dict):
    def factory():
        m = LightGBMQuantile()
        m.name = name

        # swap params for this instance only
        m._params = params
        orig_fit = m.fit

        def fit(x, y):
            import lightgbm as lgb

            from src.models.base import QUANTILES

            m._columns = list(x.columns)
            for q in QUANTILES:
                mdl = lgb.LGBMRegressor(objective="quantile", alpha=q, **params)
                mdl.fit(x, y)
                m._models[q] = mdl

        m.fit = fit
        return m

    return factory


def main() -> int:
    cfg = load_config()
    load = pd.read_parquet(cfg.paths["data_processed"] / "load.parquet").iloc[:, 0]
    tso = pd.read_parquet(cfg.paths["data_processed"] / "tso_forecast.parquet").iloc[:, 0]
    weather = load_weather_forecast_history(cfg)
    tz = cfg.timezone_local
    first = load.index[0].tz_convert(tz) + pd.Timedelta(days=30)
    last = load.index[-1].tz_convert(tz) - pd.Timedelta(days=1)
    test_start = shift_local_day(last, -365, tz)

    # Tuning runs INCLUDE the TSO feature — tuning the best-known setup.
    x = assemble_features(load, weather, tz, pd.Timestamp(first.date(), tz=tz),
                          pd.Timestamp(last.date(), tz=tz), tso=tso)
    y = load.reindex(x.index)

    results: list[BacktestResult] = []
    for name, params in CONFIGS.items():
        print(f"Backtesting {name} ...", flush=True)
        results.append(
            walk_forward_backtest(make_factory(name, params), x, y,
                                  test_start.tz_convert("UTC"))
        )
    preds_dir = cfg.paths["data_processed"] / "backtest_preds_fcst_tso"
    preds_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        r.predictions.to_parquet(preds_dir / f"{r.model_name}.parquet")
    print(summarize(results, y).round(2).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
