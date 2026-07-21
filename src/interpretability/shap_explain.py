"""SHAP explanations for the LightGBM model. Hard rule 3: every shipped
model explains itself, and the daily report speaks plain words.

Run: python -m src.interpretability.shap_explain
Outputs: reports/figures/shap_summary.png + printed top drivers.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import REPO_ROOT, load_config
from src.evaluation.run_backtest import assemble_features
from src.features.weather import load_weather_forecast_history
from src.models.gbm import LightGBMQuantile
from src.pipeline.daily_run import shift_local_day

FIGURES = REPO_ROOT / "reports" / "figures"

# Feature → plain words, for reports a manager reads in 60 seconds.
PLAIN_WORDS = {
    "load_lag_48h": "load two days ago at this hour",
    "load_lag_72h": "load three days ago at this hour",
    "load_lag_168h": "load last week at this hour",
    "load_lag_336h": "load two weeks ago at this hour",
    "load_lag_504h": "load three weeks ago at this hour",
    "load_lag_672h": "load four weeks ago at this hour",
    "load_mean_7d": "the average load level of the past week",
    "hour_local": "the hour of the day",
    "hour_sin": "the hour of the day",
    "hour_cos": "the hour of the day",
    "day_of_week": "the day of the week",
    "is_weekend": "weekend vs workday",
    "is_holiday": "a public holiday",
    "is_bridge_day": "a bridge day (workday squeezed next to a holiday)",
    "month": "the season",
    "doy_sin": "the time of year",
    "doy_cos": "the time of year",
    "temperature_2m": "temperature",
    "wind_speed_10m": "wind",
    "cloud_cover": "cloud cover",
    "shortwave_radiation": "sunshine",
    "relative_humidity_2m": "humidity",
    "heating_degrees": "heating demand (cold below 15°C)",
    "cooling_degrees": "cooling demand (heat above 22°C)",
}


def top_drivers(shap_values: np.ndarray, columns: list[str], n: int = 3) -> list[str]:
    """Top-n features by mean |SHAP|, deduplicated by plain-words phrase."""
    order = np.argsort(np.abs(shap_values).mean(axis=0))[::-1]
    phrases: list[str] = []
    for i in order:
        phrase = PLAIN_WORDS.get(columns[i], columns[i])
        if phrase not in phrases:
            phrases.append(phrase)
        if len(phrases) == n:
            break
    return phrases


def main() -> int:
    cfg = load_config()
    load = pd.read_parquet(cfg.paths["data_processed"] / "load.parquet").iloc[:, 0]
    weather = load_weather_forecast_history(cfg)
    tz = cfg.timezone_local

    end = shift_local_day(load.index[-1].tz_convert(tz), -2, tz)
    start = shift_local_day(end, -365, tz)
    x = assemble_features(load, weather, tz, start, end).dropna()
    y = load.reindex(x.index)

    model = LightGBMQuantile()
    model.fit(x, y)

    explainer = shap.TreeExplainer(model._models[0.5])
    sample = x.sample(n=min(2000, len(x)), random_state=0)
    values = explainer.shap_values(sample)

    FIGURES.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(values, sample, show=False, max_display=15)
    plt.title("SHAP — what drives the P50 forecast (last 365 days)")
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_summary.png", dpi=120)
    plt.close()

    drivers = top_drivers(values, list(sample.columns))
    print("Top 3 drivers, plain words:")
    for i, d in enumerate(drivers, 1):
        print(f"  {i}. {d}")
    print("made    reports/figures/shap_summary.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
