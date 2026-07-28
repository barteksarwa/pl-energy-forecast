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
from src.interpretability.plain_words import top_phrases
from src.features.weather import load_weather_forecast_history
from src.models.gbm import LightGBMQuantile
from src.pipeline.daily_run import shift_local_day

FIGURES = REPO_ROOT / "reports" / "figures"


def top_drivers(shap_values: np.ndarray, columns: list[str], n: int = 3) -> list[str]:
    """Top-n features by mean |SHAP|, deduplicated by plain-words phrase."""
    return top_phrases(np.abs(shap_values).mean(axis=0), columns, n)


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
