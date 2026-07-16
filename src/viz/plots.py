"""Plot functions for fetched data. One function per data kind.

All figures go to reports/figures/. Filenames are stable so re-runs overwrite.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.viz.style import BAND_ALPHA, BLUE, ORANGE, apply_style

LOCAL_TZ = "Europe/Warsaw"


def _finish(fig: plt.Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_weather_overview(weather: pd.DataFrame, city: str, out: Path) -> Path:
    """Small multiples: one panel per weather variable, single series each."""
    apply_style()
    cols = list(weather.columns)
    fig, axes = plt.subplots(len(cols), 1, figsize=(9, 1.9 * len(cols)), sharex=True)
    axes = [axes] if len(cols) == 1 else list(axes)
    units = {
        "temperature_2m": "°C",
        "wind_speed_10m": "km/h",
        "cloud_cover": "%",
        "shortwave_radiation": "W/m²",
        "relative_humidity_2m": "%",
    }
    for ax, col in zip(axes, cols):
        ax.plot(weather.index, weather[col], color=BLUE)
        ax.set_ylabel(units.get(col, ""))
        ax.set_title(col, loc="left")
    axes[-1].set_xlabel("Time (UTC)")
    fig.suptitle(f"Weather — {city} (Open-Meteo)", x=0.01, ha="left")
    return _finish(fig, out)


def plot_load_vs_tso(load: pd.Series, tso: pd.Series | None, out: Path) -> Path:
    """Actual load, with the TSO day-ahead forecast overlaid if available."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(load.index, load.values, color=BLUE, label="Actual load")
    if tso is not None and not tso.dropna().empty:
        ax.plot(tso.index, tso.values, color=ORANGE, label="TSO day-ahead forecast")
    ax.set_ylabel("Load (MW)")
    ax.set_xlabel("Time (UTC)")
    ax.set_title("Poland — actual load vs TSO forecast (ENTSO-E)", loc="left")
    ax.legend(frameon=False)
    return _finish(fig, out)


def plot_forecast_band(
    forecast: pd.DataFrame,
    target_date: str,
    out: Path,
    actual: pd.Series | None = None,
    unit: str = "Load (MW)",
) -> Path:
    """Fan chart: P50 line, P10-P90 band. Displayed in local time.

    Living figure: published with the band only; the morning after the
    target day, the daily run re-renders it WITH the realized series so
    every report's chart eventually shows forecast vs reality.
    """
    apply_style()
    local = forecast.tz_convert(LOCAL_TZ)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.fill_between(
        local.index, local["p10"], local["p90"],
        color=BLUE, alpha=BAND_ALPHA, linewidth=0, label="P10–P90",
    )
    ax.plot(local.index, local["p50"], color=BLUE, label="P50")
    title = f"Day-ahead forecast — {target_date}"
    if actual is not None:
        act = actual.tz_convert(LOCAL_TZ)
        ax.plot(act.index, act.values, color="black", linewidth=2.0, label="realized")
        title += " — vs realized"
    ax.set_ylabel(unit)
    ax.set_xlabel(f"Time ({LOCAL_TZ})")
    ax.set_title(title, loc="left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=local.index.tz))
    ax.legend(frameon=False)
    return _finish(fig, out)


def plot_temperature_history(
    city_temps: dict[str, pd.Series], weights: dict[str, float], out: Path
) -> Path:
    """Population-weighted daily mean temperature over the full backfill.

    Sanity check for the backfill and a preview of the M2 weather feature.
    """
    apply_style()
    total = sum(weights.values())
    weighted = sum(s * (weights[name] / total) for name, s in city_temps.items())
    daily = weighted.resample("1D").mean()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(daily.index, daily.values, color=BLUE, linewidth=1.0)
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Time (UTC)")
    ax.set_title(
        f"Population-weighted daily mean temperature, {len(weights)} cities (Open-Meteo ERA5)",
        loc="left",
    )
    return _finish(fig, out)


def plot_price_history(price: pd.Series, out: Path) -> Path:
    """Day-ahead price: daily mean line + daily min-max band. Spikes = the story."""
    apply_style()
    daily_mean = price.resample("1D").mean()
    lo, hi = price.resample("1D").min(), price.resample("1D").max()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.fill_between(lo.index, lo, hi, color=BLUE, alpha=BAND_ALPHA,
                    linewidth=0, label="daily min–max")
    ax.plot(daily_mean.index, daily_mean.values, color=BLUE, linewidth=1.0,
            label="daily mean")
    ax.set_ylabel("Price (PLN/MWh)")
    ax.set_xlabel("Time (UTC)")
    ax.set_title("PL day-ahead price, SDAC (PSE csdac-pln)", loc="left")
    ax.legend(frameon=False)
    return _finish(fig, out)


def plot_load_history(load: pd.Series, out: Path) -> Path:
    """Long-history overview: daily mean load. For backfill sanity checks."""
    apply_style()
    daily = load.resample("1D").mean()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(daily.index, daily.values, color=BLUE, linewidth=1.2)
    ax.set_ylabel("Daily mean load (MW)")
    ax.set_xlabel("Time (UTC)")
    ax.set_title("Poland — daily mean load, full history (ENTSO-E)", loc="left")
    return _finish(fig, out)


def plot_res_forecast(res: pd.DataFrame, out: Path) -> Path:
    """Wind + solar day-ahead forecast: daily means, stacked view of the mix."""
    apply_style()
    daily = res.resample("1D").mean()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(daily.index, daily["solar_fcst_mw"], color=ORANGE, linewidth=1.0,
            label="solar")
    wind = daily["wind_on_fcst_mw"] + daily["wind_off_fcst_mw"]
    ax.plot(daily.index, wind, color=BLUE, linewidth=1.0, label="wind (on+off)")
    ax.set_ylabel("Daily mean forecast (MW)")
    ax.set_xlabel("Time (UTC)")
    ax.set_title("PL wind + solar day-ahead forecast (ENTSO-E 14.1.D)", loc="left")
    ax.legend(frameon=False)
    return _finish(fig, out)
