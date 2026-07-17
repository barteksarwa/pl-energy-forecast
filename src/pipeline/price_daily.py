"""Daily price step: score yesterday's price forecast, forecast tomorrow.

Mirrors the desk morning routine for the price desk:
1. Pull the latest day-ahead prices + RES forecasts (incremental).
2. Score yesterday's saved forecast against the REALIZED price.
3. Fit LEAR on the trailing window, forecast tomorrow, save CSV.
4. Plot both: yesterday forecast-vs-realized, tomorrow's band.

Called from daily_run inside its own try/except — a price failure must
never kill the load report.

Timing (documented, not hidden): tomorrow's TSO load forecast
(published ~09:00 D-1) and RES forecast (~18:00 D-1) may not exist yet
when the run happens early. Both are persisted from the same clock
hour of the day before (persist_24h) — shape-preserving, flagged in
the report every time it happens.

Model choice: LEAR publishes; LightGBM waits its turn. LGBM has the
better MAE (rMAE 0.638 vs 0.660) and, since Phase 2.5, a calibrated
band too (conformal: 51% → 79% coverage). But desks do not swap the
published model on a backtest — a challenger earns promotion through
a shadow window (PLAN M9). LEAR is the incumbent; its published band
is conformally widened (config/price_conformal.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import Config
from src.features.price_matrix import build_price_features
from src.models.price import PriceLEAR


def _local_day_hours_utc(day: pd.Timestamp, tz: str) -> pd.DatetimeIndex:
    from src.pipeline.daily_run import local_day_hours_utc

    return local_day_hours_utc(day, tz)


def persist_24h(
    obj: pd.Series | pd.DataFrame, hours: pd.DatetimeIndex
) -> pd.Series | pd.DataFrame:
    """Fill `hours` missing from `obj` with the value 24h earlier.

    Shape-preserving ops proxy for series published on a daily schedule
    (TSO load forecast ~09:00 D-1, RES forecast ~18:00 D-1) when the run
    happens before publication. Callers flag the persist in the report.
    """
    missing = hours.difference(obj.index)
    if not len(missing):
        return obj
    persisted = obj.reindex(missing - pd.Timedelta(hours=24))
    persisted.index = missing
    return pd.concat([obj, persisted]).sort_index()


def _assemble(
    price: pd.Series, load: pd.Series, tso: pd.Series, res: pd.DataFrame,
    tz: str, start: pd.Timestamp, end: pd.Timestamp,
    fuel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    from src.pipeline.daily_run import shift_local_day

    frames = []
    day = start
    while day <= end:
        hours = _local_day_hours_utc(day, tz)
        frames.append(
            build_price_features(
                hours, price, load,
                price_cutoff=hours[0],
                load_cutoff=shift_local_day(day, -1, tz) + pd.Timedelta(hours=9),
                tso=tso, res=res, fuel=fuel,
            )
        )
        day = shift_local_day(day, 1, tz)
    return pd.concat(frames)


def price_daily_step(
    cfg: Config, today_local: pd.Timestamp
) -> tuple[dict[str, float], list[str], list[str]]:
    """Returns (scores, report_lines, oddities). Raises on hard failure —
    the caller isolates it."""
    from src.ingestion.backfill import (
        backfill_entsoe_prices,
        backfill_entsoe_res,
        backfill_fuel,
    )
    from src.pipeline.daily_run import shift_local_day
    from src.viz.plots import plot_forecast_band

    tz = cfg.timezone_local
    yesterday = shift_local_day(today_local, -1, tz)
    tomorrow = shift_local_day(today_local, 1, tz)
    proc = cfg.paths["data_processed"]

    # 1. Incremental data pull (resume-based, cheap after backfill).
    backfill_entsoe_prices(cfg)
    backfill_entsoe_res(cfg)
    try:
        backfill_fuel(cfg)
    except Exception:
        pass  # yfinance hiccup must not kill the price step; stale closes ffill

    price = pd.read_parquet(proc / "price_da_eur.parquet").iloc[:, 0]
    load = pd.read_parquet(proc / "load.parquet").iloc[:, 0]
    tso = pd.read_parquet(proc / "tso_forecast.parquet").iloc[:, 0]
    res = pd.read_parquet(proc / "res_forecast.parquet")
    fuel_path = proc / "fuel_daily.parquet"
    fuel = pd.read_parquet(fuel_path) if fuel_path.exists() else None


    scores: dict[str, float] = {}
    oddities: list[str] = []
    if fuel is None:
        oddities.append("Price: fuel_daily.parquet missing — running without "
                        "TTF/EUA features (DECISIONS 2026-07-17).")

    # 2. Score yesterday's saved forecasts against the realized price.
    yhours = _local_day_hours_utc(yesterday, tz)
    realized = price.reindex(yhours)
    fc_y_path = cfg.paths["forecasts"] / f"price_{yesterday.date()}.csv"
    fc_y = None
    if fc_y_path.exists():
        fc_y = pd.read_csv(fc_y_path, index_col="time_utc", parse_dates=True)
        scores["price_lear_mae"] = float((fc_y["p50"] - realized).abs().mean())
        naive_y = price.reindex(yhours - pd.Timedelta(hours=24)).to_numpy()
        scores["price_naive_mae"] = float(
            pd.Series(naive_y, index=yhours).sub(realized).abs().mean()
        )
    else:
        oddities.append("Price: no saved forecast for yesterday; first score tomorrow.")
    ch_y_path = cfg.paths["forecasts"] / f"price_{yesterday.date()}_challenger.csv"
    if ch_y_path.exists():
        ch_y = pd.read_csv(ch_y_path, index_col="time_utc", parse_dates=True)
        scores["price_lgbm_mae"] = float((ch_y["p50"] - realized).abs().mean())

    # 3. Forecast tomorrow. TSO (published ~09:00 D-1) and RES (~18:00
    # D-1) may be unpublished when the run happens early — persist
    # yesterday's same-clock-hour values (flagged below).
    thours = _local_day_hours_utc(tomorrow, tz)
    n_tso_missing = len(thours.difference(tso.index))
    tso = persist_24h(tso, thours)
    if n_tso_missing:
        oddities.append(
            f"Price: TSO load forecast for {tomorrow.date()} not yet published "
            f"({n_tso_missing} h persisted from the day before)."
        )
    n_res_missing = len(thours.difference(res.index))
    res_filled = persist_24h(res, thours)
    if n_res_missing:
        oddities.append(
            f"Price: RES forecast for {tomorrow.date()} not yet published "
            f"({n_res_missing} h persisted from the day before)."
        )

    train_start = shift_local_day(tomorrow, -365, tz)
    x = _assemble(price, load, tso, res_filled, tz, train_start, tomorrow, fuel=fuel)
    x_tr = x[x.index < thours[0]].dropna()
    y_tr = price.reindex(x_tr.index).dropna()
    x_tr = x_tr.reindex(y_tr.index)
    model = PriceLEAR()
    model.fit(x_tr, y_tr)
    x_pred = x.reindex(thours).dropna()
    if x_pred.empty:
        nan_cols = x.reindex(thours).isna().sum()
        worst = nan_cols[nan_cols > 0].index.tolist()[:5]
        raise RuntimeError(
            f"price features for {tomorrow.date()} are all-NaN "
            f"(first NaN columns: {worst}) — refusing to publish an empty forecast"
        )
    fc = model.predict(x_pred)

    # Conformal band widening (config/price_conformal.json, from the
    # trailing 90d of out-of-sample backtest errors). Without it the raw
    # LEAR band covers 72% instead of 80% — see model card.
    try:
        with open("config/price_conformal.json") as f:
            q = json.load(f)["lear"]
        fc["p10"] = (fc["p10"] - q).clip(upper=fc["p50"])
        fc["p90"] = (fc["p90"] + q).clip(lower=fc["p50"])
    except (FileNotFoundError, KeyError):
        oddities.append("Price: conformal offsets missing — publishing the RAW band.")

    cfg.paths["forecasts"].mkdir(parents=True, exist_ok=True)
    fc.rename_axis("time_utc").to_csv(
        cfg.paths["forecasts"] / f"price_{tomorrow.date()}.csv", float_format="%.2f"
    )

    # 3b. Shadow challenger (M9 gate): LightGBM + conformal. Scored daily,
    # never published; promotion only by a human after the tally window.
    try:
        from src.models.gbm import LightGBMQuantile

        ch = LightGBMQuantile()
        ch.fit(x_tr, y_tr)
        ch_fc = ch.predict(x.reindex(thours).dropna())
        with open("config/price_conformal.json") as f:
            q_ch = json.load(f)["lgbm_quantile"]
        ch_fc["p10"] = (ch_fc["p10"] - q_ch).clip(upper=ch_fc["p50"])
        ch_fc["p90"] = (ch_fc["p90"] + q_ch).clip(lower=ch_fc["p50"])
        ch_fc.rename_axis("time_utc").to_csv(
            cfg.paths["forecasts"] / f"price_{tomorrow.date()}_challenger.csv",
            float_format="%.2f",
        )
    except Exception as exc:  # noqa: BLE001 — shadow must never kill the price step
        oddities.append(f"Price challenger (lgbm) failed: {exc}")

    # 4. Living figures, one per TARGET day:
    # tomorrow's chart = band only (published forecast);
    # yesterday's chart = re-rendered WITH the realized price.
    fig_dir = cfg.paths["reports_daily"].parent / "figures" / "daily"
    plot_forecast_band(
        fc, str(tomorrow.date()), fig_dir / f"price_{tomorrow.date()}.png",
        unit="Price (EUR/MWh)",
    )
    if fc_y is not None:
        plot_forecast_band(
            fc_y, str(yesterday.date()), fig_dir / f"price_{yesterday.date()}.png",
            actual=realized, unit="Price (EUR/MWh)",
        )

    local = fc.tz_convert(tz)
    peak = local["p50"].idxmax()
    lines = [
        f"## Price — day-ahead (LEAR, shadow)",
        "",
        f"### Yesterday ({yesterday.date()}) — forecast vs realized",
        "",
        "| Model | MAE (EUR/MWh) |",
        "|---|---|",
        f"| LEAR (incumbent) | {scores.get('price_lear_mae', float('nan')):.2f} |",
        f"| LightGBM+conformal (shadow) | {scores.get('price_lgbm_mae', float('nan')):.2f} |",
        f"| naive-1d | {scores.get('price_naive_mae', float('nan')):.2f} |",
        "",
    ]
    if fc_y is not None:
        lines += [
            f"![Price yesterday vs realized](../figures/daily/price_{yesterday.date()}.png)",
            "",
        ]
    lines += [
        f"### Tomorrow ({tomorrow.date()}) — the price forecast",
        "",
        f"- Expected peak price: **{local['p50'].max():,.0f} EUR/MWh** "
        f"around {peak.strftime('%H:%M')} local.",
        f"- P50 range: {local['p50'].min():,.0f} – {local['p50'].max():,.0f} EUR/MWh; "
        f"band at peak {local.loc[peak, 'p10']:,.0f} – {local.loc[peak, 'p90']:,.0f}.",
        "",
        f"![Price forecast tomorrow](../figures/daily/price_{tomorrow.date()}.png)",
    ]
    return scores, lines, oddities
