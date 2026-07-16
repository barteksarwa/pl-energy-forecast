"""Daily price step: score yesterday's price forecast, forecast tomorrow.

Mirrors the desk morning routine for the price desk:
1. Pull the latest day-ahead prices + RES forecasts (incremental).
2. Score yesterday's saved forecast against the REALIZED price.
3. Fit LEAR on the trailing window, forecast tomorrow, save CSV.
4. Plot both: yesterday forecast-vs-realized, tomorrow's band.

Called from daily_run inside its own try/except — a price failure must
never kill the load report.

Timing at the 05:30 UTC cron (documented, not hidden):
- Tomorrow's TSO load forecast is unpublished → ffill (same fix as the
  load challenger, DECISIONS 2026-07-16).
- Tomorrow's RES forecast is published ~18:00 today → persist
  yesterday's same-local-hour value. Solar keeps its daily shape; wind
  is a weak persistence guess. The report flags it every day.

Model choice: LEAR, not LightGBM. LGBM has the better MAE (rMAE 0.638
vs 0.660) but its quantile band covers 51% instead of 80% — unusable
for a published band until conformal calibration lands
(docs/model_cards/lgbm_price.md). LEAR's band covers 72%.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config
from src.features.price_matrix import build_price_features
from src.models.price import PriceLEAR


def _local_day_hours_utc(day: pd.Timestamp, tz: str) -> pd.DatetimeIndex:
    from src.pipeline.daily_run import local_day_hours_utc

    return local_day_hours_utc(day, tz)


def _assemble(
    price: pd.Series, load: pd.Series, tso: pd.Series, res: pd.DataFrame,
    tz: str, start: pd.Timestamp, end: pd.Timestamp,
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
                tso=tso, res=res,
            )
        )
        day = shift_local_day(day, 1, tz)
    return pd.concat(frames)


def price_daily_step(
    cfg: Config, today_local: pd.Timestamp
) -> tuple[dict[str, float], list[str], list[str]]:
    """Returns (scores, report_lines, oddities). Raises on hard failure —
    the caller isolates it."""
    from src.ingestion.backfill import backfill_entsoe_prices, backfill_entsoe_res
    from src.pipeline.daily_run import shift_local_day
    from src.viz.plots import plot_price_daily

    tz = cfg.timezone_local
    yesterday = shift_local_day(today_local, -1, tz)
    tomorrow = shift_local_day(today_local, 1, tz)
    proc = cfg.paths["data_processed"]

    # 1. Incremental data pull (resume-based, cheap after backfill).
    backfill_entsoe_prices(cfg)
    backfill_entsoe_res(cfg)
    price = pd.read_parquet(proc / "price_da_eur.parquet").iloc[:, 0]
    load = pd.read_parquet(proc / "load.parquet").iloc[:, 0]
    tso = pd.read_parquet(proc / "tso_forecast.parquet").iloc[:, 0].ffill()
    res = pd.read_parquet(proc / "res_forecast.parquet")

    scores: dict[str, float] = {}
    oddities: list[str] = []

    # 2. Score yesterday's saved forecast against the realized price.
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

    # 3. Forecast tomorrow. RES for tomorrow is unpublished at the cron
    # hour — persist yesterday's same-clock-hour values (flagged below).
    thours = _local_day_hours_utc(tomorrow, tz)
    res_filled = res.copy()
    missing = thours.difference(res_filled.index)
    if len(missing):
        persisted = res.reindex(missing - pd.Timedelta(hours=24))
        persisted.index = missing
        res_filled = pd.concat([res_filled, persisted]).sort_index()
        oddities.append(
            f"Price: RES forecast for {tomorrow.date()} not yet published "
            f"({len(missing)} h persisted from the day before)."
        )

    train_start = shift_local_day(tomorrow, -365, tz)
    x = _assemble(price, load, tso, res_filled, tz, train_start, tomorrow)
    x_tr = x[x.index < thours[0]].dropna()
    y_tr = price.reindex(x_tr.index).dropna()
    x_tr = x_tr.reindex(y_tr.index)
    model = PriceLEAR()
    model.fit(x_tr, y_tr)
    fc = model.predict(x.reindex(thours).dropna())

    cfg.paths["forecasts"].mkdir(parents=True, exist_ok=True)
    fc.rename_axis("time_utc").to_csv(
        cfg.paths["forecasts"] / f"price_{tomorrow.date()}.csv", float_format="%.2f"
    )

    # 4. Plot: yesterday eval + tomorrow band, one figure.
    fig_path = (
        cfg.paths["reports_daily"].parent
        / "figures" / "daily" / f"price_{tomorrow.date()}.png"
    )
    plot_price_daily(fc_y, realized, fc, str(yesterday.date()),
                     str(tomorrow.date()), fig_path)

    local = fc.tz_convert(tz)
    peak = local["p50"].idxmax()
    lines = [
        f"## Price ({tomorrow.date()}) — day-ahead forecast (LEAR, shadow)",
        "",
        "| Yesterday | MAE (EUR/MWh) |",
        "|---|---|",
        f"| LEAR | {scores.get('price_lear_mae', float('nan')):.2f} |",
        f"| naive-1d | {scores.get('price_naive_mae', float('nan')):.2f} |",
        "",
        f"- Tomorrow's expected peak price: **{local['p50'].max():,.0f} EUR/MWh** "
        f"around {peak.strftime('%H:%M')} local.",
        f"- P50 range: {local['p50'].min():,.0f} – {local['p50'].max():,.0f} EUR/MWh; "
        f"band at peak {local.loc[peak, 'p10']:,.0f} – {local.loc[peak, 'p90']:,.0f}.",
        "",
        f"![Price: yesterday vs realized + tomorrow](../figures/daily/price_{tomorrow.date()}.png)",
    ]
    return scores, lines, oddities
