"""Portfolio POC: forecasting a SUB-national load with the national TSO
forecast as help. Question raised 2026-07-17.

Setup: a synthetic retailer portfolio built from BDEW standard load
profiles (demandlib; the jnettels/lpagg stack uses the same source) —
60% households (h0), 30% commerce (g0), 10% continuous-shift industry
(g3) — scaled to ~500 MW mean, plus AR(1) noise and 2%/year growth.
BDEW profiles are what real retail portfolios are composed from; the
noise stands in for churn and metering error.

Designs compared, identical walk-forward (12 months, weekly refits):
  (a) feature   — ridge on local lags + calendar + national TSO forecast
                  as a plain feature. Architecture unchanged.
  (b) share     — ridge forecasts the RATIO local/national on ratio lags
                  + calendar; prediction = ratio_hat x TSO forecast.
                  Top-down with dynamic proportions.
  (c) naive     — seasonal naive on the local series (the bar to clear).
  (a-) feature, no TSO — ablation: is the national signal doing anything?

Run: uv run python -m src.evaluation.run_portfolio_poc
Runtime: ~10-20 min CPU.
"""

from __future__ import annotations

import sys

import holidays as holidays_pkg
import numpy as np
import pandas as pd
from demandlib import bdew

from src.config import load_config
from src.evaluation.backtest import BacktestResult, walk_forward_backtest
from src.features.matrix import build_features
from src.models.base import REGISTRY
import src.models.baselines  # noqa: F401  (populates REGISTRY)
from src.pipeline.daily_run import local_day_hours_utc, shift_local_day

TZ = "Europe/Warsaw"
MIX_MWH = {"h0": 60_000.0, "g0": 30_000.0, "g3": 10_000.0}  # annual, per year
TARGET_MEAN_MW = 500.0
SEED = 42


def synth_portfolio(years: list[int], tz: str = TZ) -> pd.Series:
    """BDEW mix -> hourly UTC MW series with noise and growth."""
    parts = []
    for year in years:
        hols = dict(holidays_pkg.country_holidays("PL", years=[year]))
        slp = bdew.ElecSlp(year, holidays=hols)
        df = slp.get_scaled_power_profiles(MIX_MWH)
        parts.append(df.sum(axis=1))
    raw = pd.concat(parts)
    # BDEW index is naive local WITHOUT the DST duplicate hour; the
    # ambiguous autumn hour becomes NaT and is dropped (1 h/year, synthetic).
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(
        tz, ambiguous="NaT", nonexistent="shift_forward"
    )
    raw = raw[raw.index.notna()]
    hourly = raw.resample("1h").mean().tz_convert("UTC")
    hourly = hourly * (TARGET_MEAN_MW / hourly.mean())

    rng = np.random.default_rng(SEED)
    n = len(hourly)
    noise = np.zeros(n)
    eps = rng.normal(0, 0.02 * TARGET_MEAN_MW, n)
    for i in range(1, n):  # AR(1), phi=0.95
        noise[i] = 0.95 * noise[i - 1] + eps[i]
    growth = np.linspace(1.0, 1.0 + 0.02 * len(years), n)
    out = (hourly + noise) * growth
    out.name = "load_mw"  # feature builder expects load naming
    return out


def assemble(series: pd.Series, tz: str, start, end, tso=None) -> pd.DataFrame:
    frames = []
    day = start
    while day <= end:
        hours = local_day_hours_utc(day, tz)
        cutoff = shift_local_day(day, -1, tz) + pd.Timedelta(hours=9)
        wx = pd.DataFrame(index=hours)  # no weather in this POC, both designs
        frames.append(build_features(hours, series, wx, cutoff, tso=tso))
        day = shift_local_day(day, 1, tz)
    return pd.concat(frames)


def main() -> int:
    cfg = load_config()
    tz = cfg.timezone_local
    proc = cfg.paths["data_processed"]
    national = pd.read_parquet(proc / "load.parquet").iloc[:, 0]
    tso = pd.read_parquet(proc / "tso_forecast.parquet").iloc[:, 0]

    years = sorted(set(national.index.tz_convert(tz).year))
    local = synth_portfolio(years, tz).reindex(national.index).dropna()
    print(f"portfolio: {len(local)} h, mean {local.mean():.0f} MW", flush=True)

    last = local.index[-1].tz_convert(tz) - pd.Timedelta(days=1)
    first = local.index[0].tz_convert(tz) + pd.Timedelta(days=30)
    test_start = shift_local_day(last, -365, tz)
    span = (pd.Timestamp(first.date(), tz=tz), pd.Timestamp(last.date(), tz=tz))

    results: list[BacktestResult] = []

    # (a) TSO as plain feature + (a-) without, + (c) naive on the same matrix
    x = assemble(local, tz, *span, tso=tso)
    y = local.reindex(x.index)
    for name, model in [("(a) ridge + national TSO feature", "ridge"),
                        ("(c) seasonal naive", "seasonal_naive")]:
        r = walk_forward_backtest(REGISTRY[model], x, y, test_start.tz_convert("UTC"))
        results.append(BacktestResult(model_name=name, predictions=r.predictions))
        print(f"{name}: done", flush=True)
    x_no = x.drop(columns=["tso_forecast_mw"])
    r = walk_forward_backtest(REGISTRY["ridge"], x_no, y, test_start.tz_convert("UTC"))
    results.append(BacktestResult(model_name="(a-) ridge, no TSO", predictions=r.predictions))
    print("(a-): done", flush=True)

    # (b) share model: ridge on the ratio, multiplied back by the TSO forecast
    ratio = (local / national.reindex(local.index)).dropna()
    ratio.name = "load_mw"
    x_r = assemble(ratio, tz, *span)
    y_r = ratio.reindex(x_r.index)
    r = walk_forward_backtest(REGISTRY["ridge"], x_r, y_r, test_start.tz_convert("UTC"))
    share_pred = r.predictions.mul(tso.reindex(r.predictions.index), axis=0)
    results.append(BacktestResult(model_name="(b) share x TSO forecast",
                                  predictions=share_pred))
    print("(b): done", flush=True)

    rows = []
    for res in results:
        p = res.predictions
        yy = local.reindex(p.index)
        inside = (yy >= p["p10"]) & (yy <= p["p90"])
        rows.append({
            "design": res.model_name,
            "mape_pct": float((100 * (p["p50"] - yy).abs() / yy).mean()),
            "mae_mw": float((p["p50"] - yy).abs().mean()),
            "coverage_80_pct": 100.0 * float(inside.mean()),
            "n_hours": int(p["p50"].notna().sum()),
        })
    table = pd.DataFrame(rows).set_index("design").sort_values("mae_mw")

    out_dir = proc.parent.parent / "reports" / "backtests"
    stamp = f"{pd.Timestamp.now(tz).date()}_portfolio_poc"
    table.to_csv(out_dir / f"{stamp}.csv")
    md = [
        f"# Portfolio POC — sub-national load vs national TSO forecast — {stamp}",
        "",
        "Synthetic retailer portfolio (BDEW h0/g0/g3 mix, ~500 MW, AR(1)",
        "noise, 2%/yr growth). 12-month walk-forward, weekly refits, no",
        "weather in any design (level playing field).",
        "",
        table.round(2).to_markdown(),
        "",
        "Reading: (a) vs (a-) prices the national signal; (a) vs (b) decides",
        "feature-vs-share; everything must beat (c). Caveats: synthetic",
        "portfolio inherits BDEW calendar shape — correlation with the",
        "national series is optimistic vs a real portfolio; ratio",
        "non-stationarity here is mild (smooth growth), real churn is lumpier.",
        "",
    ]
    (out_dir / f"{stamp}.md").write_text("\n".join(md))
    print(table.round(2).to_string())
    print(f"Saved to {out_dir}/{stamp}.(csv|md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
