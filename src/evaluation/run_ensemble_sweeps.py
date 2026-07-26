"""Stored-preds sweeps: ensemble weight window, 4th member, CQR window.

All experiments run from stored walk-forward predictions — no model
refits. Gates pre-declared in the campaign log (2026-07-26):

1. Weight window {30,90,120} vs shipped 60: adopt only if pooled MAE
   improves >= 0.05 AND no test year gets worse.
2. Adding TimesFM as a 4th member: beat the shipped blend by >= 0.10
   MAE with DM p < 0.05, coverage 78-82%.
3. CQR window {30,60,180} vs shipped 90 (champion + blend): coverage
   distance to 80% improves AND Winkler not worse by > 0.5%.

Run: uv run python -m src.evaluation.run_ensemble_sweeps
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.evaluation.backtest import BacktestResult
from src.evaluation.conformal import rolling_conformal
from src.evaluation.ensemble import LOCAL_TZ, blend, crps3, rolling_weights
from src.evaluation.run_price_backtest import summarize_price
from src.evaluation.stats_tests import dm_test

PROC = Path("data/processed")
OUT = Path("reports/backtests/2026-07-26_ensemble_sweeps.md")


def load_members() -> tuple[dict[str, pd.DataFrame], pd.Series]:
    y = pd.read_parquet(PROC / "price_da_eur.parquet").iloc[:, 0]
    members = {
        "lgbm": pd.read_parquet(
            PROC / "backtest_preds_price_res/lgbm_quantile_conformal.parquet"),
        "lear": pd.read_parquet(
            PROC / "backtest_preds_price_res/lear_conformal.parquet"),
        "chronos": rolling_conformal(pd.read_parquet(
            PROC / "backtest_preds_price_chronos2yr/chronos_bolt_zs.parquet"), y),
    }
    return members, y


def build_blend(
    members: dict[str, pd.DataFrame], y: pd.Series, window_days: int,
) -> pd.DataFrame:
    scores = {n: crps3(p, y) for n, p in members.items()}
    idx = None
    for p in members.values():
        idx = p.index if idx is None else idx.intersection(p.index)
    days = sorted(set(idx.tz_convert(LOCAL_TZ).date))
    w = rolling_weights(scores, days, window_days=window_days)
    return rolling_conformal(blend(members, w), y)


def daily_mae(p: pd.DataFrame, y: pd.Series) -> pd.Series:
    e = (y.reindex(p.index) - p["p50"]).abs()
    return e.groupby(pd.Index(e.index.tz_convert(LOCAL_TZ).date)).mean()


def yearly(p: pd.DataFrame, y: pd.Series) -> dict[int, float]:
    d = daily_mae(p, y)
    return {yr: float(d[[x.year == yr for x in d.index]].mean())
            for yr in (2024, 2025, 2026)}


def main() -> int:
    members, y = load_members()
    lines = ["# Ensemble + CQR sweeps — 2026-07-26", ""]

    # ---- 1. weight-window sweep ----
    lines += ["## Weight-window sweep (shipped: 60d)", ""]
    blends = {w: build_blend(members, y, w) for w in (30, 60, 90, 120)}
    rows = []
    for w, p in blends.items():
        t = summarize_price([BacktestResult(f"win{w}", p)], y)
        yr = yearly(p, y)
        rows.append({"window": w, "mae": round(float(t["mae"].iloc[0]), 3),
                     "winkler": round(float(t["winkler"].iloc[0]), 2),
                     **{f"y{k}": round(v, 3) for k, v in yr.items()}})
    tbl = pd.DataFrame(rows).set_index("window")
    lines += [tbl.to_markdown(), ""]
    base = tbl.loc[60]
    verdict1 = "keep 60d"
    for w in (30, 90, 120):
        r = tbl.loc[w]
        if (base["mae"] - r["mae"] >= 0.05
                and all(r[f"y{k}"] <= base[f"y{k}"] + 1e-9
                        for k in (2024, 2025, 2026))):
            verdict1 = f"ADOPT {w}d (gate passed)"
    lines += [f"Verdict: {verdict1}", ""]

    # ---- 2. 4th member: TimesFM ----
    lines += ["## 4-member blend (+ TimesFM)", ""]
    m4 = dict(members)
    m4["timesfm"] = rolling_conformal(pd.read_parquet(
        PROC / "backtest_preds_price_timesfm2yr/timesfm_zs.parquet"), y)
    p3, p4 = blends[60], build_blend(m4, y, 60)
    t = summarize_price(
        [BacktestResult("blend3", p3), BacktestResult("blend4", p4)], y)
    lines += [t[["mae", "coverage_80_pct", "winkler"]]
              .round(3).to_markdown(), ""]
    stat, pval = dm_test(daily_mae(p4, y), daily_mae(p3, y))
    mae3 = float(t.loc["blend3", "mae"])
    mae4 = float(t.loc["blend4", "mae"])
    cov4 = float(t.loc["blend4", "coverage_80_pct"])
    ok = (mae3 - mae4 >= 0.10) and pval < 0.05 and 78 <= cov4 <= 82
    lines += [f"DM p={pval:.4f}. Verdict: "
              f"{'ADOPT 4-member (gate passed)' if ok else 'keep 3-member'}", ""]

    # ---- 3. CQR window sweep ----
    lines += ["## CQR window sweep (shipped: 90d)", ""]
    raw_lgbm = pd.read_parquet(
        PROC / "backtest_preds_price_res/lgbm_quantile.parquet")
    raw_blend = build_blend(members, y, 60)  # blend then re-CQR at each w
    for label, raw in (("lgbm", raw_lgbm), ("blend", raw_blend)):
        rows = []
        for w in (30, 60, 90, 180):
            cal = rolling_conformal(raw, y, window_days=w)
            t = summarize_price([BacktestResult(f"{label}_cqr{w}", cal)], y)
            rows.append({
                "window": w,
                "coverage": round(float(t["coverage_80_pct"].iloc[0]), 2),
                "winkler": round(float(t["winkler"].iloc[0]), 2),
                "spike_cover": round(float(t["spike_cover_pct"].iloc[0]), 1)})
        tbl = pd.DataFrame(rows).set_index("window")
        lines += [f"### {label}", "", tbl.to_markdown(), ""]
        b = tbl.loc[90]
        verdict = "keep 90d"
        for w in (30, 60, 180):
            r = tbl.loc[w]
            better_cov = abs(r["coverage"] - 80) < abs(b["coverage"] - 80)
            if better_cov and r["winkler"] <= b["winkler"] * 1.005:
                verdict = f"candidate {w}d (gate passed)"
        lines += [f"Verdict ({label}): {verdict}", ""]

    OUT.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWritten {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
