"""LGBM price HPO campaign — the champion finally gets tuned.

The model card has said "conservative untuned defaults" since M4.
This campaign tunes at the 1095d training window (the promotion-
candidate config), with the temporal-split discipline this repo
already learned the hard way:

- SCREEN on test year 1 (2024-07-16 -> 2025-07-15): all configs.
- CONFIRM on test year 2 (2025-07-16 -> end): top 3 + control only.
  The confirm year is never used for selection.

Pre-declared gate (before any run): a config becomes the champion-
config CANDIDATE only if it beats the control (shipped defaults) by
>= 0.10 MAE on BOTH years. Anything else = honest negative, defaults
stay.

Resume-safe: one CSV row per finished run; existing (config, stage)
pairs are skipped on restart.

Run: uv run python -m src.evaluation.run_lgbm_price_hpo
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.evaluation.backtest import walk_forward_backtest
from src.evaluation.run_price_backtest import assemble_price_features
from src.evaluation.run_price_backtest import summarize_price
from src.models.gbm import LightGBMQuantile

TRAIN_DAYS = 1095
SCREEN_START, SCREEN_END = "2024-07-16", "2025-07-15"
CONFIRM_START = "2025-07-16"
OUT = Path("reports/backtests/2026-07-25_lgbm_price_hpo.csv")

# name -> (param overrides, drop load_ lag columns?)
CONFIGS: dict[str, tuple[dict, bool]] = {
    "control": ({}, False),
    "leaves31": ({"num_leaves": 31}, False),
    "leaves127": ({"num_leaves": 127}, False),
    "leaves255_depth8": ({"num_leaves": 255, "max_depth": 8}, False),
    "minchild20": ({"min_child_samples": 20}, False),
    "minchild80": ({"min_child_samples": 80}, False),
    "lr03_n1000": ({"learning_rate": 0.03, "n_estimators": 1000}, False),
    "lr02_n1500": ({"learning_rate": 0.02, "n_estimators": 1500}, False),
    "colsample07": ({"colsample_bytree": 0.7}, False),
    "subsample07": ({"subsample": 0.7}, False),
    "l1_1": ({"reg_alpha": 1.0}, False),
    "l2_5": ({"reg_lambda": 5.0}, False),
    "combo_deep_slow": (
        {"num_leaves": 127, "learning_rate": 0.03, "n_estimators": 1000,
         "colsample_bytree": 0.7}, False),
    "noloadlags": ({}, True),  # confirmed -0.12 at 365d; re-check at 1095d
}


def _load_matrix(tz: str) -> tuple[pd.DataFrame, pd.Series]:
    proc = Path("data/processed")
    price = pd.read_parquet(proc / "price_da_eur.parquet").iloc[:, 0]
    load = pd.read_parquet(proc / "load.parquet").iloc[:, 0]
    tso = pd.read_parquet(proc / "tso_forecast.parquet").iloc[:, 0]
    res = pd.read_parquet(proc / "res_forecast.parquet")
    first = pd.Timestamp(SCREEN_START, tz=tz) - pd.Timedelta(days=TRAIN_DAYS + 40)
    last = min(price.index[-1], load.index[-1]).tz_convert(tz) - pd.Timedelta(days=1)
    x = assemble_price_features(
        price, load, tso, tz,
        pd.Timestamp(first.date(), tz=tz), pd.Timestamp(last.date(), tz=tz),
        res=res,
    )
    return x, price.reindex(x.index)


def _run(
    x: pd.DataFrame, y: pd.Series, overrides: dict, drop_load: bool,
    test_start: pd.Timestamp, test_end: pd.Timestamp | None, tz: str,
) -> dict:
    if drop_load:
        x = x.drop(columns=[c for c in x.columns if c.startswith("load_")])
    if test_end is not None:
        keep = x.index.tz_convert(tz) < test_end + pd.Timedelta(days=1)
        x, y = x[keep], y[keep]
    result = walk_forward_backtest(
        lambda: LightGBMQuantile(params=overrides), x, y,
        test_start.tz_convert("UTC"), train_window_days=TRAIN_DAYS,
        refit_every_days=7,
        # same information set as the reported price backtests
        target_availability="day_ahead",
    )
    table = summarize_price([result], y)
    return table.iloc[0].to_dict()


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="2-config, 2-week screen — wiring check only")
    args = parser.parse_args()

    cfg = load_config()
    tz = cfg.timezone_local
    x, y = _load_matrix(tz)

    configs = CONFIGS
    screen_end = SCREEN_END
    out = OUT
    if args.smoke:
        configs = {k: CONFIGS[k] for k in ("control", "leaves31")}
        screen_end = "2024-07-30"
        out = OUT.with_suffix(".smoke.csv")  # never pollute the real resume file

    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = set(zip(prev["config"], prev["stage"]))

    def record(config: str, stage: str, row: dict) -> None:
        entry = {"config": config, "stage": stage, **row}
        header = not out.exists()
        pd.DataFrame([entry]).to_csv(out, mode="a", header=header, index=False)
        print(f"[{stage}] {config}: MAE {row['mae']:.3f}", flush=True)

    # ---- stage 1: screen every config on test year 1 ----
    for name, (overrides, drop_load) in configs.items():
        if (name, "screen") in done:
            continue
        row = _run(x, y, overrides, drop_load,
                   pd.Timestamp(SCREEN_START, tz=tz),
                   pd.Timestamp(screen_end, tz=tz), tz)
        record(name, "screen", row)

    if args.smoke:
        print("smoke OK", flush=True)
        return 0

    # ---- stage 2: confirm top 3 + control on test year 2 ----
    df = pd.read_csv(out)
    screen = df[df["stage"] == "screen"]
    top = list(screen.sort_values("mae")["config"].head(3))
    if "control" not in top:
        top.append("control")
    for name in top:
        if (name, "confirm") in done:
            continue
        overrides, drop_load = configs[name]
        row = _run(x, y, overrides, drop_load,
                   pd.Timestamp(CONFIRM_START, tz=tz), None, tz)
        record(name, "confirm", row)

    # ---- verdict against the pre-declared gate ----
    df = pd.read_csv(out)

    def mae_of(config: str, stage: str) -> float:
        sel = df[(df["config"] == config) & (df["stage"] == stage)]
        return float(sel["mae"].iloc[0])

    ctrl = {s: mae_of("control", s) for s in ("screen", "confirm")}
    print("\ncontrol MAE:", ctrl, flush=True)
    for name in top:
        if name == "control":
            continue
        m = {s: mae_of(name, s) for s in ("screen", "confirm")}
        both = all(ctrl[s] - m[s] >= 0.10 for s in ("screen", "confirm"))
        print(f"{name}: screen {m['screen']:.3f} confirm {m['confirm']:.3f} "
              f"-> {'CANDIDATE (gate passed)' if both else 'no promotion'}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
