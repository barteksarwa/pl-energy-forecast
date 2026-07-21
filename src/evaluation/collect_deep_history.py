"""Collect the deep-history campaign results into one report.

Reads whatever the campaign produced (missing pieces are skipped, not
fatal) and writes reports/backtests/<date>_deep_history_campaign.md:

1. Training-window sweep: LGBM champion at 365/730/1095/1460-day windows.
2. Crisis-regime backtest: 5-year test through the 2021-22 price crisis.
3. Deep re-benchmark: TFT-730 and PatchTST-730 on the full 2-yr test,
   pooled + per-year.

Run: python -m src.evaluation.run_price_backtest jobs first (the
campaign script does), then python -m src.evaluation.collect_deep_history
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

OUT_DIR = Path("reports/backtests")


def _latest(pattern: str) -> Path | None:
    hits = sorted(glob.glob(str(OUT_DIR / pattern)))
    return Path(hits[-1]) if hits else None


def main() -> int:
    tz = "Europe/Warsaw"
    stamp = pd.Timestamp.now(tz).date()
    md = [f"# Deep-history campaign — {stamp}", ""]
    md += ["Data extended to 2015+ (ENTSO-E). This campaign asks what the",
           "extra history buys: longer training windows, crisis-regime",
           "evaluation, and the previously blocked deep re-benchmark on",
           "the full 2-year test.", ""]

    # 1. Window sweep
    rows = []
    for days in (365, 730, 1095, 1460):
        p = _latest(f"*_price_win{days}_summary.csv")
        if p is None:
            continue
        df = pd.read_csv(p, index_col=0)
        if "lgbm_quantile" in df.index:
            r = df.loc["lgbm_quantile"]
            rows.append({"train_days": days, "mae": r["mae"],
                         "rmae": r.get("rmae"), "spike_mae": r.get("spike_mae"),
                         "source": p.name})
    if rows:
        md += ["## 1. LGBM training-window sweep (test 2024-07-16 →)", "",
               pd.DataFrame(rows).set_index("train_days").round(3).to_markdown(),
               "", "Same model, same test window — only the training window",
               "changes. Per-year tables: the win*_summary.md files.", ""]

    # 2. Crisis regime
    p = _latest("*_price_crisis5yr_summary.csv")
    if p is not None:
        df = pd.read_csv(p, index_col=0)
        md += ["## 2. Crisis-regime backtest (test 2021-07-16 →, 5 years)", "",
               df.round(3).to_markdown(), "",
               f"Per-year breakdown (incl. 2022 crisis): {p.with_suffix('.md').name}.",
               ""]

    # 3. Deep re-benchmark
    for model in ("tft", "patchtst"):
        p = Path(f"reports/sensitivity/{model}/deep2yr_{model}.csv")
        if not p.exists():
            continue
        df = pd.read_csv(p)
        pooled = df[df.period == "pooled"][
            ["config", "seed", "mae", "rmae", "coverage_80_pct"]]
        md += [f"## 3. Deep re-benchmark — {model.upper()} "
               "(730d windows, FULL 2-yr test)", "",
               pooled.round(3).to_markdown(index=False), ""]
        years = df[df.period != "pooled"]
        if len(years):
            piv = years.pivot_table(index="period",
                                    columns=["config", "seed"], values="mae")
            md += ["Per-year MAE:", "", piv.round(2).to_markdown(), ""]

    md += ["## Reference points", "",
           "- Champion (LGBM+CQR, 365d windows, 2-yr test): MAE 17.87,",
           "  rMAE 0.640. Without load_lags: 17.755.",
           "- Best deep before this campaign: TFT-730 ens-3 MAE 18.31 on",
           "  the 1-yr window only.",
           "- Full numbers: docs/RESULTS.md.", ""]

    out = OUT_DIR / f"{stamp}_deep_history_campaign.md"
    out.write_text("\n".join(md))
    print(f"Written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
