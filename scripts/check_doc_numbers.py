"""Check hand-written doc numbers against the artifact CSVs they cite.

Why this exists: `docs/VALIDATION.md` recommendation 5. The audit found
24 documentation findings. Most were the same failure: a number was
typed into a doc, the run was regenerated, the doc was not. This script
is the control. It re-reads every artifact and compares.

How it works. Each check names five things:

1. the doc file,
2. a regex that locates the quoted number (one capture group),
3. the artifact CSV,
4. the row and column to read,
5. how to compare (`round` = decimal places, `sig` = significant figures).

Tolerance is zero. The artifact value, printed at the precision the doc
prints, must equal the doc value character for character. A pattern that
matches nothing is a failure too -- that means the doc was restructured
and the check went blind.

Numbers with no artifact are NOT listed here. Two are known and stay
unchecked on purpose (see the module-level note at CHECKS).

Run:
    make check-docs
    .venv/bin/python scripts/check_doc_numbers.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BT = "reports/backtests"

# Artifacts, named once so a rerun renames them in one place.
CONFORMAL = f"{BT}/2026-07-27_price_conformal_summary.csv"
ENS3 = f"{BT}/2026-07-27_price_ensemble_summary.csv"
ENS4 = f"{BT}/2026-07-27_price_ensemble_tft_summary.csv"
ENS1095 = f"{BT}/2026-07-24_price_ensemble_1095_summary.csv"
CHRONOS = f"{BT}/2026-07-27_price_chronos2yr_summary.csv"
TIMESFM = f"{BT}/2026-07-27_price_timesfm2yr_summary.csv"
MOIRAI = f"{BT}/2026-07-24_price_moirai2yr_summary.csv"
PNL = f"{BT}/2026-07-27_pnl_summary.csv"
SPIKE = f"{BT}/2026-07-27_spike_screen.csv"
DM_ENS = f"{BT}/2026-07-28_stats_tests_ens_dm.csv"
DM_FM = f"{BT}/2026-07-28_stats_tests_fm_dm.csv"
DM_0722 = f"{BT}/2026-07-22_stats_tests_dm.csv"

RESULTS = "docs/RESULTS.md"
BENCHMARK = "docs/BENCHMARK.md"
README = "README.md"


@dataclass(frozen=True)
class Check:
    """One doc number traced to one artifact cell."""

    name: str
    doc: str
    pattern: str
    artifact: str
    row: str
    column: str
    key_column: str | None = None  # None = the CSV's first column
    mode: str = "round"  # "round" | "sig"
    filters: tuple[tuple[str, str], ...] = ()
    scale: float = 1.0  # artifact value multiplier (e.g. 100 for a % quote)


@dataclass
class Failure:
    """One check that did not hold."""

    check: Check
    message: str
    doc_value: str = ""
    artifact_value: str = ""
    raw_value: float = field(default=float("nan"))


# --------------------------------------------------------------------
# parsing / matching helpers (unit-tested in tests/test_check_doc_numbers.py)
# --------------------------------------------------------------------


def count_decimals(text: str) -> int:
    """Digits after the decimal point, ignoring any exponent."""
    mantissa = re.split(r"[eE]", text.strip())[0]
    if "." not in mantissa:
        return 0
    return len(mantissa.split(".", 1)[1])


def count_sig_figs(text: str) -> int:
    """Significant figures in a printed number. `0.0009` -> 1."""
    mantissa = re.split(r"[eE]", text.strip())[0].lstrip("+-")
    digits = mantissa.replace(".", "").lstrip("0")
    return len(digits) or 1


def format_like(value: float, doc_text: str, mode: str) -> str:
    """Print `value` at the precision `doc_text` was printed with."""
    if mode == "sig":
        return f"{value:.{count_sig_figs(doc_text) - 1}e}"
    if mode == "round":
        return f"{value:.{count_decimals(doc_text)}f}"
    raise ValueError(f"unknown mode: {mode}")


def values_match(doc_text: str, value: float, mode: str) -> bool:
    """True when the artifact value rounds to exactly what the doc says."""
    doc_norm = format_like(float(doc_text), doc_text, mode)
    return format_like(value, doc_text, mode) == doc_norm


def find_quoted(text: str, pattern: str) -> list[str]:
    """Every number the pattern captures. Empty list means the anchor died."""
    compiled = re.compile(pattern)
    if compiled.groups != 1:
        raise ValueError(f"pattern needs exactly one capture group: {pattern}")
    return [m.group(1) for m in compiled.finditer(text)]


def lookup(
    frame: pd.DataFrame,
    key_column: str | None,
    row: str,
    column: str,
    filters: tuple[tuple[str, str], ...] = (),
) -> float:
    """Read one cell by row label, with optional extra column filters."""
    key = key_column or str(frame.columns[0])
    if key not in frame.columns:
        raise KeyError(f"no key column {key!r}")
    if column not in frame.columns:
        raise KeyError(f"no column {column!r}")
    hit = frame[frame[key].astype(str) == row]
    for col, want in filters:
        if col not in frame.columns:
            raise KeyError(f"no filter column {col!r}")
        hit = hit[hit[col].astype(str) == want]
    if len(hit) != 1:
        raise KeyError(f"row {row!r} matched {len(hit)} rows, expected 1")
    return float(hit.iloc[0][column])


# --------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------
#
# The two formerly-unverifiable numbers (ens4 P&L capture, ens3 on the
# 17,456 h intersection) now have a dedicated artifact:
# reports/backtests/2026-07-28_ens4_window_metrics.csv, produced by
# src/evaluation/run_ens4_window_metrics.py — both blends scored on the
# identical hour set. Checked below like everything else.

ENS4_WIN = "reports/backtests/2026-07-28_ens4_window_metrics.csv"
LOAD_2YR = "reports/backtests/2026-07-28_fcst_tso_load2yr_cutoff_summary.csv"

CHECKS: list[Check] = [
    # -- load master numbers (corrected-cutoff rerun) --------------------
    Check("results_load_ridge_mape", "docs/RESULTS.md",
          r"\| \*\*ridge_tso\*\* \(champion\) \| \*\*(\d+\.\d+)%\*\*",
          LOAD_2YR, "ridge", "mape_pct"),
    Check("results_load_tso_mape", "docs/RESULTS.md",
          r"\| TSO forecast \(benchmark\) \| (\d+\.\d+)%",
          LOAD_2YR, "tso_forecast", "mape_pct"),
    Check("results_load_naive_mape", "docs/RESULTS.md",
          r"\| seasonal naive \| (\d+\.\d+)%",
          LOAD_2YR, "seasonal_naive", "mape_pct"),
    Check("readme_load_ridge_mape", "README.md",
          r"\| \*\*Ridge \+ TSO forecast \(combiner\)\*\* \| \*\*(\d+\.\d+)%\*\*",
          LOAD_2YR, "ridge", "mape_pct"),
    Check("readme_load_ridge_mae", "README.md",
          r"\| \*\*Ridge \+ TSO forecast \(combiner\)\*\* \| \*\*\d+\.\d+%\*\* "
          r"\| \*\*(\d+)\*\*",
          LOAD_2YR, "ridge", "mae"),
    Check("readme_load_headline_mape", "README.md",
          r"Ridge combiner (\d+\.\d+)%",
          LOAD_2YR, "ridge", "mape_pct"),
    Check("benchmark_load_ridge_mape", "docs/BENCHMARK.md",
          r"\| \*\*Ridge \+ TSO \(combiner\)\*\* \| \*\*(\d+\.\d+)%\*\*",
          LOAD_2YR, "ridge", "mape_pct"),
    # -- ens4 / ens3 on the shared intersection window -------------------
    Check("results_ens4_capture", "docs/RESULTS.md",
          r"\| \*\*ens4_tft \(CQR\)\*\* \|.*\| \*\*(\d+\.\d+)\*\* \|",
          ENS4_WIN, "ens4_tft", "capture_rate"),
    Check("results_ens3win_mae", "docs/RESULTS.md",
          r"\| ens3 \(same 17,456h window\) \| (\d+\.\d+)",
          ENS4_WIN, "ens3", "mae"),
    Check("results_ens3win_winkler", "docs/RESULTS.md",
          r"\| ens3 \(same 17,456h window\) \| \d+\.\d+ \| \d+\.\d+ \| "
          r"\d+\.\d+% \| (\d+\.\d+)",
          ENS4_WIN, "ens3", "winkler"),
    Check("results_ens3win_capture", "docs/RESULTS.md",
          r"\| ens3 \(same 17,456h window\) \| \d+\.\d+ \| \d+\.\d+ \| "
          r"\d+\.\d+% \| \d+\.\d+ \| (\d+\.\d+)",
          ENS4_WIN, "ens3", "capture_rate"),
    Check("benchmark_ens4_capture", "docs/BENCHMARK.md",
          r"\| \*\*4-member ensemble \(\+ TFT, CQR\)\*\* \|.*\| "
          r"\*\*(\d+\.\d+)\*\* \|",
          ENS4_WIN, "ens4_tft", "capture_rate"),
    Check("readme_ens4_capture", "README.md",
          r"\| \*\*Ensemble \(4-member, \+ TFT donor\)\*\* \|.*\| "
          r"\*\*(\d+\.\d+)\*\* \|",
          ENS4_WIN, "ens4_tft", "capture_rate"),
    Check("readme_ens4_capture_prose", "README.md",
          r"ensemble captures (\d+\.\d+)% of",
          ENS4_WIN, "ens4_tft", "capture_rate", scale=100.0),
    # -- RESULTS price table vs the conformal summary --------------------
    Check("results_price_lgbm_mae", RESULTS,
          r"\*\*LGBM quantile \+ CQR\*\* \(champion\) \| \*\*(\d+\.\d+)\*\*",
          CONFORMAL, "lgbm_quantile_conformal", "mae"),
    Check("results_price_lgbm_rmae", RESULTS,
          r"\*\*LGBM quantile \+ CQR\*\* \(champion\) \| \*\*\d+\.\d+\*\* \| \*\*(\d+\.\d+)\*\*",
          CONFORMAL, "lgbm_quantile_conformal", "rmae"),
    Check("results_price_lgbm_coverage", RESULTS,
          r"\*\*LGBM quantile \+ CQR\*\* \(champion\) \| \*\*\d+\.\d+\*\* \| "
          r"\*\*\d+\.\d+\*\* \| (\d+\.\d+)%",
          CONFORMAL, "lgbm_quantile_conformal", "coverage_80_pct"),
    Check("results_price_lear_mae", RESULTS,
          r"\| LEAR \+ CQR \| (\d+\.\d+) \|",
          CONFORMAL, "lear_conformal", "mae"),
    Check("results_price_lear_rmae", RESULTS,
          r"\| LEAR \+ CQR \| \d+\.\d+ \| (\d+\.\d+) \|",
          CONFORMAL, "lear_conformal", "rmae"),
    Check("results_price_lear_coverage", RESULTS,
          r"\| LEAR \+ CQR \| \d+\.\d+ \| \d+\.\d+ \| (\d+\.\d+)%",
          CONFORMAL, "lear_conformal", "coverage_80_pct"),
    Check("results_price_naive_mae", RESULTS,
          r"\| Naive \(1-day\) \| (\d+\.\d+) \|",
          CONFORMAL, "price_naive_yesterday", "mae"),
    Check("results_price_naive_coverage", RESULTS,
          r"\| Naive \(1-day\) \| \d+\.\d+ \| \d+\.\d+ \| (\d+\.\d+)%",
          CONFORMAL, "price_naive_yesterday", "coverage_80_pct"),

    # -- BENCHMARK master table vs the conformal summary -----------------
    Check("benchmark_lgbm_mae", BENCHMARK,
          r"\| LGBM 365d \+ CQR \(champion\) \| (\d+\.\d+) \|",
          CONFORMAL, "lgbm_quantile_conformal", "mae"),
    Check("benchmark_lgbm_rmae", BENCHMARK,
          r"\| LGBM 365d \+ CQR \(champion\) \| \d+\.\d+ \| (\d+\.\d+) \|",
          CONFORMAL, "lgbm_quantile_conformal", "rmae"),
    Check("benchmark_lgbm_coverage", BENCHMARK,
          r"\| LGBM 365d \+ CQR \(champion\) \| \d+\.\d+ \| \d+\.\d+ \| (\d+\.\d+)%",
          CONFORMAL, "lgbm_quantile_conformal", "coverage_80_pct"),
    Check("benchmark_lear_mae", BENCHMARK,
          r"\| LEAR \+ CQR \(industry standard\) \| (\d+\.\d+) \|",
          CONFORMAL, "lear_conformal", "mae"),
    Check("benchmark_lear_rmae", BENCHMARK,
          r"\| LEAR \+ CQR \(industry standard\) \| \d+\.\d+ \| (\d+\.\d+) \|",
          CONFORMAL, "lear_conformal", "rmae"),
    Check("benchmark_lear_coverage", BENCHMARK,
          r"\| LEAR \+ CQR \(industry standard\) \| \d+\.\d+ \| \d+\.\d+ \| (\d+\.\d+)%",
          CONFORMAL, "lear_conformal", "coverage_80_pct"),
    Check("benchmark_naive_mae", BENCHMARK,
          r"\| Naive \(same hour yesterday\) \| (\d+\.\d+) \|",
          CONFORMAL, "price_naive_yesterday", "mae"),

    # -- README price table vs the conformal summary ---------------------
    Check("readme_lgbm_mae", README,
          r"\| LightGBM quantile \+ conformal \| (\d+\.\d+) \|",
          CONFORMAL, "lgbm_quantile_conformal", "mae"),
    Check("readme_lgbm_rmae", README,
          r"\| LightGBM quantile \+ conformal \| \d+\.\d+ \| (\d+\.\d+) \|",
          CONFORMAL, "lgbm_quantile_conformal", "rmae"),
    Check("readme_lear_mae", README,
          r"\| LEAR \+ conformal \(published daily\) \| (\d+\.\d+) \|",
          CONFORMAL, "lear_conformal", "mae"),
    Check("readme_naive_mae", README,
          r"\| Naive \(same hour yesterday\) \| (\d+\.\d+) \|",
          CONFORMAL, "price_naive_yesterday", "mae"),

    # -- RESULTS FM comparison table -------------------------------------
    Check("results_fm_champion_mae", RESULTS,
          r"\| Champion LGBM \(365d\) \| (\d+\.\d+) \|",
          CONFORMAL, "lgbm_quantile_conformal", "mae"),
    Check("results_fm_chronos_mae", RESULTS,
          r"\| \*\*Chronos-Bolt zero-shot\*\* \| \*\*(\d+\.\d+)\*\*",
          CHRONOS, "chronos_bolt_zs", "mae"),
    Check("results_fm_chronos_rmae", RESULTS,
          r"\| \*\*Chronos-Bolt zero-shot\*\* \| \*\*\d+\.\d+\*\* \| \*\*(\d+\.\d+)\*\*",
          ENS3, "chronos", "rmae"),
    Check("results_fm_timesfm_mae", RESULTS,
          r"\| TimesFM 2\.5 zero-shot \| (\d+\.\d+) \|",
          TIMESFM, "timesfm_zs", "mae"),
    Check("results_fm_timesfm_coverage", RESULTS,
          r"\| TimesFM 2\.5 zero-shot \| \d+\.\d+ \| \d+\.\d+ \| (\d+\.\d+)% raw",
          TIMESFM, "timesfm_zs", "coverage_80_pct"),

    # -- BENCHMARK FM rows -----------------------------------------------
    Check("benchmark_chronos_mae", BENCHMARK,
          r"\| Chronos-Bolt zero-shot \+ CQR \| (\d+\.\d+) \|",
          CHRONOS, "chronos_bolt_zs", "mae"),
    Check("benchmark_chronos_rmae", BENCHMARK,
          r"\| Chronos-Bolt zero-shot \+ CQR \| \d+\.\d+ \| (\d+\.\d+) \|",
          ENS3, "chronos", "rmae"),
    Check("benchmark_chronos_coverage", BENCHMARK,
          r"\| Chronos-Bolt zero-shot \+ CQR \| \d+\.\d+ \| \d+\.\d+ \| (\d+\.\d+)%",
          ENS3, "chronos", "coverage_80_pct"),
    Check("benchmark_timesfm_mae", BENCHMARK,
          r"\| TimesFM 2\.5 zero-shot \| (\d+\.\d+) \|",
          TIMESFM, "timesfm_zs", "mae"),
    Check("benchmark_timesfm_coverage", BENCHMARK,
          r"\| TimesFM 2\.5 zero-shot \| \d+\.\d+ \| \d+\.\d+ \| (\d+\.\d+)% raw",
          TIMESFM, "timesfm_zs", "coverage_80_pct"),
    Check("benchmark_moirai_uni_mae", BENCHMARK,
          r"\| Moirai univariate zero-shot \| (\d+\.\d+) \|",
          MOIRAI, "moirai_zs", "mae"),
    Check("benchmark_moirai_cov_mae", BENCHMARK,
          r"\| Moirai \+ covariates zero-shot \| (\d+\.\d+) \|",
          MOIRAI, "moirai_cov", "mae"),

    # -- README FM rows ---------------------------------------------------
    Check("readme_chronos_mae", README,
          r"\| Chronos-Bolt zero-shot \+ CQR \| (\d+\.\d+) \|",
          CHRONOS, "chronos_bolt_zs", "mae"),
    Check("readme_timesfm_mae", README,
          r"\| TimesFM 2\.5 zero-shot \| (\d+\.\d+) \|",
          TIMESFM, "timesfm_zs", "mae"),

    # -- RESULTS 3-member ensemble table ---------------------------------
    Check("results_ens3_mae", RESULTS,
          r"\| \*\*ens_crps_cqr\*\* \| \*\*(\d+\.\d+)\*\*",
          ENS3, "ens_crps_cqr", "mae"),
    Check("results_ens3_rmae", RESULTS,
          r"\| \*\*ens_crps_cqr\*\* \| \*\*\d+\.\d+\*\* \| \*\*(\d+\.\d+)\*\*",
          ENS3, "ens_crps_cqr", "rmae"),
    Check("results_ens3_coverage", RESULTS,
          r"\| \*\*ens_crps_cqr\*\* \| \*\*\d+\.\d+\*\* \| \*\*\d+\.\d+\*\* \| \*\*(\d+\.\d+)%",
          ENS3, "ens_crps_cqr", "coverage_80_pct"),
    Check("results_ens3_winkler", RESULTS,
          r"\| \*\*ens_crps_cqr\*\* \| \*\*\d+\.\d+\*\* \| \*\*\d+\.\d+\*\* \| "
          r"\*\*\d+\.\d+%\*\* \| \*\*(\d+\.\d+)\*\*",
          ENS3, "ens_crps_cqr", "winkler"),
    Check("results_ens3_raw_mae", RESULTS,
          r"\| ens_crps \(raw blend\) \| (\d+\.\d+) \|",
          ENS3, "ens_crps", "mae"),
    Check("results_ens3_raw_winkler", RESULTS,
          r"\| ens_crps \(raw blend\) \| \d+\.\d+ \| \d+\.\d+ \| \d+\.\d+% \| (\d+\.\d+) \|",
          ENS3, "ens_crps", "winkler"),
    Check("results_ens_equal_mae", RESULTS,
          r"\| ens_equal \| (\d+\.\d+) \|",
          ENS3, "ens_equal", "mae"),
    Check("results_ens_equal_rmae", RESULTS,
          r"\| ens_equal \| \d+\.\d+ \| (\d+\.\d+) \|",
          ENS3, "ens_equal", "rmae"),
    Check("results_ens_equal_winkler", RESULTS,
          r"\| ens_equal \| \d+\.\d+ \| \d+\.\d+ \| \d+\.\d+% \| (\d+\.\d+) \|",
          ENS3, "ens_equal", "winkler"),
    Check("results_ens3_lgbm_row_mae", RESULTS,
          r"\| LGBM champion \(365d\) \| (\d+\.\d+) \|",
          ENS3, "lgbm", "mae"),
    Check("results_ens3_lgbm_row_winkler", RESULTS,
          r"\| LGBM champion \(365d\) \| \d+\.\d+ \| \d+\.\d+ \| \d+\.\d+% \| (\d+\.\d+) \|",
          ENS3, "lgbm", "winkler"),
    Check("results_lgbm_1095_mae", RESULTS,
          r"\| LGBM \(1095d window\) \| (\d+\.\d+) \|",
          ENS1095, "lgbm_1095", "mae"),
    Check("results_lgbm_1095_rmae", RESULTS,
          r"\| LGBM \(1095d window\) \| \d+\.\d+ \| (\d+\.\d+) \|",
          ENS1095, "lgbm_1095", "rmae"),

    # -- BENCHMARK / README 3-member ensemble row ------------------------
    Check("benchmark_ens3_mae", BENCHMARK,
          r"\| Ensemble, 3-member variant \(CRPS-weighted \+ CQR\) \| (\d+\.\d+) \|",
          ENS3, "ens_crps_cqr", "mae"),
    Check("benchmark_ens3_rmae", BENCHMARK,
          r"\| Ensemble, 3-member variant \(CRPS-weighted \+ CQR\) \| \d+\.\d+ \| (\d+\.\d+) \|",
          ENS3, "ens_crps_cqr", "rmae"),
    Check("benchmark_ens3_coverage", BENCHMARK,
          r"\| Ensemble, 3-member variant \(CRPS-weighted \+ CQR\) \| \d+\.\d+ \| "
          r"\d+\.\d+ \| (\d+\.\d+)%",
          ENS3, "ens_crps_cqr", "coverage_80_pct"),
    Check("readme_ens3_rmae", README,
          r"\| Ensemble \(3-member variant, CRPS \+ CQR\) \| \d+\.\d+ \| (\d+\.\d+) \|",
          ENS3, "ens_crps_cqr", "rmae"),

    # -- RESULTS 4-member (ens4_tft) table -------------------------------
    Check("results_ens4_mae", RESULTS,
          r"\| \*\*ens4_tft \(CQR\)\*\* \| \*\*(\d+\.\d+)\*\*",
          ENS4, "ens_crps_cqr", "mae"),
    Check("results_ens4_rmae", RESULTS,
          r"\| \*\*ens4_tft \(CQR\)\*\* \| \*\*\d+\.\d+\*\* \| \*\*(\d+\.\d+)\*\*",
          ENS4, "ens_crps_cqr", "rmae"),
    Check("results_ens4_coverage", RESULTS,
          r"\| \*\*ens4_tft \(CQR\)\*\* \| \*\*\d+\.\d+\*\* \| \*\*\d+\.\d+\*\* \| \*\*(\d+\.\d+)%",
          ENS4, "ens_crps_cqr", "coverage_80_pct"),
    Check("results_ens4_winkler", RESULTS,
          r"\| \*\*ens4_tft \(CQR\)\*\* \| \*\*\d+\.\d+\*\* \| \*\*\d+\.\d+\*\* \| "
          r"\*\*\d+\.\d+%\*\* \| \*\*(\d+\.\d+)\*\*",
          ENS4, "ens_crps_cqr", "winkler"),
    Check("results_ens4_equal_mae", RESULTS,
          r"\| ens_equal \(4 members\) \| (\d+\.\d+) \|",
          ENS4, "ens_equal", "mae"),
    Check("results_ens4_equal_rmae", RESULTS,
          r"\| ens_equal \(4 members\) \| \d+\.\d+ \| (\d+\.\d+) \|",
          ENS4, "ens_equal", "rmae"),
    Check("results_ens4_equal_coverage", RESULTS,
          r"\| ens_equal \(4 members\) \| \d+\.\d+ \| \d+\.\d+ \| (\d+\.\d+)% raw",
          ENS4, "ens_equal", "coverage_80_pct"),
    Check("results_ens4_tft_member_mae", RESULTS,
          r"\| TFT-730 ens-3 alone \| (\d+\.\d+) \|",
          ENS4, "tft", "mae"),
    Check("results_ens4_tft_member_winkler", RESULTS,
          r"\| TFT-730 ens-3 alone \| \d+\.\d+ \| \d+\.\d+ \| \d+\.\d+% \| (\d+\.\d+) \|",
          ENS4, "tft", "winkler"),

    # -- BENCHMARK / README 4-member row ---------------------------------
    Check("benchmark_ens4_mae", BENCHMARK,
          r"\| \*\*4-member ensemble \(\+ TFT, CQR\)\*\* \| \*\*(\d+\.\d+)\*\*",
          ENS4, "ens_crps_cqr", "mae"),
    Check("benchmark_ens4_rmae", BENCHMARK,
          r"\| \*\*4-member ensemble \(\+ TFT, CQR\)\*\* \| \*\*\d+\.\d+\*\* \| \*\*(\d+\.\d+)\*\*",
          ENS4, "ens_crps_cqr", "rmae"),
    Check("benchmark_ens4_coverage", BENCHMARK,
          r"\| \*\*4-member ensemble \(\+ TFT, CQR\)\*\* \| \*\*\d+\.\d+\*\* \| "
          r"\*\*\d+\.\d+\*\* \| \*\*(\d+\.\d+)%",
          ENS4, "ens_crps_cqr", "coverage_80_pct"),
    Check("benchmark_ens4_prose_mae", BENCHMARK,
          r"it gives the new best \((\d+\.\d+) vs",
          ENS4, "ens_crps_cqr", "mae"),
    Check("readme_ens4_mae", README,
          r"\| \*\*Ensemble \(4-member, \+ TFT donor\)\*\* \| \*\*(\d+\.\d+)\*\*",
          ENS4, "ens_crps_cqr", "mae"),
    Check("readme_ens4_rmae", README,
          r"\| \*\*Ensemble \(4-member, \+ TFT donor\)\*\* \| \*\*\d+\.\d+\*\* \| \*\*(\d+\.\d+)\*\*",
          ENS4, "ens_crps_cqr", "rmae"),
    Check("readme_ens4_prose_mae", README,
          r"MAE (\d+\.\d+) EUR/MWh, 40% better than seasonal naive",
          ENS4, "ens_crps_cqr", "mae"),

    # -- Battery P&L (RESULTS table) -------------------------------------
    Check("results_pnl_perfect_eur", RESULTS,
          r"\| Perfect foresight \| (\d+) \|", PNL, "perfect", "eur_per_day"),
    Check("results_pnl_perfect_capture", RESULTS,
          r"\| Perfect foresight \| \d+ \| (\d+\.\d+) \|", PNL, "perfect", "capture_rate"),
    Check("results_pnl_ens_eur", RESULTS,
          r"\| \*\*ens_crps_cqr\*\* \| \*\*(\d+)\*\*", PNL, "ens_crps_cqr", "eur_per_day"),
    Check("results_pnl_ens_capture", RESULTS,
          r"\| \*\*ens_crps_cqr\*\* \| \*\*\d+\*\* \| \*\*(\d+\.\d+)\*\*",
          PNL, "ens_crps_cqr", "capture_rate"),
    Check("results_pnl_ens_lossdays", RESULTS,
          r"\| \*\*ens_crps_cqr\*\* \| \*\*\d+\*\* \| \*\*\d+\.\d+\*\* \| (\d+\.\d+)%",
          PNL, "ens_crps_cqr", "loss_days_pct"),
    Check("results_pnl_lgbm_eur", RESULTS,
          r"\| LGBM champion \| (\d+) \|", PNL, "lgbm", "eur_per_day"),
    Check("results_pnl_lgbm_capture", RESULTS,
          r"\| LGBM champion \| \d+ \| (\d+\.\d+) \|", PNL, "lgbm", "capture_rate"),
    Check("results_pnl_lear_eur", RESULTS,
          r"\| LEAR \| (\d+) \|", PNL, "lear", "eur_per_day"),
    Check("results_pnl_lear_capture", RESULTS,
          r"\| LEAR \| \d+ \| (\d+\.\d+) \|", PNL, "lear", "capture_rate"),
    Check("results_pnl_chronos_eur", RESULTS,
          r"\| Chronos zero-shot \| (\d+) \|", PNL, "chronos", "eur_per_day"),
    Check("results_pnl_chronos_capture", RESULTS,
          r"\| Chronos zero-shot \| \d+ \| (\d+\.\d+) \|", PNL, "chronos", "capture_rate"),
    Check("results_pnl_timesfm_eur", RESULTS,
          r"\| TimesFM zero-shot \| (\d+) \|", PNL, "timesfm", "eur_per_day"),
    Check("results_pnl_timesfm_capture", RESULTS,
          r"\| TimesFM zero-shot \| \d+ \| (\d+\.\d+) \|", PNL, "timesfm", "capture_rate"),
    Check("results_pnl_naive_eur", RESULTS,
          r"\| Naive \(yesterday\) \| (\d+) \|", PNL, "naive", "eur_per_day"),
    Check("results_pnl_naive_capture", RESULTS,
          r"\| Naive \(yesterday\) \| \d+ \| (\d+\.\d+) \|", PNL, "naive", "capture_rate"),

    # -- Battery P&L (BENCHMARK master table capture column) -------------
    Check("benchmark_pnl_ens3_capture", BENCHMARK,
          r"\| Ensemble, 3-member variant \(CRPS-weighted \+ CQR\) \| \d+\.\d+ \| "
          r"\d+\.\d+ \| \d+\.\d+% \| (\d+\.\d+) \|",
          PNL, "ens_crps_cqr", "capture_rate"),
    Check("benchmark_pnl_lgbm_capture", BENCHMARK,
          r"\| LGBM 365d \+ CQR \(champion\) \| \d+\.\d+ \| \d+\.\d+ \| \d+\.\d+% \| (\d+\.\d+) \|",
          PNL, "lgbm", "capture_rate"),
    Check("benchmark_pnl_lear_capture", BENCHMARK,
          r"\| LEAR \+ CQR \(industry standard\) \| \d+\.\d+ \| \d+\.\d+ \| "
          r"\d+\.\d+% \| (\d+\.\d+) \|",
          PNL, "lear", "capture_rate"),
    Check("benchmark_pnl_chronos_capture", BENCHMARK,
          r"\| Chronos-Bolt zero-shot \+ CQR \| \d+\.\d+ \| \d+\.\d+ \| \d+\.\d+% \| (\d+\.\d+) \|",
          PNL, "chronos", "capture_rate"),
    Check("benchmark_pnl_timesfm_capture", BENCHMARK,
          r"\| TimesFM 2\.5 zero-shot \| \d+\.\d+ \| \d+\.\d+ \| \d+\.\d+% raw \| (\d+\.\d+) \|",
          PNL, "timesfm", "capture_rate"),
    Check("benchmark_pnl_naive_capture", BENCHMARK,
          r"\| Naive \(same hour yesterday\) \| \d+\.\d+ \| \d+\.\d+ \| "
          r"\d+\.\d+% \| (\d+\.\d+) \|",
          PNL, "naive", "capture_rate"),

    # -- Battery P&L (README master table capture column) ----------------
    Check("readme_pnl_ens3_capture", README,
          r"\| Ensemble \(3-member variant, CRPS \+ CQR\) \| \d+\.\d+ \| \d+\.\d+ \| "
          r"\d+\.\d+% \| (\d+\.\d+) \|",
          PNL, "ens_crps_cqr", "capture_rate"),
    Check("readme_pnl_lgbm_capture", README,
          r"\| LightGBM quantile \+ conformal \| \d+\.\d+ \| \d+\.\d+ \| \d+\.\d+% \| (\d+\.\d+) \|",
          PNL, "lgbm", "capture_rate"),
    Check("readme_pnl_lear_capture", README,
          r"\| LEAR \+ conformal \(published daily\) \| \d+\.\d+ \| \d+\.\d+ \| "
          r"\d+\.\d+% \| (\d+\.\d+) \|",
          PNL, "lear", "capture_rate"),
    Check("readme_pnl_chronos_capture", README,
          r"\| Chronos-Bolt zero-shot \+ CQR \| \d+\.\d+ \| \d+\.\d+ \| \d+\.\d+% \| (\d+\.\d+) \|",
          PNL, "chronos", "capture_rate"),
    Check("readme_pnl_timesfm_capture", README,
          r"\| TimesFM 2\.5 zero-shot \| \d+\.\d+ \| \d+\.\d+ \| \d+\.\d+% raw \| (\d+\.\d+) \|",
          PNL, "timesfm", "capture_rate"),
    Check("readme_pnl_naive_capture", README,
          r"\| Naive \(same hour yesterday\) \| \d+\.\d+ \| \d+\.\d+ \| \d+\.\d+% \| (\d+\.\d+) \|",
          PNL, "naive", "capture_rate"),

    # -- Spike classifier -------------------------------------------------
    Check("results_spike_auc", RESULTS,
          r"labels\): AUC (\d+\.\d+)", SPIKE, "42", "auc"),
    Check("results_spike_brier", RESULTS,
          r"labels\): AUC \d+\.\d+, Brier (\d+\.\d+)", SPIKE, "42", "brier"),
    Check("results_spike_precision", RESULTS,
          r"labels\): AUC \d+\.\d+, Brier \d+\.\d+, precision@2 (\d+\.\d+)",
          SPIKE, "42", "precision_at2"),
    Check("benchmark_spike_auc", BENCHMARK,
          r"spike classifier \(AUC (\d+\.\d+)\)", SPIKE, "42", "auc"),
    Check("readme_spike_auc", README,
          r"spike classifier\s+\(AUC (\d+\.\d+)\)", SPIKE, "42", "auc"),

    # -- DM p-values, ensemble artifact -----------------------------------
    Check("results_dm_ens4_vs_ens3_table", RESULTS,
          r"\| ens4_tft beats ens3 \| ([\d.e+-]+) \|",
          DM_ENS, "ens4_tft", "p_one_sided", "a_better", "sig", (("b", "ens3"),)),
    Check("results_dm_ens4_vs_ens3_prose", RESULTS,
          r"\(gate 0\.10\), DM p=([\d.e+-]+)",
          DM_ENS, "ens4_tft", "p_one_sided", "a_better", "sig", (("b", "ens3"),)),
    Check("benchmark_dm_ens4_vs_ens3", BENCHMARK,
          r"DM p=([\d.e+-]+) \(`reports/backtests/2026-07-28_stats_tests_ens_dm\.csv`\)",
          DM_ENS, "ens4_tft", "p_one_sided", "a_better", "sig", (("b", "ens3"),)),
    Check("benchmark_dm_ens4_prose", BENCHMARK,
          r"on one window, DM p=([\d.e+-]+),",
          DM_ENS, "ens4_tft", "p_one_sided", "a_better", "sig", (("b", "ens3"),)),
    Check("readme_dm_ens4_vs_ens3", README,
          r"\(DM p=([\d.e+-]+) vs the 3-member blend",
          DM_ENS, "ens4_tft", "p_one_sided", "a_better", "sig", (("b", "ens3"),)),
    Check("results_dm_ens3_vs_lgbm_table", RESULTS,
          r"\| ens3 beats champion \| ([\d.e+-]+) \|",
          DM_ENS, "ens3", "p_one_sided", "a_better", "sig", (("b", "lgbm"),)),
    Check("results_dm_ens3_vs_lgbm_prose", RESULTS,
          r"\(gate 0\.15\), DM p=([\d.e+-]+)",
          DM_ENS, "ens3", "p_one_sided", "a_better", "sig", (("b", "lgbm"),)),
    Check("benchmark_dm_ens3_vs_lgbm", BENCHMARK,
          r"improves the blend\s+\(\d+\.\d+ vs \d+\.\d+, DM p=([\d.e+-]+)\)",
          DM_ENS, "ens3", "p_one_sided", "a_better", "sig", (("b", "lgbm"),)),
    Check("results_dm_lgbm_vs_lear_table", RESULTS,
          r"\| LGBM beats LEAR \(2-yr\) \| ([\d.e+-]+) \|",
          DM_ENS, "lgbm", "p_one_sided", "a_better", "sig",
          (("b", "lear"), ("window", "17720h full 07-27 run"))),
    Check("results_dm_lgbm_vs_lear_prose", RESULTS,
          r"corrected run \(DM p=([\d.e+-]+),",
          DM_ENS, "lgbm", "p_one_sided", "a_better", "sig",
          (("b", "lear"), ("window", "17720h full 07-27 run"))),
    Check("results_dm_lgbm_vs_lear_history", RESULTS,
          r"it\s+clears the bar \(p=([\d.e+-]+)\)\.",
          DM_ENS, "lgbm", "p_one_sided", "a_better", "sig",
          (("b", "lear"), ("window", "17720h full 07-27 run"))),
    Check("benchmark_dm_lgbm_vs_lear", BENCHMARK,
          r"it clears the bar \(p=([\d.e+-]+),",
          DM_ENS, "lgbm", "p_one_sided", "a_better", "sig",
          (("b", "lear"), ("window", "17720h full 07-27 run"))),
    Check("readme_dm_lgbm_vs_lear", README,
          r"bar \(p=([\d.e+-]+)\)",
          DM_ENS, "lgbm", "p_one_sided", "a_better", "sig",
          (("b", "lear"), ("window", "17720h full 07-27 run"))),

    # -- DM p-values, foundation-model artifact ----------------------------
    Check("results_dm_chronos_vs_timesfm", RESULTS,
          r"Chronos beats TimesFM \(DM p=([\d.e+-]+)\)",
          DM_FM, "chronos", "p_one_sided", "a_better", "sig", (("b", "timesfm"),)),
    Check("results_dm_moirai", RESULTS,
          r"covariates HURT zero-shot Moirai\s+\(\d+\.\d+ vs \d+\.\d+, DM p=([\d.e+-]+)",
          DM_FM, "moirai_zs", "p_one_sided", "a_better", "sig", (("b", "moirai_cov"),)),
    Check("results_moirai_cov_mae", RESULTS,
          r"covariates HURT zero-shot Moirai\s+\((\d+\.\d+) vs",
          MOIRAI, "moirai_cov", "mae"),
    Check("results_moirai_uni_mae", RESULTS,
          r"covariates HURT zero-shot Moirai\s+\(\d+\.\d+ vs (\d+\.\d+),",
          MOIRAI, "moirai_zs", "mae"),

    # -- DM p-values, 07-22 artifact (window + TFT + LEAR history) ---------
    Check("results_dm_win1095_table", RESULTS,
          r"\| 1095d window beats 365d \| ([\d.e+-]+) \|",
          DM_0722, "lgbm_win1095 vs lgbm_win365", "p_one_sided", "comparison", "sig"),
    Check("benchmark_dm_win1095", BENCHMARK,
          r"for free \(DM p=([\d.e+-]+)\)",
          DM_0722, "lgbm_win1095 vs lgbm_win365", "p_one_sided", "comparison", "sig"),
    Check("results_dm_lgbm_vs_tft", RESULTS,
          r"\| LGBM beats TFT-730 ens-3 \| ([\d.e+-]+) \|",
          DM_0722, "lgbm vs tft730_ens3 (2-yr)", "p_one_sided", "comparison", "sig"),
    Check("results_dm_lear_0722_history", RESULTS,
          r"NOT significant \(p=([\d.e+-]+)\) and every doc",
          DM_0722, "lgbm vs lear (2-yr)", "p_one_sided", "comparison", "sig"),
    Check("benchmark_dm_lear_0722_history", BENCHMARK,
          r"was NOT significant\s+\(p=([\d.e+-]+)\)",
          DM_0722, "lgbm vs lear (2-yr)", "p_one_sided", "comparison", "sig"),
    Check("readme_dm_lear_0722_history", README,
          r"not significant \(p=([\d.e+-]+)\)",
          DM_0722, "lgbm vs lear (2-yr)", "p_one_sided", "comparison", "sig"),
]


# --------------------------------------------------------------------
# runner
# --------------------------------------------------------------------


def run_checks(checks: list[Check], root: Path = ROOT) -> list[Failure]:
    """Run every check. Returns one Failure per broken check."""
    failures: list[Failure] = []
    docs: dict[str, str] = {}
    frames: dict[str, pd.DataFrame] = {}

    for check in checks:
        doc_path = root / check.doc
        art_path = root / check.artifact
        if check.doc not in docs:
            if not doc_path.exists():
                failures.append(Failure(check, f"missing doc {check.doc}"))
                continue
            docs[check.doc] = doc_path.read_text(encoding="utf-8")
        if check.artifact not in frames:
            if not art_path.exists():
                failures.append(Failure(check, f"missing artifact {check.artifact}"))
                continue
            frames[check.artifact] = pd.read_csv(art_path)

        try:
            quoted = find_quoted(docs[check.doc], check.pattern)
        except ValueError as exc:
            failures.append(Failure(check, str(exc)))
            continue
        if not quoted:
            failures.append(
                Failure(check, f"pattern found nothing in {check.doc} -- doc restructured?")
            )
            continue

        try:
            value = lookup(
                frames[check.artifact], check.key_column, check.row,
                check.column, check.filters,
            ) * check.scale
        except KeyError as exc:
            failures.append(Failure(check, f"{check.artifact}: {exc}"))
            continue

        for doc_text in quoted:
            if not values_match(doc_text, value, check.mode):
                failures.append(
                    Failure(
                        check,
                        "doc number does not match the artifact",
                        doc_value=doc_text,
                        artifact_value=format_like(value, doc_text, check.mode),
                        raw_value=value,
                    )
                )
    return failures


def report(checks: list[Check], failures: list[Failure]) -> str:
    """Human-readable result. One block per failure."""
    lines: list[str] = []
    for fail in failures:
        chk = fail.check
        lines.append(f"FAIL  {chk.name}")
        lines.append(f"      doc       {chk.doc}")
        lines.append(f"      artifact  {chk.artifact} [{chk.row}].{chk.column}")
        lines.append(f"      reason    {fail.message}")
        if fail.doc_value:
            lines.append(f"      doc says  {fail.doc_value}")
            lines.append(f"      artifact  {fail.artifact_value}  (raw {fail.raw_value!r})")
        lines.append("")
    passed = len(checks) - len({id(f.check) for f in failures})
    lines.append(f"{passed}/{len(checks)} checks passed, {len(failures)} failure(s).")
    return "\n".join(lines)


def main() -> int:
    failures = run_checks(CHECKS)
    print(report(CHECKS, failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
