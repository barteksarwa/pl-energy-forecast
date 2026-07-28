"""Unit tests for the doc-number checker helpers.

Synthetic strings and frames only. The real docs are checked by
`make check-docs`, not here -- a test that reads RESULTS.md would fail
every time a campaign is rerun, which is the job of the checker, not of
the test suite.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_doc_numbers.py"


def _load() -> ModuleType:
    """Import the script by path -- `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location("check_doc_numbers", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules, so register first
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cdn = _load()


# -- precision parsing -------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("17.83", 2), ("0.640", 3), ("221", 0), ("2.6e-09", 1), ("1.000", 3)],
)
def test_count_decimals(text: str, expected: int) -> None:
    assert cdn.count_decimals(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("2.6e-09", 2), ("0.0009", 1), ("0.012", 2), ("1.9e-03", 2), ("8.3e-06", 2)],
)
def test_count_sig_figs(text: str, expected: int) -> None:
    assert cdn.count_sig_figs(text) == expected


def test_count_sig_figs_of_zero_is_one() -> None:
    assert cdn.count_sig_figs("0.0") == 1


# -- formatting --------------------------------------------------------


def test_format_like_keeps_trailing_zero() -> None:
    assert cdn.format_like(0.6395201897141274, "0.640", "round") == "0.640"


def test_format_like_sig_mode() -> None:
    assert cdn.format_like(2.605e-09, "2.6e-09", "sig") == "2.6e-09"


def test_format_like_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        cdn.format_like(1.0, "1.0", "fuzzy")


# -- matching ----------------------------------------------------------


def test_values_match_rounds_to_doc_precision() -> None:
    assert cdn.values_match("17.83", 17.828845924696264, "round")


def test_values_match_rejects_one_digit_drift() -> None:
    # the real ens_equal slip: artifact 17.4347 was typed as 17.44
    assert not cdn.values_match("17.44", 17.434683133525354, "round")


def test_values_match_is_precision_aware() -> None:
    # same artifact value, two docs, two printed precisions -- both fine
    assert cdn.values_match("16.88", 16.87878247806367, "round")
    assert cdn.values_match("16.9", 16.87878247806367, "round")


def test_values_match_sig_mode_accepts_extra_artifact_digits() -> None:
    assert cdn.values_match("4.1e-04", 4.101e-04, "sig")
    assert cdn.values_match("0.012", 1.159e-02, "sig")
    assert cdn.values_match("0.0009", 9.11e-04, "sig")


def test_values_match_sig_mode_rejects_wrong_p_value() -> None:
    assert not cdn.values_match("2.6e-09", 2.605e-08, "sig")


def test_values_match_no_tolerance_band() -> None:
    # 0.005 above the doc value already flips the rounding
    assert not cdn.values_match("0.926", 0.9282460167972036, "round")


# -- locating numbers in a doc ----------------------------------------


DOC = """
| Model | MAE | rMAE |
|---|---|---|
| **LGBM quantile + CQR** (champion) | **17.83** | **0.640** |
| LEAR + CQR | 18.46 | 0.662 |

The champion scores 17.83 here and 17.83 again below.
"""


def test_find_quoted_returns_the_capture() -> None:
    found = cdn.find_quoted(DOC, r"\| LEAR \+ CQR \| (\d+\.\d+) \|")
    assert found == ["18.46"]


def test_find_quoted_returns_every_occurrence() -> None:
    # prose repeats numbers; every copy must be checked, not just the first
    assert cdn.find_quoted(DOC, r"(\d+\.\d+) (?:here|again)") == ["17.83", "17.83"]


def test_find_quoted_empty_when_anchor_dies() -> None:
    assert cdn.find_quoted(DOC, r"\| Naive \(1-day\) \| (\d+\.\d+) \|") == []


def test_find_quoted_needs_one_group() -> None:
    with pytest.raises(ValueError):
        cdn.find_quoted(DOC, r"\| LEAR \+ CQR \| (\d+\.\d+) \| (\d+\.\d+) \|")


# -- reading artifact cells -------------------------------------------


FRAME = pd.DataFrame(
    {
        "model": ["lgbm", "lear", "naive"],
        "mae": [17.828845924696264, 18.463290629458974, 27.8784723476298],
        "rmae": [0.6395201897141274, 0.6622777029971911, 1.0],
    }
)

DM = pd.DataFrame(
    {
        "a_better": ["lgbm", "lgbm", "ens3"],
        "b": ["lear", "lear", "lgbm"],
        "p_one_sided": [2.067e-03, 1.935e-03, 4.101e-04],
        "window": ["17456h", "17720h", "17456h"],
    }
)


def test_lookup_by_named_key() -> None:
    assert cdn.lookup(FRAME, "model", "lear", "mae") == pytest.approx(18.463290629458974)


def test_lookup_defaults_to_first_column() -> None:
    assert cdn.lookup(FRAME, None, "lgbm", "rmae") == pytest.approx(0.6395201897141274)


def test_lookup_filters_disambiguate_duplicate_rows() -> None:
    value = cdn.lookup(DM, "a_better", "lgbm", "p_one_sided", (("window", "17720h"),))
    assert value == pytest.approx(1.935e-03)


def test_lookup_rejects_ambiguous_row() -> None:
    with pytest.raises(KeyError, match="matched 2 rows"):
        cdn.lookup(DM, "a_better", "lgbm", "p_one_sided")


def test_lookup_rejects_missing_row() -> None:
    with pytest.raises(KeyError, match="matched 0 rows"):
        cdn.lookup(FRAME, "model", "tft", "mae")


def test_lookup_rejects_missing_column() -> None:
    with pytest.raises(KeyError, match="no column"):
        cdn.lookup(FRAME, "model", "lgbm", "winkler")


def test_lookup_matches_numeric_keys_as_strings() -> None:
    seeds = pd.DataFrame({"seed": [42], "auc": [0.9666]})
    assert cdn.lookup(seeds, "seed", "42", "auc") == pytest.approx(0.9666)


# -- the runner end to end on a temporary repo -------------------------


def _tiny_repo(tmp_path: Path, doc_text: str) -> Path:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "D.md").write_text(doc_text, encoding="utf-8")
    FRAME.to_csv(tmp_path / "a.csv", index=False)
    return tmp_path


CHECK = cdn.Check(
    name="demo", doc="docs/D.md", pattern=r"MAE (\d+\.\d+)",
    artifact="a.csv", row="lgbm", column="mae", key_column="model",
)


def test_run_checks_passes_on_a_correct_doc(tmp_path: Path) -> None:
    root = _tiny_repo(tmp_path, "champion MAE 17.83 EUR/MWh")
    assert cdn.run_checks([CHECK], root) == []


def test_run_checks_reports_a_drifted_number(tmp_path: Path) -> None:
    root = _tiny_repo(tmp_path, "champion MAE 17.87 EUR/MWh")
    failures = cdn.run_checks([CHECK], root)
    assert len(failures) == 1
    assert failures[0].doc_value == "17.87"
    assert failures[0].artifact_value == "17.83"


def test_run_checks_flags_a_repeated_number_that_drifted(tmp_path: Path) -> None:
    root = _tiny_repo(tmp_path, "MAE 17.83 in the table, MAE 17.87 in the prose")
    failures = cdn.run_checks([CHECK], root)
    assert [f.doc_value for f in failures] == ["17.87"]


def test_run_checks_fails_when_the_anchor_dies(tmp_path: Path) -> None:
    root = _tiny_repo(tmp_path, "the champion is best")
    failures = cdn.run_checks([CHECK], root)
    assert len(failures) == 1
    assert "pattern found nothing" in failures[0].message


def test_run_checks_fails_on_a_missing_artifact(tmp_path: Path) -> None:
    root = _tiny_repo(tmp_path, "champion MAE 17.83 EUR/MWh")
    (root / "a.csv").unlink()
    failures = cdn.run_checks([CHECK], root)
    assert len(failures) == 1
    assert "missing artifact" in failures[0].message


def test_report_names_the_failing_check(tmp_path: Path) -> None:
    root = _tiny_repo(tmp_path, "champion MAE 17.87 EUR/MWh")
    failures = cdn.run_checks([CHECK], root)
    text = cdn.report([CHECK], failures)
    assert "FAIL  demo" in text
    assert "0/1 checks passed" in text


# -- the real check list is well formed -------------------------------


def test_every_check_has_one_capture_group_and_a_known_mode() -> None:
    import re

    for check in cdn.CHECKS:
        assert re.compile(check.pattern).groups == 1, check.name
        assert check.mode in {"round", "sig"}, check.name


def test_check_names_are_unique() -> None:
    names = [c.name for c in cdn.CHECKS]
    assert len(names) == len(set(names))
