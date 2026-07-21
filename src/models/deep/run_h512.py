"""Q3a: capacity close-out — enc_dec h512 (~6M params), one seed.

v2 showed degradation past h64; this closes the 'maybe even bigger?' question.
"""

from __future__ import annotations

import sys

from src.config import REPO_ROOT
from src.models.deep.data import build_samples, standardize_covariates
from src.models.deep.nets import VARIANTS
from src.models.deep.screen_common import eval_and_log, load_data

RESULTS = REPO_ROOT / "outputs" / "deep_campaign_v2.csv"  # joins the capacity axis


def main() -> int:
    load, weather, _, split = load_data()
    tr = build_samples(load, weather, split["train"])
    va = build_samples(load, weather, split["val"])
    te = build_samples(load, weather, split["test"])
    standardize_covariates(tr, va, te)
    net = VARIANTS["enc_dec"](tr.enc.shape[-1], tr.fut.shape[-1], hidden=512)
    eval_and_log(net, tr, va, te, RESULTS,
                 {"variant": "enc_dec", "hidden": 512}, seed=42)
    return 0


if __name__ == "__main__":
    sys.exit(main())
