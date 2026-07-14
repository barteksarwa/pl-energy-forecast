"""Q1 for nets: TSO forecast as known-future covariate. enc_dec h64 + futmlp h256."""

from __future__ import annotations

import sys

from src.config import REPO_ROOT
from src.models.deep.data import build_samples, standardize_covariates
from src.models.deep.nets import VARIANTS
from src.models.deep.screen_common import eval_and_log, load_data

RESULTS = REPO_ROOT / "outputs" / "deep_campaign_v4_tso.csv"
JOBS = [("enc_dec", 64), ("enc_futmlp", 256)]
SEEDS = [42, 7]


def main() -> int:
    load, weather, tso, split = load_data()
    tr = build_samples(load, weather, split["train"], tso=tso)
    va = build_samples(load, weather, split["val"], tso=tso)
    te = build_samples(load, weather, split["test"], tso=tso)
    standardize_covariates(tr, va, te, n_tail=2)
    print(f"samples: {len(tr.days)}/{len(va.days)}/{len(te.days)}", flush=True)
    for vname, hidden in JOBS:
        for seed in SEEDS:
            net = VARIANTS[vname](tr.enc.shape[-1], tr.fut.shape[-1], hidden=hidden)
            eval_and_log(net, tr, va, te, RESULTS,
                         {"variant": vname, "hidden": hidden, "tag": "tso"}, seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
