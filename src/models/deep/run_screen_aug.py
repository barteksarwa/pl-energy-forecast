"""Q2: origin augmentation — 4 forecast origins per training day (~4x samples).

Validation and test stay at the true 09:00 D-1 origin. Only training grows.
"""

from __future__ import annotations

import sys

from src.config import REPO_ROOT
from src.models.deep.data import build_samples, standardize_covariates
from src.models.deep.nets import VARIANTS
from src.models.deep.screen_common import eval_and_log, load_data

RESULTS = REPO_ROOT / "outputs" / "deep_campaign_v4_aug.csv"
OFFSETS = (0, -6, -12, -18)
JOBS = [("enc_dec", 64), ("enc_dec", 128)]
SEEDS = [42, 7]


def main() -> int:
    load, weather, _, split = load_data()
    tr = build_samples(load, weather, split["train"], origin_offsets_h=OFFSETS)
    va = build_samples(load, weather, split["val"])
    te = build_samples(load, weather, split["test"])
    standardize_covariates(tr, va, te)
    print(f"samples (augmented train): {len(tr.days)}/{len(va.days)}/{len(te.days)}",
          flush=True)
    for vname, hidden in JOBS:
        for seed in SEEDS:
            net = VARIANTS[vname](tr.enc.shape[-1], tr.fut.shape[-1], hidden=hidden)
            eval_and_log(net, tr, va, te, RESULTS,
                         {"variant": vname, "hidden": hidden, "tag": "aug4"}, seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
