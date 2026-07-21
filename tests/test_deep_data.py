"""Tests for src/models/deep/data.py — standardize_covariates."""

import numpy as np
import pytest
import torch

from src.models.deep.data import DaySamples, apply_covariate_stats, standardize_covariates


def _make_samples(n: int, enc_feat: int, fut_feat: int) -> DaySamples:
    """Helper: random DaySamples with given dimensions."""
    return DaySamples(
        enc=torch.randn(n, 24, enc_feat),
        fut=torch.randn(n, 24, fut_feat),
        y=torch.randn(n, 24),
        anchor=torch.randn(n, 24),
        mean=torch.zeros(n),
        std=torch.ones(n),
        days=[],
    )


def test_standardize_fut_columns_become_unit_normal():
    """After standardize, fut non-tail training columns have mean≈0, std≈1."""
    tr = _make_samples(200, 1, 5)
    va = _make_samples(50, 1, 5)
    standardize_covariates(tr, va, n_tail=1)

    # cols 0..3 (non-tail) should be z-scored on training set
    m = tr.fut[:, :, :4].mean().item()
    s = tr.fut[:, :, :4].std().item()
    assert abs(m) < 0.01, f"training mean {m} should be ~0"
    assert abs(s - 1.0) < 0.05, f"training std {s} should be ~1"


def test_zero_variance_column_zeroed_in_val():
    """A fut column that is constant in training must be zeroed in val/test.

    This guards against the offshore-wind bug: a new RES series that first
    appears in val period has std=0 in training. Without the guard,
    standardize divides by the 1e-6 clamp and val values blow up ~1e6×.
    """
    n_tr, n_va = 100, 30
    fut_feat = 5  # col 3 will be constant in train but non-zero in val

    tr = _make_samples(n_tr, 1, fut_feat)
    va = _make_samples(n_va, 1, fut_feat)

    # col 3 = zero in training, non-zero in val (simulates new offshore wind)
    tr.fut[:, :, 3] = 0.0
    va.fut[:, :, 3] = 15.0  # ~19 MW, like Baltic wind commissioning

    standardize_covariates(tr, va, n_tail=1)

    # Without fix: va.fut[:,:,3] would be 15/1e-6 = 15,000,000
    # With fix: should be zeroed
    max_val = va.fut[:, :, 3].abs().max().item()
    assert max_val < 1e-3, (
        f"zero-variance column must be zeroed in val, got max={max_val:.1f}"
    )


def test_zero_variance_does_not_corrupt_other_columns():
    """The zero-variance guard should only zero the constant column."""
    n_tr, n_va = 100, 30
    fut_feat = 5

    # Use data well away from zero so z-scoring causes a measurable shift
    tr = _make_samples(n_tr, 1, fut_feat)
    tr.fut[:] += 100.0          # mean ~100, so z-scoring will shift by ~-100/std
    va = _make_samples(n_va, 1, fut_feat)
    va.fut[:] += 100.0

    tr.fut[:, :, 3] = 0.0       # constant in training
    va.fut[:, :, 3] = 15.0      # non-zero in val

    raw_col0_mean = va.fut[:, :, 0].mean().item()   # ~100 before standardize

    standardize_covariates(tr, va, n_tail=1)

    # Col 0 should be z-scored (mean shifted from ~100 to ~0)
    new_col0_mean = va.fut[:, :, 0].mean().item()
    assert abs(new_col0_mean) < 5.0, (
        f"z-scored col0 should be near 0, got {new_col0_mean:.1f}"
    )
    # Sanity: the raw value was ~100
    assert abs(raw_col0_mean - 100) < 5.0


def test_apply_covariate_stats_idempotent_for_normal_columns():
    """apply_covariate_stats produces consistent z-scores on a fresh sample."""
    tr = _make_samples(200, 1, 4)
    stats = standardize_covariates(tr, n_tail=1)

    new_sample = _make_samples(1, 1, 4)
    original = new_sample.fut[:, :, :-1].clone()
    apply_covariate_stats(new_sample, stats)

    # After applying, (x - mu) / sigma should differ from the raw value
    diff = (new_sample.fut[:, :, :-1] - original).abs().max().item()
    assert diff > 0, "apply_covariate_stats should have changed fut values"
