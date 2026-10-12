import json
import math
from pathlib import Path

import numpy as np

from kurt_switch.core import (
    compute_row_stats,
    get_codebook,
    quantize_block,
    quantize_row,
    select_codebook,
)


def test_switch_picks_odd_on_sparse():
    rng = np.random.RandomState(0)
    row = np.zeros(64)
    row[:4] = [9.5, -8.0, 7.0, 6.0]  # few huge outliers, rest exact zeros
    assert select_codebook(row) == "odd16"
    stats = compute_row_stats(row)
    assert stats["zero_fraction"] > 0.30
    out = quantize_row(row)
    assert out["codebook"] == "odd16"
    assert 0.0 not in get_codebook("odd16")


def test_ls_scale_optimal():
    rng = np.random.RandomState(1)
    row = rng.randn(48)
    out = quantize_row(row)
    base = np.asarray(out.get("levels", get_codebook(out["codebook"])))
    idx = np.asarray(out["indices"]).ravel()
    s_opt = float(np.dot(row, base[idx]) / np.dot(base[idx], base[idx]))
    assert math.isfinite(out["scale"])
    assert abs(out["scale"] - s_opt) < 1e-9


def test_roundtrip_error_bound():
    rng = np.random.RandomState(2)
    block = rng.randn(8, 32)
    out = quantize_block(block)
    assert out["reconstruction"].shape == (8, 32)
    assert out["mse"] < 0.05
    rel = out["mse"] / float(np.mean(block ** 2))
    assert rel < 0.10


def test_deterministic():
    rng = np.random.RandomState(3)
    row = rng.randn(32)
    a = quantize_row(row)
    b = quantize_row(row.copy())
    assert np.array_equal(a["indices"], b["indices"])
    assert a["scale"] == b["scale"]
    assert a["codebook"] == b["codebook"]


def test_readme_numbers_reproducible():
    # Fixed-seed block used in README results table; adaptive must beat static
    # uniform and land at the published MSE (tolerance guards platform noise).
    rng = np.random.RandomState(123)
    gauss = rng.randn(24, 64)
    sparse = np.zeros((4, 64))
    sparse[:, :3] = rng.randn(4, 3) * 6.0
    skew = np.abs(rng.randn(4, 64)) + 0.5
    block = np.vstack([gauss, sparse, skew])
    adapt = quantize_block(block)
    uni_mses = []
    for i in range(block.shape[0]):
        uni_mses.append(quantize_row(block[i], codebook="uniform16")["mse"])
    uni_mse = float(np.mean(uni_mses))
    assert adapt["mse"] <= uni_mse + 1e-12
    # Published reference value (regenerate with demos/demo.py seed 123 path).
    assert abs(adapt["mse"] - 0.008851) < 0.01
    assert adapt["mse"] < 0.05
