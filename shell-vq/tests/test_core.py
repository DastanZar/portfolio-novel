import math

import numpy as np

from shell_vq.core import (
    build_shell_codebook,
    quantize_shelled,
    rate_distortion_floor,
    residual_refine,
)


def _demo_block(seed=0, n=300, dim=8):
    rng = np.random.RandomState(seed)
    base = rng.randn(n, dim)
    base[n * 2 // 3:] *= 3.0  # heavy outer tail
    return base


def test_shell_quantiles_split():
    rng = np.random.RandomState(0)
    V = rng.randn(99, 8)
    C = build_shell_codebook(8, 32, seed=0)
    out = quantize_shelled(V, C)
    for s in (0, 1, 2):
        assert 28 <= out["counts"][s] <= 38
    assert out["shells"].shape == (99,)


def test_per_shell_scales_ordered():
    V = _demo_block()
    C = build_shell_codebook(8, 64, seed=0)
    out = quantize_shelled(V, C)
    assert out["scales"][0] < out["scales"][1] < out["scales"][2]
    assert all(v > 0 for v in out["scales"].values())


def test_residual_refine_improves_outer():
    V = _demo_block(seed=1)
    C = build_shell_codebook(8, 64, seed=0)
    out = quantize_shelled(V, C)
    mask = out["shells"] == 2
    assert int(mask.sum()) > 0
    ref = residual_refine(V, out["reconstruction"], mask)
    assert ref["outer_mse_after"] <= ref["outer_mse_before"] + 1e-12
    assert ref["mse_after"] <= ref["mse_before"] + 1e-12


def test_deterministic():
    V = _demo_block(seed=2)
    C = build_shell_codebook(8, 64, seed=0)
    a = quantize_shelled(V, C)
    b = quantize_shelled(V.copy(), C.copy())
    assert np.array_equal(a["indices"], b["indices"])
    assert np.array_equal(a["shells"], b["shells"])
    assert a["mse"] == b["mse"]


def test_rate_distortion_floor_monotonic():
    f4 = rate_distortion_floor(1.0, 8, 4.0)
    f8 = rate_distortion_floor(1.0, 8, 8.0)
    f0 = rate_distortion_floor(1.0, 8, 0.0)
    assert f4 > 0 and f8 > 0
    assert f8 < f4 < f0
    assert abs(f0 - 1.0) < 1e-12
    assert abs(f4 - 0.5) < 1e-12


def test_readme_numbers_reproducible():
    # Same generator as demos/demo.py (seed 7); shelled must beat flat and
    # land near the published MSE.
    rng = np.random.RandomState(7)
    n, dim = 500, 8
    V = rng.randn(n, dim)
    V[350:] *= 3.0
    C = build_shell_codebook(dim, 64, seed=0)
    out = quantize_shelled(V, C)
    mask = out["shells"] == 2
    ref = residual_refine(V, out["reconstruction"], mask)
    assert ref["mse_after"] <= out["mse"] + 1e-12
    assert abs(out["mse"] - 1.585221) < 0.05
    assert abs(ref["mse_after"] - 0.701920) < 0.05
    assert out["mse"] < 2.0
