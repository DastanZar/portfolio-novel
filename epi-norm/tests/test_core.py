"""Tests for epi-norm core (5 tests, numpy-only; torch parity when available)."""
import numpy as np

from epi_norm.core import (
    fused_epilogue_backward_recompute,
    fused_epilogue_backward_store,
    fused_epilogue_ref,
    hbm_accountant,
    pick_block,
    rmsnorm_ref,
)


def _torch_rmsnorm(x, w, eps=1e-6):
    try:
        import torch
    except Exception:
        return None
    xt = torch.as_tensor(np.asarray(x, dtype=np.float64))
    wt = torch.as_tensor(np.asarray(w, dtype=np.float64))
    ms = xt.pow(2).mean(dim=-1, keepdim=True)
    rr = 1.0 / torch.sqrt(ms + eps)
    return (xt * rr * wt).numpy(), rr.numpy()


def test_rmsnorm_matches_naive_and_torch():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 16))
    w = 1 + 0.1 * rng.normal(size=(16,))
    y, rr = rmsnorm_ref(x, w)
    # naive loop parity
    ms = (x**2).mean(axis=-1, keepdims=True)
    expected = x / np.sqrt(ms + 1e-6) * w
    assert np.allclose(y, expected, rtol=1e-12, atol=1e-12)
    assert rr.shape == (4, 1)
    t = _torch_rmsnorm(x, w)
    if t is not None:
        yt, _ = t
        assert np.allclose(y, yt, rtol=1e-9, atol=1e-9)


def test_fused_matches_composition():
    rng = np.random.default_rng(1)
    B, T, D, H = 2, 4, 8, 12
    x = rng.normal(size=(B, T, D))
    r = rng.normal(size=(B, T, D))
    w = np.ones(D) + 0.05 * rng.normal(size=D)
    Wg = rng.normal(scale=0.3, size=(D, H))
    Wu = rng.normal(scale=0.3, size=(D, H))
    f = fused_epilogue_ref(x, r, w, Wg, Wu)
    # manual composition
    fused = x + r
    n, _ = rmsnorm_ref(fused, w)
    g, u = n @ Wg, n @ Wu
    expected = (g / (1 + np.exp(-g))) * u
    assert np.allclose(f["out"], expected, rtol=1e-12, atol=1e-12)


def test_backward_recompute_matches_store():
    rng = np.random.default_rng(2)
    B, T, D, H = 2, 3, 8, 6
    x = rng.normal(size=(B, T, D))
    r = rng.normal(size=(B, T, D))
    w = np.ones(D)
    Wg = rng.normal(scale=0.3, size=(D, H))
    Wu = rng.normal(scale=0.3, size=(D, H))
    cache = fused_epilogue_ref(x, r, w, Wg, Wu)
    go = rng.normal(size=(B, T, H))
    a = fused_epilogue_backward_recompute(go, cache, w, Wg, Wu)
    b = fused_epilogue_backward_store(go, cache, w, Wg, Wu)
    for k in ("grad_x", "grad_w", "grad_Wg", "grad_Wu"):
        assert np.allclose(a[k], b[k], rtol=1e-10, atol=1e-10), k


def test_accountant_ordering():
    acc = hbm_accountant(B=2, T=128, D=1024, H=2752)
    # recompute stores far less than both alternatives
    assert acc["fused-recompute"]["stored_bytes"] < acc["fused-store"]["stored_bytes"]
    assert acc["fused-recompute"]["stored_bytes"] < acc["baseline"]["stored_bytes"]
    assert acc["fused-recompute"]["total_bytes"] < acc["baseline"]["total_bytes"]
    assert acc["fused-store"]["fwd_bytes"] < acc["baseline"]["fwd_bytes"]
    for k in ("baseline", "fused-store", "fused-recompute"):
        assert acc[k]["total_bytes"] == (
            acc[k]["fwd_bytes"] + acc[k]["stored_bytes"] + acc[k]["bwd_bytes"]
        )


def test_pick_block_pow2():
    assert pick_block(1024) == 1024
    assert pick_block(1000) == 512
    assert pick_block(37) == 32
    assert pick_block(1) == 1
    v = pick_block(2048, max_block=1024)
    assert v == 1024
    # power of two check
    for d in (7, 65, 300, 2752):
        v = pick_block(d)
        assert v & (v - 1) == 0 and v <= min(d, 1024)
