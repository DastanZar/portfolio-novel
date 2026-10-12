"""Tests for gate-keep compression (5 tests, numpy-only, deterministic)."""
import numpy as np

from gate_keep.compress import (
    compress_with_memory,
    drop_error,
    fit_gate_ridge,
    gate_predict,
    reconstruction_error,
    score_tokens,
)


def test_score_tokens_mass_ordering():
    # 6 keys; key 5 gets all attention mass from every query.
    attn = np.zeros((4, 6))
    attn[:, 5] = 1.0
    attn[:, 2] = 0.25
    scores = score_tokens(attn)
    assert scores.shape == (6,)
    assert int(np.argmax(scores)) == 5
    assert scores[5] > scores[2] > scores[0]
    # 3-D variant averages heads too
    attn3 = np.stack([attn, attn])
    s3 = score_tokens(attn3)
    assert np.allclose(s3, scores)


def test_fit_gate_ridge_closed_form():
    rng = np.random.default_rng(0)
    n, d = 64, 4
    X = rng.normal(size=(n, d))
    w_true = np.array([1.5, -2.0, 0.5, 1.0])
    y = X @ w_true + 0.01 * rng.normal(size=n)
    model = fit_gate_ridge(X, y, l2=1e-3, hidden=16, seed=1)
    pred = gate_predict(model, X)
    mse = float(np.mean((pred - y) ** 2))
    baseline = float(np.mean((y - y.mean()) ** 2))
    assert mse < 0.1 * baseline
    # closed-form check: must satisfy normal equations
    H = np.maximum(X @ model["W1"] + model["b1"], 0.0)
    lhs = (H.T @ H + model["l2"] * np.eye(model["hidden"])) @ model["W2"]
    rhs = H.T @ y
    assert np.allclose(lhs, rhs, rtol=1e-8, atol=1e-8)


def test_compress_budget_and_sink():
    rng = np.random.default_rng(2)
    n, dk, dv = 20, 8, 8
    K = rng.normal(size=(n, dk))
    V = rng.normal(size=(n, dv))
    scores = np.arange(n, dtype=float)  # later tokens score higher
    comp = compress_with_memory(K, V, scores, budget=6, mem_dim=4, sink_keep=2, seed=0)
    assert comp["K_out"].shape == (7, dk)  # budget + 1 memory key
    assert comp["V_out"].shape == (7, dv)
    kept = set(comp["kept_idx"].tolist())
    assert 0 in kept and 1 in kept  # sinks preserved
    assert len(kept) == 6
    assert comp["mem_vec_k"].shape == (4,)
    assert np.isfinite(comp["K_out"]).all()


def test_memory_beats_drop():
    rng = np.random.default_rng(3)
    n, d = 64, 16
    K = rng.normal(size=(n, d))
    V = rng.normal(size=(n, d))
    # decaying scores: head tokens matter, tail is diffuse but nonzero mass
    scores = 1.0 / (1.0 + np.arange(n))
    comp = compress_with_memory(K, V, scores, budget=8, mem_dim=8, sink_keep=2, seed=0)
    e_mem = reconstruction_error(K, V, comp)
    e_drop = drop_error(K, V, comp["kept_idx"])
    assert np.isfinite(e_mem) and np.isfinite(e_drop)
    assert e_mem < e_drop, f"memory {e_mem} should beat drop {e_drop}"


def test_no_compression_when_short():
    rng = np.random.default_rng(4)
    K = rng.normal(size=(5, 4))
    V = rng.normal(size=(5, 4))
    s = np.ones(5)
    comp = compress_with_memory(K, V, s, budget=8, sink_keep=2)
    assert comp["K_out"].shape == (5, 4)
    assert comp["evicted_idx"].size == 0
    assert comp["k_mem"] is None
