"""gate-keep: learned gate + latent memory for KV-cache compression.

Novel delta vs prior art:
  - StreamingLLM (Xiao et al. 2023) keeps attention sinks + recent window.
  - H2O (Zhang et al. 2023) keeps heavy-hitter tokens by cumulative attention mass.
  - gate-keep does NOT purely drop the tail. It keeps a top-budget by a *learned*
    gate and compresses the evicted tail into a small latent summary vector
    that is appended back as one extra key/value (so budget+1 rows total).

The gate is a tiny 2-layer probe solved in closed form with ridge regression
(random ReLU expansion + linear readout). No gradient descent, CPU-only,
deterministic. The latent memory is a random orthonormal sketch (8-dim by
default) of the score-weighted evicted barycenter, projected back to model
dim so it participates in attention as a normal key.

Inspiration disclosure: gated-eviction idea family (e.g. arXiv 2601.17668 as
briefed), latent-memory eviction discussions, and the "don't discard, sketch
it" intuition from KV-quantization threads. All code here is original.
"""

from __future__ import annotations

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z)
    e = np.exp(z)
    s = e.sum()
    if s == 0 or not np.isfinite(s):
        return np.full_like(z, 1.0 / max(len(z), 1), dtype=np.float64)
    return (e / s).astype(np.float64)


def score_tokens(attn: np.ndarray, sink_keep: int = 0) -> np.ndarray:
    """Score each key by cumulative attention mass.

    Args:
        attn: attention weights. Accepted shapes:
            - (n_q, n_k): single head / already averaged
            - (n_h, n_q, n_k): multi-head (averaged over heads and queries)
            - (n_k,): pre-aggregated (returned as-is, float copy)
        sink_keep: informational only; sinks are *preserved* by
            :func:`compress_with_memory`, not boosted here. Kept as an
            explicit argument so callers document sink policy (StreamingLLM).

    Returns:
        scores of shape (n_k,) — mean attention received per key (H2O-style
        mass scoring). Higher means more important.

    References:
        - StreamingLLM (Xiao et al. 2023): first tokens are attention sinks.
        - H2O (Zhang et al. 2023): heavy hitters by accumulated scores.
    """
    a = np.asarray(attn, dtype=np.float64)
    if a.ndim == 1:
        return a.astype(np.float64).copy()
    if a.ndim == 2:
        # (n_q, n_k) -> mean over queries
        if a.shape[1] == 0:
            return np.zeros((0,), dtype=np.float64)
        return a.mean(axis=0).astype(np.float64)
    if a.ndim == 3:
        # (n_h, n_q, n_k) -> mean over heads and queries
        if a.shape[2] == 0:
            return np.zeros((0,), dtype=np.float64)
        return a.mean(axis=(0, 1)).astype(np.float64)
    raise ValueError(f"attn must be 1/2/3-D, got shape {a.shape}")


def fit_gate_ridge(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1e-3,
    hidden: int = 16,
    seed: int = 0,
) -> dict:
    """Fit a tiny 2-layer gate probe in closed form (ridge).

    Layer 1 is a *fixed* random ReLU expansion; layer 2 is solved by ridge
    regression. This is the "learned gate": it maps past attention
    statistics (e.g. per-token mean/max/recency/position) to predicted
    future importance.

    Args:
        X: (n_tokens, d_in) past-stat features.
        y: (n_tokens,) future importance targets.
        l2: ridge penalty for the readout.
        hidden: expansion width.
        seed: RNG seed for the fixed first layer.

    Returns:
        dict with W1 (d_in, hidden), b1 (hidden,), W2 (hidden,),
        l2, hidden, d_in and train_mse.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got {X.shape}")
    n, d = X.shape
    if y.shape[0] != n:
        raise ValueError(f"X has {n} rows but y has {y.shape[0]}")
    if n == 0:
        raise ValueError("empty training set")
    if hidden <= 0:
        raise ValueError("hidden must be positive")
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0.0, 1.0 / np.sqrt(max(d, 1)), size=(d, hidden))
    b1 = np.zeros((hidden,), dtype=np.float64)
    H = np.maximum(X @ W1 + b1, 0.0)  # ReLU expansion
    A = H.T @ H + float(l2) * np.eye(hidden)
    b = H.T @ y
    W2 = np.linalg.solve(A, b)
    pred = H @ W2
    mse = float(np.mean((pred - y) ** 2))
    return {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "l2": float(l2),
        "hidden": int(hidden),
        "d_in": int(d),
        "train_mse": mse,
    }


def gate_predict(model: dict, X: np.ndarray) -> np.ndarray:
    """Apply a fitted gate probe to new features."""
    X = np.asarray(X, dtype=np.float64)
    H = np.maximum(X @ model["W1"] + model["b1"], 0.0)
    return (H @ model["W2"]).astype(np.float64)


def _orthonormal_basis(dim: int, mem_dim: int, seed: int) -> np.ndarray:
    """Random orthonormal basis (dim, k) with k = min(mem_dim, dim)."""
    k = max(1, min(int(mem_dim), int(dim)))
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(dim, k))
    Q, _ = np.linalg.qr(G)
    return Q[:, :k].astype(np.float64)


def compress_with_memory(
    K: np.ndarray,
    V: np.ndarray,
    scores: np.ndarray,
    budget: int,
    mem_dim: int = 8,
    sink_keep: int = 2,
    seed: int = 0,
    temp: float = 1.0,
) -> dict:
    """Keep top-`budget` tokens; sketch the evicted tail into 1 latent key.

    Sinks (first `sink_keep` positions) are always kept — StreamingLLM rule.
    Remaining slots go to the highest `scores` outside the sinks (H2O rule).
    The evicted tail is NOT discarded: its score-weighted barycenter is
    sketched through a random orthonormal projector to `mem_dim` dims and
    projected back, forming one extra memory key/value appended last.

    Args:
        K: (n, dk) keys. V: (n, dv) values.
        scores: (n,) importance (from :func:`score_tokens` or gate).
        budget: number of real tokens to keep (memory key is extra).
        mem_dim: latent sketch width (default 8).
        sink_keep: number of leading sinks force-kept.
        seed: RNG seed for the sketch projectors.
        temp: softmax temperature over evicted scores.

    Returns:
        dict with K_out (m+1 or n, dk), V_out, kept_idx, evicted_idx,
        mem_vec_k (k,), mem_vec_v, k_mem (dk,), v_mem (dv,), weights.
    """
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = K.shape[0]
    if V.shape[0] != n or s.shape[0] != n:
        raise ValueError("K/V/scores first dims must agree")
    if K.ndim != 2 or V.ndim != 2:
        raise ValueError("K and V must be 2-D")
    if budget < 1:
        raise ValueError("budget must be >= 1")
    dk = K.shape[1]
    dv = V.shape[1]
    sink_keep = max(0, min(int(sink_keep), n, int(budget)))
    sinks = np.arange(sink_keep, dtype=int)

    if n <= budget:
        return {
            "K_out": K.copy(),
            "V_out": V.copy(),
            "kept_idx": np.arange(n, dtype=int),
            "evicted_idx": np.zeros((0,), dtype=int),
            "mem_vec_k": np.zeros((min(mem_dim, dk),), dtype=np.float64),
            "mem_vec_v": np.zeros((min(mem_dim, dv),), dtype=np.float64),
            "k_mem": None,
            "v_mem": None,
            "weights": np.zeros((0,), dtype=np.float64),
            "Pk": None,
            "Pv": None,
            "sinks": sinks,
        }

    n_keep_rest = int(budget) - len(sinks)
    pool = np.array([i for i in range(n) if i not in set(sinks.tolist())], dtype=int)
    order = pool[np.argsort(-s[pool], kind="stable")]
    picked = order[:n_keep_rest]
    kept_idx = np.sort(np.concatenate([sinks, picked]))
    mask = np.ones((n,), dtype=bool)
    mask[kept_idx] = False
    evicted_idx = np.where(mask)[0]

    s_ev = s[evicted_idx]
    if temp <= 0:
        w = np.zeros_like(s_ev, dtype=np.float64)
        w[np.argmax(s_ev)] = 1.0
    else:
        # normalize scale so softmax is meaningful across magnitudes
        w = _softmax(s_ev / float(temp))
    k_bar = w @ K[evicted_idx]  # (dk,)
    v_bar = w @ V[evicted_idx]  # (dv,)

    Pk = _orthonormal_basis(dk, mem_dim, seed)
    Pv = _orthonormal_basis(dv, mem_dim, seed + 101)
    mem_vec_k = (Pk.T @ k_bar).astype(np.float64)
    mem_vec_v = (Pv.T @ v_bar).astype(np.float64)
    k_raw = Pk @ mem_vec_k
    # Norm-preserve: random projection shrinks energy by ~mem_dim/d;
    # rescale so the memory key carries the barycenter's logit magnitude.
    nb = float(np.linalg.norm(k_bar))
    nr = float(np.linalg.norm(k_raw))
    if nr > 1e-12 and nb > 1e-12:
        k_mem = (k_raw * (nb / nr)).astype(np.float64)
    else:
        k_mem = k_raw.astype(np.float64)
    v_mem = np.asarray(v_bar, dtype=np.float64)  # exact tail value barycenter

    K_out = np.vstack([K[kept_idx], k_mem[None, :]])
    V_out = np.vstack([V[kept_idx], v_mem[None, :]])
    return {
        "K_out": K_out,
        "V_out": V_out,
        "kept_idx": kept_idx,
        "evicted_idx": evicted_idx,
        "mem_vec_k": mem_vec_k,
        "mem_vec_v": mem_vec_v,
        "k_mem": k_mem,
        "v_mem": v_mem,
        "weights": w,
        "Pk": Pk,
        "Pv": Pv,
        "sinks": sinks,
    }


def reconstruction_error(
    K: np.ndarray,
    V: np.ndarray,
    comp: dict,
    query: np.ndarray | None = None,
    seed: int = 0,
) -> float:
    """Relative attention-output error of a compressed cache.

    Uses one probe query (default: mean key) and compares full-cache
    attention output against compressed-cache (kept + memory key) output.
    Normalized by the RMS value-vector scale (stable when outputs average
    near zero; identical denominator for drop-vs-memory comparisons).
    Lower is better; memory key should beat pure dropping.
    """
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    K_out = np.asarray(comp["K_out"], dtype=np.float64)
    V_out = np.asarray(comp["V_out"], dtype=np.float64)
    dk = K.shape[1]
    if query is None:
        q = K.mean(axis=0)
    else:
        q = np.asarray(query, dtype=np.float64).reshape(-1)
    scale = 1.0 / np.sqrt(max(dk, 1))
    lf = _softmax(K @ q * scale)
    lc = _softmax(K_out @ q * scale)
    out_full = lf @ V
    out_c = lc @ V_out
    # Normalize by value RMS scale (stable when the output mean is near zero;
    # identical denominator for drop-vs-memory so ordering is preserved).
    v_rms = float(np.sqrt(np.mean(np.sum(V**2, axis=1)))) + 1e-9
    return float(np.linalg.norm(out_full - out_c) / v_rms)


def drop_error(K: np.ndarray, V: np.ndarray, kept_idx: np.ndarray, query=None) -> float:
    """Helper: error when evicted tail is dropped (no memory key)."""
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    comp = {"K_out": K[np.asarray(kept_idx)], "V_out": V[np.asarray(kept_idx)]}
    return reconstruction_error(K, V, comp, query=query)
