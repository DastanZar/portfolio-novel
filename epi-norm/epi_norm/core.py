"""epi-norm core: RMSNorm + residual + SwiGLU epilogue fusion (numpy reference).

Novel concept: fuse RMSNorm + residual-add + SwiGLU gate into ONE epilogue
block (inspired by GEMM-epilogue fusion, e.g. briefed arXiv 2605.19269) and
explore the recompute-vs-store tradeoff: recompute `rrms` (and the normalized
activations) in backward instead of stashing them, to save HBM at the cost of
extra FLOPs — the same tradeoff Liger-Kernel / Tri-RMSNorm style fusions make.

References (ideas only, original code):
  - RMSNorm: Zhang & Sennrich 2019.
  - Liger-Kernel / fused RMSNorm + SwiGLU kernels: fused Triton blocks.
  - GEMM-epilogue fusion: attach norm/activation to the producer GEMM.

All functions are numpy-only so the project is CPU-only and dependency-light.
`kernels.py` optionally attempts a Triton forward when available.
"""

from __future__ import annotations

import math

import numpy as np


def _silu(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return (z / (1.0 + np.exp(-z))).astype(np.float64)


def rmsnorm_ref(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6):
    """Reference RMSNorm.

    Args:
        x: (..., D) input (typically x + residual already added by caller,
            or raw GEMM output).
        weight: (D,) scale.
        eps: stability term.

    Returns:
        (y, rrms): y has same shape as x,
        rrms has shape (..., 1) = 1/sqrt(mean(x^2)+eps), the value a
        recompute-style backward would recompute instead of storing.
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    if x.shape[-1] != w.shape[0]:
        raise ValueError(f"last dim {x.shape[-1]} != weight dim {w.shape[0]}")
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    rrms = (1.0 / np.sqrt(ms + float(eps))).astype(np.float64)
    y = (x * rrms * w).astype(np.float64)
    return y, rrms


def fused_epilogue_ref(
    x: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    W_gate: np.ndarray,
    W_up: np.ndarray,
    eps: float = 1e-6,
):
    """Fused RMSNorm + residual + SwiGLU-gate epilogue (reference).

    Fusion equation (our documented choice):
        fused  = x + residual            # residual stream merge
        n, rrms = RMSNorm(fused; weight) # normalized, scaled
        g = n @ W_gate ; u = n @ W_up
        out = SiLU(g) * u                # SwiGLU gate (H-dim)

    Args:
        x: (..., D) GEMM output. residual: same shape. weight: (D,).
        W_gate, W_up: (D, H) projections.
        eps: RMSNorm epsilon.

    Returns:
        dict with out (..., H), normed (..., D), rrms (..., 1),
        g, u (..., H), fused (..., D).
        `normed`/`g`/`u` are what a store-style backward would stash;
        `rrms`+`fused` are the minimal recompute set.
    """
    x = np.asarray(x, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64)
    if x.shape != r.shape:
        raise ValueError(f"x {x.shape} vs residual {r.shape}")
    Wg = np.asarray(W_gate, dtype=np.float64)
    Wu = np.asarray(W_up, dtype=np.float64)
    if Wg.shape != Wu.shape or Wg.ndim != 2 or Wg.shape[0] != x.shape[-1]:
        raise ValueError(f"W shapes {Wg.shape}/{Wu.shape} vs D {x.shape[-1]}")
    fused = x + r
    normed, rrms = rmsnorm_ref(fused, weight, eps)
    g = normed @ Wg
    u = normed @ Wu
    out = (_silu(g) * u).astype(np.float64)
    return {"out": out, "normed": normed, "rrms": rrms, "g": g, "u": u, "fused": fused}


def fused_epilogue_backward_recompute(
    grad_out: np.ndarray,
    cache: dict,
    weight: np.ndarray,
    W_gate: np.ndarray,
    W_up: np.ndarray,
    eps: float = 1e-6,
):
    """Reference backward that RECOMPUTES normed activations from rrms+fused.

    Instead of loading stashed `normed/g/u` (HBM-heavy), reload only the small
    `rrms` (+ fused input) and recompute. Returns grads for x (= grad fused),
    weight, W_gate, W_up. Mathematically identical to the store variant up to
    fp rounding; traffic accountant quantifies the saving.
    """
    go = np.asarray(grad_out, dtype=np.float64)
    fused = np.asarray(cache["fused"], dtype=np.float64)
    rrms = np.asarray(cache["rrms"], dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    Wg = np.asarray(W_gate, dtype=np.float64)
    Wu = np.asarray(W_up, dtype=np.float64)
    # recompute normed, g, u from stored fused + rrms
    normed = (fused * rrms * w).astype(np.float64)
    g = normed @ Wg
    u = normed @ Wu
    sig = 1.0 / (1.0 + np.exp(-g))
    silu_g = g * sig
    d_silu = sig * (1.0 + g * (1.0 - sig))
    d_g = go * u * d_silu
    d_u = go * silu_g
    d_normed = d_g @ Wg.T + d_u @ Wu.T
    # RMSNorm backward (weight + input)
    x_f = fused
    inv = rrms  # 1/sqrt(mu2+eps)
    x_hat = x_f * inv
    grad_w = (d_normed * x_f * inv).sum(axis=tuple(range(d_normed.ndim - 1)))
    # Correct RMSNorm dx formula:
    # dx = (w*rrms) * (d_normed - x_hat * mean(d_normed*x_hat)/D... careful)
    m = np.mean(d_normed * w * x_hat, axis=-1, keepdims=True)
    grad_fused = ((d_normed * w - x_hat * m) * inv).astype(np.float64)
    dWg = normed.reshape(-1, normed.shape[-1]).T @ d_g.reshape(-1, d_g.shape[-1])
    dWu = normed.reshape(-1, normed.shape[-1]).T @ d_u.reshape(-1, d_u.shape[-1])
    return {"grad_x": grad_fused, "grad_w": grad_w, "grad_Wg": dWg, "grad_Wu": dWu}


def fused_epilogue_backward_store(
    grad_out: np.ndarray,
    cache: dict,
    weight: np.ndarray,
    W_gate: np.ndarray,
    W_up: np.ndarray,
    eps: float = 1e-6,
):
    """Reference backward that USES stashed normed/g/u (no recompute).

    Same math as :func:`fused_epilogue_backward_recompute` but loads the
    stashed activations instead of recomputing them. Used to verify parity.
    """
    go = np.asarray(grad_out, dtype=np.float64)
    fused = np.asarray(cache["fused"], dtype=np.float64)
    normed = np.asarray(cache["normed"], dtype=np.float64)
    g = np.asarray(cache["g"], dtype=np.float64)
    u = np.asarray(cache["u"], dtype=np.float64)
    rrms = np.asarray(cache["rrms"], dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    Wg = np.asarray(W_gate, dtype=np.float64)
    Wu = np.asarray(W_up, dtype=np.float64)
    sig = 1.0 / (1.0 + np.exp(-g))
    silu_g = g * sig
    d_silu = sig * (1.0 + g * (1.0 - sig))
    d_g = go * u * d_silu
    d_u = go * silu_g
    d_normed = d_g @ Wg.T + d_u @ Wu.T
    x_hat = fused * rrms
    grad_w = (d_normed * fused * rrms).sum(axis=tuple(range(d_normed.ndim - 1)))
    m = np.mean(d_normed * w * x_hat, axis=-1, keepdims=True)
    grad_fused = ((d_normed * w - x_hat * m) * rrms).astype(np.float64)
    dWg = normed.reshape(-1, normed.shape[-1]).T @ d_g.reshape(-1, d_g.shape[-1])
    dWu = normed.reshape(-1, normed.shape[-1]).T @ d_u.reshape(-1, d_u.shape[-1])
    return {"grad_x": grad_fused, "grad_w": grad_w, "grad_Wg": dWg, "grad_Wu": dWu}


def hbm_accountant(
    B: int, T: int, D: int, H: int, dtype_bytes: int = 2
) -> dict:
    """Traffic accountant comparing 3 designs (elements -> bytes model).

    Designs:
      A baseline: separate add / RMSNorm / SwiGLU kernels; forward does 3
        passes; backward stashes fused(D), normed(D), g(H), u(H) per token.
      B fused-store: single epilogue kernel (1 pass); still stashes
        normed(D)+g(H)+u(H).
      C fused-recompute (our recommended): single kernel; stashes only
        fused(D) + rrms(1) and recomputes normed/g/u in backward.

    Counts are analytic traffic estimates (reads+writes), not measurements.
    Keys: per-design {fwd_bytes, stored_bytes, bwd_bytes, total_bytes,
    passes, flops_recompute}.
    """
    N = int(B) * int(T)
    db = int(dtype_bytes)

    def b(elems: int) -> int:
        return int(elems) * db

    # Forward traffic (reads + writes, per token element counts)
    # A: read x,res (2D), write fused (D); read fused+ w, write normed (D);
    #    read normed (D)x2 for two GEMMs + write g,u (2H). Approx:
    fwd_A = b(N * (2 * D + D + D + D + 2 * D + 2 * H))
    fwd_B = b(N * (2 * D + 2 * H))  # one pass: read x,res, write g,u-ish out
    fwd_C = fwd_B
    stored_A = b(N * (D + D + H + H))
    stored_B = b(N * (D + H + H))
    stored_C = b(N * (D + 1))
    # Backward traffic: A/B reload stashed; C reloads small + recomputes.
    bwd_A = b(N * (D + D + H + H + D + H * D * 0)) + b(N * (2 * D + 2 * H))
    bwd_B = b(N * (D + H + H)) + b(N * (2 * D + 2 * H))
    # C: reload fused+rrms, recompute (extra flops), write same grads
    bwd_C = b(N * (D + 1)) + b(N * (2 * D + 2 * H))
    flops_recompute = N * (D + D * H)  # rough recompute flops for C
    return {
        "config": {"B": B, "T": T, "D": D, "H": H, "dtype_bytes": db, "N": N},
        "baseline": {
            "fwd_bytes": fwd_A,
            "stored_bytes": stored_A,
            "bwd_bytes": bwd_A,
            "total_bytes": fwd_A + stored_A + bwd_A,
            "passes": 3,
            "flops_recompute": 0,
        },
        "fused-store": {
            "fwd_bytes": fwd_B,
            "stored_bytes": stored_B,
            "bwd_bytes": bwd_B,
            "total_bytes": fwd_B + stored_B + bwd_B,
            "passes": 1,
            "flops_recompute": 0,
        },
        "fused-recompute": {
            "fwd_bytes": fwd_C,
            "stored_bytes": stored_C,
            "bwd_bytes": bwd_C,
            "total_bytes": fwd_C + stored_C + bwd_C,
            "passes": 1,
            "flops_recompute": int(flops_recompute),
        },
    }


def pick_block(D: int, max_block: int = 1024) -> int:
    """Pick largest power-of-two block <= min(D, max_block). Triton helper."""
    cap = max(1, min(int(D), int(max_block)))
    p = 1
    while p * 2 <= cap:
        p *= 2
    return int(p)
