"""Core for shell-vq.

Novelty: norm-shelled grouped VQ. Instead of one flat codebook/scale for all
vectors (as in plain lattice brute force), weight vectors are grouped by
radius shells - inner / mid / outer via norm quantiles - a separate scale is
learned per shell (closed-form LS), and outer-shell vectors get an extra 1-bit
residual sign refinement (per-vector magnitude + sign, gated so it never
hurts). The delta vs flat lattice VQ is the shell grouping + per-shell scales.

Prior art (idea-level only, no code reuse):
- AQLM / QTIP (lattice / trellis VQ for LLM weights).
- Conway-Sloane lattices; Leech-lattice VQ (arXiv 2603.11021).
- Sub-1-bit ideas such as BTC-LLM (binary + scale refinement).
See README for full attribution.
"""

from __future__ import annotations

import numpy as np

try:
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


def _to_numpy(a) -> np.ndarray:
    if _HAS_TORCH and isinstance(a, torch.Tensor):
        return a.detach().to("cpu").numpy().astype(np.float64)
    return np.asarray(a, dtype=np.float64)


def build_shell_codebook(dim: int, n_points: int = 64, seed: int = 0) -> np.ndarray:
    """Build a deterministic unit-sphere codebook (n_points x dim).

    Points come from a fixed Gaussian RNG stream and are L2-normalized, so
    the same seed returns bit-identical entries. Radius info lives in the
    per-shell scales learned by :func:`quantize_shelled`.
    """
    if dim <= 0 or n_points <= 0:
        raise ValueError("dim and n_points must be positive")
    rng = np.random.RandomState(seed)
    pts = rng.randn(n_points, dim).astype(np.float64)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return (pts / norms).astype(np.float64)


def _assign(vectors: np.ndarray, codebook: np.ndarray, scales_per_vec: np.ndarray):
    normed = vectors / scales_per_vec[:, None]
    # squared Euclidean to each unit codebook entry
    d2 = (np.sum(normed ** 2, axis=1, keepdims=True)
          + np.sum(codebook ** 2, axis=1)[None, :]
          - 2.0 * normed @ codebook.T)
    return np.argmin(d2, axis=1).astype(np.int64)


def _ls_scales_per_shell(vectors, codebook, idx, shells):
    scales = {}
    for s in (0, 1, 2):
        m = shells == s
        if int(np.sum(m)) == 0:
            scales[s] = 0.0
            continue
        c = codebook[idx[m]]  # unit-norm entries
        num = float(np.sum(vectors[m] * c))
        den = float(np.sum(c * c))  # == count, since |c| == 1
        scales[s] = float(num / den) if den > 1e-18 else 0.0
        if not np.isfinite(scales[s]) or scales[s] <= 0:
            n = np.linalg.norm(vectors[m], axis=1)
            scales[s] = float(np.mean(n))
    return scales


def quantize_shelled(vectors, codebook, q_low: float = 1.0 / 3.0,
                     q_high: float = 2.0 / 3.0) -> dict:
    """Quantize (N, dim) vectors with norm shells + per-shell LS scales.

    Shells come from norm quantiles: inner (norm <= t1), mid (<= t2), outer
    (> t2). Each shell gets its own scale: seeded by mean norm, then refined
    in closed form ``s = sum(v.c)/sum(c.c)`` given the assignments, with one
    re-assign + re-scale pass. Returns indices, shells, thresholds, scales,
    reconstruction and MSE stats.
    """
    was_torch = _HAS_TORCH and isinstance(vectors, torch.Tensor)
    V = _to_numpy(vectors)
    C = _to_numpy(codebook)
    if V.ndim != 2 or C.ndim != 2 or V.shape[1] != C.shape[1]:
        raise ValueError("vectors (N,d) and codebook (P,d) must share dim d")
    n = V.shape[0]
    if n == 0:
        raise ValueError("empty vectors")
    norms = np.linalg.norm(V, axis=1)
    t1 = float(np.quantile(norms, q_low))
    t2 = float(np.quantile(norms, q_high))
    shells = (norms > t1).astype(np.int64) + (norms > t2).astype(np.int64)
    shells = np.clip(shells, 0, 2)
    # seed scales by mean norm per shell
    seed = {}
    global_mean = float(np.mean(norms)) if n else 1.0
    for s in (0, 1, 2):
        m = shells == s
        seed[s] = float(np.mean(norms[m])) if int(np.sum(m)) else global_mean
        if not np.isfinite(seed[s]) or seed[s] <= 1e-12:
            seed[s] = global_mean if global_mean > 1e-12 else 1.0
    spv = np.array([seed[int(s)] for s in shells])
    idx = _assign(V, C, spv)
    scales = _ls_scales_per_shell(V, C, idx, shells)
    spv2 = np.array([scales[int(s)] for s in shells])
    spv2[spv2 <= 1e-12] = spv[spv2 <= 1e-12]
    idx2 = _assign(V, C, spv2)
    scales2 = _ls_scales_per_shell(V, C, idx2, shells)
    spv3 = np.array([scales2[int(s)] for s in shells])
    recon = spv3[:, None] * C[idx2]
    mse = float(np.mean((V - recon) ** 2))
    per_shell_mse = {}
    counts = {}
    for s in (0, 1, 2):
        m = shells == s
        counts[s] = int(np.sum(m))
        per_shell_mse[s] = float(np.mean((V[m] - recon[m]) ** 2)) if counts[s] else 0.0
    out = {"indices": idx2, "shells": shells,
           "thresholds": (t1, t2), "scales": scales2,
           "scales_per_vector": spv3, "reconstruction": recon,
           "mse": mse, "per_shell_mse": per_shell_mse,
           "counts": counts, "shape": tuple(V.shape)}
    if was_torch:
        out["reconstruction"] = torch.from_numpy(np.asarray(recon)).to(torch.float32)
    return out


def residual_refine(original, reconstruction, outer_mask) -> dict:
    """1-bit residual sign refinement for outer-shell vectors.

    For each outer vector: magnitude ``m = mean(|residual|)`` plus per-coordinate
    ``sign(residual)`` approximates the residual with a single extra bit per
    coordinate (plus one float magnitude). The update is gated per vector: it
    is applied only when it strictly reduces that vector's squared error, so
    outer MSE never gets worse.
    """
    O = _to_numpy(original)
    R = _to_numpy(reconstruction)
    mask = np.asarray(outer_mask, dtype=bool)
    if O.shape != R.shape or mask.shape[0] != O.shape[0]:
        raise ValueError("shape mismatch")
    res = O - R
    refined = R.copy()
    mags = np.zeros(O.shape[0])
    applied = np.zeros(O.shape[0], dtype=bool)
    for i in np.flatnonzero(mask):
        r = res[i]
        m = float(np.mean(np.abs(r)))
        mags[i] = m
        if m <= 1e-18:
            continue
        s = np.sign(r)
        s[s == 0] = 1.0
        cand = R[i] + m * s
        if float(np.sum((O[i] - cand) ** 2)) < float(np.sum(r ** 2)):
            refined[i] = cand
            applied[i] = True
    def _mse(a, b, m):
        return float(np.mean((a[m] - b[m]) ** 2)) if int(np.sum(m)) else 0.0
    return {"refined": refined, "magnitudes": mags, "applied": applied,
            "mse_before": float(np.mean((O - R) ** 2)),
            "mse_after": float(np.mean((O - refined) ** 2)),
            "outer_mse_before": _mse(O, R, mask),
            "outer_mse_after": _mse(O, refined, mask)}


def rate_distortion_floor(var: float, dim: int, bits_per_vec: float) -> float:
    """Gaussian rate-distortion lower bound ``D = var * 2^(-2R/d)``.

    ``var`` is the per-coordinate variance, ``d`` the vector dim and ``R``
    the bits per vector. Useful as a theoretical floor to compare shelled-VQ
    MSEs against. ``R = 0`` returns ``var``.
    """
    if dim <= 0:
        raise ValueError("dim must be positive")
    if var <= 0:
        return 0.0
    if bits_per_vec <= 0:
        return float(var)
    return float(var * (2.0 ** (-2.0 * float(bits_per_vec) / float(dim))))
