"""Core for kurt-switch.

Novelty: per-row adaptive switch among three 4-bit codebooks driven by
kurtosis / outlier-fraction / skew statistics, plus a closed-form
least-squares scale per row. Static 4-bit grids (uniform) waste levels on
rows that are sparse/heavy-tailed or skewed; the selector is the contribution.

Prior art this builds on (idea-level only, no code reuse):
- GPTQ / AWQ (per-row scaling for 4-bit LLM weights).
- Grouped lattice vector quantization (arXiv 2510.20984).
- Practical 4-bit LLM benchmark discussion on Reddit r/LocalLLaMA.
See README for full attribution.
"""

from __future__ import annotations

import numpy as np

try:  # torch is optional; numpy path is the default and is fully tested.
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

CODEBOOK_NAMES = ("uniform16", "odd16", "shifted")


def get_codebook(name: str) -> np.ndarray:
    """Return the 16-entry base codebook for ``name`` (float64, ~[-1, 1]).

    - ``uniform16``: 16 uniform levels in [-1, 1] (static-grid baseline).
    - ``odd16``: 16 odd-symmetric square-law companded levels,
      ``sign(u)*|u|^2`` with ``u`` uniform in [-1, 1]. Zero-excluding with
      very fine resolution near zero (smallest |level| ~0.004). Suits sparse
      / heavy-tailed rows where most mass clusters near zero plus a few big
      outliers: fine center + full-range tails beat a uniform grid.
    - ``shifted``: 16 uniform levels in [-0.75, 1.25], i.e. biased toward
      positive values for right-skewed rows (mirrored use covers left skew
      via the LS scale sign handling — scale stays positive, grid is asymmetric).
    """
    if name == "uniform16":
        return np.linspace(-1.0, 1.0, 16).astype(np.float64)
    if name == "odd16":
        u = np.linspace(-1.0, 1.0, 16).astype(np.float64)
        # Odd-symmetric square-law companding grid, zero-excluding:
        # fine resolution near zero (smallest |level| ~0.004) for
        # sparse/heavy-tailed rows, full +-1 range for outliers.
        return (np.sign(u) * (np.abs(u) ** 2)).astype(np.float64)
    if name == "shifted":
        return np.linspace(-0.75, 1.25, 16).astype(np.float64)
    raise ValueError(f"unknown codebook {name!r}; choose from {CODEBOOK_NAMES}")


def _to_numpy(row) -> np.ndarray:
    if _HAS_TORCH and isinstance(row, torch.Tensor):
        return row.detach().to("cpu").numpy().astype(np.float64)
    return np.asarray(row, dtype=np.float64)


def compute_row_stats(row) -> dict:
    """Compute per-row statistics used by the switch.

    Returns dict with mean, std, var, kurtosis (excess), skew,
    outlier_fraction (|z| > 3) and zero_fraction (|x| < 1e-6).
    Accepts numpy arrays or CPU torch tensors.
    """
    x = _to_numpy(row).ravel()
    n = x.size
    mean = float(np.mean(x)) if n else 0.0
    var = float(np.mean((x - mean) ** 2)) if n else 0.0
    std = float(np.sqrt(var))
    if std < 1e-12:
        return {
            "mean": mean, "std": 0.0, "var": 0.0,
            "kurtosis": 0.0, "skew": 0.0,
            "outlier_fraction": 0.0, "zero_fraction": 1.0 if n else 0.0,
            "n": int(n),
        }
    z = (x - mean) / std
    kurtosis = float(np.mean(z ** 4) - 3.0)
    skew = float(np.mean(z ** 3))
    outlier_fraction = float(np.mean(np.abs(z) > 3.0))
    zero_fraction = float(np.mean(np.abs(x) < 1e-6))
    return {
        "mean": mean, "std": std, "var": var,
        "kurtosis": kurtosis, "skew": skew,
        "outlier_fraction": outlier_fraction, "zero_fraction": zero_fraction,
        "n": int(n),
    }


def select_codebook(stats_or_row) -> str:
    """Choose among 'uniform16' / 'odd16' / 'shifted'.

    Accepts either a stats dict (from :func:`compute_row_stats`) or a raw
    row vector (numpy / torch). Rule — the paper contribution in miniature:
    heavy-tailed or sparse rows go to the zero-excluding ``odd16`` grid,
    skewed rows go to the biased ``shifted`` grid, everything else stays on
    the ``uniform16`` baseline.
    """
    if isinstance(stats_or_row, dict):
        s = stats_or_row
    else:
        s = compute_row_stats(stats_or_row)
    if s.get("zero_fraction", 0.0) > 0.30 or s.get("kurtosis", 0.0) > 5.0 \
            or s.get("outlier_fraction", 0.0) > 0.02:
        return "odd16"
    if abs(s.get("skew", 0.0)) > 1.0:
        return "shifted"
    return "uniform16"


def _ls_scale(row: np.ndarray, levels: np.ndarray) -> float:
    denom = float(np.dot(levels, levels))
    if denom < 1e-18:
        return 0.0
    return float(np.dot(row, levels) / denom)


def quantize_row(row, codebook: str | None = None) -> dict:
    """Quantize one row to 4 bit with closed-form LS scale.

    Steps: pick codebook (or use ``codebook`` override), seed scale from
    range, nearest-neighbour assign, closed-form LS re-scale, re-assign,
    final LS re-scale so the returned scale is exactly optimal for the
    returned indices: ``s = (x.c) / (c.c)``.
    """
    was_torch = _HAS_TORCH and isinstance(row, torch.Tensor)
    x = _to_numpy(row).astype(np.float64)
    orig_shape = x.shape
    flat = x.ravel()
    if flat.size == 0:
        raise ValueError("empty row")
    stats = compute_row_stats(flat)
    name = codebook or select_codebook(stats)
    base = get_codebook(name)
    if name == "shifted" and stats.get("skew", 0.0) < 0 and codebook is None:
        # Mirror the biased grid for left-skewed rows so the dense side
        # tracks the long tail (range becomes [-1.25, 0.75]).
        base = (-base[::-1]).astype(np.float64)
    if np.max(np.abs(flat)) < 1e-12:
        recon = np.zeros_like(flat)
        out = {
            "indices": np.zeros_like(flat, dtype=np.int64),
            "scale": 0.0, "codebook": name,
            "reconstruction": recon.reshape(orig_shape),
            "mse": 0.0, "stats": stats,
        }
        if was_torch:
            out["reconstruction"] = torch.from_numpy(
                np.asarray(out["reconstruction"])).to(torch.float32)
        return out
    cb_max = float(np.max(np.abs(base)))
    s_init = float(np.max(np.abs(flat)) / (cb_max if cb_max > 1e-12 else 1.0))
    if not np.isfinite(s_init) or s_init <= 0:
        s_init = float(np.max(np.abs(flat)))
    d1 = np.abs(flat[:, None] - s_init * base[None, :])
    idx = np.argmin(d1, axis=1)
    s_ls = _ls_scale(flat, base[idx])
    if not np.isfinite(s_ls) or s_ls <= 0:
        s_ls = s_init
    d2 = np.abs(flat[:, None] - s_ls * base[None, :])
    idx2 = np.argmin(d2, axis=1)
    s_final = _ls_scale(flat, base[idx2])
    if not np.isfinite(s_final) or s_final <= 0:
        s_final = s_ls
        idx2 = idx
    recon_flat = s_final * base[idx2]
    mse = float(np.mean((flat - recon_flat) ** 2))
    recon = recon_flat.reshape(orig_shape)
    if was_torch:
        recon = torch.from_numpy(np.asarray(recon)).to(torch.float32)
    return {
        "indices": idx2.astype(np.int64).reshape(orig_shape),
        "scale": float(s_final), "codebook": name,
        "levels": np.asarray(base, dtype=np.float64),
        "reconstruction": recon, "mse": mse, "stats": stats,
    }


def quantize_block(block) -> dict:
    """Quantize each row of a 2-D weight block independently."""
    was_torch = _HAS_TORCH and isinstance(block, torch.Tensor)
    arr = _to_numpy(block).astype(np.float64)
    if arr.ndim != 2:
        raise ValueError("block must be 2-D")
    rows = []
    for i in range(arr.shape[0]):
        rows.append(quantize_row(arr[i]))
    indices = np.stack([r["indices"] for r in rows]).astype(np.int64)
    scales = np.array([r["scale"] for r in rows], dtype=np.float64)
    codebooks = [r["codebook"] for r in rows]
    recon = np.stack([np.asarray(r["reconstruction"]) for r in rows])
    mse = float(np.mean((arr - recon) ** 2))
    out = {"indices": indices, "scales": scales, "codebooks": codebooks,
           "reconstruction": recon, "mse": mse,
           "bits_per_weight": 4.0, "shape": tuple(arr.shape)}
    if was_torch:
        out["reconstruction"] = torch.from_numpy(np.asarray(recon)).to(torch.float32)
    return out
