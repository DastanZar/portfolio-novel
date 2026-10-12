"""Per-block dual format + dual exponent-bias FP8.

Novelty: each 32-element K-block independently chooses {E5M2, E4M3}
by a dynamic-range test, plus an integer power-of-two bias shift
B in [11, 19] (nominal 15) that replaces a float scale. Dequant and
GEMV dot are fused in the reference kernel (block decode -> partial
dot, no full fp32 materialization).

Grids follow OCP FP8 (E5M2 bias 15, E4M3 bias 8, FN); E5M2 max 57344,
E4M3 finite max 240 (0x7E saturating; NaN code 0x7F excluded).
"""

from __future__ import annotations

import math

import numpy as np

BLOCK = 32
E5M2_MAX = 57344.0
E4M3_MAX = 240.0
BIAS_STD = 15
BIAS_MIN = 11
BIAS_MAX = 19
RANGE_RATIO_THRESH = 12.0

try:  # optional Triton path; CPU reference is canonical
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

# ---------------------------------------------------------------- grids
_grid_cache: dict = {}


def _build_e5m2():
    codes, vals = [], []
    for b in range(256):
        s = (b >> 7) & 1
        e = (b >> 2) & 0x1F
        m = b & 0x3
        sign = -1.0 if s else 1.0
        if e == 0:
            v = sign * (2.0**-14) * (m / 4.0)
        elif e == 31:
            continue  # Inf / NaN excluded from finite grid
        else:
            v = sign * (2.0 ** (e - 15)) * (1.0 + m / 4.0)
        codes.append(b)
        vals.append(v)
    return np.array(codes, dtype=np.uint8), np.array(vals, dtype=np.float64)


def _build_e4m3():
    codes, vals = [], []
    for b in range(256):
        s = (b >> 7) & 1
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 15 and m == 7:
            continue  # NaN code excluded
        sign = -1.0 if s else 1.0
        if e == 0:
            v = sign * (2.0**-7) * (m / 8.0)
        else:
            v = sign * (2.0 ** (e - 8)) * (1.0 + m / 8.0)
        # saturate the reserved-code magnitude to documented max
        if abs(v) > E4M3_MAX:
            v = math.copysign(E4M3_MAX, v)
        codes.append(b)
        vals.append(v)
    return np.array(codes, dtype=np.uint8), np.array(vals, dtype=np.float64)


def _grid(fmt: str):
    if fmt not in _grid_cache:
        _grid_cache[fmt] = _build_e5m2() if fmt == "e5m2" else _build_e4m3()
    return _grid_cache[fmt]


def _encode(xs: np.ndarray, fmt: str) -> np.ndarray:
    codes, vals = _grid(fmt)
    xs = np.asarray(xs, dtype=np.float64)
    # nearest-grid rounding via broadcast (32 x ~255, cheap)
    d = np.abs(xs[:, None] - vals[None, :])
    idx = np.argmin(d, axis=1)
    # saturate overflows to max-magnitude code (argmax: E4M3 grid tops at 224)
    fmax = E5M2_MAX if fmt == "e5m2" else E4M3_MAX
    over = np.abs(xs) > fmax
    if np.any(over):
        pos_code = codes[int(np.argmax(vals))]
        neg_code = codes[int(np.argmin(vals))]
        out = codes[idx]
        out[over] = np.where(xs[over] > 0, pos_code, neg_code)
        return out
    return codes[idx]


def _decode(codes: np.ndarray, fmt: str) -> np.ndarray:
    gc, gv = _grid(fmt)
    lut = np.zeros(256, dtype=np.float64)
    for c, v in zip(gc.tolist(), gv.tolist()):
        lut[c] = v
    # map reserved codes to saturated max (avoid NaN leaks)
    if fmt == "e5m2":
        for b in range(256):
            e, m = (b >> 2) & 0x1F, b & 0x3
            if e == 31:
                lut[b] = math.copysign(E5M2_MAX, -1.0 if (b >> 7) & 1 else 1.0)
    else:
        lut[0x7F] = E4M3_MAX
        lut[0xFF] = -E4M3_MAX
    return lut[np.asarray(codes, dtype=np.uint8)]


# ---------------------------------------------------------------- policy
def choose_format(block: np.ndarray) -> str:
    """Dynamic-range test: wide range or huge amax -> E5M2, else E4M3."""
    x = np.asarray(block, dtype=np.float64)
    amax = float(np.max(np.abs(x))) if x.size else 0.0
    if not np.isfinite(amax) or amax == 0.0:
        return "e4m3"
    if amax > E4M3_MAX:
        return "e5m2"
    rms = float(np.sqrt(np.mean(x * x)))
    ratio = amax / (rms + 1e-12)
    return "e5m2" if ratio > RANGE_RATIO_THRESH else "e4m3"


def calibrate_bias(block: np.ndarray, fmt: str) -> int:
    """Pick integer bias B in [11,19] mapping amax near 90% of fmt max."""
    x = np.asarray(block, dtype=np.float64)
    amax = float(np.max(np.abs(x))) if x.size else 0.0
    if not np.isfinite(amax) or amax == 0.0:
        return BIAS_STD
    fmax = E5M2_MAX if fmt == "e5m2" else E4M3_MAX
    bias = round(BIAS_STD - math.log2(amax / (fmax * 0.9)))
    return max(BIAS_MIN, min(BIAS_MAX, bias))


def quantize_block(block: np.ndarray, fmt: str | None = None, bias: int | None = None):
    """Quantize one BLOCK-vector. Returns (codes uint8, fmt, bias)."""
    x = np.asarray(block, dtype=np.float64)
    if fmt is None:
        fmt = choose_format(x)
    if bias is None:
        bias = calibrate_bias(x, fmt)
    scale = 2.0 ** (BIAS_STD - bias)
    xs = x / scale
    codes = _encode(xs, fmt)
    return codes.astype(np.uint8), fmt, int(bias)


def dequantize_block(codes: np.ndarray, fmt: str, bias: int) -> np.ndarray:
    """Inverse of quantize_block (fused-scale multiply)."""
    scale = 2.0 ** (BIAS_STD - int(bias))
    return (_decode(np.asarray(codes, dtype=np.uint8), fmt) * scale).astype(np.float64)


# ---------------------------------------------------------------- packed GEMV
class PackedWeights:
    def __init__(self, codes, fmts, biases, shape, k_padded):
        self.codes = codes  # (M, nblk, BLOCK) uint8
        self.fmts = fmts  # (M, nblk) bool True=E5M2
        self.biases = biases  # (M, nblk) int
        self.shape = shape
        self.k_padded = k_padded


def pack_weights(W: np.ndarray) -> PackedWeights:
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError("W must be 2-D (M, K)")
    M, K = W.shape
    nblk = math.ceil(K / BLOCK)
    Kp = nblk * BLOCK
    Wp = np.zeros((M, Kp), dtype=np.float64)
    Wp[:, :K] = W
    codes = np.zeros((M, nblk, BLOCK), dtype=np.uint8)
    fmts = np.zeros((M, nblk), dtype=bool)
    biases = np.zeros((M, nblk), dtype=np.int64)
    for i in range(M):
        for j in range(nblk):
            blk = Wp[i, j * BLOCK : (j + 1) * BLOCK]
            c, f, b = quantize_block(blk)
            codes[i, j] = c
            fmts[i, j] = f == "e5m2"
            biases[i, j] = b
    return PackedWeights(codes, fmts, biases, (M, K), Kp)


def gemv_reference(packed: PackedWeights, x: np.ndarray) -> np.ndarray:
    """Fused dequant+dot: decode one block, partial dot, accumulate."""
    x = np.asarray(x, dtype=np.float64).ravel()
    M, K = packed.shape
    if x.size != K:
        raise ValueError(f"x len {x.size} != K {K}")
    nblk = packed.codes.shape[1]
    Xp = np.zeros(packed.k_padded, dtype=np.float64)
    Xp[:K] = x
    y = np.zeros(M, dtype=np.float64)
    for i in range(M):
        acc = 0.0
        for j in range(nblk):
            fmt = "e5m2" if packed.fmts[i, j] else "e4m3"
            w = dequantize_block(packed.codes[i, j], fmt, int(packed.biases[i, j]))
            acc += float(w @ Xp[j * BLOCK : (j + 1) * BLOCK])
        y[i] = acc
    return y


def gemv_triton(packed: PackedWeights, x: np.ndarray) -> np.ndarray:
    """GPU path when Triton+CUDA exist; otherwise reference fallback."""
    if not HAS_TRITON:
        return gemv_reference(packed, x)
    try:
        import torch

        if not torch.cuda.is_available():
            return gemv_reference(packed, x)
    except Exception:
        return gemv_reference(packed, x)
    # Fused Triton kernel is out of scope for the CPU artifact;
    # dispatch here would launch it. Fall back for now.
    return gemv_reference(packed, x)


def bytes_model(M: int, K: int) -> dict:
    nblk = math.ceil(K / BLOCK)
    return {
        "fp32": 4 * M * K,
        "fp16": 2 * M * K,
        "int8_per_block": M * nblk * (BLOCK + 4),
        "fp8_uniform": M * K + 4,
        "dual_bias_fp8": M * nblk * (BLOCK + 2),  # 32B data + fmt + bias
    }
