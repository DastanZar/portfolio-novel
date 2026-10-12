import math

import numpy as np
import pytest

from dual_bias_fp8.core import (
    BLOCK,
    calibrate_bias,
    choose_format,
    dequantize_block,
    pack_weights,
    gemv_reference,
    quantize_block,
    bytes_model,
)


def test_block_choice_large_goes_e5m2():
    rng = np.random.default_rng(0)
    # one huge outlier exceeds E4M3 range -> must pick E5M2
    blk = rng.normal(0, 1, BLOCK)
    blk[0] = 1000.0
    assert choose_format(blk) == "e5m2"


def test_block_choice_small_well_conditioned_goes_e4m3():
    rng = np.random.default_rng(1)
    blk = rng.normal(0, 1, BLOCK)  # amax small, ratio low -> E4M3 precision
    assert choose_format(blk) == "e4m3"


def test_calibrate_bias_range_and_monotonic():
    big = np.full(BLOCK, 20000.0)
    small = np.full(BLOCK, 1e-4)
    mid = np.full(BLOCK, 1.0)
    bb = calibrate_bias(big, "e5m2")
    bs = calibrate_bias(small, "e5m2")
    bm = calibrate_bias(mid, "e5m2")
    assert 11 <= bb <= 19 and 11 <= bs <= 19 and 11 <= bm <= 19
    # larger magnitudes need smaller bias (larger descaling)
    assert bb <= bm <= bs
    assert calibrate_bias(np.zeros(BLOCK), "e5m2") == 15


def test_roundtrip_and_gemv_fused_matches_manual():
    rng = np.random.default_rng(2)
    W = rng.normal(0, 2, (4, 64))
    x = rng.normal(0, 1, 64)
    packed = pack_weights(W)
    y = gemv_reference(packed, x)
    # manual: decode each block then dot
    M, K = W.shape
    ref = np.zeros(M)
    nblk = packed.codes.shape[1]
    Xp = np.zeros(packed.k_padded)
    Xp[:K] = x
    for i in range(M):
        for j in range(nblk):
            fmt = "e5m2" if packed.fmts[i, j] else "e4m3"
            w = dequantize_block(packed.codes[i, j], fmt, int(packed.biases[i, j]))
            ref[i] += w @ Xp[j * BLOCK : (j + 1) * BLOCK]
    assert np.allclose(y, ref, rtol=1e-12), "fused GEMV must match manual decode+dot"
    # accuracy sanity: dual-bias GEMV within 5% of fp32 on tame data
    y32 = W @ x
    rel = np.linalg.norm(y - y32) / (np.linalg.norm(y32) + 1e-12)
    assert rel < 0.05


def test_torch_e5m2_bit_exact_where_available():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e5m2"):
        pytest.skip("torch without float8_e5m2")
    from dual_bias_fp8.core import _decode, _encode

    vals = np.array([0.5, 1.0, -2.0, 100.0, 0.04, -0.0], dtype=np.float64)
    codes = _encode(vals, "e5m2")
    ours = _decode(codes, "e5m2")
    t = torch.tensor(vals, dtype=torch.float32).to(torch.float8_e5m2).to(torch.float32).numpy().astype(np.float64)
    # exact match where both finite; torch saturates inf -> allow our saturation
    for o, e in zip(ours.tolist(), t.tolist()):
        if math.isnan(e) or math.isinf(e):
            assert abs(o) <= 57344.0
        else:
            assert o == pytest.approx(e, abs=1e-9)


def test_bytes_model_savings():
    b = bytes_model(64, 128)
    assert b["dual_bias_fp8"] < b["fp32"]
    assert b["dual_bias_fp8"] == 64 * 4 * (32 + 2)
    assert b["fp32"] == 4 * 64 * 128
