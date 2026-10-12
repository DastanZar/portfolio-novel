"""Cache-aware roofline model.

Classic roofline (Williams et al. 2009):
    attainable = min(peak_flops, AI * peak_bandwidth)

Novelty here (the "cache roof"):
  1. L2-hit-rate model: memory time is split into DRAM-miss traffic and
     L2-hit traffic served at L2 bandwidth:
         t_mem = B_dram / BW_dram + B_l2 / BW_l2
     so kernels with high temporal reuse (e.g. decode attention over a
     resident KV tile) get a higher effective roof than the DRAM-only
     model predicts.
  2. Dequant-overlap model: for quantized KV / weight kernels, a fraction
     ``q`` of the memory time is hidden behind compute (async dequant /
     KV-dequant skip discussed in recent 5090 + vLLM bench threads):
         t_mem_eff = t_mem * (1 - q)
         roof = FLOPs / max(t_compute, t_mem_eff)

Grades a measured kernel as % of this cache-adjusted peak.
"""
from __future__ import annotations

from typing import Any, Dict


def classic_roof(
    oi_dram: float, peak_gflops: float, bw_dram_gbps: float
) -> float:
    """Classic DRAM-only roofline in GFLOP/s."""
    if peak_gflops <= 0 or bw_dram_gbps <= 0:
        raise ValueError("peaks must be positive")
    if oi_dram < 0:
        raise ValueError("OI must be non-negative")
    return float(min(peak_gflops, oi_dram * bw_dram_gbps))


def cache_adjusted_roof(
    *,
    flops: float,
    bytes_dram: float,
    bytes_l2: float = 0.0,
    peak_gflops: float,
    bw_dram_gbps: float,
    bw_l2_gbps: float,
    dequant_overlap: float = 0.0,
) -> Dict[str, Any]:
    """Cache-adjusted roof for one kernel shape.

    Args:
        flops: FLOPs per invocation.
        bytes_dram: bytes that must come from DRAM (misses).
        bytes_l2: bytes served from L2 (hits).
        peak_gflops: compute ceiling (GFLOP/s).
        bw_dram_gbps: DRAM ceiling (GB/s).
        bw_l2_gbps: L2 ceiling (GB/s).
        dequant_overlap: fraction of memory time hidden by overlapped
            dequant (0 = none, 0.5 = half hidden). Must be in [0, 1).

    Returns dict with roof_gflops, classic_gflops, times, bound label,
    effective bandwidth and implied L2 hit rate.
    """
    for label, v in (
        ("peak_gflops", peak_gflops),
        ("bw_dram_gbps", bw_dram_gbps),
        ("bw_l2_gbps", bw_l2_gbps),
    ):
        if v <= 0:
            raise ValueError(f"{label} must be positive")
    if flops < 0 or bytes_dram < 0 or bytes_l2 < 0:
        raise ValueError("flops/bytes must be non-negative")
    if not 0.0 <= dequant_overlap < 1.0:
        raise ValueError("dequant_overlap must be in [0, 1)")

    total = bytes_dram + bytes_l2
    hit_rate = (bytes_l2 / total) if total > 0 else 0.0

    # Times in nanoseconds-ish consistent units: FLOP/GFLOP/s -> 1e-9 s etc.
    # Use seconds: t = bytes / (GB/s * 1e9).
    t_compute = flops / (peak_gflops * 1e9) if flops else 0.0
    t_mem_raw = bytes_dram / (bw_dram_gbps * 1e9) + bytes_l2 / (bw_l2_gbps * 1e9)
    t_mem_eff = t_mem_raw * (1.0 - dequant_overlap)
    t_bound = max(t_compute, t_mem_eff)
    roof = flops / t_bound / 1e9 if t_bound > 0 else peak_gflops

    oi_dram = flops / bytes_dram if bytes_dram > 0 else float("inf")
    classic = classic_roof(oi_dram, peak_gflops, bw_dram_gbps) if bytes_dram > 0 else peak_gflops

    # Effective bandwidth: total bytes / raw memory time.
    bw_eff = (total / t_mem_raw / 1e9) if t_mem_raw > 0 else bw_dram_gbps

    if flops == 0 or t_bound == 0:
        bound = "none"
    elif t_compute >= t_mem_eff:
        bound = "compute"
    elif bytes_l2 > bytes_dram:
        bound = "cache"
    else:
        bound = "memory"

    return {
        "roof_gflops": float(min(roof, peak_gflops)),
        "classic_gflops": float(min(classic, peak_gflops)),
        "t_compute_s": float(t_compute),
        "t_mem_raw_s": float(t_mem_raw),
        "t_mem_eff_s": float(t_mem_eff),
        "bound": bound,
        "l2_hit_rate": float(hit_rate),
        "bw_eff_gbps": float(bw_eff),
        "oi_dram": float(oi_dram),
        "dequant_overlap": float(dequant_overlap),
    }


def analyze(case: Dict[str, Any], roof: Dict[str, Any]) -> Dict[str, Any]:
    """Grade a measured case as % of cache-adjusted peak.

    Args:
        case: output of harness.run_case (needs achieved_gflops + CI).
        roof: output of cache_adjusted_roof (needs roof_gflops).

    Returns dict with grade_pct, CI band on grade, headroom, verdict.
    """
    peak = float(roof["roof_gflops"])
    if peak <= 0:
        raise ValueError("roof must be positive")
    ach = float(case["achieved_gflops"])
    lo = float(case.get("achieved_gflops_lo", ach))
    hi = float(case.get("achieved_gflops_hi", ach))
    grade = 100.0 * ach / peak
    grade_lo = 100.0 * min(lo, hi) / peak
    grade_hi = 100.0 * max(lo, hi) / peak
    headroom = peak - ach
    if grade >= 80:
        verdict = "at-roof: well tuned"
    elif grade >= 50:
        verdict = f"{roof.get('bound', 'memory')}-bound: moderate headroom"
    elif grade >= 20:
        verdict = f"{roof.get('bound', 'memory')}-bound: significant headroom"
    else:
        verdict = "overhead-bound: likely launch/scheduling overhead, not BW"
    return {
        "name": case.get("name", "?"),
        "grade_pct": float(grade),
        "grade_lo": float(min(grade_lo, grade_hi)),
        "grade_hi": float(max(grade_lo, grade_hi)),
        "headroom_gflops": float(headroom),
        "bound": roof.get("bound", "?"),
        "verdict": verdict,
        "roof_gflops": peak,
        "achieved_gflops": ach,
    }


def peak_measure(
    size_mb: float = 8.0, matmul_n: int = 256, repeats: int = 10
) -> Dict[str, float]:
    """Best-effort CPU peak measurement (no GPU needed).

    - DRAM bandwidth: time ``np.copy`` of a ``size_mb`` float32 array.
    - Compute peak: time ``matmul_n x matmul_n`` float32 GEMM
      (2*N^3 FLOPs per multiply).
    - L2 bandwidth: modelled as 4x measured DRAM bandwidth, a
      conservative ratio for modern client CPUs (documented in README
      as an assumption, not a measurement).

    Returns dict(peak_gflops, bw_dram_gbps, bw_l2_gbps).
    """
    import time

    import numpy as np

    n_elem = max(int(size_mb * 1e6 / 4), 1_000)
    a = np.random.rand(n_elem).astype(np.float32)
    # Warmup then timed copies.
    for _ in range(3):
        _ = np.copy(a)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = np.copy(a)
        ts.append(time.perf_counter() - t0)
    best = max(min(ts), 1e-9)
    # copy moves 2x array bytes (read + write).
    bw_dram = (2 * n_elem * 4) / best / 1e9

    m = np.random.rand(matmul_n, matmul_n).astype(np.float32)
    for _ in range(2):
        _ = m @ m
    flops = 2 * matmul_n ** 3
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = m @ m
        ts.append(time.perf_counter() - t0)
    best = max(min(ts), 1e-9)
    peak = flops / best / 1e9
    return {
        "peak_gflops": float(peak),
        "bw_dram_gbps": float(bw_dram),
        "bw_l2_gbps": float(bw_dram * 4.0),
    }
