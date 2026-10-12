"""Timing harness with confidence intervals (CPU-only).

Inspired by Triton's ``do_bench`` warmup + repeated-trial pattern, but
returns full statistics (mean/median/std + 95% CI) so downstream
roofline grades can carry error bars instead of point estimates.
"""
from __future__ import annotations

import statistics
import time
from typing import Any, Callable, Dict, List, Optional


def _ci95_mean(samples: List[float]) -> tuple[float, float, float]:
    """Mean + normal-approx 95% CI. Returns (mean, lo, hi)."""
    n = len(samples)
    mean = float(statistics.fmean(samples))
    if n < 2:
        return mean, mean, mean
    stdev = float(statistics.pstdev(samples)) if n > 1 else 0.0
    # Use sample stdev for CI width (unbiased enough for n>=5).
    try:
        stdev = float(statistics.stdev(samples))
    except statistics.StatisticsError:
        stdev = 0.0
    half = 1.96 * stdev / (n ** 0.5)
    return mean, mean - half, mean + half


def time_fn(
    fn: Callable[[], Any],
    trials: int = 30,
    warmup: int = 5,
    timer: Callable[[], float] = time.perf_counter,
) -> Dict[str, Any]:
    """Time a zero-arg callable with warmup + repeated trials.

    Args:
        fn: callable to benchmark (its return value is ignored).
        trials: number of timed trials (must be >= 1).
        warmup: number of untimed warmup calls (cache/TLB warmup).
        timer: clock function (injectable for tests).

    Returns:
        Dict with keys: mean, median, stdev, min, max,
        ci95_lo, ci95_hi, ci95_half, trials, samples.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    for _ in range(warmup):
        fn()
    samples: List[float] = []
    for _ in range(trials):
        t0 = timer()
        fn()
        t1 = timer()
        samples.append(float(t1 - t0))
    mean, lo, hi = _ci95_mean(samples)
    return {
        "mean": mean,
        "median": float(statistics.median(samples)),
        "stdev": float(statistics.stdev(samples)) if len(samples) > 1 else 0.0,
        "min": float(min(samples)),
        "max": float(max(samples)),
        "ci95_lo": float(min(lo, hi)),
        "ci95_hi": float(max(lo, hi)),
        "ci95_half": float((hi - lo) / 2.0),
        "trials": int(trials),
        "samples": samples,
    }


def run_case(
    name: str,
    fn: Callable[[], Any],
    *,
    flops: float,
    bytes_dram: float,
    bytes_l2: float = 0.0,
    trials: int = 30,
    warmup: int = 5,
    timer: Optional[Callable[[], float]] = None,
) -> Dict[str, Any]:
    """Benchmark one kernel case and convert time -> achieved rates.

    Achieved GFLOP/s = flops / mean_seconds / 1e9.
    CI on time maps to an (inverted) interval on rate:
    rate_lo uses the *slow* end (ci95_hi time), rate_hi the fast end.

    Returns a dict with timing + achieved metrics + CI band.
    """
    if flops < 0 or bytes_dram < 0 or bytes_l2 < 0:
        raise ValueError("flops/bytes must be non-negative")
    timing = time_fn(fn, trials=trials, warmup=warmup,
                     **({"timer": timer} if timer is not None else {}))
    mean = timing["mean"]
    if mean <= 0:
        raise RuntimeError("non-positive mean time; clock resolution issue")
    # Guard against zero-duration CI edge (ultra-fast fn): clamp lo > 0.
    lo_t = max(timing["ci95_lo"], 1e-12)
    hi_t = max(timing["ci95_hi"], 1e-12)
    achieved = flops / mean / 1e9
    # Inverted interval: slower time -> lower rate.
    rate_hi = flops / lo_t / 1e9
    rate_lo = flops / hi_t / 1e9
    bw_dram = bytes_dram / mean / 1e9
    bw_dram_lo = bytes_dram / hi_t / 1e9
    bw_dram_hi = bytes_dram / lo_t / 1e9
    total_bytes = bytes_dram + bytes_l2
    oi_dram = flops / bytes_dram if bytes_dram > 0 else float("inf")
    oi_total = flops / total_bytes if total_bytes > 0 else float("inf")
    return {
        "name": name,
        "flops": float(flops),
        "bytes_dram": float(bytes_dram),
        "bytes_l2": float(bytes_l2),
        "oi_dram": float(oi_dram),
        "oi_total": float(oi_total),
        "timing": timing,
        "achieved_gflops": float(achieved),
        "achieved_gflops_lo": float(min(rate_lo, rate_hi)),
        "achieved_gflops_hi": float(max(rate_lo, rate_hi)),
        "achieved_bw_dram_gbps": float(bw_dram),
        "achieved_bw_dram_lo": float(min(bw_dram_lo, bw_dram_hi)),
        "achieved_bw_dram_hi": float(max(bw_dram_lo, bw_dram_hi)),
    }
