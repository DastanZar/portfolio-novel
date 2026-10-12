"""5 tests for the cache-roof harness + cache-adjusted roofline."""
from __future__ import annotations

from cache_roof.harness import run_case, time_fn
from cache_roof.roofline import analyze, cache_adjusted_roof, classic_roof


def test_time_fn_stats_shape():
    out = time_fn(lambda: sum(range(100)), trials=10, warmup=2)
    for key in ("mean", "median", "stdev", "min", "max", "ci95_lo", "ci95_hi", "samples"):
        assert key in out
    assert out["mean"] > 0
    assert out["ci95_lo"] <= out["mean"] <= out["ci95_hi"]
    assert len(out["samples"]) == 10
    assert out["min"] <= out["median"] <= out["max"]


def test_time_fn_ci_valid_and_ordered():
    out = time_fn(lambda: None, trials=15, warmup=1)
    assert out["stdev"] >= 0
    assert out["ci95_lo"] <= out["ci95_hi"]
    assert out["ci95_half"] >= 0
    # CI must bracket the mean by construction.
    assert out["ci95_lo"] <= out["mean"] <= out["ci95_hi"]


def test_run_case_achieved_math():
    # Deterministic fake clock: each trial takes exactly 0.01 s.
    ticks = {"t": 0.0}

    def timer():
        ticks["t"] += 0.01
        return ticks["t"]

    out = run_case("fake", lambda: None, flops=1e9, bytes_dram=1e9,
                   trials=5, warmup=0, timer=timer)
    # 1e9 flop / 0.01 s per trial (timer ticks 0.01 per call, 2 calls per trial).
    import math

    assert math.isclose(out["achieved_gflops"], 100.0, rel_tol=1e-9)
    assert out["achieved_gflops_lo"] <= out["achieved_gflops"] <= out["achieved_gflops_hi"]
    assert math.isclose(out["achieved_bw_dram_gbps"], 100.0, rel_tol=1e-9)
    assert out["oi_dram"] == 1.0


def test_cache_adjusted_roof_bounds():
    # Total traffic held fixed (10 GB); reuse shifts bytes DRAM -> L2.
    base = dict(flops=10e9, peak_gflops=100.0, bw_dram_gbps=20.0, bw_l2_gbps=200.0)
    no_reuse = cache_adjusted_roof(bytes_dram=10e9, bytes_l2=0.0, **base)
    high_reuse = cache_adjusted_roof(bytes_dram=1e9, bytes_l2=9e9, **base)  # 90% L2 hits
    assert no_reuse["roof_gflops"] <= 100.0
    assert high_reuse["roof_gflops"] >= no_reuse["roof_gflops"]
    # Classic DRAM-only roof at OI=1 is 20 GFLOP/s; cache reuse must beat it.
    assert classic_roof(1.0, 100.0, 20.0) == 20.0
    assert high_reuse["roof_gflops"] > 20.0
    assert high_reuse["l2_hit_rate"] == 0.9


def test_dequant_overlap_monotonic_and_analyze():
    kw = dict(flops=2e9, bytes_dram=1e9, bytes_l2=0.0, peak_gflops=200.0,
              bw_dram_gbps=20.0, bw_l2_gbps=200.0)
    r0 = cache_adjusted_roof(dequant_overlap=0.0, **kw)
    r5 = cache_adjusted_roof(dequant_overlap=0.5, **kw)
    assert r5["roof_gflops"] >= r0["roof_gflops"]
    assert r5["t_mem_eff_s"] < r0["t_mem_raw_s"]
    case = {
        "name": "k", "achieved_gflops": r0["roof_gflops"] / 2,
        "achieved_gflops_lo": r0["roof_gflops"] / 2,
        "achieved_gflops_hi": r0["roof_gflops"] / 2,
    }
    graded = analyze(case, r0)
    assert 49.0 < graded["grade_pct"] < 51.0
    assert graded["grade_lo"] <= graded["grade_pct"] <= graded["grade_hi"]
    assert "verdict" in graded and graded["bound"] == r0["bound"]
