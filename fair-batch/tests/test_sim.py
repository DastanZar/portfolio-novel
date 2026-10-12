"""Tests for fair-batch simulator (6 tests)."""
import math

from fair_batch.sim import (
    Request,
    jain_index,
    power_model,
    simulate_static,
    simulate_fair_continuous,
)


def _mk(req_id, user, arrival, prompt=8, dec=4):
    return Request(req_id=req_id, user=user, arrival=float(arrival), prompt_len=prompt, decode_len=dec)


def test_jain_perfect_fairness_is_one():
    assert jain_index([5.0, 5.0, 5.0, 5.0]) == 1.0


def test_jain_unfair_less_than_one():
    j = jain_index([10.0, 1.0, 1.0, 1.0])
    assert 0.0 < j < 1.0


def test_power_model_clock_drops_over_threshold():
    lo = power_model(4, threshold=8)
    hi = power_model(16, threshold=8)
    assert lo["clock"] == 1.0
    assert hi["clock"] < 1.0
    # DVFS cubic: capped clock must not explode peak power per token unboundedly;
    # iteration gets slower under cap
    assert hi["iter_ms"] > power_model(8, threshold=8)["iter_ms"]
    assert hi["power_w"] > 0 and hi["j_per_token"] > 0


def test_static_completes_all_with_sane_times():
    reqs = [_mk(f"r{i}", "alice" if i % 2 == 0 else "bob", arrival=i * 2) for i in range(6)]
    out = simulate_static(reqs, batch_size=2)
    assert len(out["latencies"]) == 6
    assert all(v >= 0 for v in out["ttfts"])
    assert all(v >= 0 for v in out["latencies"])
    assert out["p99"] >= 0 and out["j_per_token"] > 0
    assert 0.0 <= out["jain"] <= 1.0


def test_fair_continuous_respects_kv_and_completes():
    reqs = [_mk(f"r{i}", "alice" if i % 2 == 0 else "bob", arrival=0, prompt=16, dec=4) for i in range(8)]
    cap = 64  # tokens; each req reserves 20 -> forces admission control
    out = simulate_fair_continuous(reqs, max_batch=8, kv_capacity_tokens=cap, power_threshold=8)
    assert len(out["latencies"]) == len(reqs)
    assert out["kv_peak"] <= cap  # reservation-based admission: never oversubscribed
    assert out["p99"] > 0 and out["j_per_token"] > 0


def test_fair_drr_interleaves_and_is_fairer_under_skew():
    # Skewed: alice floods at t=0, bob arrives at t=0 too but static FIFO serves alice first.
    reqs = [_mk(f"a{i}", "alice", arrival=0) for i in range(6)] + [_mk(f"b{i}", "bob", arrival=0) for i in range(2)]
    static = simulate_static(reqs, batch_size=2)
    fair = simulate_fair_continuous(reqs, max_batch=2, kv_capacity_tokens=512, power_threshold=8, quantum=1)
    assert len(fair["latencies"]) == 8
    # DRR must interleave: at least one bob request completes before the last alice one
    order = [c["req_id"] for c in fair["completions"]]
    last_alice = max(order.index(c) for c in order if c.startswith("a"))
    first_bob = min(order.index(c) for c in order if c.startswith("b"))
    assert first_bob < last_alice
    # fairness should not be worse than FIFO under skew
    assert fair["jain"] >= static["jain"] - 1e-9
