"""7 tests for the green-roof planner."""
from __future__ import annotations

import math

from green_roof.gpu import GPU_TABLE, get_gpu
from green_roof.model import MODEL_ZOO, get_model
from green_roof.planner import (
    co2e_per_million,
    cost_per_million,
    energy_per_million,
    fair_split,
    full_plan,
    predict,
    replicas_for_slo,
)


def test_predict_positive_and_bound():
    m, g = get_model("llama-3-8b"), get_gpu("A100-80")
    p = predict(m, g, seq_len=2048, batch=8)
    assert p["tok_per_s"] > 0 and p["ms_per_token"] > 0
    assert p["bound"] in ("compute", "memory")
    assert p["bytes_per_token"] == m.bytes_per_token(2048)


def test_decode_is_memory_bound_small_batch():
    m, g = get_model("llama-3-70b"), get_gpu("H100")
    p = predict(m, g, seq_len=2048, batch=1)
    # 70B decode at batch 1 streams ~140GB weights per token: memory-bound.
    assert p["bound"] == "memory"
    assert p["t_mem_ms"] >= p["t_compute_ms"]


def test_cost_math_matches_formula():
    m, g = get_model("mistral-7b"), get_gpu("L4")
    p = predict(m, g)
    expect = g.price_per_hr / (p["tok_per_s"] * 3600.0) * 1e6
    assert math.isclose(cost_per_million(p, g), expect, rel_tol=1e-12)
    assert cost_per_million(p, g) > 0


def test_energy_math_matches_formula():
    m, g = get_model("llama-3-8b"), get_gpu("A10")
    p = predict(m, g)
    expect_j = g.tdp_w * 1.15 / p["tok_per_s"] * 1e6
    assert math.isclose(energy_per_million(p, g), expect_j, rel_tol=1e-12)
    expect_g = expect_j / 3.6e6 * g.co2_g_per_kwh
    assert math.isclose(co2e_per_million(p, g), expect_g, rel_tol=1e-12)


def test_replicas_scale_with_qps():
    m, g = get_model("llama-3-8b"), get_gpu("A100-80")
    r1 = replicas_for_slo(m, g, qps_target=5, p99_budget_ms=10_000)
    r2 = replicas_for_slo(m, g, qps_target=50, p99_budget_ms=10_000)
    assert r1["feasible"] and r2["feasible"]
    assert r2["replicas"] >= r1["replicas"]
    assert r2["replicas"] > r1["replicas"]  # 10x load must need more replicas
    assert r2["usd_per_hr"] == r2["replicas"] * g.price_per_hr


def test_tight_slo_is_infeasible_with_reason():
    m, g = get_model("llama-3-70b"), get_gpu("T4")
    r = replicas_for_slo(m, g, qps_target=1, p99_budget_ms=1.0)  # impossibly tight
    assert r["feasible"] is False
    assert r["replicas"] >= 1
    assert "budget" in r["reason"]


def test_fair_split_max_min():
    # Small demand fully met, remainder shared equally.
    alloc = fair_split([2.0, 8.0, 10.0], 12.0)
    assert alloc[0] == 2.0
    assert math.isclose(alloc[1], alloc[2], rel_tol=1e-9)
    assert math.isclose(sum(alloc), 12.0, rel_tol=1e-9)
    # Over-demand: equal shares; under-demand: everyone met.
    assert fair_split([10.0, 10.0, 10.0], 15.0) == [5.0, 5.0, 5.0]
    assert fair_split([1.0, 2.0], 10.0) == [1.0, 2.0]
    assert fair_split([], 5.0) == []
    # Never exceeds capacity or any demand.
    a = fair_split([3.0, 3.0, 3.0, 3.0], 10.0)
    assert all(v <= 3.0 for v in a) and sum(a) <= 10.0 + 1e-9


def test_full_plan_keys_and_tables_nonempty():
    assert len(GPU_TABLE) >= 8 and len(MODEL_ZOO) >= 4
    plan = full_plan(get_model("phi-3-mini"), get_gpu("T4"))
    for key in ("tok_per_s", "usd_per_1m", "joules_per_1m", "co2e_g_per_1m"):
        assert plan[key] > 0
