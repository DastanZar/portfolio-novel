"""Roofline-based latency/cost/carbon planner + SLO replica solver."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from .gpu import GPU
from .model import LLMModel

PUE = 1.15  # data-center power usage effectiveness (assumption, in README)


def predict(
    model: LLMModel,
    gpu: GPU,
    *,
    seq_len: int = 2048,
    batch: int = 8,
    util_compute: float = 0.6,
    util_bw: float = 0.7,
) -> Dict[str, Any]:
    """Predict per-token decode latency/throughput via roofline.

    Per-token work scales ~linearly with batch (each sequence streams
    weights + its KV cache). Time per batch-step = max(compute, memory),
    per-token latency = step_time, throughput = batch / step_time.
    """
    if batch < 1 or seq_len < 0:
        raise ValueError("batch must be >= 1 and seq_len >= 0")
    if not 0 < util_compute <= 1.0 or not 0 < util_bw <= 1.0:
        raise ValueError("utils must be in (0, 1]")
    flops = model.flops_per_token() * batch
    read_bytes = model.bytes_per_token(seq_len) * batch
    t_compute = flops / (gpu.tflops_bf16 * 1e12 * util_compute)
    t_mem = read_bytes / (gpu.bw_tbs * 1e12 * util_bw)
    t_step = max(t_compute, t_mem)
    bound = "compute" if t_compute >= t_mem else "memory"
    tok_per_s = batch / t_step if t_step > 0 else float("inf")
    return {
        "ms_per_token": t_step * 1e3,
        "tok_per_s": tok_per_s,
        "t_compute_ms": t_compute * 1e3,
        "t_mem_ms": t_mem * 1e3,
        "bound": bound,
        "flops_per_token": model.flops_per_token(),
        "bytes_per_token": model.bytes_per_token(seq_len),
        "batch": batch,
        "seq_len": seq_len,
    }


def cost_per_million(pred: Dict[str, Any], gpu: GPU) -> float:
    """$/1M tokens from throughput and $/hr."""
    tp = pred["tok_per_s"]
    if tp <= 0:
        raise ValueError("throughput must be positive")
    return gpu.price_per_hr / (tp * 3600.0) * 1e6


def energy_per_million(pred: Dict[str, Any], gpu: GPU, pue: float = PUE) -> float:
    """Joules per 1M tokens (board power x PUE / throughput)."""
    tp = pred["tok_per_s"]
    if tp <= 0:
        raise ValueError("throughput must be positive")
    return gpu.tdp_w * pue / tp * 1e6


def co2e_per_million(pred: Dict[str, Any], gpu: GPU, pue: float = PUE) -> float:
    """Grams CO2e per 1M tokens from energy and grid intensity."""
    joules = energy_per_million(pred, gpu, pue)
    kwh = joules / 3.6e6
    return kwh * gpu.co2_g_per_kwh


def full_plan(
    model: LLMModel, gpu: GPU, *, seq_len: int = 2048, batch: int = 8
) -> Dict[str, Any]:
    """predict + cost/energy/carbon in one dict (used by CLI + demo)."""
    pred = predict(model, gpu, seq_len=seq_len, batch=batch)
    return {
        **pred,
        "model": model.name,
        "gpu": gpu.name,
        "usd_per_1m": cost_per_million(pred, gpu),
        "joules_per_1m": energy_per_million(pred, gpu),
        "co2e_g_per_1m": co2e_per_million(pred, gpu),
    }


def replicas_for_slo(
    model: LLMModel,
    gpu: GPU,
    *,
    qps_target: float,
    p99_budget_ms: float,
    seq_len: int = 2048,
    batch: int = 8,
    tok_per_request: int = 128,
) -> Dict[str, Any]:
    """Autoscale replica solver for an SLO.

    Feasibility requires per-token latency under budget (approx p99 ~= 2x
    mean step under bursty serving — documented assumption). Per-replica
    request throughput = tok_per_s / tok_per_request. Replicas =
    ceil(qps / per_replica_qps). Infeasible SLOs return feasible=False
    with the limiting reason instead of silently rounding up.
    """
    if qps_target <= 0 or p99_budget_ms <= 0 or tok_per_request <= 0:
        raise ValueError("qps/budget/tokens must be positive")
    pred = predict(model, gpu, seq_len=seq_len, batch=batch)
    p99_est = 2.0 * pred["ms_per_token"]
    latency_ok = p99_est <= p99_budget_ms
    per_replica_qps = pred["tok_per_s"] / tok_per_request
    replicas = math.ceil(qps_target / per_replica_qps) if per_replica_qps > 0 else 0
    replicas = max(replicas, 1)
    feasible = bool(latency_ok)
    reason = "ok" if feasible else (
        f"p99-est {p99_est:.1f}ms > budget {p99_budget_ms:.1f}ms on {gpu.name}; "
        "try smaller model, bigger batch, or faster GPU")
    return {
        "model": model.name, "gpu": gpu.name,
        "qps_target": qps_target, "p99_budget_ms": p99_budget_ms,
        "p99_est_ms": p99_est, "ms_per_token": pred["ms_per_token"],
        "tok_per_s_per_replica": pred["tok_per_s"],
        "replicas": replicas, "feasible": feasible, "reason": reason,
        "usd_per_hr": replicas * gpu.price_per_hr,
        "watts": replicas * gpu.tdp_w,
    }


def fair_split(demands: Sequence[float], capacity: float) -> List[float]:
    """Max-min fair allocation across tenants.

    Classic water-filling: sort demands, raise the fair share until
    capacity runs out or every demand is met. Returns allocation per
    tenant (same order as input), summing to <= capacity.
    """
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    if any(d < 0 for d in demands):
        raise ValueError("demands must be non-negative")
    n = len(demands)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: demands[i])
    alloc = [0.0] * n
    remaining = float(capacity)
    for rank, idx in enumerate(order):
        active = n - rank
        share = remaining / active if active else 0.0
        give = min(float(demands[idx]), share)
        alloc[idx] = give
        remaining -= give
    return alloc
