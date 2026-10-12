"""Fair-batch: fairness + power-aware continuous-batching simulator.

Novelty vs classic Orca-style continuous batching:
  * deficit-round-robin (DRR) across users (fair queueing, not FIFO),
  * DVFS-style power cap: decode clock scales down when batch > threshold,
  * KV-memory pressure admission control (no oversubscription).

Original code written for this portfolio project (2026).
Inspired by / cites:
  * Orca (OSDI '22) — continuous / iteration-level batching.
  * NanoFlow — nano-batching execution.
  * Fairness-Aware Batch Formation (arXiv 2510.14392).
  * KernelSight-LM — LLM inference simulator methodology.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import math


@dataclass
class Request:
    req_id: str
    user: str
    arrival: float          # ms, arrival time
    prompt_len: int         # prefill tokens
    decode_len: int         # tokens to generate

    def total_tokens(self) -> int:
        return self.prompt_len + self.decode_len


def jain_index(values) -> float:
    """Jain's fairness index over non-negative values.

    J = (sum x)^2 / (n * sum x^2). Returns 1.0 for empty input
    (vacuously fair) and 0.0 when all values are zero.
    """
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return 1.0
    if any(v < 0 for v in vals):
        raise ValueError("jain_index requires non-negative values")
    s = sum(vals)
    sq = sum(v * v for v in vals)
    if sq == 0:
        return 0.0
    return (s * s) / (n * sq)


def power_model(
    batch_size: int,
    threshold: int = 8,
    base_ms: float = 5.0,
    per_token_ms: float = 1.2,
    p_idle: float = 50.0,
    p_dyn: float = 200.0,
    alpha: float = 0.6,
    min_clock: float = 0.5,
) -> dict:
    """DVFS-style power model for one decode iteration.

    When batch_size > threshold the decode clock is scaled down
    (thermal / power cap), so per-iteration latency grows but peak
    power is capped by the cubic DVFS law P ~ C * f^3.

    Returns dict with clock, iter_ms, power_w, energy_j, j_per_token.
    """
    b = int(batch_size)
    if b < 0:
        raise ValueError("batch_size must be >= 0")
    if threshold <= 0:
        raise ValueError("threshold must be > 0")
    if b == 0:
        return {
            "clock": 1.0,
            "iter_ms": 0.0,
            "power_w": float(p_idle),
            "energy_j": 0.0,
            "j_per_token": 0.0,
        }
    if b <= threshold:
        clock = 1.0
    else:
        clock = max(float(min_clock), 1.0 - float(alpha) * (b - threshold) / threshold)
    iter_ms = float(base_ms) + float(per_token_ms) * b / clock
    util = min(1.0, b / float(threshold))
    power_w = float(p_idle) + float(p_dyn) * (clock ** 3) * util
    energy_j = power_w * (iter_ms / 1000.0)
    j_per_token = energy_j / float(b)
    return {
        "clock": clock,
        "iter_ms": iter_ms,
        "power_w": power_w,
        "energy_j": energy_j,
        "j_per_token": j_per_token,
    }


def _p99(values) -> float:
    if not values:
        return 0.0
    s = sorted(float(v) for v in values)
    # nearest-rank method
    k = math.ceil(0.99 * len(s)) - 1
    k = max(0, min(k, len(s) - 1))
    return s[k]


def _per_user_jain(lat_by_user: dict) -> float:
    """Fairness over users from mean latencies.

    Uses goodput proxy x_u = 1 / mean_latency_u so equal waiting -> J=1.
    """
    xs = []
    for _u, lats in lat_by_user.items():
        m = sum(lats) / len(lats) if lats else 0.0
        xs.append(1.0 / (m + 1e-9))
    return jain_index(xs)


def simulate_static(
    requests,
    batch_size: int = 4,
    prefill_ms_per_token: float = 0.5,
    power_threshold: int = 8,
) -> dict:
    """Static batching baseline (FIFO, fixed batches, run-to-completion).

    Batches are formed in arrival order; each batch starts when all its
    members have arrived and the previous batch finished. Every request
    in a batch shares the batch completion time (head-of-line blocking).
    """
    reqs = sorted(list(requests), key=lambda r: (r.arrival, r.req_id))
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    ttfts: list[float] = []
    latencies: list[float] = []
    lat_by_user: dict[str, list[float]] = defaultdict(list)
    total_energy = 0.0
    total_tokens = 0
    kv_peak = 0
    t = 0.0
    completions = []

    for i in range(0, len(reqs), batch_size):
        batch = reqs[i : i + batch_size]
        if not batch:
            continue
        start = max([t] + [r.arrival for r in batch])
        max_prompt = max(r.prompt_len for r in batch)
        prefill_ms = max_prompt * prefill_ms_per_token
        # decode: max decode steps, one iter per step
        max_dec = max(r.decode_len for r in batch)
        dec_ms = 0.0
        for _ in range(max_dec):
            pm = power_model(len(batch), threshold=power_threshold)
            dec_ms += pm["iter_ms"]
            total_energy += pm["energy_j"]
        end = start + prefill_ms + dec_ms
        kv_need = sum(r.total_tokens() for r in batch)
        kv_peak = max(kv_peak, kv_need)
        for r in batch:
            ttft = (start + r.prompt_len * prefill_ms_per_token) - r.arrival
            lat = end - r.arrival
            ttfts.append(ttft)
            latencies.append(lat)
            lat_by_user[r.user].append(lat)
            total_tokens += r.total_tokens()
            completions.append({"req_id": r.req_id, "user": r.user, "ttft": ttft, "latency": lat})
        t = end

    jain = _per_user_jain(lat_by_user)
    return {
        "scheduler": "static",
        "ttfts": ttfts,
        "latencies": latencies,
        "completions": completions,
        "jain": jain,
        "p99": _p99(latencies),
        "total_energy_j": total_energy,
        "j_per_token": (total_energy / total_tokens) if total_tokens else 0.0,
        "makespan_ms": t - (min(r.arrival for r in reqs) if reqs else 0.0),
        "kv_peak": kv_peak,
        "per_user_lat": {u: sum(v) / len(v) for u, v in lat_by_user.items()},
    }


def simulate_fair_continuous(
    requests,
    max_batch: int = 4,
    kv_capacity_tokens: int = 512,
    power_threshold: int = 8,
    quantum: int = 1,
    prefill_ms_per_token: float = 0.5,
) -> dict:
    """Fair continuous-batching with DRR + DVFS power cap + KV admission.

    * DRR across users: each user accumulates `quantum` deficit per round;
      one deficit unit admits one request (if arrived + KV fits + batch free).
    * Admission control: admit only if sum(active reserved KV) + total <= capacity
      and len(active) < max_batch. Reservation covers prompt + max decode so
      in-flight generation can never oversubscribe KV.
    * Each loop iteration runs one decode step for the whole active batch;
      step time/energy come from power_model(len(active)).
    """
    if max_batch <= 0:
        raise ValueError("max_batch must be > 0")
    if kv_capacity_tokens <= 0:
        raise ValueError("kv_capacity_tokens must be > 0")
    if quantum <= 0:
        raise ValueError("quantum must be > 0")

    waiting: deque = deque(sorted(list(requests), key=lambda r: (r.arrival, r.req_id)))
    users = sorted({r.user for r in waiting})
    if not users:
        return {
            "scheduler": "fair_continuous",
            "ttfts": [],
            "latencies": [],
            "completions": [],
            "jain": 1.0,
            "p99": 0.0,
            "total_energy_j": 0.0,
            "j_per_token": 0.0,
            "makespan_ms": 0.0,
            "kv_peak": 0,
            "per_user_lat": {},
            "num_iters": 0,
        }
    deficit: dict[str, int] = {u: 0 for u in users}
    # per-user FIFO of not-yet-admitted (kept in global waiting but filtered)
    active: list[dict] = []  # {req, generated, admit_time}
    completions: list[dict] = []
    ttfts: list[float] = []
    latencies: list[float] = []
    lat_by_user: dict[str, list[float]] = defaultdict(list)
    total_energy = 0.0
    total_tokens = sum(r.total_tokens() for r in waiting)
    t = float(waiting[0].arrival) if waiting else 0.0
    t0 = t
    kv_peak = 0
    num_iters = 0

    def kv_used() -> int:
        return sum(a["req"].prompt_len + a["generated"] for a in active)

    def kv_reserved() -> int:
        # worst-case footprint of active work (prompt + full decode each)
        return sum(a["req"].total_tokens() for a in active)

    def arrived(req) -> bool:
        return req.arrival <= t + 1e-9

    # user -> queue for quick lookup; rebuild view each admit phase is fine at sim scale
    while waiting or active:
        # If nothing active and next arrival is in the future, jump.
        if not active and waiting and waiting[0].arrival > t + 1e-9:
            t = float(waiting[0].arrival)

        # ---- admission phase (DRR rounds until no progress) ----
        admitted_this_round: list = []
        progress = True
        rounds = 0
        while progress and len(active) + len(admitted_this_round) < max_batch:
            progress = False
            rounds += 1
            if rounds > 200:  # safety for pathological inputs
                break
            for u in users:
                if len(active) + len(admitted_this_round) >= max_batch:
                    break
                deficit[u] += quantum
                # admit up to deficit while head-of-user-queue arrived & fits
                while deficit[u] > 0:
                    # find first waiting req of user u that has arrived
                    idx = None
                    for j, r in enumerate(waiting):
                        if r.user == u and arrived(r):
                            idx = j
                            break
                        if r.user == u and not arrived(r):
                            # user queue is arrival-sorted; later ones arrive later too
                            # but other users may still have arrived work -> stop for u
                            break
                    if idx is None:
                        break
                    cand = waiting[idx]
                    pending = sum(r.total_tokens() for r in admitted_this_round)
                    if kv_reserved() + pending + cand.total_tokens() > kv_capacity_tokens:
                        break  # KV pressure: wait for completions
                    # admit (remove from waiting)
                    del waiting[idx]
                    admitted_this_round.append(cand)
                    deficit[u] -= 1
                    progress = True
                    if len(active) + len(admitted_this_round) >= max_batch:
                        break
            # if no user could admit, stop
        # materialize admissions with batched prefill
        if admitted_this_round:
            max_prompt = max(r.prompt_len for r in admitted_this_round)
            # prefill clock: use power model at post-admission batch size
            pm_pre = power_model(len(active) + len(admitted_this_round), threshold=power_threshold)
            prefill_ms = max_prompt * prefill_ms_per_token / pm_pre["clock"]
            t += prefill_ms
            total_energy += pm_pre["power_w"] * (prefill_ms / 1000.0)
            for r in admitted_this_round:
                active.append({"req": r, "generated": 0, "admit_time": t})
                ttft = t - r.arrival  # includes queueing + prefill
                # store ttft at completion time (need prefill per-req prompt diff? use max-prefill batch effect:
                # per-request TTFT uses its own prompt length within the batched prefill window)
                # approximate: ttft_adj = (t - prefill_ms) + r.prompt_len*prefill/pm.clock - arrival
                ttft_adj = (t - prefill_ms) + r.prompt_len * prefill_ms_per_token / pm_pre["clock"] - r.arrival
                # stash pending ttft on the active entry
                active[-1]["ttft"] = ttft_adj
            kv_peak = max(kv_peak, kv_used())

        if not active:
            # No active and no admittable (future arrivals handled at loop top);
            # if waiting remains but none arrived (shouldn't happen after jump), advance.
            if waiting:
                t = max(t, float(waiting[0].arrival))
                continue
            break

        # ---- one decode iteration over active batch ----
        pm = power_model(len(active), threshold=power_threshold)
        t += pm["iter_ms"]
        total_energy += pm["energy_j"]
        num_iters += 1
        done_idx = []
        for k, a in enumerate(active):
            a["generated"] += 1
            if a["generated"] >= a["req"].decode_len:
                done_idx.append(k)
        kv_peak = max(kv_peak, kv_used())  # actual live KV after generating
        for k in reversed(done_idx):
            a = active.pop(k)
            r = a["req"]
            lat = t - r.arrival
            ttft = float(a.get("ttft", lat))
            ttfts.append(ttft)
            latencies.append(lat)
            lat_by_user[r.user].append(lat)
            completions.append({"req_id": r.req_id, "user": r.user, "ttft": ttft, "latency": lat})

    jain = _per_user_jain(lat_by_user)
    return {
        "scheduler": "fair_continuous",
        "ttfts": ttfts,
        "latencies": latencies,
        "completions": completions,
        "jain": jain,
        "p99": _p99(latencies),
        "total_energy_j": total_energy,
        "j_per_token": (total_energy / total_tokens) if total_tokens else 0.0,
        "makespan_ms": t - t0,
        "kv_peak": kv_peak,
        "per_user_lat": {u: sum(v) / len(v) for u, v in lat_by_user.items()},
        "num_iters": num_iters,
    }
