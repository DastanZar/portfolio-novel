"""Thompson-sampling bandit over speculative-decoding draft lengths.

Novelty vs fixed analytic argmax: each draft length K keeps its own
Beta posterior over per-token acceptance; each step samples
theta_K ~ Beta(a_K, b_K) and picks argmax expected_tokens(theta_K, K)
/ (1 + c*K). A p99-latency SLO guard vetoes/shrinks K when the tail
exceeds budget. This learns non-stationary acceptance where a closed
form with a single global alpha cannot.
"""

from __future__ import annotations

import numpy as np

KS_DEFAULT = tuple(range(1, 8))


def expected_tokens(alpha: float, K: int) -> float:
    """Expected accepted tokens for draft length K, per-token accept prob alpha.

    Geometric model: E = 1 + a + a^2 + ... + a^K = (1 - a^{K+1}) / (1 - a).
    """
    if K < 0:
        raise ValueError("K must be >= 0")
    a = float(alpha)
    if a >= 1.0 - 1e-12:
        return float(K + 1)
    if a <= 0.0:
        return 1.0
    a = min(max(a, 0.0), 1.0)
    return (1.0 - a ** (K + 1)) / (1.0 - a)


def thompson_pick(rng: np.random.Generator, a_posts, b_posts, Ks, cost_c: float) -> int:
    """Sample acceptance per K from Beta posteriors; pick max reward.

    reward(K) = expected_tokens(theta_K, K) / (1 + c*K).
    """
    Ks = list(Ks)
    best_K, best_r = Ks[0], -1.0
    for K, a, b in zip(Ks, a_posts, b_posts):
        theta = float(rng.beta(max(a, 1e-9), max(b, 1e-9)))
        r = expected_tokens(theta, int(K)) / (1.0 + cost_c * int(K))
        if r > best_r:
            best_r, best_K = r, int(K)
    return best_K


def slo_guard(latencies_ms, budget_ms: float, current_K: int, min_K: int = 1) -> int:
    """p99 tail veto: shrink K by 1 when p99 latency exceeds budget."""
    lat = np.asarray(list(latencies_ms), dtype=np.float64)
    lat = lat[np.isfinite(lat)]
    if lat.size == 0 or budget_ms is None:
        return int(current_K)
    p99 = float(np.quantile(lat, 0.99))
    if p99 > float(budget_ms):
        # far over budget -> shrink faster
        step = 2 if p99 > 1.5 * float(budget_ms) else 1
        return max(int(min_K), int(current_K) - step)
    return int(current_K)


class BanditSpeculator:
    """Beta--Bernoulli Thompson bandit over draft lengths + SLO guard."""

    def __init__(self, ks=KS_DEFAULT, cost: float = 0.15,
                 prior_a: float = 1.0, prior_b: float = 1.0,
                 slo_budget_ms: float | None = None, seed: int = 0,
                 latency_window: int = 200):
        self.ks = [int(k) for k in ks]
        if not self.ks:
            raise ValueError("ks must be non-empty")
        self.cost = float(cost)
        self.prior_a = float(prior_a)
        self.prior_b = float(prior_b)
        self.a = {k: float(prior_a) for k in self.ks}
        self.b = {k: float(prior_b) for k in self.ks}
        self.slo_budget_ms = slo_budget_ms
        self.rng = np.random.default_rng(seed)
        self.latencies: list[float] = []
        self.latency_window = int(latency_window)
        self.history: list[dict] = []

    def posterior_mean(self, K: int) -> float:
        a, b = self.a[int(K)], self.b[int(K)]
        return a / (a + b)

    def pick(self) -> int:
        a_posts = [self.a[k] for k in self.ks]
        b_posts = [self.b[k] for k in self.ks]
        K = thompson_pick(self.rng, a_posts, b_posts, self.ks, self.cost)
        if self.slo_budget_ms is not None and self.latencies:
            K = slo_guard(self.latencies, self.slo_budget_ms, K)
        return int(K)

    def observe(self, K: int, n_accepted: int, n_proposed: int,
                latency_ms: float | None = None) -> None:
        K = int(K)
        if K not in self.a:
            raise ValueError(f"unknown K={K}")
        n_accepted = max(0, min(int(n_accepted), int(n_proposed)))
        n_rejected = int(n_proposed) - n_accepted
        self.a[K] += n_accepted
        self.b[K] += n_rejected
        if latency_ms is not None and np.isfinite(latency_ms):
            self.latencies.append(float(latency_ms))
            if len(self.latencies) > self.latency_window:
                self.latencies = self.latencies[-self.latency_window :]
        self.history.append({"K": K, "acc": n_accepted, "n": int(n_proposed),
                             "lat": latency_ms})

    # alias required by spec wording
    def update(self, K: int, n_accepted: int, n_proposed: int,
               latency_ms: float | None = None) -> None:
        self.observe(K, n_accepted, n_proposed, latency_ms)
