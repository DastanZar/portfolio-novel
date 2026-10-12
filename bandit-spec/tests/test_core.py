import numpy as np

from bandit_spec.core import BanditSpeculator, expected_tokens, slo_guard, thompson_pick


def test_expected_tokens_edges():
    assert expected_tokens(0.0, 5) == 1.0
    assert expected_tokens(1.0, 5) == 6.0
    assert expected_tokens(0.5, 1) == 1.5
    # monotonic in alpha and K
    assert expected_tokens(0.9, 5) > expected_tokens(0.5, 5)
    assert expected_tokens(0.8, 5) > expected_tokens(0.8, 2)


def test_thompson_pick_cost_sensitivity():
    rng = np.random.default_rng(0)
    Ks = [1, 2, 3, 4, 5, 6, 7]
    # strong posteriors everywhere -> cheap cost favors large K, dear cost small K
    a = [20.0] * 7
    b = [5.0] * 7
    picks_cheap = [thompson_pick(rng, a, b, Ks, 0.0) for _ in range(30)]
    rng2 = np.random.default_rng(0)
    picks_dear = [thompson_pick(rng2, a, b, Ks, 2.0) for _ in range(30)]
    assert sum(picks_cheap) / len(picks_cheap) > sum(picks_dear) / len(picks_dear)


def test_update_posterior_math():
    s = BanditSpeculator(seed=0)
    s.observe(3, n_accepted=4, n_proposed=5, latency_ms=10.0)
    assert s.a[3] == 1.0 + 4
    assert s.b[3] == 1.0 + 1
    assert abs(s.posterior_mean(3) - 5 / 7) < 1e-12
    s.update(3, 1, 2)
    assert s.a[3] == 6.0 and s.b[3] == 3.0


def test_slo_guard_veto_and_pass():
    assert slo_guard([10.0] * 50, budget_ms=50.0, current_K=5) == 5
    assert slo_guard([100.0] * 50, budget_ms=50.0, current_K=5) < 5
    assert slo_guard([], budget_ms=50.0, current_K=5) == 5
    # far-over-budget shrinks faster than mild
    mild = slo_guard([60.0] * 50, budget_ms=50.0, current_K=5)
    far = slo_guard([200.0] * 50, budget_ms=50.0, current_K=5)
    assert far <= mild


def test_bandit_learns_high_acceptance_arm():
    s = BanditSpeculator(seed=1, cost=0.05)
    # arm K=6 looks great, K=1 looks poor
    for _ in range(30):
        s.observe(6, 9, 10)
        s.observe(1, 2, 10)
    picks = [s.pick() for _ in range(50)]
    assert sum(picks) / len(picks) > 3.0


def test_pick_observe_loop_runs_and_bounds():
    s = BanditSpeculator(seed=2, cost=0.15, slo_budget_ms=25.0)
    rng = np.random.default_rng(2)
    for _ in range(50):
        K = s.pick()
        assert 1 <= K <= 7
        acc = int(rng.binomial(K, 0.7))
        s.observe(K, acc, K, latency_ms=float(8 + 2 * K + rng.normal()))
    assert len(s.history) == 50
