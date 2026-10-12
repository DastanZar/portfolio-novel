"""Demo: Thompson bandit vs fixed-K drafting + SLO guard, race + regret plot."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bandit_spec.core import BanditSpeculator, expected_tokens

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def simulate(T=300, seed=0):
    rng = np.random.default_rng(seed)
    Ks = list(range(1, 8))
    # ground truth per-token acceptance decays with K (longer drafts harder)
    true_alpha = {k: 0.88 - 0.035 * k for k in Ks}
    cost = 0.15
    # oracle: best fixed-K by true reward
    oracle_K = max(Ks, key=lambda k: expected_tokens(true_alpha[k], k) / (1 + cost * k))

    bandit = BanditSpeculator(cost=cost, slo_budget_ms=28.0, seed=seed)
    cum_bandit, cum_fixed, cum_oracle = [], [], []
    regret = []
    tot_b = tot_f3 = tot_o = 0.0
    fixed_K = 3
    lat_base = 6.0
    for t in range(T):
        K = bandit.pick()
        # sample accepted run: geometric-like via binomial proxy
        acc_b = int(rng.binomial(K, true_alpha[K]))
        lat = lat_base + 2.0 * K + rng.normal(0, 1.5)
        # SLO spike regime in middle third
        if 100 <= t < 200:
            lat += 12.0
        bandit.observe(K, acc_b, K, latency_ms=max(1.0, lat))
        rew_b = (acc_b + 1) / (1 + cost * K)
        acc_f = int(rng.binomial(fixed_K, true_alpha[fixed_K]))
        rew_f = (acc_f + 1) / (1 + cost * fixed_K)
        acc_o = int(rng.binomial(oracle_K, true_alpha[oracle_K]))
        rew_o = (acc_o + 1) / (1 + cost * oracle_K)
        tot_b += rew_b
        tot_f3 += rew_f
        tot_o += rew_o
        cum_bandit.append(tot_b)
        cum_fixed.append(tot_f3)
        cum_oracle.append(tot_o)
        regret.append(tot_o - tot_b)
    return {
        "steps": list(range(T)),
        "cum_bandit": cum_bandit,
        "cum_fixed3": cum_fixed,
        "cum_oracle": cum_oracle,
        "regret": regret,
        "oracle_K": oracle_K,
        "picks": [h["K"] for h in bandit.history],
    }


def main():
    os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "receipts"), exist_ok=True)
    out = simulate()
    steps = out["steps"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(steps, out["cum_oracle"], label=f"oracle K={out['oracle_K']}", ls="--", color="black")
    ax1.plot(steps, out["cum_bandit"], label="bandit-spec (ours)", color="tab:red")
    ax1.plot(steps, out["cum_fixed3"], label="fixed K=3", color="tab:blue", alpha=0.7)
    ax1.set_xlabel("step")
    ax1.set_ylabel("cumulative reward")
    ax1.set_title("Draft-length race: bandit vs fixed")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(steps, out["regret"], color="tab:red")
    ax2.set_xlabel("step")
    ax2.set_ylabel("regret vs oracle")
    ax2.set_title("Regret curve (sublinear = learning)")
    ax2.grid(alpha=0.3)
    fig.suptitle("Bandit speculative decoding: Thompson sampling + SLO guard")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "assets/bandit_race.png"), dpi=150)

    picks = out.pop("picks")
    receipt = {k: v for k, v in out.items() if k != "steps"}
    receipt["final_cum_bandit"] = out["cum_bandit"][-1]
    receipt["final_cum_fixed3"] = out["cum_fixed3"][-1]
    receipt["final_regret"] = out["regret"][-1]
    receipt["mean_pick"] = float(np.mean(picks))
    receipt["artifact"] = "assets/bandit_race.png"
    with open(os.path.join(ROOT, "receipts/receipt.json"), "w") as f:
        json.dump(receipt, f, indent=2)
    print("wrote assets/bandit_race.png + receipts/receipt.json")
    print(f"oracle_K={out['oracle_K']} mean_pick={np.mean(picks):.2f} "
          f"final_regret={out['regret'][-1]:.1f}")


if __name__ == "__main__":
    main()
