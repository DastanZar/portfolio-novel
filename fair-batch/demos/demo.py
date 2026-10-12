"""Demo: skewed multi-user workload -> static vs fair comparison.

Outputs:
  receipts/receipt.json
  assets/fairness.png  (TTFT scatter per user, static vs fair)
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fair_batch.sim import Request, simulate_static, simulate_fair_continuous

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def build_workload(seed: int = 7):
    rng = random.Random(seed)
    reqs = []
    # alice floods, bob/bonus light — classic fairness stress
    for i in range(14):
        reqs.append(Request(f"alice-{i}", "alice", arrival=rng.uniform(0, 10),
                            prompt_len=rng.randint(8, 32), decode_len=rng.randint(4, 12)))
    for i in range(5):
        reqs.append(Request(f"bob-{i}", "bob", arrival=rng.uniform(0, 10),
                            prompt_len=rng.randint(8, 24), decode_len=rng.randint(4, 10)))
    for i in range(3):
        reqs.append(Request(f"cara-{i}", "cara", arrival=rng.uniform(2, 12),
                            prompt_len=rng.randint(8, 16), decode_len=rng.randint(2, 6)))
    return reqs


def main():
    reqs = build_workload()
    static = simulate_static(reqs, batch_size=4, power_threshold=8)
    fair = simulate_fair_continuous(reqs, max_batch=4, kv_capacity_tokens=256,
                                    power_threshold=8, quantum=1)
    os.makedirs(os.path.join(ROOT, "receipts"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True)
    receipt = {
        "workload": {"n": len(reqs), "users": sorted({r.user for r in reqs})},
        "static": {k: static[k] for k in ("jain", "p99", "j_per_token", "makespan_ms", "total_energy_j")},
        "fair": {k: fair[k] for k in ("jain", "p99", "j_per_token", "makespan_ms", "total_energy_j")},
        "per_user_lat_static": static["per_user_lat"],
        "per_user_lat_fair": fair["per_user_lat"],
        "config": {"max_batch": 4, "kv_capacity": 256, "power_threshold": 8, "quantum": 1},
    }
    with open(os.path.join(ROOT, "receipts", "receipt.json"), "w") as f:
        json.dump(receipt, f, indent=2)

    # TTFT scatter: x = completion order index, y = TTFT, color = user
    users = sorted({c["user"] for c in static["completions"]})
    cmap = {u: c for u, c in zip(users, plt.cm.tab10.colors)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, res, title in ((axes[0], static, f"static  Jain={static['jain']:.3f} p99={static['p99']:.1f}ms"),
                           (axes[1], fair, f"fair-DRR Jain={fair['jain']:.3f} p99={fair['p99']:.1f}ms")):
        for i, c in enumerate(res["completions"]):
            ax.scatter(i, c["ttft"], color=cmap[c["user"]], s=26, alpha=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("completion order")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("TTFT (ms)")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap[u], markersize=8, label=u)
               for u in users]
    fig.legend(handles=handles, loc="upper center", ncol=len(users), fontsize=9)
    fig.suptitle("TTFT vs completion order — static FIFO vs fair DRR + DVFS cap + KV admission")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(ROOT, "assets", "fairness.png"), dpi=130)
    print(json.dumps(receipt, indent=2))
    print("wrote receipts/receipt.json + assets/fairness.png")


if __name__ == "__main__":
    main()
