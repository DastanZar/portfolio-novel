"""Demo: norm-shelled grouped VQ receipt + chart (matplotlib Agg)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shell_vq.core import (
    build_shell_codebook,
    quantize_shelled,
    rate_distortion_floor,
    residual_refine,
)

RECEIPTS = ROOT / "receipts"
ASSETS = ROOT / "assets"


def flat_baseline(V, C):
    norms = np.linalg.norm(V, axis=1)
    s = float(np.mean(norms))
    normed = V / s
    d2 = (np.sum(normed ** 2, axis=1, keepdims=True)
          + np.sum(C ** 2, axis=1)[None, :]
          - 2.0 * normed @ C.T)
    idx = np.argmin(d2, axis=1)
    c = C[idx]
    s_ls = float(np.sum(V * c) / np.sum(c * c))
    recon = s_ls * c
    return float(np.mean((V - recon) ** 2)), recon


def main():
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    seed, n, dim = 7, 500, 8
    rng = np.random.RandomState(seed)
    V = rng.randn(n, dim)
    V[350:] *= 3.0
    C = build_shell_codebook(dim, 64, seed=0)
    out = quantize_shelled(V, C)
    mask = out["shells"] == 2
    ref = residual_refine(V, out["reconstruction"], mask)
    flat_mse, _ = flat_baseline(V, C)
    var = float(np.mean(V ** 2))
    floor = rate_distortion_floor(var, dim, 6.0 + 2.0 / 3.0)
    receipt = {
        "seed": seed, "n_vectors": n, "dim": dim, "codebook_points": 64,
        "thresholds": [out["thresholds"][0], out["thresholds"][1]],
        "counts": {str(k): v for k, v in out["counts"].items()},
        "scales": {str(k): v for k, v in out["scales"].items()},
        "per_shell_mse": {str(k): v for k, v in out["per_shell_mse"].items()},
        "shelled_mse": out["mse"],
        "refined_mse": ref["mse_after"],
        "outer_mse_before": ref["outer_mse_before"],
        "outer_mse_after": ref["outer_mse_after"],
        "flat_mse": flat_mse,
        "rd_floor_6p67b": floor,
        "variance": var,
    }
    receipt["improvement_vs_flat_pct"] = float(100.0 * (flat_mse - ref["mse_after"]) / flat_mse)
    with open(RECEIPTS / "receipt.json", "w") as f:
        json.dump(receipt, f, indent=2)

    norms = np.linalg.norm(V, axis=1)
    t1, t2 = out["thresholds"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    fig.patch.set_facecolor("#0b1020")
    for ax in axes:
        ax.set_facecolor("#0b1020")
    # left: norm histogram with shell cuts
    axes[0].hist(norms, bins=40, color="#22d3ee", edgecolor="white", linewidth=0.4, alpha=0.9)
    axes[0].axvline(t1, color="#fbbf24", linewidth=2.2, label=f"t1={t1:.2f}")
    axes[0].axvline(t2, color="#fb7185", linewidth=2.2, label=f"t2={t2:.2f}")
    axes[0].set_title("radius shells from norm quantiles", color="white", fontweight="bold")
    axes[0].set_xlabel("vector L2 norm", color="white")
    axes[0].set_ylabel("count", color="white")
    axes[0].legend(facecolor="#111827", labelcolor="white")
    # right: per-shell MSE bars
    labels = ["inner", "mid", "outer"]
    shelled = [out["per_shell_mse"][s] for s in (0, 1, 2)]
    # flat per-shell mse for comparison
    _, flat_recon = flat_baseline(V, C)
    flat_shell = []
    for s in (0, 1, 2):
        m = out["shells"] == s
        flat_shell.append(float(np.mean((V[m] - flat_recon[m]) ** 2)))
    x = np.arange(3)
    axes[1].bar(x - 0.2, flat_shell, width=0.4, label="flat", color="#64748b", edgecolor="white")
    axes[1].bar(x + 0.2, shelled, width=0.4, label="shelled", color=["#22d3ee", "#a78bfa", "#fb7185"], edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title(f"per-shell MSE  •  flat {flat_mse:.3f} vs shelled+refined {ref['mse_after']:.3f}", color="white", fontweight="bold")
    axes[1].set_ylabel("MSE", color="white")
    axes[1].legend(facecolor="#111827", labelcolor="white")
    for ax in axes:
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(ASSETS / "shells.png", dpi=150, facecolor=fig.get_facecolor())
    print(json.dumps(receipt, indent=2))
    print(f"wrote {RECEIPTS/'receipt.json'} and {ASSETS/'shells.png'}")


if __name__ == "__main__":
    main()
