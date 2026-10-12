"""Demo: kurtosis-switched 4-bit quantization receipt + chart (matplotlib Agg)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kurt_switch.core import compute_row_stats, quantize_row
RECEIPTS = ROOT / "receipts"
ASSETS = ROOT / "assets"


def build_rows(seed=0, n_gauss=120, n_sparse=40, n_skew=40, dim=64):
    rng = np.random.RandomState(seed)
    gauss = rng.randn(n_gauss, dim)
    sparse = np.zeros((n_sparse, dim))
    for i in range(n_sparse):
        k = rng.randint(1, 5)
        cols = rng.choice(dim, size=k, replace=False)
        sparse[i, cols] = rng.randn(k) * 6.0
    skew = rng.standard_gamma(2.0, size=(n_skew, dim)).astype(float)
    # center-ish but keep skew: subtract mean so rows straddle zero yet stay skewed
    skew = skew - 2.0
    # a few left-skewed rows
    skew[: n_skew // 2] *= -1.0
    block = np.vstack([gauss, sparse, skew])
    kinds = (["gauss"] * n_gauss) + (["sparse"] * n_sparse) + (["skew"] * n_skew)
    return block, kinds


def main():
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    block, kinds = build_rows(seed=0)
    kurt = []
    adapt_mses, uni_mses, picks = [], [], []
    for row in block:
        st = compute_row_stats(row)
        kurt.append(float(st["kurtosis"]))
        a = quantize_row(row)  # adaptive switch
        u = quantize_row(row, codebook="uniform16")
        adapt_mses.append(a["mse"])
        uni_mses.append(u["mse"])
        picks.append(a["codebook"])
    adapt_mses = np.array(adapt_mses)
    uni_mses = np.array(uni_mses)
    kurt = np.array(kurt)
    bins = [(-1e9, 0.0, "kurt<0"), (0.0, 3.0, "0-3"), (3.0, 10.0, "3-10"), (10.0, 1e9, ">10")]
    per_bin = {}
    win_rates, labels = [], []
    for lo, hi, label in bins:
        m = (kurt >= lo) & (kurt < hi)
        labels.append(label)
        if int(m.sum()) == 0:
            per_bin[label] = {"n": 0, "win_rate": 0.0}
            win_rates.append(0.0)
            continue
        wins = np.mean(adapt_mses[m] < uni_mses[m] - 1e-12)
        ties = np.mean(np.abs(adapt_mses[m] - uni_mses[m]) <= 1e-12)
        per_bin[label] = {"n": int(m.sum()), "win_rate": float(wins),
                          "tie_rate": float(ties),
                          "adapt_mse": float(np.mean(adapt_mses[m])),
                          "uniform_mse": float(np.mean(uni_mses[m]))}
        win_rates.append(float(wins))
    overall = {
        "seed": 0, "n_rows": int(block.shape[0]), "dim": int(block.shape[1]),
        "adaptive_mse": float(np.mean(adapt_mses)),
        "uniform_mse": float(np.mean(uni_mses)),
        "win_rate": float(np.mean(adapt_mses < uni_mses - 1e-12)),
        "codebook_counts": {c: int(np.sum(np.array(picks) == c)) for c in ("uniform16", "odd16", "shifted")},
        "per_bin": per_bin,
    }
    overall["improvement_pct"] = float(100.0 * (overall["uniform_mse"] - overall["adaptive_mse"]) / overall["uniform_mse"])
    with open(RECEIPTS / "receipt.json", "w") as f:
        json.dump(overall, f, indent=2)

    # Punchy hero chart: adaptive win-rate vs kurtosis bins.
    fig, ax = plt.subplots(figsize=(8, 4.6))
    fig.patch.set_facecolor("#0b1020")
    ax.set_facecolor("#0b1020")
    colors = ["#22d3ee", "#a78bfa", "#fb7185", "#fbbf24"]
    bars = ax.bar(labels, [w * 100 for w in win_rates], color=colors, edgecolor="white", linewidth=1.2, width=0.62)
    for b, w, lab in zip(bars, win_rates, labels):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{w*100:.1f}%",
                ha="center", va="bottom", color="white", fontsize=11, fontweight="bold")
        n = per_bin[lab]["n"]
        ax.text(b.get_x() + b.get_width() / 2, 4, f"n={n}", ha="center", va="bottom", color="white", fontsize=9, alpha=0.85)
    ax.set_ylim(0, 115)
    ax.set_ylabel("adaptive beats uniform (%)", color="white", fontsize=11)
    ax.set_xlabel("excess kurtosis bin", color="white", fontsize=11)
    ax.set_title("kurt-switch: adaptive codebook win-rate vs kurtosis  •  "
                 f"overall MSE {overall['adaptive_mse']:.4f} vs {overall['uniform_mse']:.4f} "
                 f"({overall['improvement_pct']:.1f}% better)", color="white", fontsize=11, fontweight="bold", pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(ASSETS / "switch_map.png", dpi=150, facecolor=fig.get_facecolor())
    print(json.dumps(overall, indent=2))
    print(f"wrote {RECEIPTS/'receipt.json'} and {ASSETS/'switch_map.png'}")


if __name__ == "__main__":
    main()
