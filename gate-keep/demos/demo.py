"""gate-keep demo: learned gate + latent memory beats pure dropping.

Generates synthetic KV + attention stats, fits the ridge gate, sweeps
budgets, compares drop-vs-memory reconstruction error, saves
assets/gate_gain.png and receipts/receipt.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from gate_keep.compress import (
    compress_with_memory,
    drop_error,
    fit_gate_ridge,
    gate_predict,
    reconstruction_error,
    score_tokens,
)

ASSETS = ROOT / "assets"
RECEIPTS = ROOT / "receipts"

def build_synthetic(n: int = 256, d: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    K = rng.normal(size=(n, d))
    V = rng.normal(size=(n, d))
    # Fake attention: heavy hitters at sinks + a few mid positions, diffuse tail.
    n_q = 32
    attn = np.abs(rng.normal(scale=0.2, size=(n_q, n)))
    attn[:, 0:2] += 2.0  # sinks (StreamingLLM)
    attn[:, 40:45] += 1.2  # heavy hitters (H2O)
    attn[:, 120:126] += 0.8
    attn = attn / attn.sum(axis=1, keepdims=True)
    return K, V, attn


def stat_features(attn: np.ndarray) -> np.ndarray:
    # Past-stat features per key: mean, max, recency ramp, position norm.
    mean = attn.mean(axis=0)
    mx = attn.max(axis=0)
    n = attn.shape[1]
    pos = np.arange(n) / max(n - 1, 1)
    recency = np.linspace(0, 1, n)  # recent tokens get higher prior
    X = np.stack([mean, mx, recency, pos], axis=1)
    # standardize for stable ridge
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    return (X - mu) / sd


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    K, V, attn = build_synthetic()
    scores_mass = score_tokens(attn)
    X = stat_features(attn)
    # Future importance: same mass + small shift (gate must learn mapping).
    rng = np.random.default_rng(7)
    y_future = scores_mass + 0.05 * rng.normal(size=scores_mass.shape)
    model = fit_gate_ridge(X, y_future, l2=1e-3, hidden=16, seed=0)
    scores_gate = gate_predict(model, X)

    budgets = [8, 16, 32, 64]
    rows = []
    for b in budgets:
        comp = compress_with_memory(
            K, V, scores_gate, budget=b, mem_dim=8, sink_keep=2, seed=0
        )
        e_mem = reconstruction_error(K, V, comp)
        e_drop = drop_error(K, V, comp["kept_idx"])
        gain = (e_drop - e_mem) / (e_drop + 1e-12)
        rows.append({"budget": b, "drop": float(e_drop), "memory": float(e_mem), "gain": float(gain)})
        print(f"budget={b:3d} drop={e_drop:.4f} memory={e_mem:.4f} gain={gain*100:.1f}%")

    # Plot
    x = np.array([r["budget"] for r in rows])
    drop = np.array([r["drop"] for r in rows])
    mem = np.array([r["memory"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, drop, "o--", label="drop tail (H2O-style)")
    ax.plot(x, mem, "s-", label="gate-keep + 8-d memory")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, [str(v) for v in x])
    ax.set_xlabel("KV budget (tokens kept, +1 memory key)")
    ax.set_ylabel("attention-output error (normed by value RMS)")
    ax.set_title("gate-keep: latent memory beats pure dropping")
    ax.legend()
    ax.grid(True, alpha=0.3)
    for r in rows:
        ax.annotate(
            f"{r['gain']*100:.0f}% gain",
            (r["budget"], r["memory"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    out_png = ASSETS / "gate_gain.png"
    fig.savefig(out_png, dpi=150)
    print(f"saved {out_png}")

    receipt = {
        "project": "gate-keep",
        "n": 256,
        "d": 32,
        "mem_dim": 8,
        "sink_keep": 2,
        "hidden": 16,
        "l2": 1e-3,
        "train_mse": float(model["train_mse"]),
        "rows": rows,
        "assets": ["assets/gate_gain.png"],
        "method": "ridge gate (closed-form) + orthonormal sketch memory",
    }
    (RECEIPTS / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print("saved receipts/receipt.json")


if __name__ == "__main__":
    main()
